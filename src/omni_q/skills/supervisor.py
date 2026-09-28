"""Hard safety boundary between skill proposals and actuator access."""

from __future__ import annotations

import math

from .contracts import (
    ControllerType,
    ProposedAction,
    SkillManifest,
    SkillObservation,
    SkillRequest,
    SkillStage,
    SupervisorDecision,
    SupervisorReceipt,
)

__all__ = ["SkillSupervisor"]


class SkillSupervisor:
    """Validate every controller proposal immediately before actuation.

    The supervisor is intentionally stateless.  It consumes a revision-bound
    request, a scoped observation, a promoted manifest, and one proposal.  It
    returns a receipt plus a safe action; it never calls an actuator and never
    changes a goal, constraint, or skill selection.
    """

    _TRANSLATIONS = {
        "dx_mm": ("x", "linear"),
        "dy_mm": ("y", "linear"),
        "dz_mm": ("z", "linear"),
        "droll_deg": ("roll", "rotation"),
        "dpitch_deg": ("pitch", "rotation"),
        "dyaw_deg": ("yaw", "rotation"),
        "gripper_delta": ("gripper", "gripper"),
    }

    def evaluate(
        self,
        request: SkillRequest,
        manifest: SkillManifest,
        observation: SkillObservation,
        proposal: ProposedAction,
    ) -> SupervisorReceipt:
        requested = tuple(proposal.values)
        base = dict(
            request_id=request.request_id,
            skill_id=manifest.skill_id,
            org_id=request.org_id,
            world_revision=observation.world_revision,
            requested_action=requested,
            controller_backend=proposal.controller_backend,
            fallback_skill_id=manifest.fallback_skill_id,
        )

        if request.org_id != manifest.org_id or request.org_id != observation.org_id:
            return self._denied(base, "organization scope mismatch")
        if observation.arm_id != request.arm_id:
            return self._denied(base, "observation arm does not match request arm")
        if manifest.capability != request.capability:
            return self._denied(base, "skill capability does not match request")
        if manifest.stage is not SkillStage.ACTIVE:
            return self._denied(base, f"skill is {manifest.stage.value}, not ACTIVE")
        if observation.safety_stop:
            return SupervisorReceipt(
                **base,
                decision=SupervisorDecision.STOP,
                reason="observation carries an active safety stop",
            )
        if not manifest.authority.may_move_arm:
            return self._denied(base, "skill manifest does not grant arm motion")
        if not request.authorized:
            return self._denied(base, "request lacks an ALLOW authorization")
        if request.authorization.op != request.operation:
            return self._denied(base, "authorization operation does not match request")
        if (request.authorization.state_revision != request.expected_world_revision
                or observation.world_revision != request.expected_world_revision):
            return self._denied(base, "world revision is stale for this authorization")
        if proposal.request_id != request.request_id:
            return self._denied(base, "proposal request id does not match")
        if proposal.skill_id != manifest.skill_id:
            return self._denied(base, "controller proposal attempted to select another skill")
        if (manifest.arm_ids and request.arm_id not in manifest.arm_ids):
            return self._denied(base, "skill is not bound to the requested arm")
        if (manifest.control.max_contact_force_n is not None
                and observation.contact_force_n > manifest.control.max_contact_force_n):
            return SupervisorReceipt(
                **base,
                decision=SupervisorDecision.STOP,
                reason="contact force exceeded the skill limit",
            )

        values = dict(requested)
        if manifest.controller_type is ControllerType.RESIDUAL_RL:
            scale = proposal.residual_scale
            if not math.isfinite(scale) or scale < 0 or scale > manifest.control.residual_scale_max:
                return self._denied(base, "residual scale exceeded the skill envelope")
            values = {name: value * scale for name, value in values.items()}
        elif proposal.residual_scale != 1.0:
            return self._denied(base, "non-residual skill supplied a residual scale")

        allowed = set(self._TRANSLATIONS)
        unknown = sorted(set(values) - allowed)
        if unknown:
            return self._denied(base, f"unsupported action fields: {', '.join(unknown)}")
        if not values:
            return self._denied(base, "controller proposed an empty action")

        workspace_reason = self._workspace_violation(
            manifest, observation, values,
        )
        if workspace_reason is not None:
            return self._denied(base, workspace_reason)

        clamped: dict[str, float] = {}
        changed = values != dict(requested)
        for name, value in values.items():
            limit = self._limit(manifest, observation, name)
            safe = max(-limit, min(limit, value))
            clamped[name] = safe
            changed = changed or safe != value

        return SupervisorReceipt(
            **base,
            decision=SupervisorDecision.CLAMP if changed else SupervisorDecision.ALLOW,
            reason=("action clamped to the skill envelope" if changed
                     else "action accepted by the skill supervisor"),
            safe_action=tuple(sorted(clamped.items())),
        )

    @staticmethod
    def _denied(base: dict, reason: str) -> SupervisorReceipt:
        return SupervisorReceipt(
            **base,
            decision=SupervisorDecision.DENY,
            reason=reason,
        )

    def _limit(
        self,
        manifest: SkillManifest,
        observation: SkillObservation,
        name: str,
    ) -> float:
        _, kind = self._TRANSLATIONS[name]
        control = manifest.control
        if kind == "linear":
            limit = control.max_delta_mm
            if control.max_speed_mm_s is not None:
                speed_limit = control.max_speed_mm_s * min(
                    control.action_interval_ms,
                    observation.action_interval_ms,
                ) / 1000.0
                limit = min(limit, speed_limit)
            return limit
        if kind == "rotation":
            return control.max_rotation_deg
        return control.max_gripper_delta

    @classmethod
    def _workspace_violation(
        cls,
        manifest: SkillManifest,
        observation: SkillObservation,
        values: dict[str, float],
    ) -> str | None:
        bounds = {axis: (lower, upper)
                  for axis, lower, upper in manifest.control.workspace_mm}
        position = dict(zip(("x", "y", "z"), observation.tcp_position_mm))
        for name, value in values.items():
            axis, kind = cls._TRANSLATIONS[name]
            if kind != "linear" or axis not in bounds:
                continue
            lower, upper = bounds[axis]
            proposed = position[axis] + value
            if proposed < lower or proposed > upper:
                return f"{axis} workspace envelope violation"
        return None
