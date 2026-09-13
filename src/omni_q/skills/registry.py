"""Promotion-gated skill registry and deterministic selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..events import EventBus
from .contracts import (
    PromotionEvidence,
    SkillManifest,
    SkillRequest,
    SkillStage,
)

__all__ = [
    "SkillRegistry",
    "SkillRegistryError",
    "SkillSelection",
    "SkillUnavailable",
]


class SkillRegistryError(ValueError):
    """A registry mutation or promotion is invalid."""


class SkillUnavailable(SkillRegistryError):
    """No ACTIVE skill satisfies a governed request."""


@dataclass(frozen=True)
class SkillSelection:
    """The skill selected for one request; fallback is metadata, not execution."""

    manifest: SkillManifest
    fallback_skill_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill": self.manifest.as_dict(),
            "fallback_skill_id": self.fallback_skill_id,
        }


class SkillRegistry:
    """Own skill identity and promotion state, never controller execution."""

    def __init__(self, *, bus: EventBus | None = None) -> None:
        self.bus = bus or EventBus()
        self._skills: dict[str, SkillManifest] = {}
        self._promotion_history: dict[str, list[dict[str, Any]]] = {}

    def register(self, manifest: SkillManifest) -> None:
        if not isinstance(manifest, SkillManifest):
            raise SkillRegistryError("registry accepts SkillManifest values only")
        if manifest.stage is not SkillStage.TRAINED:
            raise SkillRegistryError(
                "new skills must enter the registry at TRAINED; use promote() for evidence"
            )
        if manifest.skill_id in self._skills:
            raise SkillRegistryError(f"skill already registered: {manifest.skill_id}")
        self._skills[manifest.skill_id] = manifest
        self._promotion_history[manifest.skill_id] = []
        self.bus.publish(
            "skill.registered",
            source="skill-registry",
            skill=manifest.as_dict(),
        )

    def get(self, skill_id: str) -> SkillManifest:
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            raise SkillUnavailable(f"unknown skill: {skill_id}") from exc

    def all(self, *, org_id: str | None = None) -> tuple[SkillManifest, ...]:
        values = tuple(self._skills.values())
        if org_id is not None:
            values = tuple(skill for skill in values if skill.org_id == org_id)
        return tuple(sorted(values, key=lambda skill: skill.skill_id))

    def history(self, skill_id: str) -> tuple[dict[str, Any], ...]:
        self.get(skill_id)
        return tuple(dict(item) for item in self._promotion_history[skill_id])

    def promote(self, skill_id: str, evidence: PromotionEvidence) -> SkillManifest:
        manifest = self.get(skill_id)
        if not isinstance(evidence, PromotionEvidence):
            raise SkillRegistryError("promotion requires PromotionEvidence")
        if manifest.stage is SkillStage.ACTIVE:
            raise SkillRegistryError("ACTIVE skills cannot be promoted again")
        try:
            index = _PROMOTION_ORDER.index(manifest.stage)
        except ValueError as exc:  # defensive if the enum grows without the order
            raise SkillRegistryError(f"unsupported skill stage: {manifest.stage}") from exc
        if index + 1 >= len(_PROMOTION_ORDER):
            raise SkillRegistryError("skill has no further promotion stage")
        target = _PROMOTION_ORDER[index + 1]
        if evidence.artifact_digest != manifest.artifact_digest:
            raise SkillRegistryError("promotion evidence artifact does not match the skill")
        if target is SkillStage.ACTIVE and not all(value for _, value in evidence.checks):
            raise SkillRegistryError("ACTIVE promotion requires every validation check to pass")

        updated = replace(manifest, stage=target)
        self._skills[skill_id] = updated
        record = {
            "from": manifest.stage.value,
            "to": target.value,
            "skill_id": skill_id,
            "evidence": evidence.as_dict(),
        }
        self._promotion_history[skill_id].append(record)
        self.bus.publish("skill.promoted", source="skill-registry", **record)
        return updated

    def select(self, request: SkillRequest) -> SkillSelection:
        if not isinstance(request, SkillRequest):
            raise SkillRegistryError("selection requires SkillRequest")
        if request.skill_id is not None:
            manifest = self.get(request.skill_id)
            self._check_candidate(manifest, request)
            return SkillSelection(manifest, manifest.fallback_skill_id)

        candidates = [
            manifest
            for manifest in self._skills.values()
            if manifest.stage is SkillStage.ACTIVE
            and manifest.org_id == request.org_id
            and manifest.capability == request.capability
            and (not manifest.arm_ids or request.arm_id in manifest.arm_ids)
        ]
        if request.preferred_skill_ids:
            rank = {skill_id: index for index, skill_id in
                    enumerate(request.preferred_skill_ids)}
            candidates.sort(key=lambda manifest: (
                rank.get(manifest.skill_id, len(rank)), manifest.skill_id))
        else:
            candidates.sort(key=lambda manifest: manifest.skill_id)
        if not candidates:
            same_capability = [
                manifest for manifest in self._skills.values()
                if manifest.stage is SkillStage.ACTIVE
                and manifest.capability == request.capability
            ]
            if same_capability and all(
                manifest.org_id != request.org_id for manifest in same_capability
            ):
                raise SkillUnavailable(
                    f"no ACTIVE skill in organization scope {request.org_id}"
                )
            raise SkillUnavailable(
                f"no ACTIVE skill for {request.org_id}/{request.capability}/{request.arm_id}"
            )
        manifest = candidates[0]
        return SkillSelection(manifest, manifest.fallback_skill_id)

    @staticmethod
    def _check_candidate(manifest: SkillManifest, request: SkillRequest) -> None:
        if manifest.stage is not SkillStage.ACTIVE:
            raise SkillUnavailable(
                f"skill {manifest.skill_id} is {manifest.stage.value}, not ACTIVE"
            )
        if manifest.org_id != request.org_id:
            raise SkillUnavailable("skill organization scope does not match request")
        if manifest.capability != request.capability:
            raise SkillUnavailable("skill capability does not match request")
        if manifest.arm_ids and request.arm_id not in manifest.arm_ids:
            raise SkillUnavailable("skill is not bound to the requested arm")


_PROMOTION_ORDER: tuple[SkillStage, ...] = (
    SkillStage.TRAINED,
    SkillStage.SIM_EVAL,
    SkillStage.DOMAIN_RANDOMIZATION,
    SkillStage.SHADOW,
    SkillStage.SUPERVISED_HARDWARE,
    SkillStage.VALIDATED,
    SkillStage.ACTIVE,
)
