"""Device provider — placement is separate from function (OQ-001 ``Device``).

A :class:`~omni_q.contracts.Step` says *what* to do; :class:`DeviceRouter` says
*where* it runs, honouring a ``keep_local`` constraint and per-op affinity.
Device-to-device routing between Snapdragon and Arduino (OQ-031) extends this.
"""

from __future__ import annotations

from .contracts import DeviceSpec, Step, WorldState

# op / contract-kind -> ordered candidate device kinds
_AFFINITY: dict[str, tuple[str, ...]] = {
    "observe": ("perception",),
    "verify": ("perception",),
    "PICK": ("arm",), "PLACE": ("arm",), "MOVE": ("arm",),
    "OPEN": ("arm",), "CLOSE": ("arm",), "ROTATE": ("arm",),
    "PRESENT": ("arm",), "STABILIZE": ("arm",), "REGRASP": ("arm",),
    "HANDOFF": ("arm",), "COOPERATIVE_ROTATE": ("arm",), "EXPRESS": ("arm",),
    "LOCATE": ("reasoning",),
}


class DeviceRouter:
    def __init__(self, specs: list[DeviceSpec] | None = None) -> None:
        self._specs: dict[str, DeviceSpec] = {d.name: d for d in (specs or _default_specs())}

    def devices(self) -> list[DeviceSpec]:
        return list(self._specs.values())

    def set_online(self, name: str, online: bool) -> None:
        d = self._specs[name]
        self._specs[name] = DeviceSpec(d.name, d.kinds, d.local, online)

    def route(self, step: Step, world: WorldState) -> str:
        need_key = step.op if step.op in _AFFINITY else step.contract
        wanted = _AFFINITY.get(need_key, (step.contract,))
        local_only = world.local_only()

        candidates = [
            d for d in self._specs.values()
            if d.online
            and (d.local or not local_only)
            and any(d.can_host(k) for k in wanted)
        ]
        if step.arm:
            armed = [d for d in candidates if step.arm in d.name]
            if armed:
                candidates = armed
        if not candidates:
            raise RuntimeError(f"no device can host {step.id} ({step.op})")
        # prefer local, then earliest-declared
        candidates.sort(key=lambda d: (not d.local,))
        return candidates[0].name


def _default_specs() -> list[DeviceSpec]:
    return [
        DeviceSpec("snapdragon.npu", ("perception", "verify"), local=True),
        DeviceSpec("snapdragon.gpu", ("reasoning", "perception"), local=True),
        DeviceSpec("snapdragon.cpu", ("reasoning",), local=True),
        DeviceSpec("arduino.left_arm", ("arm",), local=True),
        DeviceSpec("arduino.right_arm", ("arm",), local=True),
        DeviceSpec("cloud.gpu", ("reasoning", "perception"), local=False),
    ]


def default_devices() -> DeviceRouter:
    return DeviceRouter()
