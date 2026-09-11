"""OQ-031 / OQ-034 — sponsor providers under one Omni graph.

A ``Provider`` is a named bundle of ``DeviceSpec``s for one sponsor track
(Intel, Qualcomm). ``ProviderRouter`` implements the ``Device`` contract and
routes every ``Step`` across **all** registered providers at once, so a single
Omni graph runs on Intel *or* Qualcomm capabilities without being rebuilt
(OQ-034). Because perception, reasoning and each arm land on distinct devices,
the router assigns **complementary** work rather than a single serial command
stream (OQ-031).

The Intel track is real (dual SO-101 arms + MuJoCo perception + OpenVINO
reasoning on Core Ultra). The Qualcomm track is a placeholder wired the same
way, ``available=False`` until OQ-028 / OQ-030 land.

Standalone: reads ``Step`` / ``WorldState`` / ``DeviceSpec`` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import DeviceSpec, Step, WorldState

# op -> the device kinds that can host it (mirrors devices._AFFINITY)
_OP_KINDS: dict[str, tuple[str, ...]] = {
    "PICK": ("arm",), "PLACE": ("arm",), "MOVE": ("arm",), "OPEN": ("arm",),
    "CLOSE": ("arm",), "ROTATE": ("arm",), "PRESENT": ("arm",), "STABILIZE": ("arm",),
    "REGRASP": ("arm",), "HANDOFF": ("arm",), "COOPERATIVE_ROTATE": ("arm",),
    "EXPRESS": ("arm",),
    "SPIN_SHOW": ("arm",), "LOCATE": ("reasoning",),
}
_CONTRACT_KINDS: dict[str, tuple[str, ...]] = {
    "observe": ("perception",),
    "verify": ("perception",),
    "manipulate": ("arm",),
    "reasoning": ("reasoning",),
}


def _need(step: Step) -> tuple[str, ...]:
    return _OP_KINDS.get(step.op) or _CONTRACT_KINDS.get(step.contract) or (step.contract,)


@dataclass(frozen=True)
class Provider:
    name: str                       # "intel" | "qualcomm"
    track: str                      # human-facing label
    devices: tuple[DeviceSpec, ...]
    available: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "track": self.track, "available": self.available,
            "devices": [{"name": d.name, "kinds": list(d.kinds), "local": d.local}
                        for d in self.devices],
        }


INTEL_PROVIDER = Provider(
    name="intel",
    track="Intel Core Ultra — dual SO-101 (MuJoCo) + OpenVINO",
    devices=(
        DeviceSpec("intel.arm_left", ("arm",), local=True),
        DeviceSpec("intel.arm_right", ("arm",), local=True),
        DeviceSpec("intel.perception", ("perception", "verify"), local=True),
        DeviceSpec("intel.reason", ("reasoning",), local=True),
    ),
)

QUALCOMM_PROVIDER = Provider(
    name="qualcomm",
    track="Qualcomm Snapdragon X Elite + Arduino UNO Q (pending OQ-028/OQ-030)",
    devices=(
        DeviceSpec("qualcomm.x_elite", ("perception", "verify", "reasoning"), local=True),
        DeviceSpec("qualcomm.arduino", ("arm", "actuation"), local=True),
        DeviceSpec("qualcomm.cloud", ("reasoning", "perception"), local=False),
    ),
    available=False,
)


class NoDeviceForStep(RuntimeError):
    """No available provider can host a step (the engine treats this as a
    placement failure and replans)."""


class ProviderRouter:
    """Device provider spanning every sponsor track (OQ-034)."""

    def __init__(self, providers: list[Provider] | None = None) -> None:
        self._providers: dict[str, Provider] = {
            p.name: p for p in (providers or [INTEL_PROVIDER, QUALCOMM_PROVIDER])
        }

    # -- introspection ------------------------------------------------
    def providers(self) -> list[Provider]:
        return list(self._providers.values())

    def set_available(self, name: str, available: bool) -> None:
        p = self._providers[name]
        self._providers[name] = Provider(p.name, p.track, p.devices, available)

    def provider_of(self, device_name: str) -> str:
        return device_name.split(".", 1)[0]

    def _live_specs(self) -> list[tuple[str, DeviceSpec]]:
        return [(p.name, d) for p in self._providers.values() if p.available
                for d in p.devices]

    # -- Device contract -------------------------------------------
    def devices(self) -> list[DeviceSpec]:
        return [d for _p, d in self._live_specs()]

    def route(self, step: Step, world: WorldState) -> str:
        need = _need(step)
        local_only = world.local_only()
        candidates = [
            (pname, d) for pname, d in self._live_specs()
            if d.online and (d.local or not local_only)
            and any(d.can_host(k) for k in need)
        ]
        if step.arm in {"left", "right"}:
            armed = [c for c in candidates if c[1].name.endswith(f"arm_{step.arm}")
                     or (step.arm in c[1].name)]
            if armed:
                candidates = armed
            elif any("arm" in k for k in need):
                # a single-arm device (e.g. qualcomm.arduino) still counts
                pass
        if not candidates:
            raise NoDeviceForStep(
                f"no available device for {step.id} ({step.op}); need {need}, "
                f"local_only={local_only}")
        candidates.sort(key=lambda c: (not c[1].local,))
        return candidates[0][1].name

    # -- OQ-034 view ---------------------------------------------
    def route_graph(self, graph: Any, world: WorldState) -> dict[str, str]:
        """Device per step for a whole graph (or anything with ``.steps``)."""
        out: dict[str, str] = {}
        for s in graph.steps:
            try:
                out[s.id] = self.route(s, world)
            except NoDeviceForStep:
                out[s.id] = "UNROUTED"
        return out

    def to_dict(self, graph: Any | None = None, world: WorldState | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {"providers": [p.as_dict() for p in self._providers.values()]}
        if graph is not None and world is not None:
            routed = self.route_graph(graph, world)
            d["routing"] = routed
            d["by_provider"] = _group_by_provider(routed, self.provider_of)
        return d


def _group_by_provider(routing: dict[str, str], provider_of) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for step_id, device in routing.items():
        grouped.setdefault(provider_of(device), []).append(step_id)
    return grouped


def _main() -> None:  # pragma: no cover - manual smoke
    from .fakes import RulePlanner
    from .scheduler import schedule
    from .world import MockWorld

    world = MockWorld.sample().state()
    base = RulePlanner().plan("inspect and correct the workspace", world)
    graph = schedule(base, world).annotate(base)
    router = ProviderRouter()
    print("routing (intel only live):")
    for sid, dev in router.route_graph(graph, world).items():
        print(f"  {sid:20} -> {dev}")
    router.set_available("intel", False)
    router.set_available("qualcomm", True)
    print("same graph, qualcomm track:")
    for sid, dev in router.route_graph(graph, world).items():
        print(f"  {sid:20} -> {dev}")


if __name__ == "__main__":
    _main()
