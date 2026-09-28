# Sponsor providers (`omni_q.providers`)

> OQ-034 (Intel + Qualcomm as one graph) · OQ-031 (device-to-device task
> routing — complementary work, not one serial stream).

A **`Provider`** is a named bundle of `DeviceSpec`s for one sponsor track.
**`ProviderRouter`** implements the `Device` contract and routes every `Step`
across *all* registered providers at once, so one Omni graph runs on Intel *or*
Qualcomm capabilities without being rebuilt.

```python
from omni_q import build_mock_engine
from omni_q.fakes import RulePlanner
from omni_q.scheduler import ScheduledPlanner
from omni_q.providers import ProviderRouter, INTEL_PROVIDER

engine = build_mock_engine()
engine.planner = ScheduledPlanner(RulePlanner())
engine.device  = ProviderRouter([INTEL_PROVIDER])     # the full Intel stack
engine.run("inspect and correct the workspace")
```

## Tracks

| Provider | Devices | State |
|----------|---------|-------|
| **`INTEL_PROVIDER`** | `intel.arm_left`, `intel.arm_right` (SO-101 in MuJoCo), `intel.perception` (camera + OpenVINO detector), `intel.reason` (OpenVINO on Core Ultra) | live |
| **`QUALCOMM_PROVIDER`** | `qualcomm.x_elite` (AI Hub / QAIRT + GenieX), `qualcomm.arduino` (physical I/O), `qualcomm.cloud` (non-local) | `available=False` until OQ-028 / OQ-030 |

## Routing rules

- op → device kinds (`PICK/MOVE/HANDOFF/... → arm`, `LOCATE → reasoning`), else
  contract → kinds (`observe`/`verify → perception`, `manipulate → arm`).
- `Step.arm` (`left`/`right`, set by the scheduler) pins the step to that
  provider's matching arm device.
- `keep_local` on the world drops non-local devices (e.g. `qualcomm.cloud`).
- Only devices from `available` providers are considered. No candidate →
  `NoDeviceForStep` (a `RuntimeError` — the engine treats it as a placement
  failure and replans).

## OQ-031 — complementary work

Because perception, reasoning and each arm land on **distinct** devices, a
routed plan spreads across several devices and verification never shares a
device with manipulation:

```
pick_connector_2 -> intel.arm_right     move_connector_2 -> intel.arm_right
pick_plate_1     -> intel.arm_left      move_plate_1     -> intel.arm_left
verify_final     -> intel.perception
```

## OQ-034 — one graph, two tracks

`router.set_available("intel", False); router.set_available("qualcomm", True)`
re-routes the **same graph object** entirely onto Qualcomm devices. A track that
drops out mid-run just triggers the engine's normal replan onto the other track
(`tests/test_providers.py::test_engine_replans_when_a_track_goes_offline_mid_run`).

`router.route_graph(graph, world) -> {step_id: device}` and
`router.to_dict(graph, world)` (adds `routing` + `by_provider`) feed the
Qualcomm device-graph UI (OQ-032).

## Try it

```
PYTHONPATH=src python -m omni_q.providers
PYTHONPATH=src python -m pytest -q tests/test_providers.py
```

## Not yet

- Real Qualcomm devices (OQ-028 X Elite inference, OQ-030 Arduino node).
- An `actuation` kind beyond the placeholder on `qualcomm.arduino`.
