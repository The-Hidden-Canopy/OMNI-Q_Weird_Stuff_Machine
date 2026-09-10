# SO-101 / SO-ARM100 Capability Map

> OQ-003 (Gerron/Kimi). Every number below is measured by
> [`scripts/probe_so101.py`](scripts/probe_so101.py) from the pinned MJCF and
> traceable to
> [`evidence/benchmark_results/so101_capability_map_2026-09-10/so101_spec.json`](../../evidence/benchmark_results/so101_capability_map_2026-09-10/so101_spec.json).
> Re-run the probe after any model change; do not hand-edit numbers here.

## Model & lineage

The Intel online challenge targets the **SO-101** arm. MuJoCo Menagerie has no
dedicated SO-101 entry; this map is measured from **`trs_so_arm100`**
(The Robot Studio SO-ARM100, menagerie commit
`8161bba264d7fa7c99ca301e91e7fb44737676ad`, vendored under
[`assets/menagerie_so_arm100/`](assets/menagerie_so_arm100/SOURCE.md),
Apache-2.0). The SO-101 is the assembly/wiring/gripper update of the SO-100 —
**joint architecture is identical**, which is why this model is the reference
sim until the challenge's own assets land.

- MuJoCo model: 6 DOF (`nq=6 nv=6 nu=6`), timestep 0.002 s, total arm mass
  1.1714 kg, position actuators `kp=50`, `forcerange ±3.5`.
- Cross-check source: LeRobot `so_follower.py` @
  `b6ec0060779550c0a157ae34feb89e0cf86012a8` — same 6-joint layout, all
  Feetech **STS3215** servos (ids 1–6), `wrist_roll` flagged as the
  full-turn motor, gripper normalized 0–100 (% open), body joints in degrees.

## Joints

| # | XML name | LeRobot name | Range (rad) | Range (deg) | Axis |
|---|----------|--------------|-------------|-------------|------|
| 1 | `Rotation`   | `shoulder_pan`  | −1.92 … 1.92     | ±110.0          | 0 1 0 |
| 2 | `Pitch`      | `shoulder_lift` | −3.32 … 0.174    | −190.2 … +10.0  | 1 0 0 |
| 3 | `Elbow`      | `elbow_flex`    | −0.174 … 3.14    | −10.0 … +179.9  | 1 0 0 |
| 4 | `Wrist_Pitch`| `wrist_flex`    | ±1.66            | ±95.1           | 1 0 0 |
| 5 | `Wrist_Roll` | `wrist_roll`    | ±2.79            | ±159.9          | 0 1 0 |
| 6 | `Jaw`        | `gripper`       | −0.174 … 1.75    | −10.0 … +100.3  | 0 0 1 |

**Limit source caveat:** the menagerie README states these are *approximate*
joint limits, and on real SO-101 hardware usable range is calibration-dependent
(LeRobot stores per-motor calibration offsets; Feetech servos also support
`max_relative_target` safety clamping). Treat the sim ranges as the envelope
and expect the onsite arm to be a few degrees narrower per joint.

Keyframes (rad, joint order as table): `home = [0, −1.57, 1.57, 1.57, −1.57, 0]`,
`rest = [0, −3.32, 3.11, 1.18, 0, −0.174]`.

## Gripper

- Jaw travel −0.174 … 1.75 rad (110.3°); measured pad-tip gap **21.3 mm closed
  → 77.1 mm fully open** (near-linear, ~0.29 mm per 0.01 rad).
- Comfortably spans tableware: plates (~150–300 mm diameter are grasped at the
  rim, well under 77 mm jaw width), cups, cutlery.
- LeRobot API works in 0–100 "percent open" on hardware; the sim joint is rad.
  Conversion: `rad = lo + (pct/100) * (hi − lo)` with lo/hi from the table.

## Wrist roll — the spin-primitive number (feeds OQ-014)

- Range ±159.9°, **total travel 319.7° — 0.89 of a full turn**. A plate cannot
  be rotated 360° in one grasp; spins beyond ~320° (with margin: plan ~±150°
  per grasp) require the rotate → handoff → regrasp → rotate sequence from
  BACKLOG OQ-014.
- `wrist_pitch` (±95°) covers the remaining orientation freedom for plate
  face-up poses.

## Workspace (measured: 40k random FK samples, TCP = `Fixed_Jaw` body, gripper
parked at home)

The arm **faces −y** at the home keyframe (TCP at [0, −0.239, 0.177] m).

| Region | x (m) | y (m) | z (m) | max radial (m) |
|--------|-------|-------|-------|----------------|
| Full envelope | −0.34 … 0.34 | −0.39 … 0.23 | −0.12 … 0.43 | 0.386 |
| Front (y < −0.05) | −0.34 … 0.34 | −0.39 … −0.05 | −0.12 … 0.43 | 0.386 |
| z ≈ 0 ± 1 cm (base height) | −0.32 … 0.32 | −0.36 … 0.06 | — | 0.365 |

Planning numbers:

- **Reach radius ~0.39 m** from the base origin; a **~0.6–0.7 m wide shared
  table** between two arms is workable with base spacing ~0.4–0.5 m.
- 64% of uniformly random joint configs put the TCP in the front half-space —
  the front region is not constraint-bound; scheduling should worry about
  **collisions and gripper occupancy, not reach** (feeds OQ-012/OQ-013).
- Bimanual note: two arms are a **mirrored pair** (left/right). Mirror the
  mount about the table centerline; the −y front direction and all envelope
  numbers above apply per-arm in arm-local coordinates.

## Control API

**Sim (MuJoCo):** six position actuators, one per joint, same names as joints.
`ctrl` vector = target joint positions in rad, `ctrlrange` = joint range,
`gear 1.0`, `kp 50` (dampratio 1), `forcerange ±3.5`. Set `data.ctrl`,
step with `mujoco.mj_step`. Keyframes above are valid startup poses.

**Hardware (LeRobot SO-101, onsite track):** FeetechMotorsBus over serial;
motors `shoulder_pan…gripper` = ids 1–6 (`sts3215`). Position-mode PID written
at connect: P=16, I=0, D=32. Observation/actuation dictionaries keyed by
LeRobot joint names; body joints in degrees (`use_degrees=True`), gripper in
0–100 %. `max_relative_target` clamps per-step motion — enable for safety.
Torque disabled on disconnect by default.

## Cameras

**None in the MJCF** (`cameras: []`, `sensors: []`) — measured, not assumed.
OQ-006 must add them to the scene:

- **third-person** camera facing the table (the OQ-008 detector input), and
- optional **wrist camera** on `Fixed_Jaw` (matches SO-101 hardware options).
LeRobot configs cameras by `CameraConfig` (width/height/fps) per named camera.

## Open questions / risks

1. **SO-100 vs SO-101 gripper delta** — the sim uses the SO-100 two-jaw
   gripper; the SO-101 production gripper differs mechanically. Verify against
   the physical arm before claiming fingertip fidelity onsite.
2. **Approximate limits** (menagerie's own caveat) — recalibrate per-arm on
   hardware; keep a sim-vs-hardware range audit for OQ-021's deliberate
   breakage tests.
3. **Mounting assumptions** — envelope numbers are in arm-local frame with the
   base at the origin; table height, arm spacing, and the left/right mirror are
   OQ-006/OQ-007 scene decisions, not arm facts.
4. **Challenge assets** — if the Intel challenge ships its own SO-101 MJCF,
   re-point the probe at it; the spec schema stays identical.

## Reproduce

```bash
py -3.12 -m venv .venv && .venv/Scripts/pip install -r integrations/intel/requirements.txt
.venv/Scripts/python integrations/intel/scripts/probe_so101.py
```

Writes `evidence/benchmark_results/so101_capability_map_<date>/so101_spec.json`
and appends `so101_probe_history.jsonl`.
