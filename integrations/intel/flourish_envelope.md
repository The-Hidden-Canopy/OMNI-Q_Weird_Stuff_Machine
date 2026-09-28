# Flourish Envelope — measured bounds (OQ-026 pre-work)

> Gerron/Kimi, 2026-09-10. All numbers measured by
> [`scripts/probe_flourish_envelope.py`](scripts/probe_flourish_envelope.py)
> from the pinned SO-ARM100 MJCF; traceable to
> [`evidence/benchmark_results/flourish_envelope_2026-09-10/flourish_envelope.json`](../../evidence/benchmark_results/flourish_envelope_2026-09-10/flourish_envelope.json).
> **Failure-rate trials remain OQ-026 proper** — they need the working OQ-014
> spin primitive; this document supplies the bounds those trials must respect
> and their protocol.

## 1. Safe rotation amounts (from the capability map)

- Wrist-roll travel: **319.7° total** (±159.9°) — 0.89 of a full turn.
- Per-grasp planning margin for spins: **≤ ±150°**, then handoff → regrasp.
- Wrist-pitch ±95.1° covers plate face-up orientation freedom.
- Source: `so101_capability_map.md` §Wrist roll.

## 2. Velocity limits (measured)

Position-actuator step commands (model kp=50, forcerange ±3.5), 1.2 s sweeps,
peak values:

| Joint | Peak speed (deg/s) | Peak TCP speed (m/s) |
|---|---|---|
| Rotation | 463 | 1.57 |
| Pitch | 463 (to max) / 429 (to min) | 1.34 / 1.24 |
| Elbow | 415 / 436 | 1.15 / 1.21 |
| Wrist_Pitch | 42 (to max, lifting) / 677 (to min) | 0.07 / 1.04 |
| Wrist_Roll | 824 (to max) / 370 (to min) | ≤ 0.08 |

Reading:
- The big wrist asymmetry is gravity: wrist-pitch **down** is fast, **up**
  while loaded is heavily damped (42°/s with the full lower arm hanging on it).
  Flourishes that swing a plate upward are the slow, power-limited direction —
  choreograph spins around wrist **roll** (fast both ways, minimal TCP
  translation ≤ 0.08 m/s) rather than pitch.
- Flourish velocity cap for "safe + visible": **≤ 90°/s joint speed,
  ≤ 0.3 m/s TCP** — roughly half the unloaded elbow/pitch rate, well under
  every measured peak, so the cap is achievable on the slowest (loaded
  wrist-up) axis too. (Design proposal; the OQ-026 trials confirm or tighten.)

## 3. Grip-force envelope (measured)

Static 40 mm probe cylinder between open pads; jaw closed in steps:

- First pad contact at jaw ≈ 93° with a **21.7 N impact transient**.
- Force rises through the low-20s N within ~30° of further closure; probe
  stopped at the 25 N bound.

Implication: the jaw position actuator generates **20–25 N almost immediately**
on a 40 mm object — far above what tableware clamping needs. Fragile-object
grasps must land in the **compliant clamp band** (close to a target jaw angle
that just brackets the object, as the contact-handoff controller does with
`gripper_closed_rad=0.75`), never drive toward the jaw limit against an
object. A force-limited grip needs either current sensing (hardware) or
contact-stop descent (controller); sim alone cannot certify "plate-safe" N.

## 4. Shared bimanual workspace (measured)

20k random FK samples per arm at the dual-scene mounts (left/right at
x = ∓0.26, y = 0.20, both facing −y), 2 cm grid overlap:

- **549 overlapping cells**, world bbox x ∈ [−0.08, 0.08], y ∈ [−0.06, 0.26],
  z ∈ [−0.06, 0.32] m.
- Practical handoff zone (table-front subset, y ≤ −0.04, z ≥ 0.02): the
  central strip x ∈ [−0.08, 0.08] in front of the table centre — consistent
  with the contact controller's shared target (−0.04, −0.11, 0.134).
- Plan handoffs at |x| ≤ 0.08: either arm can hand or receive there without
  re-mounting, which is what makes rotate → handoff → regrasp chains possible
  without moving bases.

## 5. OQ-014/OQ-026 trial protocol (ready to run when the spin primitive lands)

1. **Spin success trial**: plate grasped at centre, rotate wrist_roll at
   ≤ 90°/s to +150°, handoff at (±0.04, −0.11, 0.134), regrasp, rotate to
   +300°. Success = evaluator PASS (position ≤ 10 mm, orientation ≤ 5°) with
   plate upright throughout.
2. **Failure rates**: 20 seeded trials, randomized plate start (±20 mm,
   ±15° per the handoff controller's jitter scales); record drop / slip /
   timeout categories exactly as the contact-handoff report does.
3. **Flourish envelope search**: sweep rotation speed {45, 90, 135}°/s ×
   per-grasp angle {120, 150, 160}°; the largest pair with 0 failures in 20
   trials **per arm pair** defines the demo envelope; everything larger is
   choreography-only (OQ-045) and must never be a dependency of task success.

## Reproduce

```bash
.venv/Scripts/python integrations/intel/scripts/probe_flourish_envelope.py
```
