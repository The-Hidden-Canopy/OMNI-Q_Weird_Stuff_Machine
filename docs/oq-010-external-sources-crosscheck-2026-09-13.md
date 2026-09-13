# OQ-010 cross-check against two host-provided sources (2026-09-13)

The contest hosts circulated two references today. Both were read in full and
checked **against this repo's actual scene and code**, not summarized in the
abstract. The useful result is mostly confirmation — with two concrete deltas
and one path we have not taken.

Sources:

1. [`jeongeun980906/lerobot-mujoco-tutorial`](https://github.com/jeongeun980906/lerobot-mujoco-tutorial)
   — teleop → `LeRobotDataset` → ACT / pi_0 / SmolVLA, in MuJoCo.
2. [SO-101 RL lift agent](https://ggando.com/blog/so101-rl-lift/) — SAC trained
   to grasp and lift a 3 cm cube with the **same arm we use**.

## What the SO-101 post confirms we already do

Its headline finding is that **mesh-to-mesh collisions generate unstable single
contact points**, and the fix is small box geoms at the fingertips for stable
multi-point contact, with elliptic friction cones.

We already do all of this, and more of it:

| | SO-101 post | This repo |
| --- | --- | --- |
| Fingertip collision primitive | 2 boxes (1 per jaw) | **8 boxes** — `fixed_jaw_pad_1..4`, `moving_jaw_pad_1..4` |
| Friction cone | `elliptic` | `cone="elliptic"` (`so_arm100.xml:4`) |
| Contact/friction ratio | not stated | `impratio="10"` |

Its second finding — jittery motion fixed by a **four-step sequence (above →
descend → close → lift)** — is also what `_do_pick` already does: transit to a
safe height, descend to `clear_z = start_z + half_h + GRASP_CLEARANCE`, close,
lift, then verify the object is genuinely carried.

Its third — mixing up **graspframe (contact point) vs gripperframe (TCP)** — is
the same class of bug `_grasp_frame` / `_ik_reach_pad_pose` were written to fix
here. Independent arrival at the same answer is worth something.

So: the contact model is not where our remaining grasp failures live. That is
now externally corroborated rather than assumed.

## Two concrete deltas worth testing

**1. Torsional friction is 2.5x lower than theirs.** Their working pads used
`friction="1 0.05 0.001"`. Our pads are overridden at scene-build time in
`intel_sim.py` to `friction="3.00 0.020 0.001"` — *higher* sliding friction
(3.0 vs 1.0) but **lower torsional** (0.020 vs 0.05).

Torsional friction is what stops a pinched object rotating out of the grasp.
That is precisely the failure mode for thin, flat objects, which is precisely
the set still unsolved (`fork_1`, `spoon_1`, `napkin_1`). This is a cheap,
falsifiable probe: raise the middle term toward 0.05 and re-run the 10-seed
randomized harness. **Not yet run** — it changes physics parameters and
deserves its own before/after evidence bundle, not a drive-by edit.

Note the raw `so_arm100.xml` values (`friction="1 0.005 0.0001"`,
`solimp="2 1 0.01"`) are **not** what runs: `intel_sim.py` overrides friction,
`solref` and `solimp` on every `jaw_pad_` geom when it builds the dual-arm
scene. Read the override, not the asset, when reasoning about contact.

**2. Their action space was Cartesian, not joint.** They report Cartesian XYZ
clearly outperformed joint-space control for exploration. We already command
Cartesian targets through IK, so we are aligned — but it is worth remembering
if anyone proposes moving to joint-space control for the RL/learned path.

## What does *not* transfer

- The **tutorial repo is a different robot** — ROBOTIS OMY (6 joints +
  gripper), not SO-101. The scene, the IK and the reach envelope are not
  reusable. Its value is the *pipeline*, not the model.
- It pins **MuJoCo 3.1.6**; our `intel` extra is `mujoco>=3.13`. Borrowing code
  from it means checking API drift, not copying.
- The RL post used **no domain randomization**, and its **curriculum learning
  failed** (pre-grasped mid-air cube "failed to transfer well"; training the
  full task from scratch is what worked). Both are useful negative results: do
  not spend time on a grasp curriculum on this evidence.
- Its 200k-step / ~4-hour SAC run produced a 100% success rate **on a 3 cm cube
  only**. A cube is the easy case. Nothing in it addresses flat cutlery.

## The path we have not taken

The tutorial is a complete, working imitation-learning loop on the Intel stack
we already depend on (`lerobot` is in our `intel` extra):

- keyboard teleop collection (`1.collect_data.ipynb`) — WASD xy, RF z, QE tilt,
  arrows rotate, SPACE toggles gripper;
- `LeRobotDataset` at 20 fps: `observation.image` and `observation.wrist_image`
  (256×256), `observation.state` (6-D pose), `action` (7-D: 6 joints + gripper);
- ACT with `chunk_size: 10`, **30–60 minutes** to train;
- pi_0 / SmolVLA via `pi0_omy.yaml` / `smolvla_omy.yaml`
  (`chunk_size: 5`, `n_action_steps: 5`, `batch_size: 16`, `steps: 20_000`).

`BACKLOG.md` already names "a learned policy" as one of three ways to close the
`fork_1` / `spoon_1` / `napkin_1` gap, alongside analytical multi-solution IK
and a precomputed configuration library. This is the first concrete, costed
recipe for that option: a 30–60 minute ACT train is cheap enough to try, and
the teleop step is the real cost — someone has to demonstrate the grasps.

**Caveat before anyone budgets on that number:** 30–60 min is *their* figure on
*their* hardware for *their* task. This box trains the planner reasoner at
~10 s/example; no ACT timing has been measured here.

## Status

Nothing in this document has been implemented. It exists so the two sources are
usable as evidence rather than re-read from scratch, and so the torsional-
friction probe is recorded as a specific hypothesis with a specific test,
instead of being rediscovered later.
