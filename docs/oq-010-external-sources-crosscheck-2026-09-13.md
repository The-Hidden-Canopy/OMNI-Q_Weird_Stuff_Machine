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
the set still unsolved (`fork_1`, `spoon_1`, `napkin_1`).

> **RUN AND FALSIFIED, same day.** Ten randomized trials per arm, same seed,
> only the middle term changed: **byte-identical** per-object outcomes
> (`cup_1` 10/10, `plate_1` 9/10, flat objects 0). Friction was never the
> binding constraint — the receipts show the gripper never reaches these
> objects (pinch position error up to **147 mm**, pressing the table at
> **32.7 N** against the successful grasp's 0.28 N). Full evidence:
> [`evidence/benchmark_results/pad_friction_probe_2026-09-13/`](../evidence/benchmark_results/pad_friction_probe_2026-09-13/README.md).
> The blocker is IK convergence on the final descent to a low flat target.

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

## The tutorial's code, read properly (default branch is `master`, not `main`)

`mujoco_env/` is the part worth studying — `y_env.py`, `ik.py`,
`mujoco_parser.py`, `transforms.py`, `utils.py`. `train_model.py` is **LeRobot's
own training script, vendored** (Apache-2.0); copying it means carrying its
licence and attribution, not treating it as tutorial code.

`SimpleEnv(xml_path, action_type='eef_pose', state_type='joint_angle', seed)`:

- **`action_type='eef_pose'` means the policy emits Cartesian deltas**
  (`dx,dy,dz,dr,dp,dy`) and IK converts them. The SO-101 RL post independently
  concluded Cartesian beats joint-space. **Two unrelated sources, same
  conclusion** — that is the strongest single signal across both, and we are
  already on the right side of it.
- IK: augmented Jacobian, damped least squares (`damped_ls`), `stepsize=1.0`,
  `eps=1e-2`, `th=1°`, `max_ik_tick=50` inside `step()` (1000 available),
  joint angles clipped to `[q_mins, q_maxs]` every iteration. Worth comparing
  against our own damping sweep in `_ik_reach_pad`, which tested 4 start/end
  damping pairs; theirs is a single fixed `eps`.
- Reset randomization: `sample_xyzs()` over `x ∈ [0.24, 0.4]`,
  `y ∈ [-0.2, 0.2]`, `z = 0.82`, **minimum 0.2 m separation between objects**,
  then 100 settle steps. The separation constraint is a detail our
  `IntelSceneConfig` jitter does not enforce.
- Success is plain geometry: mug-plate XY < 0.1 m, Z < 0.6 m, gripper opening
  < 0.1, TCP height > 0.9.
- Gripper fans one command to four finger joints as `[cmd, cmd*0.8, cmd,
  cmd*0.8]`. Ours is a single `Jaw` joint — not transferable, but it shows the
  shape.

### The schema we would have to emit

To use ACT / pi_0 / SmolVLA unmodified, `IntelTableWorld` must produce exactly:
`observation.image` and `observation.wrist_image` (256×256 RGB),
`observation.state` (6-D pose), `action` (7-D), at 20 fps, as `LeRobotDataset`
parquet. We render cameras already; the wrist view and the 20 fps cadence are
the work.

### The objection that decides whether this path is worth starting

**Imitation learning needs demonstrations of the thing we cannot currently do.**
`fork_1` / `spoon_1` / `napkin_1` have no reliable scripted grasp. An ACT policy
trained on teleop data can only imitate successful episodes — so someone has to
first produce successful fork grasps *by hand*, through keyboard teleop.

That may well be easier than our scripted IK, because a human closes the loop
visually and nudges — note the RL agent independently learned to nudge the cube
before grasping, which nobody rewarded. But it may also be impossible, in which
case the blocker is geometry and contact, and **no policy will learn around it**.

**Proposed de-risking experiment, ~30 minutes, before committing to the VLA
path:** wire keyboard teleop (their control scheme: WASD xy, RF z, QE tilt,
arrows rotate, SPACE gripper, Z reset) over *our existing* dual-SO-101 scene and
try to grasp `fork_1` by hand. One bit of information, and it decides the whole
branch:

- *human succeeds* → demonstrations are collectable, ACT is worth the 30–60 min
  train, and the scripted-IK gap is a control problem;
- *human fails* → the problem is physics/geometry (start with the torsional
  friction delta above), and the imitation path would have been dead on arrival.

This test does not exist in `BACKLOG.md` and is cheaper than either branch it
chooses between.

## Status

Nothing in this document has been implemented. It exists so the two sources are
usable as evidence rather than re-read from scratch, and so the torsional-
friction probe is recorded as a specific hypothesis with a specific test,
instead of being rediscovered later.
