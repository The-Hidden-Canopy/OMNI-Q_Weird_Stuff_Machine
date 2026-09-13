# Skill Runtime: CEO directive mapped onto what OMNI-Q already has

**Directive (2026-09-13):** *OMNI owns intent and authority. Controllers own
motion.* A learned policy never gets authority over the robot; it is handed a
bounded control problem, its output passes a hard supervisor, and only
`ACTIVE`-promoted policies exist to production OMNI.

This document does not restate the directive. It checks it against the codebase
and says what is already built, what is genuinely new, and in what order the new
parts are worth doing.

**Headline: the supervisor pattern already exists and is tested — for exactly
one skill family.** The work is generalizing it, not inventing it. That is a
materially smaller and lower-risk job than "add an RL subsystem", and it means
the architecture the directive describes is one OMNI-Q has already committed to
once and found workable.

## Already built

| Directive element | Existing implementation |
| --- | --- |
| **Control Supervisor** (clamp / deny / stop) | `expressive.py:248` `validate_proposal` — see below |
| Supervisor re-checked at the actuator | `expressive.py:289` `validate_expressive_command`, whose docstring already gives the directive's own reason: *"Providers must repeat the safety boundary because a caller can construct a `Step` without going through `generate_expression_plan`"* |
| **Skill Selector** (capability → provider) | `providers.py:115` `ProviderRouter.route(step, world)`; `route_graph` for a whole plan |
| Provider eligibility / capability declaration | `fleet.py:181` manipulator identity with `capabilities: frozenset[str]`; capability leases, workspace reservations, fail-closed eligibility (OQ-FLEET-001) |
| Consequence-level vocabulary (**not** `move_joint_3`) | `contracts.py` seven contracts; plans are `PICK / MOVE / PRESENT / VERIFY` over `object_id` — OMNI already thinks in consequences, never in joint deltas |
| Workspace envelope at execution | `intel_sim.py` `_workspace_safety(...)` gating approach and carry, `TransitionRejected` on violation |
| Verification after every manipulation | `OmniQ` verifies after every manipulate step and replans on mismatch (OQ-018) |
| Deterministic four-stage pick | `_do_pick`: transit height → descend to `clear_z` → close → lift → confirm carried |
| Cartesian action space | already the interface: IK tracks a Cartesian pad pose (`_grasp_frame` / `_ik_reach_pad_pose`) |
| Credible contact geometry | 8 fingertip box pads (4/jaw), `cone="elliptic"`, `impratio="10"` — see [`oq-010-external-sources-crosscheck-2026-09-13.md`](oq-010-external-sources-crosscheck-2026-09-13.md) |

### The supervisor, concretely

`validate_proposal` already enforces, and fails closed on, every category the
directive lists:

| Directive check | `ExpressiveWindow` equivalent |
| --- | --- |
| `current_authority` | `base_world_revision` + `current_world_revision` staleness |
| `workspace` | `allowed_regions` |
| `active_skill` | `allowed_primitives` |
| `may_move_arm` | `allowed_arms` |
| `max_delta` | `max_amplitude`, `axis` bounded 0–5 |
| `timeout_ms` | `start_ms` / `return_deadline_ms` budget |
| force/contact limits | `max_contact_duration_ms`, contact-primitive allowance |
| org/tenant scope | `org_id` / `session_id` match |

So "put a hard supervisor after every learned policy" is, in this codebase,
"give `Manipulate` the treatment `Express` already has".

**Two of the three things the directive says to steal from pick-101 are already
in place** (Cartesian action space, fingertip contact geometry), both verified
against the source today. The third — wrist-camera DrQ-v2 — is genuinely future
work.

## Genuinely new

1. **Skill contracts as registered artifacts.** Nothing today declares a
   controller with a digest, preconditions (`object_class`, `object_size_mm`),
   an action envelope (`max_delta_mm`, `max_rotation_deg`), a fallback
   controller, and an explicit `authority` block. `Provider`/`DeviceSpec`
   declare *devices and kinds*, not *skills and envelopes*. This is the
   keystone: the supervisor needs a contract to validate against, and the
   selector needs preconditions to choose on.
2. **Residual control (`u = u_IK + α·Δu_RL`, `0 ≤ α ≤ α_max`).** No blending
   layer exists. Note this composes with the existing four-stage pick almost
   exactly as the directive describes: deterministic pre-grasp → bounded
   residual at contact → deterministic lift, with `α = 0` making it a no-op —
   which is also how it should ship by default.
3. **Promotion lifecycle** TRAINED → SIM EVAL → DOMAIN RANDOMIZATION → SHADOW →
   SUPERVISED HARDWARE → VALIDATED → ACTIVE, with OMNI able to select only
   `ACTIVE`. The pieces to build it on exist — `evidence/benchmark_results/`
   bundles, the 10-seed randomized harness, `IntelSceneConfig` jitter axes — but
   there is no status field and nothing gates on one.
4. **Learning/execution plane separation.** Nothing captures
   `(observation, skill requested, controller used, actions, outcome, failure,
   human correction, world-state delta)` as a training corpus. The receipts
   contain much of it already but are shaped for evidence, not for training.

## Sequence I would actually follow

**1. Skill contracts + registry first, with zero RL.** Register the *existing*
deterministic controllers (`so101.top_down_grasp`, the `_do_pick` sequence)
as contracted skills and route `GRASP` through the registry. This is testable
immediately, changes no behaviour, and forces the envelope vocabulary to be
right before anything learned depends on it. Building the supervisor generalization
alongside it is cheap because `validate_proposal` is the template.

**2. Then the promotion status field**, with every existing controller
`ACTIVE` by definition. Now a future policy has somewhere to *not* be.

**3. Then residual blending with `α = 0` shipped**, and a single validated skill
allowed a small `α_max` under the supervisor.

**4. RL training last**, and only after OQ-010-TELEOP answers whether a human
can demonstrate the grasps at all — see
[`oq-010-external-sources-crosscheck-2026-09-13.md`](oq-010-external-sources-crosscheck-2026-09-13.md).
An RL policy for cutlery has the same prerequisite an imitation policy does: the
simulator must make the grasp physically achievable. The SO-101 RL post's 100%
success was on a **3 cm cube**, the easy case, and its curriculum learning
*failed*. Nothing in it demonstrates flat-cutlery grasping.

## One honest tension with the directive

The directive says residual RL is "where I think you get the most value", and
architecturally I agree. But on current evidence the binding constraint for
`fork_1` / `spoon_1` / `napkin_1` is not controller quality — those objects have
**no successful grasp from any controller**, deterministic or otherwise, and
four independent search strategies failed on them. A residual corrector
improves a controller that is nearly right; it does not discover a grasp that
does not exist in the contact model.

So the architecture is worth building on its own merits — it is the right shape,
and steps 1–3 have value with no RL at all — but it should not be sequenced as
*the fix for cutlery*. The torsional-friction probe and the teleop experiment
are the cheap tests that tell us whether cutlery is a physics problem or a
control problem, and both should land before RL training is budgeted.

## Status

Design mapping only — no code written. Recorded so the directive is actionable
against this codebase rather than re-derived, and so the "already built" column
is not rebuilt by mistake.
