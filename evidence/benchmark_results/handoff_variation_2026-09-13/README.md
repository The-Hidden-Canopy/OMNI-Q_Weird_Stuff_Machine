# Bimanual handoff robustness — 2026-09-13

This bundle is the wider-variation follow-up to the calibrated deterministic
handoff gate. It is exploratory robustness evidence, not a promotion claim.

## Result

Twenty retained trials (`seed=700..719`) varied all of the following:

- cup start position: `position_jitter_m=0.02`
- cup orientation: `orientation_jitter_rad=0.15`
- handoff location: `shared_point_jitter_m=0.03`
- receiving-arm posture: `receiver_pose_jitter_rad=0.15`

| outcome | trials |
| --- | ---: |
| successful handoff and final settle | **6/20** |
| receiving-arm IK timeout after left lift transfer | 8 |
| receiver joint-limit margin reached during initial settle | 3 |
| cup unstable during final settle | 3 |

The earlier deterministic seed-19 gate remains useful as a calibrated
acceptance check, but it is not the robustness envelope. The engineering task
is recovery and receiver targeting under variation, not inventing bimanual
capability from zero.

The machine-readable summary is [`report.json`](report.json), with one retained
receipt per trial. The source note explains the contact-honesty measurements,
the current firm default, and the remaining recovery work:
[`docs/contact-honesty-2026-09-13.md`](../../../docs/contact-honesty-2026-09-13.md).

## Provenance boundary

This bundle exercises the contact-handoff controller and its MuJoCo physics.
It is not evidence that a learned motor policy generated the joint actions.
The public trained artifact is the multimodal OMNI **reasoner** at
[`KissTheHabit/IDA_OMNI_Q`](https://huggingface.co/KissTheHabit/IDA_OMNI_Q);
the model-composed rollout is entered through
`src/omni_q/demo_intel_reasoner.py`, where
`OmniReferenceReasoner → OmniPlanner → governed Intel engine` is recorded in
the planner decisions. A GIF from `watch_sim.py` alone is controller/physics
evidence and must not be labelled as an HF-model rollout without the matching
receipt decision and backend label.

The remaining flat/deformable-object corner is `napkin_1`; its evidence is
tracked separately in [`flat_object_grasp_2026-09-13`](../flat_object_grasp_2026-09-13/README.md).
