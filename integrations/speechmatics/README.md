# Speechmatics integration (stackable bonus)

Not a primary track — a bonus award that stacks on whichever primary track we
enter. One project competes twice.

## Role in the loop

Speech → transcription → intent → **graph mutation** on the running Omni Q graph.

| Utterance | Effect |
| --- | --- |
| *"Omni, don't use the left arm anymore."* | remove `LEFT_ARM` node → recompile |
| *"Keep inference on-device."* | placement constraint → re-place graph |
| *"Watch the second camera too."* | add sensory node → extend topology |
| *"Inspect this / don't move that / use the other device."* | goal or constraint update |

## TODO

- [ ] Speechmatics streaming transcription hookup
- [ ] Intent parser: utterance → {goal update, constraint, node add/remove}
- [ ] Feed mutations into the Omni Q graph compiler
- [ ] On-screen confirmation of the applied change
