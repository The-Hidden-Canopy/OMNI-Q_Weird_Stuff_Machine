# Speechmatics integration (stackable bonus)

Not an entry track — a bonus award that stacks on our Intel online entry (the
only track we qualify for). One project competes twice.

## Role in the loop

Speech → transcription → intent → **graph mutation** on the running Omni Q graph.

| Utterance | Effect |
| --- | --- |
| *"Omni, don't use the left arm anymore."* | remove `LEFT_ARM` node → recompile |
| *"Keep inference on-device."* | placement constraint → re-place graph |
| *"Watch the second camera too."* | add sensory node → extend topology |
| *"Inspect this / don't move that / use the other device."* | goal or constraint update |

## Current bounded slice

`src/omni_q/voice.py` now provides the provider-neutral boundary used by a
Speechmatics transport:

```
Speechmatics payload → attributed/timestamped claim → speaker/turn state
                      → intent candidate → explicit authority decision
                      → RuntimeMutator or injected OMNI handler
```

The `SpeechmaticsRealtimeAdapter` only translates payloads; it does not import
the Speechmatics SDK or become part of the core architecture. Partial speech
publishes a preparation event and cannot execute. Final speech is still an
observation claim. Unknown speakers remain unauthorized, cross-session and
cross-organization events are rejected, and unresolved references are kept
ambiguous rather than guessed.

The existing `RuntimeMutator` remains the only path for voice-driven constraint
changes. A successful mutation is reported as committed; deferred or rejected
mutations are explicitly not committed. STOP/HOLD/BACKOFF is a separate local
reflex request and still requires an explicitly registered interrupt role.

## Remaining integration work

- [ ] Speechmatics websocket/audio transport and measured latency receipt
- [ ] Connect diarization/visual person association to a live camera provider
- [ ] Map authorized command candidates into OMNI's capability planner
- [ ] Surface voice claims, authority, and commitment events in the UI
- [ ] Benchmark overlap, interruption, ambiguity, stale world, and cross-org cases
