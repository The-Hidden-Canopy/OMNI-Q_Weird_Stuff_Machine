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

## Result-to-speech response seam

`integrations/speechmatics/run_voice_transport.py` closes the loop without
creating a second command or state architecture:

```text
Speechmatics final payload
        -> SpeechmaticsRealtimeAdapter
        -> VoiceRuntime.ingest_final()
        -> real VoiceDispatchResult / OMNI mutation result
        -> VoiceResponseRenderer
        -> VoiceSink
        -> SpeechmaticsTTS + injected audio player
```

Construct the sink around the existing adapter:

```python
adapter = SpeechmaticsRealtimeAdapter(runtime)
tts = SpeechmaticsTTS(
    api_key=os.environ["SPEECHMATICS_API_KEY"],
    player=audio_player,  # optional; Windows defaults to in-memory winsound playback
)
sink = VoiceSink(adapter, on_result=print_result, speech_output=tts)
result = sink.on_final(payload, world=world)
```

`VoiceResponseRenderer` only speaks committed, denied, authorized-but-not-
committed, or interruption results. Partials and observation-only finals stay
silent. It does not infer completion from a transcript, invent a receipt, or
mutate the world. `SpeechmaticsTTS` calls the official preview TTS endpoint
with a bearer key and returns a `SpeechOutputReceipt` containing session/org
scope, response ID, text hash, format, and byte count. Playback is injected so
microphone/audio-device code remains outside the governed core. Transport
failures emit `voice.response.failed` and do not change the original OMNI
result.

For a direct local audio check, set the key in the process environment and run
the transport smoke command from the repository root:

```powershell
$env:SPEECHMATICS_API_KEY = 'set-this-in-your-process-only'
$env:PYTHONPATH = 'src'
& .venv/Scripts/python.exe -m integrations.speechmatics.run_voice_transport --text 'OMNI response audio is working.'
```

The command uses `wav_16000` and the native Windows in-memory player by
default. It does not print or persist the API key. On Linux/macOS, inject an
`AudioPlayer` implementation for the local device; the TTS request itself is
platform-neutral.

## Remaining integration work

- [ ] Speechmatics realtime websocket/audio input transport and measured latency receipt
- [x] Result-driven Speechmatics TTS response seam with scoped output receipt
- [ ] Connect diarization/visual person association to a live camera provider
- [ ] Map authorized command candidates into OMNI's capability planner
- [ ] Surface voice claims, authority, and commitment events in the UI
- [ ] Benchmark overlap, interruption, ambiguity, stale world, and cross-org cases
