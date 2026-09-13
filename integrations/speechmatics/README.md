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

## Transport (built)

| File | Role |
| --- | --- |
| `mapper.py` | Provider message → adapter payload. Pure; no socket, no SDK, injectable clock. |
| `transport.py` | Realtime v2 WebSocket, audio pacing, offline replay, latency receipt. |
| `scripts/run_voice_transport.py` | CLI: file / mic / replay → the real voice boundary. |
| `samples/session_replay.jsonl` | Synthetic provider messages (`make_sample_replay.py` regenerates them). |
| `samples/live_session_2026-09-13.jsonl` | **Real** recorded session — exposed the fragmentation defect. |
| `samples/live_halting_2026-09-13.jsonl` | **Real** halting-speech session — exposed everything in "Intent accumulation". |
| `samples/live_stop_using_2026-09-13.jsonl` | **Real** session — exposed the safety-gate collision. |
| `tests/test_speechmatics_transport.py` | 42 tests, no network and no key. |

The pipeline, end to end:

```
Speechmatics ─┬─ partials ──────────────────> UI / anticipation only
              └─ finals
                   │
                   ▼
          UtteranceAggregator      (acoustic: silence, speaker change)
                   │
                   ▼
          IntentAccumulator        (semantic: is this an instruction yet?)
                   │
                   ▼
          VoiceRuntime.ingest_final
                   │  claim → interruption gate → parse → references
                   ▼
             Authority decision    (unknown speakers denied here, not earlier)
                   │
                   ▼
             RuntimeMutator ──────> OMNI graph
```

Protocol: `StartRecognition` (once) → `RecognitionStarted` → binary PCM chunks
(`AudioAdded` acks) → `AddPartialTranscript` / `AddTranscript` → `EndOfStream`
→ `EndOfTranscript`.

### The mapping rules that matter

`voice.py` validates hard, so the mapper does the conversion work:

- provider float-seconds → **integer nanoseconds** anchored at
  `RecognitionStarted`, with `t_end_ns > t_start_ns` forced (a zero-length
  partial is legal for Speechmatics and illegal for `SpeechPartial`);
- **one sequence counter** shared by partials and finals, because
  `VoiceRuntime._check_event` requires strict increase across both;
- `confidence` is the mean of the word confidences the provider actually sent,
  punctuation excluded. When a message carries none, the transcript is counted
  under `transcripts_without_word_confidence` in the receipt rather than
  quietly reported as 1.0;
- speaker label `UU` ("unknown") is **not** treated as an identity — it falls
  back to the configured default id so no phantom speaker enters the registry;
- empty transcripts are dropped without burning a sequence number.

### Run it

Offline, no key and no microphone — this is the reviewable path:

```
.venv/Scripts/python integrations/speechmatics/scripts/run_voice_transport.py \
    --replay integrations/speechmatics/samples/session_replay.jsonl \
    --operator S1 --mock-engine
```

Live, from a 16-bit mono WAV paced at real time:

```
$env:SPEECHMATICS_API_KEY = "<key from portal.speechmatics.com>"
.venv/Scripts/python integrations/speechmatics/scripts/run_voice_transport.py \
    --file audio.wav --operator S1 --receipt evidence/voice_latency.json
```

Live from the microphone: `--mic` (needs `pip install -e .[speechmatics-mic]`).
`--record tmp/msgs.jsonl` captures every provider message so a real session can
be replayed offline afterwards.

**Credentials:** `SPEECHMATICS_API_KEY` only. There is deliberately no
`--api-key` flag (shell history, process table), the key is never written to a
receipt or log, and it is scrubbed from any exception this transport raises.

### Authority is opt-in

`--operator S1` grants that diarization label `command` + `graph_mutation` the
first time it speaks. Every unlisted speaker stays unauthorized: their finals
are observed or denied, never committed. The sample replay demonstrates both —
S1's two constraints commit through `RuntimeMutator`, S2's request is denied.

### Utterance segmentation (why raw provider finals are not enough)

**Measured on the first live microphone session, 2026-09-13.** One spoken
sentence — *"Omni, don't use your left arm anymore"* — came back as **five**
separate `AddTranscript` messages:

```
0.00-1.08  "Omni."      1.08-1.32  "Don't"     1.32-1.68  "use your"
1.68-2.16  "left arm"   2.16-2.84  "anymore."
```

Realtime finalizes on its own latency budget, not on sentence boundaries. Each
fragment reached `nlu.parse` alone, where `"use your"` and `"left arm"` mean
nothing, so a clear operator instruction was recorded as five observations and
**never became a graph mutation**. Raising `max_delay` reduces fragmentation
but cannot fix it — the provider does not know where the operator's sentence
ends, and fewer fragments costs partial latency on every utterance.

`UtteranceAggregator` therefore segments on the speaker's own silence: finals
accumulate until an 800 ms gap (`--silence-ms`), a speaker change, or
`max_utterance_ms`, then dispatch as one claim. Partials are never buffered, so
the "N ms behind live" feedback stays responsive. Two details that are easy to
get wrong and are covered by tests:

- the flushed utterance takes a **fresh** sequence number, because partials
  emitted while it buffered already moved the counter past the fragment's own;
- **empty finals are silence evidence.** Speechmatics reports silence as empty
  `AddTranscript` messages; dropping the message is right, dropping its timing
  is not — without it an entire session merges into one utterance.

`--no-aggregate` restores the raw fragmented behaviour (there is a test that
documents it).

**Flushing early, where waiting buys nothing.** The measured 3.3 s an operator
experienced was ~1.1 s of provider finalization plus ~2.2 s of idle timeout
waiting for silence that had already been signalled. Two early exits:

- **terminal punctuation** — the provider sends sentence-final punctuation as
  its own fragment; once it arrives the utterance is over. Narrowed by
  `min_words_for_punctuation_flush` (3), because Speechmatics punctuates the
  vocative: the real capture's first fragment is literally `"Omni."`, and
  treating that period as a sentence end split the address off from the command
  that followed it.
- **urgency** — a bare `"Stop!"` is too short for the punctuation rule and
  would otherwise sit in the buffer for the full idle timeout, which is the one
  utterance that must never wait. The aggregator takes an `urgent` predicate
  and the CLI passes the voice boundary's own `InterruptionGate`, so safety
  policy is asked for, never re-implemented in the vendor layer.

### Intent accumulation (silence segmentation is not enough)

**Second live session, 2026-09-13.** The speaker paused mid-instruction, so the
acoustic gaps were *real* — segmentation was correct and the result was still
garbage (`fragments_per_utterance: 1.0`, ten fragments, ten claims):

```
"They" / "don't use" / "your left" / "arm" / "anymore" / "."
"Keep" / "everything" / "local" / "."
```

No silence-based rule can join those; the speaker genuinely stopped. So
`omni_q.voice.IntentAccumulator` sits between segmentation and the runtime and
holds an utterance that **is not yet an instruction**, joining it with the next
one until the parse is actionable. It is runtime-shaped, so it drops in
wherever a `VoiceRuntime` goes:

```python
adapter = SpeechmaticsRealtimeAdapter(IntentAccumulator(runtime))
```

Completeness is decided by `voice.intent_capability` — the *same* function
`ingest_final` uses, so "complete" can never drift from what the runtime acts
on. The rules, each from an observed failure:

- **a constraint or mutation acts immediately** (unambiguous);
- **a bare goal waits for the sentence to end.** `COMMAND` comes from goal
  classification alone, which fires on fragments: `"Keep"` dispatched as an
  authorized command a second before `"everything local"` arrived, and
  `"They don't use"` before `"your left arm anymore"`. Terminal punctuation
  arrives as its own fragment moments later; if it never comes, the window
  backstop releases the text anyway;
- **safety words are never held.** Anything the interruption gate recognizes
  (STOP/HOLD/BACKOFF) is complete by definition — delaying one would be the
  worst bug in the file;
- **nothing is silently discarded.** A hold that expires is dispatched as an
  ordinary observation, so unexecuted speech still appears in the claim record;
- **authority is not consulted here.** It stays inside `ingest_final`, so
  unauthorized speakers still produce claims and explicit denials instead of
  vanishing before the boundary sees them.

Bounds: `--intent-window-ms` (default 4000) between utterances, 8 utterances
max. The count is deliberately generous — halting speech produced *six*
fragments for one instruction, and a limit of 3 cut it in half.

### Three bugs this found in code that was not the transport

0. **The safety gate swallowed a preference constraint.** Spoken live:
   *"Stop using the left arm."* `InterruptionGate` matches bare `\bstop\b`, so
   it claimed the whole sentence as an emergency STOP, `nlu` never parsed it,
   and with no interrupt handler registered the result was
   `interrupt_denied` — the instruction neither stopped anything **nor**
   changed the graph. `nlu._rule_stop_using_arm` exists precisely to parse
   *"stop using the right hand"*, so the two layers disagreed about the same
   phrase. The STOP pattern now excludes `stop use/using …` and nothing else:
   `"stop"`, `"stop now"`, `"stop moving"`, `"emergency stop"` and `"freeze"`
   all still fire, and *"stop using the left arm"* still stops using it — via
   `prefer_arm=right`, the correct path.

1. **`_rule_prefer_arm` inverted a negated instruction.** `"don't use your
   left"` — the state of the text before `"arm"` is transcribed — matched the
   positive phrase `"use your left"` and emitted `prefer_arm=left`, the exact
   opposite of what the operator said. `_rule_stop_using_arm` handles negation
   and wins on ordering for a *complete* sentence, which is why this hid until
   partial text was parsed. Now suppressed by an explicit negation guard.
2. **`UtteranceAggregator.tick` mistook provider latency for silence.** It
   compared wall-now against the transcript's *stream* end; a transcript
   arrives ~1.1 s after the audio it describes, so an 800 ms threshold read as
   idle the instant each fragment landed. Wall idle is now measured from
   fragment arrival and is only a safety net — stream-time gaps do the real
   segmentation.

### Measured live, 2026-09-13

First real session, HyperX QuadCast → 16 kHz mono → `eu.rt.speechmatics.com`,
128 s wall, 1266 chunks, receipt at `evidence/voice_latency.json`:

| | n | mean | p50 | p95 | max |
|---|---|---|---|---|---|
| partials | 7 | 416 ms | 423 ms | 453 ms | 453 ms |
| finals | 5 | 1043 ms | 1066 ms | 1116 ms | 1116 ms |

Zero adapter rejections, zero transcripts missing word confidence or speaker
labels. 415 empty transcripts over the silent stretches. **These are pre-
aggregation numbers**; utterance finals now additionally carry up to
`--silence-ms` of deliberate hold, which the receipt reports honestly rather
than quoting the fragment's figure.

### Latency receipt

`latency_ms` = monotonic wall-clock at message arrival minus the stream
position the transcript claims, i.e. how far behind live it was. The receipt
reports n / mean / p50 / p95 / max separately for partials and finals, plus
chunks sent, dropped-empty count, and adapter rejections. Replay-mode latency
is explicitly labelled as *not* a service number.

## Result-to-speech response seam

`VoiceSink` in `transport.py` is the existing mapped-transcript/result boundary.
The response output attaches there, after `VoiceRuntime` or the injected OMNI
handler returns its real result:

```text
Speechmatics final -> aggregator -> VoiceRuntime -> real result/receipt
                   -> VoiceSink.on_result -> SpeechmaticsTTS -> audio player
```

`SpeechResponseRenderer` speaks only committed, denied, authorized-but-not-
committed, or interruption results. Partials and observation-only finals stay
silent. It never treats a transcript as proof of execution or invents a
receipt. `SpeechmaticsTTS` calls the documented preview endpoint, returns a
scoped `SpeechOutputReceipt`, and defaults to in-memory Windows WAV playback;
other platforms can inject an `AudioPlayer`. The API key comes only from the
process environment. TTS failures are recorded as response failures without
changing the OMNI result.

For a live run, set `SPEECHMATICS_API_KEY` in the process environment and use
the existing transport CLI:

```powershell
$env:SPEECHMATICS_API_KEY = 'set-this-in-your-process-only'
& .venv/Scripts/python.exe integrations/speechmatics/scripts/run_voice_transport.py --mic --operator S1
```

## Remaining integration work

- [x] Speechmatics websocket/audio transport and measured latency receipt —
      built, tested offline, and **measured live** on 2026-09-13 (table above,
      `evidence/voice_latency.json`).
- [ ] Re-measure the receipt **after** utterance segmentation landed — the
      table above is pre-aggregation, so the final-latency column no longer
      describes what an operator experiences.
- [ ] Connect diarization/visual person association to a live camera provider
- [ ] Map authorized command candidates into OMNI's capability planner
- [ ] Surface voice claims, authority, and commitment events in the UI
- [ ] Benchmark overlap, interruption, ambiguity, stale world, and cross-org cases
