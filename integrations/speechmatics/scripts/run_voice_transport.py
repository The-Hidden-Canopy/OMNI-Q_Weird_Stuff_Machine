"""Run the Speechmatics realtime transport into the OMNI-Q voice boundary.

Speech -> transcription -> attributed claim -> intent candidate -> authority
decision -> (optionally) a graph mutation.  This script is the transport end of
that chain; the policy end lives in ``src/omni_q/voice.py`` and is not
duplicated here.

Modes
-----

Offline replay -- no API key, no network, no microphone::

    .venv/Scripts/python integrations/speechmatics/scripts/run_voice_transport.py \\
        --replay integrations/speechmatics/samples/session_replay.jsonl \\
        --operator S1

Live from a 16-bit mono WAV (paced at real time, so the latency receipt is
meaningful)::

    $env:SPEECHMATICS_API_KEY = "<key from portal.speechmatics.com>"
    .venv/Scripts/python integrations/speechmatics/scripts/run_voice_transport.py \\
        --file audio.wav --operator S1 --receipt evidence/voice_latency.json

Live from the default microphone (needs the optional ``sounddevice`` extra)::

    .venv/Scripts/python integrations/speechmatics/scripts/run_voice_transport.py \\
        --mic --operator S1 --record tmp/provider_messages.jsonl

``--record`` writes every provider message to JSON lines, which can then be fed
back through ``--replay`` to re-run the boundary offline against real audio.

The API key comes from ``SPEECHMATICS_API_KEY`` only.  There is deliberately no
``--api-key`` flag: it would leak the key into shell history and the process
table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

OMNIQ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OMNIQ_ROOT / "src"))
sys.path.insert(0, str(OMNIQ_ROOT))

from omni_q.voice import (  # noqa: E402
    IntentAccumulator,
    SpeechmaticsRealtimeAdapter,
    VoiceCapability,
    VoiceRuntime,
)
from integrations.speechmatics.mapper import MappedTranscript, TranscriptMapper  # noqa: E402
from integrations.speechmatics.response import SpeechmaticsTTS  # noqa: E402
from integrations.speechmatics.transport import (  # noqa: E402
    AudioStream,
    SpeechmaticsConfig,
    SpeechmaticsError,
    SpeechmaticsTransport,
    UtteranceAggregator,
    VoiceSink,
    api_key_from_env,
    microphone_audio_stream,
    raw_audio_stream,
    read_jsonl,
    replay_messages,
    wav_audio_stream,
)


def _print_result(mapped: MappedTranscript, result) -> None:
    payload = mapped.payload
    speaker = payload["speaker_id"]
    latency = payload["latency_ms"]
    if not mapped.final:
        print(f"  ~ [{speaker}] {mapped.text}   ({latency:.0f} ms behind live)")
        return
    if result is None:
        # Held by the IntentAccumulator: not yet something to act on.
        print(f"  . [{speaker}] {mapped.text}   (held, not an instruction yet)")
        return
    status = getattr(result, "status", "?")
    candidate = getattr(result, "candidate", None)
    action = getattr(candidate, "action", None) if candidate else None
    authority = getattr(result, "authority", None)
    allowed = getattr(authority, "allowed", None) if authority else None
    detail = f"status={status}"
    if action:
        detail += f" action={action}"
    if allowed is not None:
        detail += f" authorized={allowed}"
    mutation = getattr(result, "mutation", None)
    if mutation:
        detail += f" mutation={mutation}"
    # The claim may carry more than this fragment: the accumulator joins held
    # speech, so show what was actually acted on, not the last piece of it.
    claim = getattr(result, "claim", None)
    print(f"* [{speaker}] {getattr(claim, 'text', None) or mapped.text}")
    print(f"    {detail}   conf={payload['confidence']:.2f}  "
          f"{latency:.0f} ms behind live")


def _authorize_on_first_appearance(runtime: VoiceRuntime,
                                   operators: list[str]) -> None:
    """Grant operator authority the first time a configured speaker is heard.

    ``SpeakerRegistry.set_authority`` rejects a speaker it has never observed,
    and diarization labels (S1, S2, ...) only exist once the provider has
    assigned them -- so the grant is deferred to the ``on_final_speech`` hook,
    which ``VoiceRuntime.ingest_final`` fires after the speaker is registered
    and before the authority decision for that same utterance.  Speakers not
    named on the command line are never granted anything.
    """
    pending = {s for s in operators}

    def _grant(*, event, **_ignored) -> None:
        if event.speaker_id in pending:
            runtime.registry.set_authority(
                event.speaker_id, "operator",
                [VoiceCapability.COMMAND.value,
                 VoiceCapability.GRAPH_MUTATION.value],
            )
            pending.discard(event.speaker_id)
            print(f"[voice] {event.speaker_id} granted operator authority")

    runtime.on("on_final_speech", _grant)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Speechmatics realtime -> OMNI-Q voice boundary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Not required at parse time so --list-devices can stand alone; main()
    # enforces "exactly one source" for every other invocation.
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--file", type=Path, help="16-bit mono WAV to stream")
    source.add_argument("--raw", type=Path, help="headerless 16-bit LE PCM file")
    source.add_argument("--mic", action="store_true", help="default input device")
    source.add_argument("--replay", type=Path,
                        help="recorded provider messages (JSON lines); offline")

    parser.add_argument("--session-id", default="voice_session_01")
    parser.add_argument("--org-id", default="omni_q_local")
    parser.add_argument("--operator", action="append", default=[],
                        metavar="SPEAKER_ID",
                        help="grant this Speechmatics speaker label (e.g. S1) "
                             "command + graph_mutation authority; repeatable. "
                             "Unlisted speakers stay unauthorized by design.")
    parser.add_argument("--mock-engine", action="store_true",
                        help="attach a RuntimeMutator over omni_q's mock engine "
                             "so authorized constraint changes actually commit "
                             "(without it, finals stop at 'authorized')")
    parser.add_argument("--no-response", action="store_true",
                        help="disable Speechmatics response speech for live runs")
    parser.add_argument("--response-voice", default="sarah",
                        help="Speechmatics TTS voice for live responses (default: sarah)")
    parser.add_argument("--intent-window-ms", type=float, default=4000.0,
                        help="how long an incomplete utterance is held waiting "
                             "for the rest of the instruction (default 4000)")
    parser.add_argument("--no-accumulate", action="store_true",
                        help="dispatch every utterance immediately, even when "
                             "it is only part of an instruction")
    parser.add_argument("--silence-ms", type=float, default=800.0,
                        help="silence that ends an utterance; provider finals "
                             "are reassembled up to this gap (default 800)")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="dispatch raw provider finals instead of whole "
                             "utterances -- fragments one sentence into several "
                             "claims, so the NLU sees 'use your' alone")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-delay", type=float, default=1.5,
                        help="provider finalization delay in seconds (0.7-4.0)")
    parser.add_argument("--no-partials", action="store_true")
    parser.add_argument("--no-diarization", action="store_true")
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--device", default=None,
                        help="input device index or name for --mic "
                             "(default: the system default input)")
    parser.add_argument("--list-devices", action="store_true",
                        help="print input devices with a live level check and exit")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="only used for --raw and --mic")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--url", default=SpeechmaticsConfig.url)
    parser.add_argument("--record", type=Path,
                        help="write every provider message here as JSON lines")
    parser.add_argument("--receipt", type=Path,
                        help="write the measured latency receipt here")
    parser.add_argument("--no-realtime-pacing", action="store_true",
                        help="send file audio as fast as the socket accepts it "
                             "(latency numbers then mean throughput, not lag)")
    return parser


def list_input_devices(seconds: float = 1.0) -> int:
    """Print input devices and measure the default one.

    The level check exists because a hardware-muted microphone produces a
    perfectly healthy-looking stream of near-silence, and the resulting empty
    transcript looks exactly like a transport bug.
    """
    try:
        import sounddevice  # type: ignore import-not-found
    except Exception:
        print("sounddevice is not installed: pip install -e .[speechmatics-mic]")
        return 2
    default_in = sounddevice.default.device[0]
    for index, device in enumerate(sounddevice.query_devices()):
        if device["max_input_channels"] > 0:
            mark = "*" if index == default_in else " "
            print(f"{mark} [{index}] {device['name']}  "
                  f"in={device['max_input_channels']} "
                  f"sr={int(device['default_samplerate'])}")
    print(f"\nmeasuring the default input for {seconds:g}s -- say something...")
    try:
        with sounddevice.RawInputStream(samplerate=16000, blocksize=1600,
                                        dtype="int16", channels=1) as stream:
            raw, overflowed = stream.read(int(16000 * seconds))
    except Exception as exc:
        print(f"could not open the default input at 16 kHz mono: {exc}")
        return 2
    samples = memoryview(bytes(raw)).cast("h")
    peak = max((abs(s) for s in samples), default=0)
    print(f"peak amplitude {peak} / 32767" + (" (overflow)" if overflowed else ""))
    if peak < 200:
        print("that is effectively silence -- check the hardware mute "
              "(the QuadCast's tap-to-mute kills the stream without an error) "
              "and Windows' microphone privacy setting")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # cp1252 consoles drop transcripts
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.list_devices:
        return list_input_devices()
    if not (args.file or args.raw or args.mic or args.replay):
        parser.error("one of --file, --raw, --mic or --replay is required")

    mutator = None
    if args.mock_engine:
        from omni_q import build_mock_engine  # noqa: PLC0415 - optional path
        from omni_q.mutation import RuntimeMutator

        mutator = RuntimeMutator(build_mock_engine())
        print("[voice] mock engine attached: authorized constraint changes "
              "will be applied through RuntimeMutator")

    runtime = VoiceRuntime(args.session_id, args.org_id, mutator=mutator)
    if args.operator:
        _authorize_on_first_appearance(runtime, args.operator)
        print(f"[voice] operator roles pending for {', '.join(args.operator)} "
              "(command + graph_mutation, applied when each first speaks)")
    else:
        print("[voice] no --operator given: every speaker stays unauthorized, "
              "so finals will be observed/denied, never committed")

    mapper = TranscriptMapper(session_id=args.session_id, org_id=args.org_id)
    accumulator = None
    if not args.no_accumulate:
        accumulator = IntentAccumulator(runtime, window_ms=args.intent_window_ms,
                                        sequence_source=mapper.next_sequence)
        print(f"[voice] intent accumulation on: utterances that are not yet an "
              f"instruction are held up to {args.intent_window_ms:.0f} ms and joined")
    adapter = SpeechmaticsRealtimeAdapter(accumulator or runtime)
    speech_output = None
    if not args.replay and not args.no_response:
        # Read the key before opening the microphone or websocket. It remains
        # process-local and is never printed or written into a receipt.
        speech_output = SpeechmaticsTTS(
            api_key_from_env(),
            voice=args.response_voice,
        )
        print(f"[voice] response playback on: Speechmatics TTS voice={args.response_voice}")
    sink = VoiceSink(
        adapter,
        on_result=_print_result,
        speech_output=speech_output,
        response_async=speech_output is not None,
    )
    aggregator = None
    if not args.no_aggregate:
        aggregator = UtteranceAggregator(
            sink, mapper, silence_ms=args.silence_ms,
            # The voice boundary's own gate decides what is urgent; the
            # transport only asks, so safety policy stays in one place.
            urgent=lambda text: runtime.interruption_gate.inspect(text).detected,
        )

    config = SpeechmaticsConfig(
        language=args.language,
        enable_partials=not args.no_partials,
        max_delay=args.max_delay,
        diarization=None if args.no_diarization else "speaker",
        max_speakers=args.max_speakers,
        sample_rate=args.sample_rate,
        chunk_ms=args.chunk_ms,
        url=args.url,
    )
    transport = SpeechmaticsTransport(config=config, record_path=args.record)

    started_wall = time.time()
    started_mono = time.monotonic()

    if args.replay:
        mapper.start_stream(epoch_ns=0)
        stats = replay_messages(read_jsonl(args.replay), sink, mapper, aggregator)
        if accumulator is not None:
            accumulator.flush()  # nothing held is silently discarded
        audio = AudioStream(sample_rate=config.sample_rate, encoding=config.encoding,
                            chunks=lambda: iter(()),  # type: ignore[arg-type]
                            description=f"replay:{args.replay.name}")
        receipt = transport.receipt(
            sink, mapper, audio, chunks_sent=0,
            wall_s=time.monotonic() - started_mono,
            started_wall=started_wall, mode="replay", aggregator=aggregator,
        )
        receipt["replay"] = stats
        # A replay's clock has no relation to when the audio was captured, so
        # its latency numbers describe this machine's mapping cost only.
        receipt["latency_definition"] = (
            "replay mode: elapsed time against a synthetic epoch, NOT provider "
            "latency -- do not quote these as service numbers"
        )
    else:
        realtime = not args.no_realtime_pacing
        if args.file:
            audio = wav_audio_stream(args.file, chunk_ms=args.chunk_ms,
                                     realtime=realtime)
        elif args.raw:
            audio = raw_audio_stream(args.raw, sample_rate=args.sample_rate,
                                     chunk_ms=args.chunk_ms, realtime=realtime)
        else:
            device: int | str | None = args.device
            if isinstance(device, str) and device.isdigit():
                device = int(device)
            audio = microphone_audio_stream(sample_rate=args.sample_rate,
                                            chunk_ms=args.chunk_ms,
                                            device=device)
        api_key_from_env()  # fail before opening a socket if the key is absent
        print(f"[speechmatics] streaming {audio.description} at "
              f"{audio.sample_rate} Hz to {config.url}")
        if args.mic:
            print("[speechmatics] speak now; Ctrl+C to stop")
        try:
            receipt = asyncio.run(transport.run(
                audio, sink, mapper, aggregator,
                on_tick=accumulator.tick if accumulator else None,
                on_close=accumulator.flush if accumulator else None))
        except KeyboardInterrupt:
            print("\n[speechmatics] stopped by operator")
            # Order matters: the aggregator feeds the accumulator, so draining
            # it second would leave the last utterance stuck in the buffer.
            if aggregator is not None:
                aggregator.close()
            if accumulator is not None:
                accumulator.flush()
            receipt = transport.receipt(
                sink, mapper, audio, chunks_sent=0,
                wall_s=time.monotonic() - started_mono,
                started_wall=started_wall, mode="interrupted",
                aggregator=aggregator,
            )

    if accumulator is not None:
        receipt["intent_accumulation"] = accumulator.stats()

    print("\n--- receipt ---")
    print(json.dumps(receipt, indent=2))
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(f"[speechmatics] receipt written to {args.receipt}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpeechmaticsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
