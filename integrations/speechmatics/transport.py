"""Speechmatics realtime WebSocket transport for the OMNI-Q voice boundary.

This is the vendor half of the boundary described in
``integrations/speechmatics/README.md``.  It owns the socket, the audio pacing,
and the latency receipt; it owns no policy.  Everything it learns is handed to
``omni_q.voice.SpeechmaticsRealtimeAdapter`` as a plain mapping, so the
authority/arbitration rules stay in ``src/omni_q/voice.py`` and stay
provider-neutral.

Protocol implemented (Speechmatics realtime v2)::

    connect wss://<region>.rt.speechmatics.com/v2/   Authorization: Bearer <key>
      -> StartRecognition {audio_format, transcription_config}   (exactly once)
      <- RecognitionStarted
      -> binary PCM chunks            <- AudioAdded {seq_no}
                                      <- AddPartialTranscript / AddTranscript
      -> EndOfStream {last_seq_no}    <- EndOfTranscript
                                      <- Error {type, reason, code}

Credentials
-----------
The API key is read from the ``SPEECHMATICS_API_KEY`` environment variable and
from nowhere else.  It is never written to a file, a receipt, a log line, or an
exception message -- ``_redact`` scrubs it from anything this module raises.
Do not add a ``--api-key`` flag: that would put the key in shell history and in
the process table.

Three run modes, in increasing order of what they need:

``replay``
    Feed recorded provider messages (JSON lines) through the mapper and the
    real adapter.  No key, no network, no audio device.  This is what the tests
    use and what makes the mapping rules reviewable.
``file``
    Stream a WAV/raw-PCM file to the live service, paced at real time so the
    measured latency means something.  Needs a key and network.
``mic``
    Stream the default input device.  Additionally needs ``sounddevice``.
"""

from __future__ import annotations

import asyncio
import contextlib
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import statistics
import time
import wave
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterable, Iterator, Mapping

from .mapper import FINAL_MESSAGE, MappedTranscript, PARTIAL_MESSAGE, TranscriptMapper
from .response import (
    SpeechOutputReceipt,
    SpeechResponseRenderer,
    SpeechmaticsTTS,
)

__all__ = [
    "AudioStream",
    "SpeechmaticsConfig",
    "SpeechmaticsError",
    "SpeechmaticsTransport",
    "UtteranceAggregator",
    "VoiceSink",
    "api_key_from_env",
    "raw_audio_stream",
    "replay_messages",
    "wav_audio_stream",
]

API_KEY_ENV = "SPEECHMATICS_API_KEY"
DEFAULT_URL = "wss://eu.rt.speechmatics.com/v2"


class SpeechmaticsError(RuntimeError):
    """Transport-level failure (auth, protocol, or provider Error message)."""


# ---------------------------------------------------------------------------
# credentials
# ---------------------------------------------------------------------------

def api_key_from_env(env: Mapping[str, str] | None = None) -> str:
    """Read the API key from the environment.  Never logged, never echoed."""
    source = os.environ if env is None else env
    key = (source.get(API_KEY_ENV) or "").strip()
    if not key:
        raise SpeechmaticsError(
            f"{API_KEY_ENV} is not set. Export it in this shell only, e.g.\n"
            f'  PowerShell:  $env:{API_KEY_ENV} = "<key from portal.speechmatics.com>"\n'
            f'  bash:        export {API_KEY_ENV}="<key>"\n'
            "The key must not be passed as a CLI flag or written into any file "
            "in this repository."
        )
    return key


def _redact(text: str, key: str | None) -> str:
    if key and key in text:
        return text.replace(key, "<redacted>")
    return text


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass
class SpeechmaticsConfig:
    """Everything that goes into ``StartRecognition``, plus transport knobs."""

    language: str = "en"
    enable_partials: bool = True
    #: Provider-side finalization delay. Lower = faster finals, slightly worse
    #: accuracy. 1.0 is the realtime-agent end of the documented 0.7-4.0 range.
    max_delay: float = 1.5
    max_delay_mode: str = "flexible"
    #: "enhanced" is materially more accurate than the "standard" default and
    #: is what a demo should run. Measured 2026-09-13 on standard: the operator's
    #: name "Omni" came back as "They", "Deep" and "Omnipresent" across three
    #: sessions.
    operating_point: str | None = "enhanced"
    #: ``None`` disables diarization; "speaker" makes the provider label S1/S2,
    #: which is what the voice registry needs for multi-speaker authority.
    diarization: str | None = "speaker"
    max_speakers: int | None = None
    sample_rate: int = 16000
    encoding: str = "pcm_s16le"
    chunk_ms: int = 100
    url: str = DEFAULT_URL
    #: Domain words the recognizer would otherwise never guess. "Omni" is the
    #: wake word for this system and was mis-transcribed in every session
    #: recorded on 2026-09-13; custom vocabulary is the documented fix.
    additional_vocab: tuple[str, ...] = (
        "Omni", "OMNI-Q", "gripper", "bimanual", "handoff", "regrasp",
    )

    def start_recognition(self) -> dict[str, Any]:
        transcription: dict[str, Any] = {
            "language": self.language,
            "enable_partials": bool(self.enable_partials),
            "max_delay": float(self.max_delay),
            "max_delay_mode": self.max_delay_mode,
        }
        if self.operating_point:
            transcription["operating_point"] = self.operating_point
        if self.additional_vocab:
            transcription["additional_vocab"] = [
                {"content": word} for word in self.additional_vocab
            ]
        if self.diarization:
            transcription["diarization"] = self.diarization
            if self.max_speakers:
                transcription["speaker_diarization_config"] = {
                    "max_speakers": int(self.max_speakers)
                }
        return {
            "message": "StartRecognition",
            "audio_format": {
                "type": "raw",
                "encoding": self.encoding,
                "sample_rate": int(self.sample_rate),
            },
            "transcription_config": transcription,
        }

    def as_receipt(self) -> dict[str, Any]:
        """Config as recorded in the receipt.  Contains no credentials."""
        return asdict(self)


# ---------------------------------------------------------------------------
# audio sources
# ---------------------------------------------------------------------------

@dataclass
class AudioStream:
    """A paced source of raw PCM chunks plus the facts the provider needs."""

    sample_rate: int
    encoding: str
    chunks: Callable[[], AsyncIterator[bytes]]
    description: str
    duration_s: float | None = None


def _pace(chunk_bytes: int, sample_rate: int, bytes_per_sample: int) -> float:
    return chunk_bytes / float(sample_rate * bytes_per_sample)


def wav_audio_stream(path: str | Path, *, chunk_ms: int = 100,
                     realtime: bool = True) -> AudioStream:
    """Stream a 16-bit PCM WAV file.

    The file's own sample rate is reported to the provider rather than
    resampled here -- resampling in Python would be a silent quality change
    that the latency receipt could not account for.
    """
    wav_path = Path(path)
    with contextlib.closing(wave.open(str(wav_path), "rb")) as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
    if width != 2:
        raise SpeechmaticsError(
            f"{wav_path.name}: expected 16-bit PCM (sampwidth 2), got {width}."
        )
    if channels != 1:
        raise SpeechmaticsError(
            f"{wav_path.name}: expected mono, got {channels} channels. "
            "Convert first (e.g. ffmpeg -i in.wav -ac 1 -ar 16000 out.wav) so "
            "no lossy downmix happens inside the transport."
        )
    frames_per_chunk = max(1, int(rate * chunk_ms / 1000))

    async def _chunks() -> AsyncIterator[bytes]:
        deadline = time.monotonic()
        with contextlib.closing(wave.open(str(wav_path), "rb")) as source:
            while True:
                data = source.readframes(frames_per_chunk)
                if not data:
                    return
                yield data
                if realtime:
                    deadline += _pace(len(data), rate, 2)
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)

    return AudioStream(
        sample_rate=rate,
        encoding="pcm_s16le",
        chunks=_chunks,
        description=f"wav:{wav_path.name}",
        duration_s=frames / float(rate) if rate else None,
    )


def raw_audio_stream(path: str | Path, *, sample_rate: int = 16000,
                     chunk_ms: int = 100, realtime: bool = True) -> AudioStream:
    """Stream a headerless 16-bit little-endian PCM file at ``sample_rate``."""
    raw_path = Path(path)
    size = raw_path.stat().st_size
    chunk_bytes = max(2, int(sample_rate * 2 * chunk_ms / 1000))

    async def _chunks() -> AsyncIterator[bytes]:
        deadline = time.monotonic()
        with raw_path.open("rb") as handle:
            while True:
                data = handle.read(chunk_bytes)
                if not data:
                    return
                yield data
                if realtime:
                    deadline += _pace(len(data), sample_rate, 2)
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)

    return AudioStream(
        sample_rate=sample_rate,
        encoding="pcm_s16le",
        chunks=_chunks,
        description=f"raw:{raw_path.name}",
        duration_s=size / float(sample_rate * 2),
    )


def microphone_audio_stream(*, sample_rate: int = 16000,
                            chunk_ms: int = 100,
                            device: int | str | None = None) -> AudioStream:
    """Stream the default (or named) input device via ``sounddevice``.

    Optional dependency on purpose: the file and replay modes must keep working
    on a box with no PortAudio and no microphone.
    """
    try:
        import sounddevice  # type: ignore import-not-found
    except Exception as exc:  # pragma: no cover - depends on local audio stack
        raise SpeechmaticsError(
            "microphone capture needs the optional 'sounddevice' package: "
            "pip install -e .[speechmatics-mic]"
        ) from exc

    frames_per_chunk = max(1, int(sample_rate * chunk_ms / 1000))

    async def _chunks() -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)

        def _callback(indata, _frames, _time, status) -> None:  # pragma: no cover
            if status:
                # Overflow means we are not draining fast enough; the receipt's
                # latency numbers would be meaningless if we hid it.
                print(f"[speechmatics] input stream status: {status}", flush=True)
            loop.call_soon_threadsafe(_offer, bytes(indata))

        def _offer(data: bytes) -> None:  # pragma: no cover
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(data)

        stream = sounddevice.RawInputStream(
            samplerate=sample_rate, blocksize=frames_per_chunk, device=device,
            dtype="int16", channels=1, callback=_callback,
        )
        with stream:  # pragma: no cover - hardware path
            while True:
                yield await queue.get()

    return AudioStream(
        sample_rate=sample_rate,
        encoding="pcm_s16le",
        chunks=_chunks,
        description=f"mic:{device if device is not None else 'default'}",
        duration_s=None,
    )


# ---------------------------------------------------------------------------
# sink: mapped transcript -> voice adapter
# ---------------------------------------------------------------------------

class VoiceSink:
    """Deliver mapped transcripts to the adapter and measure what happened.

    Adapter rejections are counted, not swallowed and not fatal: one
    out-of-scope or malformed transcript must not tear down a live session, but
    it must show up in the receipt.
    """

    def __init__(self, adapter: Any, *,
                 on_result: Callable[[MappedTranscript, Any], None] | None = None,
                 final_kwargs: Mapping[str, Any] | None = None,
                 response_renderer: SpeechResponseRenderer | None = None,
                 speech_output: SpeechmaticsTTS | None = None,
                 response_async: bool = False) -> None:
        if not isinstance(response_async, bool):
            raise TypeError("response_async must be a bool")
        self.adapter = adapter
        self.on_result = on_result
        self.final_kwargs = dict(final_kwargs or {})
        self.response_renderer = response_renderer or SpeechResponseRenderer()
        self.speech_output = speech_output
        self.response_async = response_async and speech_output is not None
        self._response_executor = (
            ThreadPoolExecutor(max_workers=1,
                               thread_name_prefix="omni-q-voice-response")
            if self.response_async else None
        )
        self._response_futures: list[Future[None]] = []
        self.partials = 0
        self.finals = 0
        self.held = 0
        self.rejected: list[dict[str, Any]] = []
        self.outputs: list[SpeechOutputReceipt] = []
        self.response_errors: list[dict[str, Any]] = []
        self.partial_latency_ms: list[float] = []
        self.final_latency_ms: list[float] = []
        self.unmeasured_confidence = 0
        self.unlabelled_speaker = 0

    def deliver(self, mapped: MappedTranscript) -> Any:
        if not mapped.confidence_measured:
            self.unmeasured_confidence += 1
        if not mapped.speaker_labelled:
            self.unlabelled_speaker += 1
        try:
            if mapped.final:
                result = self.adapter.on_final(mapped.payload, **self.final_kwargs)
                if result is None:
                    # An IntentAccumulator is holding this utterance for the
                    # next one. Not a dispatch and not a failure.
                    self.held += 1
                else:
                    self.finals += 1
                    self.final_latency_ms.append(mapped.latency_ms)
            else:
                result = self.adapter.on_partial(mapped.payload)
                self.partials += 1
                self.partial_latency_ms.append(mapped.latency_ms)
        except Exception as exc:  # adapter/runtime validation errors
            self.rejected.append({
                "final": mapped.final,
                "sequence": mapped.payload.get("sequence"),
                "error": f"{type(exc).__name__}: {exc}",
            })
            return None
        if self.on_result is not None:
            self.on_result(mapped, result)
        if mapped.final and result is not None and self.speech_output is not None:
            if self._response_executor is None:
                self._speak(result)
            else:
                self._response_futures.append(
                    self._response_executor.submit(self._speak, result)
                )
        return result

    def _speak(self, result: Any) -> None:
        """Render and play only a completed result from the real voice boundary.

        Response playback is deliberately best-effort at the transport edge:
        an unavailable speaker or TTS endpoint must be visible in the receipt,
        but must not turn a valid authority/mutation decision into a transport
        failure or retry the mutation.
        """
        response = None
        runtime = getattr(self.adapter, "runtime", None)
        bus = getattr(runtime, "bus", None)
        started = False
        try:
            response = self.response_renderer.render(result)
            if response is None:
                return
            if bus is not None:
                bus.publish(
                    "voice.response.started",
                    source="speechmatics-tts",
                    response_id=response.response_id,
                    status=response.source_status,
                )
                started = True
            receipt = self.speech_output.speak(response)
        except Exception as exc:  # noqa: BLE001 - response is a side-effect boundary
            response_id = getattr(response, "response_id", None)
            self.response_errors.append({
                "response_id": response_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
            if bus is not None:
                bus.publish(
                    "voice.response.failed",
                    source="speechmatics-tts",
                    response_id=response_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if started:
                    bus.publish(
                        "voice.response.stopped",
                        source="speechmatics-tts",
                        response_id=response_id,
                        status=getattr(response, "source_status", None),
                        ok=False,
                    )
            return
        self.outputs.append(receipt)
        if bus is not None:
            bus.publish(
                "voice.response.sent",
                source="speechmatics-tts",
                response=receipt.as_dict(),
            )
            bus.publish(
                "voice.response.stopped",
                source="speechmatics-tts",
                response_id=receipt.response_id,
                status=getattr(response, "source_status", None),
                ok=True,
            )

    def wait_for_responses(self, timeout_s: float | None = None) -> None:
        """Drain queued response playback before a receipt is finalized."""
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive when provided")
        deadline = (time.monotonic() + timeout_s
                    if timeout_s is not None else None)
        futures = tuple(self._response_futures)
        for future in futures:
            remaining = (None if deadline is None
                         else max(0.0, deadline - time.monotonic()))
            future.result(timeout=remaining)
        if futures:
            self._response_futures = [
                future for future in self._response_futures
                if not future.done()
            ]

    def stats(self) -> dict[str, Any]:
        return {
            "partials": self.partials,
            "finals": self.finals,
            "held_for_accumulation": self.held,
            "rejected": len(self.rejected),
            "rejections": self.rejected[:20],
            "transcripts_without_word_confidence": self.unmeasured_confidence,
            "transcripts_without_speaker_label": self.unlabelled_speaker,
            "partial_latency_ms": _latency_summary(self.partial_latency_ms),
            "final_latency_ms": _latency_summary(self.final_latency_ms),
            "responses_sent": len(self.outputs),
            "response_failures": len(self.response_errors),
            "response_errors": self.response_errors[:20],
        }


class UtteranceAggregator:
    """Reassemble provider finals into whole utterances.

    Measured on a live session, 2026-09-13: "Omni, don't use your left arm
    anymore" came back as **five** separate ``AddTranscript`` messages --
    "Omni.", "Don't", "use your", "left arm", "anymore." -- because realtime
    finalizes on its own latency budget, not on sentence boundaries.  Each
    fragment reached ``nlu.parse`` alone, where "use your" and "left arm" are
    meaningless, so an instruction the operator clearly gave was recorded as
    five observations and never became a graph mutation.

    Raising ``max_delay`` reduces fragmentation but cannot fix it: the provider
    does not know where the operator's sentence ends, and buying fewer
    fragments costs partial latency on every utterance.  So segmentation is
    done here, on the speaker's own silence:

    * fragments accumulate while the same speaker keeps talking;
    * a gap of ``silence_ms`` in *stream* time, a speaker change, or
      ``max_utterance_ms`` of continuous speech closes the utterance;
    * ``tick()`` closes a trailing utterance on wall-clock idle, so the last
      thing said does not hang until the socket closes.

    Partials are never buffered -- they pass straight through, which is what
    keeps the "N ms behind live" feedback responsive while finals wait.
    """

    def __init__(self, sink: "VoiceSink", mapper: TranscriptMapper, *,
                 silence_ms: float = 800.0,
                 max_utterance_ms: float = 15000.0,
                 idle_grace_ms: float = 1400.0,
                 flush_on_terminal_punctuation: bool = True,
                 min_words_for_punctuation_flush: int = 3,
                 urgent: Callable[[str], bool] | None = None) -> None:
        self.flush_on_terminal_punctuation = bool(flush_on_terminal_punctuation)
        # Speechmatics punctuates a vocative: the real capture starts with the
        # fragment "Omni." Treating that period as a sentence end split the
        # address off from the instruction that followed it, so a punctuation
        # flush additionally requires enough words to be a sentence.
        self.min_words_for_punctuation_flush = int(min_words_for_punctuation_flush)
        # Predicate for "dispatch this right now" -- the CLI passes the voice
        # boundary's own interruption gate. Without it a bare "Stop!" is too
        # short to trigger the punctuation flush and would sit in the buffer
        # for the full idle timeout, which is the one utterance that must never
        # wait. Policy stays in voice.py; the transport only asks.
        self.urgent = urgent
        self.sink = sink
        self.mapper = mapper
        self.silence_ms = float(silence_ms)
        self.max_utterance_ms = float(max_utterance_ms)
        # Wall-clock flushing has to allow for the provider's own cadence: a
        # transcript arrives ~max_delay + network after the audio it describes
        # (measured p50 1066 ms on 2026-09-13). Comparing wall-now against the
        # transcript's *stream* end therefore reads ~1.1 s of idle the instant
        # a fragment lands, and an 800 ms threshold flushed after every single
        # fragment -- the live receipt showed 10 fragments becoming 10
        # utterances. Wall idle is now measured from fragment *arrival*, and
        # only as a safety net; real segmentation uses stream-time gaps.
        self.idle_grace_ms = float(idle_grace_ms)
        self._last_arrival_ns: int | None = None
        self._latest_partial = ""
        self._buffer: list[MappedTranscript] = []
        self._last_flush_end_ns: int | None = None
        self.utterances = 0
        self.fragments = 0

    # -- input ----------------------------------------------------------

    def offer(self, mapped: MappedTranscript) -> Any:
        if not mapped.final:
            # Kept as the provider's current hypothesis for in-flight audio.
            # Partials run ~400 ms ahead of finals, so this is how the urgent
            # check below can know a sentence is still going.
            self._latest_partial = mapped.text
            return self.sink.deliver(mapped)  # partials are never held
        self.fragments += 1
        if self._buffer and self._should_close(mapped):
            self.flush()
        self._buffer.append(mapped)
        self._last_arrival_ns = int(self.mapper.clock())
        if self._span_ms() >= self.max_utterance_ms:
            return self.flush()
        if self.urgent is not None and self._is_urgent():
            return self.flush()
        if (self.flush_on_terminal_punctuation
                and _ends_sentence(mapped.text)
                and len(self._buffered_text().split())
                >= self.min_words_for_punctuation_flush):
            # The provider sends sentence-final punctuation as its own
            # fragment. Waiting out silence after that buys nothing and cost
            # 2.2 s of the 3.3 s an operator actually experienced (measured
            # live 2026-09-13). If the speaker carries on, the continuation is
            # simply a new utterance -- which is what it is.
            return self.flush()
        return None

    def tick(self, now_ns: int | None = None) -> Any:
        """Safety net: flush when nothing has *arrived* for long enough.

        Measured against arrival, not against the transcript's stream end --
        see ``idle_grace_ms``. Stream-time gaps (``_should_close`` and
        ``mark_silence``) are what actually segment speech; this only stops the
        last utterance of a session from waiting forever.
        """
        if not self._buffer or self._last_arrival_ns is None:
            return None
        now = int(self.mapper.clock() if now_ns is None else now_ns)
        idle_ms = (now - self._last_arrival_ns) / 1_000_000.0
        if idle_ms >= self.silence_ms + self.idle_grace_ms:
            return self.flush()
        return None

    def mark_silence(self, stream_ns: int | None) -> Any:
        """Close an utterance using the provider's own silence evidence.

        Speechmatics reports silence as empty ``AddTranscript`` messages, which
        the mapper drops before they reach the voice boundary.  Dropping the
        message is right; dropping its *timing* is not -- without this, a whole
        session's utterances merge into one, because nothing else tells a
        replay that the speaker stopped talking.
        """
        if stream_ns is None or not self._buffer:
            return None
        # Stream time against stream time -- no provider latency in either
        # term, unlike tick()'s wall-clock safety net.
        quiet_ms = (stream_ns - self._buffer[-1].payload["t_end_ns"]) / 1_000_000.0
        return self.flush() if quiet_ms >= self.silence_ms else None

    def close(self) -> Any:
        return self.flush()

    # -- internals ------------------------------------------------------

    def _should_close(self, incoming: MappedTranscript) -> bool:
        last = self._buffer[-1]
        if incoming.payload["speaker_id"] != last.payload["speaker_id"]:
            return True
        gap_ms = (incoming.payload["t_start_ns"]
                  - last.payload["t_end_ns"]) / 1_000_000.0
        return gap_ms >= self.silence_ms

    def _is_urgent(self) -> bool:
        """Is the buffer an emergency, given what is still being said?

        A bare "Stop!" must dispatch at once. But "stop using the left arm" is
        an ordinary constraint whose first fragment is also the word "stop" --
        and flushing there severs the sentence, leaving "using the left arm" to
        parse as a preference *for* the left arm. That happened live on
        2026-09-13: the operator said stop using it and the graph committed
        use it.

        The provider already tells us which one it is. Partials lead finals by
        ~400 ms, so by the time the final "Stop" arrives the partial reads
        "Stop using" -- and the interruption gate rejects that. When the
        in-flight hypothesis is not urgent, neither is the fragment.
        """
        assert self.urgent is not None
        if not self.urgent(self._buffered_text()):
            return False
        hypothesis = self._latest_partial.strip()
        if hypothesis and not self.urgent(hypothesis):
            return False
        return True

    def _buffered_text(self) -> str:
        return _join_fragments(f.text for f in self._buffer)

    def _span_ms(self) -> float:
        if not self._buffer:
            return 0.0
        return (self._buffer[-1].payload["t_end_ns"]
                - self._buffer[0].payload["t_start_ns"]) / 1_000_000.0

    def flush(self) -> Any:
        if not self._buffer:
            return None
        fragments, self._buffer = self._buffer, []
        self._latest_partial = ""  # belongs to the utterance just closed
        first, last = fragments[0].payload, fragments[-1].payload
        text = _join_fragments(f.text for f in fragments)
        measured = [f for f in fragments if f.confidence_measured]
        scored = measured or fragments
        weights = [max(1, len(f.text.split())) for f in scored]
        confidence = sum(f.payload["confidence"] * w
                         for f, w in zip(scored, weights)) / sum(weights)
        pause_ms = None
        if self._last_flush_end_ns is not None:
            pause_ms = max(0.0, (first["t_start_ns"]
                                 - self._last_flush_end_ns) / 1_000_000.0)
        now_ns = int(self.mapper.clock())
        payload = dict(first)
        payload.update({
            "text": text,
            "t_end_ns": max(last["t_end_ns"], first["t_start_ns"] + 1),
            # A fresh number, not the fragment's: partials emitted while this
            # utterance buffered have already moved the counter past it.
            "sequence": self.mapper.next_sequence(),
            "confidence": confidence,
            "pause_ms": pause_ms,
            # Recomputed at flush, so it includes the time the utterance was
            # deliberately held. This is the honest operator-visible latency,
            # and it is larger than any single fragment's.
            "latency_ms": max(0.0, (now_ns - last["t_end_ns"]) / 1_000_000.0),
        })
        self._last_flush_end_ns = payload["t_end_ns"]
        self.utterances += 1
        merged = MappedTranscript(
            payload=payload, final=True,
            confidence_measured=all(f.confidence_measured for f in fragments),
            speaker_labelled=all(f.speaker_labelled for f in fragments),
        )
        return self.sink.deliver(merged)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "silence_ms": self.silence_ms,
            "max_utterance_ms": self.max_utterance_ms,
            "provider_final_fragments": self.fragments,
            "utterances_dispatched": self.utterances,
            "fragments_per_utterance": (
                round(self.fragments / self.utterances, 2) if self.utterances else None
            ),
        }


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?"))


def _join_fragments(parts: Iterable[str]) -> str:
    """Join fragments without gluing a space in front of punctuation."""
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out and part[0] not in ",.!?;:":
            out += " "
        out += part
    return out


def _latency_summary(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"n": 0}
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 2),
        "p50": round(ordered[len(ordered) // 2], 2),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
        "max": round(ordered[-1], 2),
    }


# ---------------------------------------------------------------------------
# offline replay
# ---------------------------------------------------------------------------

def replay_messages(messages: Iterable[Mapping[str, Any]], sink: VoiceSink,
                    mapper: TranscriptMapper,
                    aggregator: UtteranceAggregator | None = None) -> dict[str, Any]:
    """Run recorded provider messages through the mapper and the adapter.

    No socket and no key: this is the path that proves the mapping rules and
    the voice boundary agree, and the one the tests exercise.
    """
    seen = 0
    for message in messages:
        kind = message.get("message")
        if kind == "RecognitionStarted":
            mapper.start_stream()
            continue
        if kind == "Error":
            raise SpeechmaticsError(_provider_error(message))
        mapped = mapper.map_message(message)
        if mapped is None:
            if aggregator is not None and kind == FINAL_MESSAGE:
                aggregator.mark_silence(mapper.last_empty_end_ns)
            continue
        seen += 1
        if aggregator is not None:
            aggregator.offer(mapped)
        else:
            sink.deliver(mapped)
    if aggregator is not None:
        aggregator.close()  # the last utterance has no following gap to close it
    return {"messages_mapped": seen, "dropped_empty": mapper.dropped_empty}


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _provider_error(message: Mapping[str, Any]) -> str:
    return (
        f"Speechmatics error: type={message.get('type')} "
        f"code={message.get('code')} reason={message.get('reason')}"
    )


# ---------------------------------------------------------------------------
# live transport
# ---------------------------------------------------------------------------

@dataclass
class SpeechmaticsTransport:
    """One live realtime session: socket in, adapter calls out, receipt back."""

    config: SpeechmaticsConfig = field(default_factory=SpeechmaticsConfig)
    api_key: str | None = None
    record_path: Path | None = None
    verbose: bool = True

    def _key(self) -> str:
        return self.api_key or api_key_from_env()

    async def run(self, audio: AudioStream, sink: VoiceSink,
                  mapper: TranscriptMapper,
                  aggregator: UtteranceAggregator | None = None,
                  on_tick: Callable[[], Any] | None = None,
                  on_close: Callable[[], Any] | None = None) -> dict[str, Any]:
        """Stream ``audio`` until it ends, dispatching transcripts to ``sink``.

        ``on_tick`` runs on the same idle timer as the aggregator -- an
        ``IntentAccumulator.tick`` goes here, so a held instruction is released
        while the session is still open rather than at shutdown. ``on_close``
        runs after the aggregator is closed and **before** the receipt is
        built, so a final flush is counted rather than silently omitted.
        """
        try:
            import websockets  # type: ignore import-not-found
        except Exception as exc:
            raise SpeechmaticsError(
                "the live transport needs the optional 'websockets' package: "
                "pip install -e .[speechmatics]"
            ) from exc

        key = self._key()
        config = self.config
        # The stream's own rate wins over the configured default: a 44.1 kHz
        # file announced as 16 kHz transcribes as gibberish.
        config.sample_rate = audio.sample_rate
        config.encoding = audio.encoding
        headers = {"Authorization": f"Bearer {key}"}
        recorder = self.record_path.open("w", encoding="utf-8") if self.record_path else None
        started_wall = time.time()
        started_mono = time.monotonic()
        audio_seq = 0
        end_of_transcript = asyncio.Event()
        provider_error: dict[str, Any] | None = None

        try:
            connection = _connect(websockets, config.url, headers)
            async with connection as socket:
                await socket.send(json.dumps(config.start_recognition()))

                async def _receive() -> None:
                    nonlocal provider_error
                    async for raw in socket:
                        if isinstance(raw, (bytes, bytearray)):
                            continue
                        message = json.loads(raw)
                        if recorder is not None:
                            recorder.write(json.dumps(message) + "\n")
                        kind = message.get("message")
                        if kind == "RecognitionStarted":
                            mapper.start_stream()
                            self._log(f"recognition started (id={message.get('id')})")
                            continue
                        if kind == "Error":
                            provider_error = dict(message)
                            end_of_transcript.set()
                            return
                        if kind == "Warning":
                            self._log(f"provider warning: {message.get('reason')}")
                            continue
                        if kind == "EndOfTranscript":
                            end_of_transcript.set()
                            return
                        if kind in (PARTIAL_MESSAGE, FINAL_MESSAGE):
                            mapped = mapper.map_message(message)
                            if mapped is None:
                                if aggregator is not None and kind == FINAL_MESSAGE:
                                    aggregator.mark_silence(mapper.last_empty_end_ns)
                            elif aggregator is not None:
                                aggregator.offer(mapped)
                            else:
                                sink.deliver(mapped)

                async def _flush_on_silence() -> None:
                    # Without this the final utterance of a session waits for
                    # the socket to close, which on a live mic is forever.
                    while True:
                        await asyncio.sleep(0.1)
                        if aggregator is not None:
                            aggregator.tick()
                        if on_tick is not None:
                            on_tick()

                receiver = asyncio.create_task(_receive())
                ticker = (asyncio.create_task(_flush_on_silence())
                          if (aggregator is not None or on_tick is not None)
                          else None)
                try:
                    async for chunk in audio.chunks():
                        if receiver.done():
                            break
                        await socket.send(chunk)
                        audio_seq += 1
                except asyncio.CancelledError:  # Ctrl+C during mic capture
                    pass
                if not receiver.done():
                    await socket.send(json.dumps(
                        {"message": "EndOfStream", "last_seq_no": audio_seq}
                    ))
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(end_of_transcript.wait(), timeout=30)
                receiver.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receiver
                if ticker is not None:
                    ticker.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ticker
                # Order matters: the aggregator feeds the accumulator, so it
                # must drain first or a mid-utterance fragment is lost.
                if aggregator is not None:
                    aggregator.close()
                if on_close is not None:
                    on_close()
        except SpeechmaticsError:
            raise
        except Exception as exc:
            raise SpeechmaticsError(_redact(f"{type(exc).__name__}: {exc}", key)) from None
        finally:
            if recorder is not None:
                recorder.close()

        if provider_error is not None:
            raise SpeechmaticsError(_provider_error(provider_error))

        return self.receipt(sink, mapper, audio,
                            chunks_sent=audio_seq,
                            wall_s=time.monotonic() - started_mono,
                            started_wall=started_wall,
                            aggregator=aggregator)

    def receipt(self, sink: VoiceSink, mapper: TranscriptMapper,
                audio: AudioStream, *, chunks_sent: int, wall_s: float,
                started_wall: float, mode: str = "live",
                aggregator: UtteranceAggregator | None = None) -> dict[str, Any]:
        """Measured latency receipt.  Contains no credentials."""
        sink.wait_for_responses()
        return {
            "provider": "speechmatics",
            "mode": mode,
            "url": self.config.url,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime(started_wall)),
            "wall_s": round(wall_s, 3),
            "audio": {
                "source": audio.description,
                "sample_rate": audio.sample_rate,
                "encoding": audio.encoding,
                "duration_s": (round(audio.duration_s, 3)
                               if audio.duration_s is not None else None),
                "chunks_sent": chunks_sent,
            },
            "config": self.config.as_receipt(),
            "utterance_segmentation": (aggregator.stats() if aggregator is not None
                                       else {"enabled": False}),
            "transcripts": sink.stats(),
            "dropped_empty_transcripts": mapper.dropped_empty,
            "latency_definition": (
                "monotonic wall-clock at message arrival minus the stream "
                "position (transcript end_time anchored at RecognitionStarted); "
                "i.e. how far behind live the transcript was, in ms"
            ),
        }

    def _log(self, text: str) -> None:
        if self.verbose:
            print(f"[speechmatics] {text}", flush=True)


def _connect(websockets_module: Any, url: str, headers: Mapping[str, str]) -> Any:
    """``websockets`` renamed the header kwarg in v14; support both."""
    try:
        return websockets_module.connect(url, additional_headers=dict(headers),
                                         max_size=None)
    except TypeError:
        return websockets_module.connect(url, extra_headers=dict(headers),
                                         max_size=None)
