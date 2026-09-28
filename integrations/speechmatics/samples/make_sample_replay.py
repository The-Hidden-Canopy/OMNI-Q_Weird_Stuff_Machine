"""Generate ``session_replay.jsonl`` -- a hand-authored Speechmatics session.

These are *synthetic* provider messages shaped exactly like the realtime v2
protocol, not a recording of a real session.  They exist so the voice boundary
can be exercised offline, in tests and in the demo, with no API key and no
microphone.  Replace the file with a real ``--record`` capture whenever one is
available; the replay path does not care which it is reading.

    .venv/Scripts/python integrations/speechmatics/samples/make_sample_replay.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "session_replay.jsonl"


def words(text: str, start: float, speaker: str, confidence: float = 0.93):
    """Lay the sentence out as word/punctuation results over ``[start, ...)``."""
    results = []
    cursor = start
    for token in text.split():
        bare = token.rstrip(",.")
        trailing = token[len(bare):]
        span = 0.12 + 0.035 * len(bare)
        results.append({
            "type": "word",
            "start_time": round(cursor, 3),
            "end_time": round(cursor + span, 3),
            "alternatives": [
                {"content": bare, "confidence": confidence, "speaker": speaker}
            ],
        })
        cursor += span + 0.04
        for mark in trailing:
            results.append({
                "type": "punctuation",
                "start_time": round(cursor, 3),
                "end_time": round(cursor, 3),
                "alternatives": [
                    {"content": mark, "confidence": 1.0, "speaker": speaker}
                ],
            })
    return results, cursor


def transcript(kind: str, text: str, start: float, speaker: str,
               confidence: float = 0.93):
    results, end = words(text, start, speaker, confidence)
    return {
        "message": kind,
        "metadata": {"start_time": round(start, 3),
                     "end_time": round(end, 3),
                     "transcript": text},
        "results": results,
    }, end


def build() -> list[dict]:
    messages: list[dict] = [
        {"message": "RecognitionStarted", "id": "sample-session-0001"},
        {"message": "AudioAdded", "seq_no": 1},
    ]
    cursor = 0.4

    # An operator constraint: this one should reach graph_mutation.
    first = "Omni, don't use the left arm anymore."
    partial_a, _ = transcript("AddPartialTranscript", "Omni don't use", cursor, "S1")
    partial_b, _ = transcript("AddPartialTranscript", "Omni don't use the left", cursor, "S1")
    final_a, cursor = transcript("AddTranscript", first, cursor, "S1")
    messages += [partial_a, partial_b, final_a]

    # A placement constraint from the same operator.
    cursor += 0.9
    second = "Keep inference on-device."
    partial_c, _ = transcript("AddPartialTranscript", "Keep inference", cursor, "S1")
    final_b, cursor = transcript("AddTranscript", second, cursor, "S1", 0.89)
    messages += [partial_c, final_b]

    # A second, unregistered speaker: must be observed, never committed.
    cursor += 1.4
    third = "Watch the second camera too."
    final_c, cursor = transcript("AddTranscript", third, cursor, "S2", 0.81)
    messages.append(final_c)

    # A silent partial: the mapper drops it rather than emitting empty text.
    messages.append({"message": "AddPartialTranscript",
                     "metadata": {"start_time": cursor, "end_time": cursor,
                                  "transcript": ""},
                     "results": []})
    messages.append({"message": "EndOfTranscript"})
    return messages


def main() -> None:
    OUT.write_text(
        "\n".join(json.dumps(m) for m in build()) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
