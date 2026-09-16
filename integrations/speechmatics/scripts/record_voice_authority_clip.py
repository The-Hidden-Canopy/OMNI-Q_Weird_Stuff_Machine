"""Speechmatics bonus clip: a real Speechmatics realtime session drives the
authority change in the MuJoCo table run (2026-09-15).

The audio was spoken live on 2026-09-13 and transcribed by Speechmatics
Realtime v2 (every provider message was recorded to
``integrations/speechmatics/samples/live_session_2026-09-13.jsonl``); this
script replays those provider messages -- partials, finals, speaker labels --
into the SAME voice boundary the live transport uses (mapper -> utterance
aggregator -> intent accumulator -> VoiceRuntime authority -> RuntimeMutator),
so the constraint that withdraws the left arm is produced by the voice path,
not typed in.  The HUD shows the transcript arriving word by word, the speaker,
the authority decision and the resulting plan change.

Set ``SPEECHMATICS_API_KEY`` and pass ``--file <wav>`` to make a fresh live
session instead (``--record`` in run_voice_transport.py captures it); then
point ``--session`` at that recording.

    OMNIQ_VLA_CHECKPOINT=<dir> .venv/Scripts/python integrations/speechmatics/scripts/record_voice_authority_clip.py --seed 905
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations/intel/scripts"))

import record_demo_videos as rdv  # noqa: E402  (installs VLA mode + fixed director cameras from env)
from omni_q.voice import IntentAccumulator, SpeechmaticsRealtimeAdapter, VoiceRuntime  # noqa: E402
from omni_q.mutation import RuntimeMutator  # noqa: E402
from integrations.speechmatics.transport import (  # noqa: E402
    FINAL_MESSAGE, TranscriptMapper, UtteranceAggregator, VoiceSink, _is_turn_signal, read_jsonl,
)
from run_voice_transport import _authorize_on_first_appearance  # noqa: E402

SESSION = ROOT / "integrations/speechmatics/samples/live_session_2026-09-13.jsonl"
DEFAULT_OUT = Path.home() / "OneDrive" / "Desktop" / "OMNI-Q_demo_videos_VLA"


def voice_authority(seed: int, session: Path, fps: int = 25):
    from omni_q.intel_sim import IntelSceneConfig, TABLE_SETTING_PHRASINGS, IntelTablePlanner, build_intel_sim_engine, _per_object_pick_place_outcomes

    def world():
        eng = build_intel_sim_engine(IntelSceneConfig(seed=seed, randomized=True))
        eng.world._engine = eng
        eng.parallel_arms = False
        return eng.world

    def run(w):
        eng = w._engine
        rec = w._rec
        orig = w.apply_transition
        done = {"x": False, "decision": None}

        def speak() -> None:
            """Replay the recorded Speechmatics session into the voice boundary,
            rendering the transcript as it arrives (paced by the provider's own
            timestamps) while the arms hold still."""
            runtime = VoiceRuntime("voice_session_01", "omni_q_local", mutator=RuntimeMutator(eng))
            _authorize_on_first_appearance(runtime, ["S1"])
            mapper = TranscriptMapper(session_id="voice_session_01", org_id="omni_q_local", language="en")
            accumulator = IntentAccumulator(runtime, window_ms=4000.0, sequence_source=mapper.next_sequence)
            adapter = SpeechmaticsRealtimeAdapter(accumulator)
            state = {"partial": "", "finals": [], "speaker": "S1", "result": ""}

            def on_result(mapped, result):
                if mapped is None:
                    claim = getattr(getattr(result, "claim", None), "text", "")
                    state["result"] = f"semantic turn: \"{claim}\"  status={getattr(result, 'status', '?')}"
                    return
                state["speaker"] = mapped.payload.get("speaker_id", "S1")
                if not mapped.final or result is None:
                    return
                cand = getattr(result, "candidate", None)
                auth = getattr(result, "authority", None)
                state["result"] = (f"status={getattr(result, 'status', '?')}"
                                   + (f"  intent={getattr(cand, 'action', None)}" if cand else "")
                                   + (f"  authorized={getattr(auth, 'allowed', None)}" if auth else ""))

            sink = VoiceSink(adapter, on_result=on_result)
            aggregator = UtteranceAggregator(sink, mapper, silence_ms=800.0,
                                             urgent=lambda text: runtime.interruption_gate.inspect(text).detected)
            head = "SPEECHMATICS realtime v2 -- live session recorded 2026-09-13 (real audio), replayed into the voice boundary"

            def hud():
                text = " ".join(t.strip() for t in state["finals"]) + (("  " + state["partial"]) if state["partial"] else "")
                rec.hud_extra = [head, f"{state['speaker']}: \"{text.strip()}\"" + ("   (partial)" if state["partial"] else ""),
                                 f"voice boundary: {state['result']}" if state["result"] else "voice boundary: listening"]
                rec.render_hud()

            last_t = 0.0
            hud()
            for _ in range(fps):
                rec.frame()
            for message in read_jsonl(session):
                kind = message.get("message")
                meta = message.get("metadata") or {}
                t = float(meta.get("end_time", last_t) or last_t)
                for _ in range(max(0, int((t - last_t) * fps))):   # provider timestamps pace the replay
                    rec.frame()
                last_t = max(last_t, t)
                if kind == "AddPartialTranscript":
                    state["partial"] = meta.get("transcript", "")
                elif kind == FINAL_MESSAGE:
                    if meta.get("transcript"):
                        state["finals"].append(meta["transcript"])
                    state["partial"] = ""
                if kind == "RecognitionStarted":
                    mapper.start_stream()
                    hud()
                    continue
                if _is_turn_signal(message):
                    aggregator.flush()
                    sink.on_provider_event(message)
                    hud()
                    continue
                mapped = mapper.map_message(message)
                if mapped is None:
                    if kind == FINAL_MESSAGE:
                        aggregator.mark_silence(mapper.last_empty_end_ns)
                    hud()
                    continue
                aggregator.offer(mapped)
                hud()
            aggregator.close()
            accumulator.flush()
            hud()
            done["decision"] = state["result"]
            applied = [c for c in getattr(eng, "_pending_constraints", [])]
            rec.notify(f"VOICE -> {state['result']}  -> constraint {[(c.kind, c.value) for c in applied]} -> plan recompiled, right arm takes over", 6)
            for _ in range(2 * fps):
                rec.frame()

        def apply(req):
            res = orig(req)
            if not done["x"] and req.op == "MOVE" and req.args.get("object") == "napkin_1" and res.ok:
                done["x"] = True
                speak()
            return res
        w.apply_transition = apply
        saved = IntelTablePlanner._MUST_PRECEDE
        IntelTablePlanner._MUST_PRECEDE = ("plate_1", "fork_1", "napkin_1")
        try:
            r = eng.run(TABLE_SETTING_PHRASINGS[(seed - 900) % len(TABLE_SETTING_PHRASINGS)])
        finally:
            IntelTablePlanner._MUST_PRECEDE = saved
        arms = [a.get("arm") for a in r.as_dict()["actions"] if a["op"] in ("PICK", "MOVE")]
        po = _per_object_pick_place_outcomes(r)
        tally = "".join("P" if v["placed"] else "-" for v in po.values())
        return f"placed {tally} resolved={bool(r.metrics.get('resolved'))}  voice: {done['decision']}  arms in order: {arms}"
    return world, run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=905)
    ap.add_argument("--session", type=Path, default=SESSION)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--views", nargs="*", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    world, run = voice_authority(args.seed, args.session)
    views = [("director", rdv.DIRECTOR), ("third_person", "third_person"), ("grid", rdv.GRID)]
    if args.views:
        views = [v for v in views if v[0] in args.views]
    name = f"10_speechmatics_authority_seed{args.seed}"
    print(f"== {name}", flush=True)
    for line in rdv._with_recorders(world, args.out, name, run, views):
        print("   ", line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
