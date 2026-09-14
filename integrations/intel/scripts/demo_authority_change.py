"""Mid-run change of authority: "don't use the left arm anymore".

Runs the dual-arm table-setting goal in the MuJoCo scene and, once the
operator's cue step has completed (default: after the two-arm plate is set),
injects the operator instruction through the SAME path the voice runtime
uses -- ``omni_q.nlu.parse`` -> ``engine.add_constraint(prefer_arm=right)``.
The engine applies it at its next control boundary and recompiles; the
planner reassigns what the right arm can reach, prunes what it cannot (the
left-side objects, the two-arm plate), and records that in the receipt.

Nothing here is staged: the constraint is queued exactly as a live operator
utterance would queue it, and every arm assignment printed is read back from
the receipt's recorded graphs.

    python integrations/intel/scripts/demo_authority_change.py --seed 901
    python integrations/intel/scripts/demo_authority_change.py --seed 901 --after cup_1
    python integrations/intel/scripts/demo_authority_change.py --seed 901 --live
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from omni_q import nlu  # noqa: E402
from omni_q.intel_sim import (  # noqa: E402
    IntelSceneConfig, IntelTableWorld, TABLE_SETTING_PHRASINGS, build_intel_sim_engine,
)

PHRASE = "don't use the left arm anymore"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=901)
    ap.add_argument("--after", default="plate_1", help="inject the instruction after this object's MOVE completes")
    ap.add_argument("--phrase", default=PHRASE)
    ap.add_argument("--live", action="store_true", help="open the MuJoCo viewer (real-time paced)")
    ap.add_argument("--out", type=Path, default=Path("tmp") / "authority_change")
    args = ap.parse_args()

    engine = build_intel_sim_engine(IntelSceneConfig(seed=args.seed, randomized=True))
    world: IntelTableWorld = engine.world  # type: ignore[assignment]

    viewer = None
    if args.live:
        import mujoco.viewer  # noqa: PLC0415
        viewer = mujoco.viewer.launch_passive(world.model, world.data)
        real = world._mujoco
        wall0, count = time.time(), [0]
        dt = float(world.model.opt.timestep)

        class Spy:
            def __getattr__(self, n):
                return getattr(real, n)

            def mj_step(self, m, d, nstep=1):
                for _ in range(nstep):
                    real.mj_step(m, d)
                    count[0] += 1
                    if count[0] % 8 == 0:
                        viewer.sync()
                        ahead = count[0] * dt - (time.time() - wall0)
                        if ahead > 0:
                            time.sleep(min(ahead, 0.05))
        world._mujoco = Spy()

    # The operator speaks mid-run. Hook the world's transition boundary so the
    # instruction is queued the moment the cue step finishes -- the engine
    # picks it up at its next control boundary, exactly as a live utterance.
    injected = {"at": None}
    orig_apply = world.apply_transition
    orig_parallel = world.apply_transitions_parallel

    def after(request, result):
        if (injected["at"] is None and request.op in {"MOVE", "PLACE"}
                and request.args.get("object") == args.after and result.ok):
            parsed = nlu.parse(args.phrase)
            for kind, value in parsed.constraints:
                engine.add_constraint(kind, value, source="operator", justification=f"voice: {args.phrase!r}")
            injected["at"] = {"after_step": request.step_id, "revision": world.revision,
                              "constraints": list(parsed.constraints)}
            print(f"\n>>> operator: {args.phrase!r}  ->  {parsed.constraints}  (queued after {request.step_id})\n", flush=True)
        state = "ok" if result.ok else "FAIL"
        print(f"   {request.op:5s} {str(request.args.get('object') or ''):9s} {str(request.actor)[-9:]:9s} {state}", flush=True)

    def apply_one(request):
        result = orig_apply(request)
        after(request, result)
        return result

    def apply_pair(requests):
        results = orig_parallel(requests)
        print("   [both arms at once]", flush=True)
        for request, result in zip(requests, results):
            after(request, result)
        return results
    world.apply_transition = apply_one
    world.apply_transitions_parallel = apply_pair

    goal = TABLE_SETTING_PHRASINGS[0]
    print(f"goal: {goal!r}  seed {args.seed}\n")
    receipt = engine.run(goal)

    # Read the authority change back from the receipt, not from local state.
    rd = receipt.as_dict()
    graphs = [d for d in rd.get("decisions", []) if isinstance(d, dict) and "steps" in d] or []
    report = getattr(engine.planner, "last_authority_report", None)
    actions = rd.get("actions", [])
    # Actions are recorded in execution order; everything after the cue step
    # ran under the new authority.
    cut = next((i for i, a in enumerate(actions)
                if injected["at"] and a.get("step") == injected["at"]["after_step"]), len(actions))
    used_before = sorted({str(a.get("arm")) for a in actions[:cut + 1] if a.get("op") in {"PICK", "MOVE", "PLACE", "OPEN"}})
    used_after = sorted({str(a.get("arm")) for a in actions[cut + 1:] if a.get("op") in {"PICK", "MOVE", "PLACE", "OPEN"}})
    steps_after = [(a.get("op"), a.get("arm"), a.get("state")) for a in actions[cut + 1:]]
    summary = {
        "seed": args.seed, "goal": goal, "phrase": args.phrase, "injected": injected["at"],
        "authority_report": report,
        "arms_used_before_instruction": used_before,
        "arms_used_after_instruction": used_after,
        "steps_after_instruction": steps_after,
        "resolved": bool(rd.get("metrics", {}).get("resolved")),
        "graph_revisions": rd.get("metrics", {}).get("revisions"),
        "run_id": rd.get("run_id"), "content_hash": rd.get("content_hash"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"seed-{args.seed}.json").write_text(json.dumps({"summary": summary, "receipt": rd}, indent=1, sort_keys=True))
    print("\n=== authority change ===")
    print(json.dumps(summary, indent=1))
    print(f"\nreceipt: {args.out / f'seed-{args.seed}.json'}")
    if viewer is not None:
        print("physics keeps running; close the viewer to exit", flush=True)
        import mujoco  # noqa: PLC0415
        while viewer.is_running():
            mujoco.mj_step(world.model, world.data)
            viewer.sync()
            time.sleep(float(world.model.opt.timestep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
