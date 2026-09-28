"""One arm fails mid-run; the other finishes what it can reach (2026-09-14).

Full table-setting run. After the left arm has set the fork, its servo bus
goes silent: `IntelTableWorld.fail_arm` freezes the left arm's six actuator
commands on every physics step from then on -- a real, physical failure, not
a label. The fault handler withdraws the arm from OMNI's authority through
the same constraint path an operator's "don't use the left arm" takes
(`engine.add_constraint(prefer_arm=right)` under operator authority, fault as justification); the engine
recompiles at its next control boundary; the planner re-routes what the
right arm can reach (its own cup and spoon, plus the napkin in the shared
band) and lists what it cannot with the reason. The frozen arm stays in
frame, not moving, for the whole rest of the run.

    python integrations/intel/scripts/demo_arm_failure.py --seed 903
    python integrations/intel/scripts/demo_arm_failure.py --seed 903 --live
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from omni_q.intel_sim import (  # noqa: E402
    IntelSceneConfig, IntelTableWorld, TABLE_SETTING_PHRASINGS, _per_object_pick_place_outcomes, build_intel_sim_engine,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=903)
    ap.add_argument("--fail-after", default="fork_1", help="freeze the left arm after this object's MOVE completes")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("tmp") / "arm_failure")
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

    failure = {"at": None}
    orig_apply = world.apply_transition
    orig_parallel = world.apply_transitions_parallel

    def after(request, result):
        """Runs after every transition, single or paired (both arms at once)."""
        if (failure["at"] is None and request.op in {"MOVE", "PLACE"}
                and request.args.get("object") == args.fail_after and result.ok):
            info = world.fail_arm(0, reason="left arm servo bus: no response")
            # The fault handler withdraws the arm from OMNI's authority. The
            # constraint contract lets only an operator change graph
            # authority (a health monitor cannot rewrite it on its own), so
            # the handler submits under the operator's standing authority
            # with the fault as the justification -- the receipt records both.
            engine.add_constraint("prefer_arm", "right", source="operator",
                                  justification="fault handler: left arm servo bus no response; withdrawn from authority")
            failure["at"] = {"after_step": request.step_id, "revision": world.revision, "fault": info}
            print(f"\n!!! LEFT ARM FAILED after {request.step_id}: {info['reason']} (commands frozen) -> withdrawn from authority\n", flush=True)
        print(f"   {request.op:5s} {str(request.args.get('object') or ''):9s} {str(request.actor)[-9:]:9s} {'ok' if result.ok else 'FAIL'}", flush=True)

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
    rd = receipt.as_dict()
    actions = rd["actions"]
    cut = next((i for i, a in enumerate(actions) if failure["at"] and a.get("step") == failure["at"]["after_step"]), len(actions))
    after = [(a["op"], a.get("arm"), a["state"]) for a in actions[cut + 1:] if a["op"] in {"PICK", "MOVE", "PLACE", "OPEN"}]
    report = getattr(engine.planner, "last_authority_report", None)
    summary = {
        "seed": args.seed, "goal": goal, "failure": failure["at"],
        "steps_after_failure": after,
        "arms_used_after_failure": sorted({a for _, a, _ in after}),
        "planner_report": report,
        "per_object": {k: {"held": v["held"], "placed": v["placed"]} for k, v in _per_object_pick_place_outcomes(receipt).items()},
        "resolved": bool(rd.get("metrics", {}).get("resolved")),
        "run_id": rd.get("run_id"), "content_hash": rd.get("content_hash"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"seed-{args.seed}.json").write_text(json.dumps({"summary": summary, "receipt": rd}, indent=1, sort_keys=True))
    print("\n=== arm failure ===")
    print(json.dumps(summary, indent=1))
    if viewer is not None:
        import mujoco  # noqa: PLC0415
        print("physics keeps running; close the viewer to exit", flush=True)
        while viewer.is_running():
            world._mujoco.mj_step(world.model, world.data)
            viewer.sync()
            time.sleep(float(world.model.opt.timestep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
