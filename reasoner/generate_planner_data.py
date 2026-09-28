"""Generate (prompt, PLAN) pairs for the Omni planner-advisor from the project's own oracle.

The prompt is rendered by ``OmniPlanner._render_prompt`` -- the exact text the
reasoner sees at inference -- and the target is the deterministic
``RulePlanner`` plan for that world, formatted in the reasoner grammar the
planner parses. Worlds are randomized over the tabletop classes the YOLO
detector emits, zones, misplaced flags, held objects, operator constraints,
and replan reasons. ``docs/datasets.md`` ("Planner data") describes this
lane; every row is checkable against the same validator that governs the
model at run time.

Usage:
    PYTHONPATH=src .venv/Scripts/python reasoner/generate_planner_data.py \
        --out reasoner/data/planner_pairs.jsonl --n 3300 --seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from omni_q.contracts import Constraint, Detection
from omni_q.fakes import RulePlanner
from omni_q.omni_planner import OmniPlanner
from omni_q.omni_reasoner import MockReasoner
from omni_q.world import MockWorld

MOVABLE = ["plate", "cup", "fork", "spoon", "knife", "napkin"]
HOLD_ZONES = ["drawer", "tray", "counter", "sink", "bin", "cart"]
SETTING_ZONES = ["setting_1", "setting_2", "setting_3", "setting_4", "table"]
ARMS = ("left", "right")
GOALS = [
    "set the table", "please set the table for dinner", "set the table for two",
    "tidy up the table", "tidy the place setting", "put away the cutlery",
    "clear the table", "clear the table and put everything away",
    "correct the place setting", "inspect and fix the table setting",
    "set the table and show off",
]


def _rng_world(rng: random.Random) -> tuple[list[Detection], dict[str, str | None]]:
    n = rng.randint(2, 6)
    objects: list[Detection] = []
    counts: dict[str, int] = {}
    put_away = rng.random() < 0.3   # "clear/put away": targets are storage zones
    for _ in range(n):
        cls = rng.choice(MOVABLE)
        counts[cls] = counts.get(cls, 0) + 1
        oid = f"{cls}_{counts[cls]}"
        if put_away:
            target = rng.choice(HOLD_ZONES[:3])
            zone = target if rng.random() < 0.35 else rng.choice(SETTING_ZONES + HOLD_ZONES)
        else:
            target = rng.choice(SETTING_ZONES)
            zone = target if rng.random() < 0.35 else rng.choice(HOLD_ZONES + SETTING_ZONES)
        objects.append(Detection(oid, cls, zone=zone, target_zone=target))
    if not any(o.zone != o.target_zone for o in objects):
        o = objects[0]
        objects[0] = Detection(o.object_id, o.cls, zone=rng.choice(HOLD_ZONES), target_zone=o.target_zone)
    ownership: dict[str, str | None] = {o.object_id: None for o in objects}
    misplaced = [o.object_id for o in objects if o.zone != o.target_zone]
    if misplaced and rng.random() < 0.2:
        ownership[rng.choice(misplaced)] = f"intel.{rng.choice(ARMS)}_arm"
    return objects, ownership


def _rng_constraints(rng: random.Random, objects: list[Detection], goal: str) -> tuple[Constraint, ...]:
    cons: list[Constraint] = []
    misplaced = [o.object_id for o in objects if o.zone != o.target_zone]
    r = rng.random()
    if r < 0.22 and misplaced:
        cons.append(Constraint("forbid_object", rng.choice(misplaced), "operator", "operator said leave it"))
    elif r < 0.42:
        cons.append(Constraint("prefer_arm", rng.choice(ARMS), "operator", "operator preference"))
    elif r < 0.52:
        cons.append(Constraint("keep_local", None, "operator", "no cloud"))
    if "show off" in goal or rng.random() < 0.12:
        cons.append(Constraint("style", "show_off", "operator", "entertaining guests"))
    return tuple(cons)


def _rng_reason(rng: random.Random, objects: list[Detection]) -> str | None:
    if rng.random() < 0.6:
        return None
    oid = rng.choice(objects).object_id
    return rng.choice([
        f"{oid}: grasp_failure", f"{oid} moved", f"verify mismatch: {oid} not at target",
        "capability intel.left_arm offline", "capability intel.right_arm offline",
        f"step pick_{oid} failed",
    ])


def format_plan(graph, world) -> str:
    lines = ["PLAN"]
    for s in graph.steps:
        if s.op == "VERIFY":
            lines.append("STEP VERIFY")
            continue
        oid = s.args.get("object", "")
        holder = world.ownership.get(oid)
        arm = ("left" if "left" in holder else "right") if holder else s.arm
        arm_txt = f" arm={arm}" if arm else ""
        if s.op == "PICK":
            lines.append(f"STEP PICK object={oid}{arm_txt}")
        elif s.op == "MOVE":
            lines.append(f"STEP MOVE object={oid} to={s.args['to']}{arm_txt}")
        elif s.op == "PRESENT":
            lines.append(f"STEP PRESENT object={oid}")
        elif s.op == "ROTATE":
            lines.append(f"STEP ROTATE object={oid}")
    lines.append("END")
    # Short, templated rationale: it is display-only for the planner, and every
    # target token is a separate sequential forward step in the reference
    # model, so a long sentence is pure training cost.
    moved = [s.args["object"] for s in graph.steps if s.op == "MOVE"]
    skipped = sorted(world.forbidden() & {d.object_id for d in world.misplaced()})
    if moved:
        tail = f" skip {' '.join(skipped)}." if skipped else "."
        lines.append(f"RATIONALE: move {len(moved)} misplaced to target zones{tail}")
    else:
        lines.append("RATIONALE: nothing misplaced; verify.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--holdout", type=float, default=0.1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    renderer = OmniPlanner(MockReasoner())
    oracle = RulePlanner()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n_written = 0
    with out.open("w", encoding="utf-8") as f:
        while n_written < args.n:
            objects, ownership = _rng_world(rng)
            goal = rng.choice(GOALS)
            cons = _rng_constraints(rng, objects, goal)
            world = MockWorld(objects, cons)
            world.goal = goal
            for oid, holder in ownership.items():
                world._ownership[oid] = holder
            state = world.state()
            reason = _rng_reason(rng, objects)
            prompt = renderer._render_prompt(goal, state, reason)
            graph = oracle.plan(goal, state)
            if not any(s.contract == "manipulate" for s in graph.steps):
                continue
            target = format_plan(graph, state)
            key = hashlib.sha256(prompt.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            split = "eval" if int(key[:8], 16) / 0xFFFFFFFF < args.holdout else "train"
            meta = {
                "goal": goal, "reason": reason,
                "objects": [{"object_id": o.object_id, "cls": o.cls, "zone": o.zone,
                             "target_zone": o.target_zone} for o in objects],
                "constraints": [c.as_dict() for c in cons],
                "ownership": ownership,
            }
            f.write(json.dumps({"prompt": prompt, "target": target, "split": split, "meta": meta}) + "\n")
            n_written += 1
    print(f"wrote {n_written} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
