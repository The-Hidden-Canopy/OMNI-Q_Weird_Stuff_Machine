"""Record LeRobot demonstrations for a motor-level VLA from the governed
expert (2026-09-15).

Runs the submission's own table-setting stack (planner + contact
controllers, both arms at once) on randomized seeds and logs, at 10 Hz and
per arm, what a VLA policy would see and what the expert actually did:

* observation.images.overhead / .front / .wrist  -- the scene cameras and
  that arm's wrist camera, 320x240 RGB
* observation.state (9)  -- the arm's six joint positions, the jaw command,
  and the fixed-pad xyz
* action (7)  -- the Cartesian delta of the fixed pad over the next tick
  (mm, deg) and the jaw command delta: the same seven fields the governed
  SmolVLA seam (src/omni_q/skills/controllers/smolvla.py) proposes
* task  -- the step the arm is executing right now, in language
  ("pick up the cup with the right arm", "hold position", ...)

One episode per (seed, arm), written with scripts/record_smolvla_dataset.py
(the teammate's recorder, current LeRobot API). Nothing in the submission
stack is modified; the run is the real one.

    python integrations/intel/vla/record_expert_demos.py --seeds 900 901 ... --root datasets/so101_table_vla
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from record_smolvla_dataset import ACTION_FIELDS, SmolVLADatasetRecorder  # noqa: E402

STATE_NAMES = ("rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll", "jaw_cmd", "pad_x", "pad_y", "pad_z")
ARMS = {0: "left", 6: "right"}
W, H = 320, 240


def _rpy(R: np.ndarray) -> np.ndarray:
    """xyz roll/pitch/yaw of a rotation matrix (matches the actuator's _rotation_xyz)."""
    sy = -R[2, 0]
    pitch = math.asin(max(-1.0, min(1.0, sy)))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + math.pi) % (2 * math.pi) - math.pi


def instruction(step_text: str | None, arm: str) -> str:
    if not step_text:
        return f"hold the {arm} arm still"
    return step_text


class Sampler:
    """Renders on the main thread; under paired execution the arm threads
    queue physics snapshots and the world's pump drains them (same pattern
    as the demo recorder)."""

    def __init__(self, world, every: int) -> None:
        import mujoco
        self.world, self.every, self.mujoco = world, every, mujoco
        self.renderer = mujoco.Renderer(world.model, height=H, width=W)
        self.count = 0
        self.rows: dict[int, list[dict]] = {0: [], 6: []}
        self.step_text: dict[int, str | None] = {0: None, 6: None}
        self.queue: collections.deque = collections.deque()
        self.lock = threading.Lock()
        self.main = threading.main_thread()
        self.pad = world._pad_geom

    def render(self, data, cam: str) -> np.ndarray:
        self.renderer.update_scene(data, camera=cam)
        return self.renderer.render().copy()

    def sample(self, data) -> None:
        imgs = {"overhead": self.render(data, "table_overhead"), "front": self.render(data, "third_person")}
        for off, arm in ARMS.items():
            pos = data.geom_xpos[self.pad[off]].copy()
            R = data.geom_xmat[self.pad[off]].reshape(3, 3).copy()
            self.rows[off].append({
                "t": float(data.time),
                "state": np.concatenate([data.qpos[off:off + 5], [float(data.ctrl[off + 5])], pos]).astype(np.float32),
                "pos": pos, "R": R, "jaw": float(data.ctrl[off + 5]),
                "images": {"overhead": imgs["overhead"], "front": imgs["front"], "wrist": self.render(data, f"{arm}_wrist")},
                "task": instruction(self.step_text[off], arm),
            })

    def spy(self):
        real = self.world._mujoco
        s = self

        class Spy:
            def __getattr__(self, n):
                return getattr(real, n)

            def mj_step(self, m, d, nstep=1):
                for _ in range(int(nstep)):
                    real.mj_step(m, d)
                    s.count += 1
                    if s.count % s.every == 0:
                        if threading.current_thread() is s.main:
                            s.sample(d)
                        else:
                            snap = s.mujoco.MjData(m)
                            s.mujoco.mj_copyData(snap, m, d)
                            with s.lock:
                                s.queue.append(snap)
                            while len(s.queue) > 40:
                                time.sleep(0.002)

            def main_thread_pump(self):
                while True:
                    with s.lock:
                        if not s.queue:
                            return
                        snap = s.queue.popleft()
                    s.sample(snap)
        return Spy()


def record_seed(seed: int, recorder: SmolVLADatasetRecorder, hz: int = 10) -> dict:
    from omni_q.intel_sim import IntelSceneConfig, TABLE_SETTING_PHRASINGS, build_intel_sim_engine, _per_object_pick_place_outcomes

    goal = TABLE_SETTING_PHRASINGS[(seed - 900) % len(TABLE_SETTING_PHRASINGS)]
    eng = build_intel_sim_engine(IntelSceneConfig(seed=seed, randomized=True))
    w = eng.world
    every = int(round(1.0 / (hz * float(w.model.opt.timestep))))
    s = Sampler(w, every)
    w._mujoco = s.spy()

    names = {"plate_1": "plate", "cup_1": "cup", "fork_1": "fork", "spoon_1": "spoon", "napkin_1": "napkin"}
    zones = {"center": "the centre", "upper_right": "the upper right", "left": "the left", "right": "the right", "lower_left": "the lower left"}

    def text(req, arm):
        obj = names.get(req.args.get("object"), req.args.get("object"))
        if req.op == "PICK":
            return f"pick up the {obj} with {'both arms' if req.args.get('object') == 'plate_1' else 'the ' + arm + ' arm'}"
        if req.op in {"MOVE", "PLACE"}:
            return f"move the {obj} to {zones.get(req.args.get('to'), req.args.get('to'))} with {'both arms' if req.args.get('object') == 'plate_1' else 'the ' + arm + ' arm'}"
        return f"{req.op.lower()} {obj}"

    def begin(reqs):
        for req in reqs:
            off = 6 if (req.actor and "right" in req.actor) else 0
            s.step_text[off] = text(req, ARMS[off])
            if req.args.get("object") == "plate_1":
                s.step_text[0] = s.step_text[6] = text(req, "left")

    def end(reqs):
        for req in reqs:
            off = 6 if (req.actor and "right" in req.actor) else 0
            s.step_text[off] = None
            if req.args.get("object") == "plate_1":
                s.step_text[0] = s.step_text[6] = None

    orig, orig_par = w.apply_transition, w.apply_transitions_parallel

    def apply(req):
        begin([req]); res = orig(req); end([req]); return res

    def apply_pair(reqs):
        begin(reqs); out = orig_par(reqs); end(reqs); return out
    w.apply_transition, w.apply_transitions_parallel = apply, apply_pair

    t0 = time.time()
    r = eng.run(goal)
    po = _per_object_pick_place_outcomes(r)
    placed = sum(int(v["placed"]) for v in po.values())
    # write one episode per arm: action[i] = delta from row i to row i+1
    frames = 0
    for off, arm in ARMS.items():
        rows = s.rows[off]
        for i in range(len(rows) - 1):
            a, b = rows[i], rows[i + 1]
            dpos = (b["pos"] - a["pos"]) * 1000.0
            drpy = np.degrees(_rpy(b["R"] @ a["R"].T))   # relative rotation: small angles, no wrap
            action = dict(zip(ACTION_FIELDS, [*dpos, *drpy, b["jaw"] - a["jaw"]]))
            recorder.add(state=a["state"], action=action, task=a["task"],
                         images={"observation.images.overhead": a["images"]["overhead"],
                                 "observation.images.front": a["images"]["front"],
                                 "observation.images.wrist": a["images"]["wrist"]})
            frames += 1
        recorder.finish_episode()
    s.renderer.close()
    return {"seed": seed, "goal": goal, "placed": placed, "resolved": bool(r.metrics.get("resolved")),
            "frames": frames, "wall_s": round(time.time() - t0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(900, 920)))
    ap.add_argument("--root", type=Path, default=ROOT / "datasets" / "so101_table_vla")
    ap.add_argument("--repo-id", default="omni-q/so101_table_vla")
    args = ap.parse_args()
    if args.root.exists():
        raise SystemExit(f"{args.root} exists; choose a new --root (LeRobot datasets are append-only)")
    recorder = SmolVLADatasetRecorder(repo_id=args.repo_id, root=args.root, fps=10, state_names=STATE_NAMES,
                                      camera_sources={"observation.images.overhead": None, "observation.images.front": None,
                                                      "observation.images.wrist": None},
                                      width=W, height=H)
    # the teammate's recorder captures from live sources at add() time; ours
    # are already-rendered arrays, so hand them in per frame instead
    orig_add = recorder.add

    def add(*, state, action, task, images):
        recorder.camera_sources = {k: (lambda img=v: img) for k, v in images.items()}
        orig_add(state=state, action=action, task=task)
    recorder.add = add
    results = []
    for seed in args.seeds:
        res = record_seed(seed, recorder)
        results.append(res)
        print(res, flush=True)
    recorder.finalize() if hasattr(recorder, "finalize") else None
    (args.root / "expert_runs.json").write_text(__import__("json").dumps(results, indent=1))
    print("episodes", 2 * len(results), "frames", sum(r["frames"] for r in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
