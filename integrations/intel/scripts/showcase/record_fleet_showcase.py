"""SHOWCASE (not part of the competition submission) -- the escalation clip.

Shot 1 tight on the real two-arm table setting (the submission stack, OMNI
planning both arms); pull back to reveal a second unit working; a voice
command ("OMNI, show off") goes through the voice boundary + NLU and every
arm does a 3-second dance, then goes back to work; pull back to 8 arms with
a fleet HUD; the drone takes off and surveys the row; the rover drives in
with a delivery; final pull-back on the whole ecosystem.

Honesty, printed on the HUD: unit 1 is the submission stack; units 2-4 are
physically simulated arms on scripted choreography; the drone and the rover
are kinematic showcase actors.

    python integrations/intel/scripts/showcase/record_fleet_showcase.py --seed 903
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3] / "src"))

import fleet_scene  # noqa: E402
from _recording import Recorder, draw_text_block  # noqa: E402

DEFAULT_OUT = Path.home() / "OneDrive" / "Desktop" / "OMNI-Q_demo_videos" / "showcase_experiments"
HOME5 = np.array([0.0, -1.57, 1.57, 1.57, -1.57])
JAW_OPEN, JAW_PART, JAW_CLOSED = 1.5, 0.30, -0.17


# ---------------------------------------------------------------- extra arms
class ExtraArm:
    """One scripted SO-101 on a showcase unit: joint addresses, a vertical-
    finger FK scan (same idea as the submission's _topdown_seed_joints), and a
    keyframe player that writes ctrl every physics step."""

    def __init__(self, model, prefix: str) -> None:
        import mujoco
        self.m, self.prefix = model, prefix
        joints = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        self.qadr = [int(model.jnt_qposadr[model.joint(f"{prefix}_{j}").id]) for j in joints]
        self.uadr = [int(model.actuator(f"{prefix}_{j}").id) for j in joints]
        self.base_id = model.body(f"{prefix}_Base").id
        self.pad1 = model.geom(f"{prefix}_fixed_jaw_pad_1").id
        self.pad4 = model.geom(f"{prefix}_fixed_jaw_pad_4").id
        self.rows = None
        self.keys: list[tuple[np.ndarray, float, float]] = []   # (q6, start_s, end_s)
        self.cycle = 0.0
        self.t0 = 0.0

    def scan(self, data) -> None:
        import mujoco
        scratch = mujoco.MjData(self.m)
        base = data.xpos[self.base_id].copy()
        rng = [self.m.jnt_range[self.m.joint(f"{self.prefix}_{j}").id] for j in ("Pitch", "Elbow", "Wrist_Pitch")]
        grid = [np.linspace(lo, hi, 25) for lo, hi in rng]
        rows = []
        for pitch, elbow, wp in itertools.product(*grid):
            scratch.qpos[:] = data.qpos
            for a, v in zip(self.qadr[:5], (0.0, pitch, elbow, wp, -1.57)):
                scratch.qpos[a] = v
            mujoco.mj_kinematics(self.m, scratch)
            R = scratch.geom_xmat[self.pad4].reshape(3, 3)
            if R[2, 1] > 0.95:
                tip = scratch.geom_xpos[self.pad1] - base
                rows.append((pitch, elbow, wp, float(np.hypot(tip[0], tip[1])), float(tip[2]), float(np.arctan2(tip[1], tip[0]))))
        self.rows = np.array(rows)
        self._base = base

    def pose(self, tip_target, jaw: float) -> np.ndarray:
        rel = np.asarray(tip_target, float) - self._base
        want_r, want_z, heading = float(np.hypot(rel[0], rel[1])), float(rel[2]), float(np.arctan2(rel[1], rel[0]))
        err = np.hypot(self.rows[:, 3] - want_r, self.rows[:, 4] - want_z)
        best = self.rows[int(np.argmin(err))]
        yaw = heading - best[5]
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        return np.array([yaw, best[0], best[1], best[2], -1.57, jaw])

    def program(self, keys: list[tuple[np.ndarray, float]], t0: float) -> None:
        """keys: (q6, seconds to get there); the program loops."""
        t = 0.0
        self.keys = []
        for q, dur in keys:
            self.keys.append((q, t, t + dur))
            t += dur
        self.cycle, self.t0 = t, t0

    def ctrl_at(self, t: float, data) -> np.ndarray:
        if not self.keys:
            return np.append(HOME5, JAW_OPEN)
        tt = (t - self.t0) % self.cycle
        prev = self.keys[-1][0]
        for q, a, b in self.keys:
            if a <= tt < b:
                f = (tt - a) / max(1e-6, b - a)
                f = f * f * (3 - 2 * f)
                return prev + (q - prev) * f
            prev = q
        return self.keys[-1][0]

    def write(self, data, q6: np.ndarray) -> None:
        for u, v in zip(self.uadr, q6):
            data.ctrl[u] = v


def unit_choreography(left: ExtraArm, right: ExtraArm, ux: float, t0: float) -> None:
    """A mimed pass on one unit: left lifts cup_a, carries it to a hand-off
    point between the arms, right takes it and sets it by the plate."""
    cup = np.array([ux - 0.10, 0.02, 0.0])
    hand = np.array([ux - 0.02, -0.02, 0.0])
    plate = np.array([ux + 0.12, -0.06, 0.0])
    up = np.array([0, 0, 0.12]); grip = np.array([0, 0, 0.055]); carry = np.array([0, 0, 0.16])
    home = np.append(HOME5, JAW_OPEN)
    L = [(home, 1.0), (left.pose(cup + up, JAW_PART), 1.6), (left.pose(cup + grip, JAW_PART), 1.2),
         (left.pose(cup + grip, JAW_CLOSED), 0.6), (left.pose(cup + carry, JAW_CLOSED), 1.0),
         (left.pose(hand + carry, JAW_CLOSED), 1.6), (left.pose(hand + carry, JAW_CLOSED), 2.0),
         (left.pose(hand + carry, JAW_OPEN), 0.6), (left.pose(hand + carry + up, JAW_OPEN), 1.0), (home, 2.0)]
    R = [(home, 5.4), (right.pose(hand + carry + np.array([0.09, 0, 0.02]), JAW_OPEN), 1.6),
         (right.pose(hand + carry + np.array([0.035, 0, 0.0]), JAW_OPEN), 1.0),
         (right.pose(hand + carry + np.array([0.035, 0, 0.0]), JAW_CLOSED), 0.8),
         (right.pose(hand + carry + np.array([0.035, 0, 0.0]), JAW_CLOSED), 0.8),
         (right.pose(plate + carry, JAW_CLOSED), 1.6), (right.pose(plate + grip, JAW_CLOSED), 1.0),
         (right.pose(plate + grip, JAW_OPEN), 0.5), (right.pose(plate + up, JAW_OPEN), 0.8), (home, 1.5)]
    left.program(L, t0)
    right.program(R, t0)


# ---------------------------------------------------------------- the take
class FleetRecorder(Recorder):
    """Recorder with an animated director camera and a fleet HUD block."""

    def __init__(self, world, path, **kw):
        super().__init__(world, "free:75,-32,0.8,0.05,-0.15,0.05", path, **kw)
        self.cam = np.array([75.0, -32.0, 0.8, 0.05, -0.15, 0.05])   # az, el, dist, lookat
        self.cam_target = self.cam.copy()
        self.follow = None      # callable -> lookat xyz, or None
        self.fleet: list[str] = []
        self.footer = ""

    def _render(self, camera, data=None):
        # ease the camera toward its target; follow a moving body if asked
        if self.follow is not None:
            self.cam_target[3:] = self.follow()
        self.cam += (self.cam_target - self.cam) * 0.06
        spec = "free:" + ",".join("%.4f" % v for v in self.cam)
        return super()._render(spec, data)

    def _draw_hud(self, bgr):
        super()._draw_hud(bgr)
        if self.fleet:
            w = max(self._cv2.getTextSize(t, self._cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for t in self.fleet)
            draw_text_block(bgr, self.fleet, self.size[0] - w - 30, 12, scales=[0.62] + [0.55] * (len(self.fleet) - 1), alpha=0.6)
        if self.footer:
            draw_text_block(bgr, [self.footer], None, self.size[1] - 72, scales=[0.42], alpha=0.5, pad=6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=903)
    ap.add_argument("--units", type=int, default=4)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tail", type=float, default=52.0, help="seconds of drone/rover/finale after the run")
    ap.add_argument("--quick", action="store_true", help="640x360 test render")
    args = ap.parse_args()

    import mujoco
    fleet_scene.install(units=args.units)
    from omni_q import nlu
    from omni_q.intel_sim import IntelSceneConfig, TABLE_SETTING_PHRASINGS, build_intel_sim_engine
    from omni_q.voice import SpeakerRegistry, SpeechFinal

    args.out.mkdir(parents=True, exist_ok=True)
    engine = build_intel_sim_engine(IntelSceneConfig(seed=args.seed, randomized=True))
    world = engine.world
    m, d = world.model, world.data
    dt = float(m.opt.timestep)
    goal = TABLE_SETTING_PHRASINGS[0]

    rec = FleetRecorder(world, args.out / f"fleet_escalation_seed{args.seed}{'_quick' if args.quick else ''}.mp4",
                        label="OMNI-Q  showcase: fleet escalation", size=(640, 360) if args.quick else (1280, 720))
    rec.attach(world, goal)
    rec.footer = "unit 1 (centre): OMNI-planned, the submission stack  |  units 2-4: physically simulated arms on scripted choreography  |  drone & rover: kinematic showcase actors"
    n_arms = 2 * args.units
    rec.fleet = [f"FLEET: {n_arms} MANIPULATORS", "ACTIVE UNITS: 1", "OBJECTIVES: 1", "CONFLICTS: 0"]

    # extra units
    arms: dict[int, tuple[ExtraArm, ExtraArm]] = {}
    for unit in range(2, args.units + 1):
        ln, rn = fleet_scene.unit_arm_names(unit)
        L, R = ExtraArm(m, ln), ExtraArm(m, rn)
        L.scan(d); R.scan(d)
        arms[unit] = (L, R)
    active_units = {1}
    drone_id = m.body("drone_01").mocapid[0] if m.nmocap else -1
    rover_id = m.body("mobile_01").mocapid[0] if m.nmocap > 1 else -1
    drone_pos0 = d.mocap_pos[drone_id].copy() if drone_id >= 0 else None
    rover_pos0 = d.mocap_pos[rover_id].copy() if rover_id >= 0 else None
    state = {"dance_until": -1.0, "dance_t0": 0.0, "drone_t0": None, "rover_t0": None}

    def sim_t() -> float:
        return float(d.time)

    def drone_path(t: float):
        # takeoff 3 s, sweep the row 9 s, hover over unit 1 for 4 s, return 5 s, land 3 s
        p0 = drone_pos0
        if t < 3:
            f = t / 3; return p0 + np.array([0, 0, 1.6 * f * f * (3 - 2 * f)]), 0.0
        if t < 12:
            f = (t - 3) / 9
            x = p0[0] + (2.3 - p0[0]) * f
            return np.array([x, 0.55, p0[2] + 1.6 + 0.1 * math.sin(f * math.pi)]), 0.0
        if t < 16:
            f = (t - 12) / 4
            return np.array([2.3 + (0.0 - 2.3) * f, 0.55 + (0.35 - 0.55) * f, p0[2] + 1.6 - 0.2 * f]), math.pi
        if t < 20:
            return np.array([0.0, 0.35, p0[2] + 1.4]), math.pi
        if t < 25:
            f = (t - 20) / 5
            return np.array([0.0 + (p0[0] - 0.0) * f, 0.35 + (p0[1] - 0.35) * f, p0[2] + 1.4 + 0.2 * f]), math.pi
        f = min(1.0, (t - 25) / 3)
        return p0 + np.array([0, 0, 1.6 * (1 - f)]), math.pi

    def rover_path(t: float):
        p0 = rover_pos0
        goal_x = 0.05
        travel = abs(p0[0] - goal_x) / 0.35
        f = min(1.0, t / travel)
        f = f * f * (3 - 2 * f) if f < 1 else 1.0
        return np.array([p0[0] + (goal_x - p0[0]) * f, p0[1] - 0.25 * f, p0[2]]), math.pi

    def yaw_quat(yaw: float):
        return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])

    real = world._mujoco
    rec_spy = rec.spy()

    class Spy:
        def __getattr__(self, n):
            return getattr(real, n)

        def mj_step(self, mm, dd, nstep=1):
            for _ in range(int(nstep)):
                t = sim_t()
                if t < state["dance_until"]:
                    tau = t - state["dance_t0"]
                    for k, (L, R) in enumerate(arms.values()):
                        for j, arm in enumerate((L, R)):
                            ph = 0.9 * (2 * k + j)
                            q = HOME5 + np.array([0.45 * math.sin(2 * math.pi * 1.2 * tau + ph), 0.35 * math.sin(2 * math.pi * 1.2 * tau + ph),
                                                  -0.35 * math.sin(2 * math.pi * 1.2 * tau + ph), 0.0, 0.9 * math.sin(2 * math.pi * 2.0 * tau)])
                            arm.write(dd, np.append(q, 0.75 + 0.75 * math.sin(2 * math.pi * 2.0 * tau + ph)))
                else:
                    for unit, (L, R) in arms.items():
                        if unit in active_units:
                            L.write(dd, L.ctrl_at(t, dd)); R.write(dd, R.ctrl_at(t, dd))
                        else:
                            L.write(dd, np.append(HOME5, JAW_OPEN)); R.write(dd, np.append(HOME5, JAW_OPEN))
                if drone_id >= 0 and state["drone_t0"] is not None:
                    p, yaw = drone_path(t - state["drone_t0"])
                    dd.mocap_pos[drone_id] = p; dd.mocap_quat[drone_id] = yaw_quat(yaw)
                if rover_id >= 0 and state["rover_t0"] is not None:
                    p, yaw = rover_path(t - state["rover_t0"])
                    dd.mocap_pos[rover_id] = p; dd.mocap_quat[rover_id] = yaw_quat(yaw)
                rec_spy.mj_step(mm, dd, 1)

        def main_thread_pump(self):
            rec_spy.main_thread_pump()

    world._mujoco = Spy()

    def dance_all(seconds: float = 3.0) -> None:
        """Voice-triggered: every arm, including the submission pair, for 3 s."""
        state["dance_t0"] = sim_t(); state["dance_until"] = sim_t() + seconds
        t_end = state["dance_until"]
        while sim_t() < t_end:
            tau = sim_t() - state["dance_t0"]
            for j, off in enumerate((0, 6)):
                ph = 0.9 * j + 0.3
                q = HOME5 + np.array([0.45 * math.sin(2 * math.pi * 1.2 * tau + ph), 0.35 * math.sin(2 * math.pi * 1.2 * tau + ph),
                                      -0.35 * math.sin(2 * math.pi * 1.2 * tau + ph), 0.0, 0.9 * math.sin(2 * math.pi * 2.0 * tau)])
                d.ctrl[off:off + 5] = q; d.ctrl[off + 5] = 0.75 + 0.75 * math.sin(2 * math.pi * 2.0 * tau + ph)
            world._mujoco.mj_step(m, d)
        for off in (0, 6):
            world._go_home(off)

    def voice_show_off() -> None:
        reg = SpeakerRegistry("showcase-session", "omni-q")
        ev = SpeechFinal(session_id="showcase-session", org_id="omni-q", speaker_id="operator", text="OMNI, show off",
                         t_start_ns=int(sim_t() * 1e9), t_end_ns=int(sim_t() * 1e9) + 900_000_000, sequence=1, confidence=0.96)
        reg.observe(ev)
        parsed = nlu.parse(ev.text)
        style = dict(parsed.constraints).get("style")
        rec.notify(f"VOICE  operator (conf {ev.confidence:.2f}): \"{ev.text}\"   ->  NLU: style={style}   ->  fleet choreography, 3 s", 6)
        if style == "show_off":
            dance_all(3.0)
        rec.notify("back to work", 2)

    # shot plan keyed to the submission run's own progress
    events = {"n": 0}
    orig, orig_par = world.apply_transition, world.apply_transitions_parallel

    def after(reqs):
        events["n"] += 1
        n = events["n"]
        if n == 3:      # plate set: reveal unit 2
            active_units.add(2)
            L, R = arms[2]; unit_choreography(L, R, fleet_scene.UNIT_X[2], sim_t())
            rec.cam_target[:] = [120.0, -28.0, 2.4, 0.55, -0.10, 0.0]
            rec.fleet = [f"FLEET: {n_arms} MANIPULATORS", "ACTIVE UNITS: 2", "OBJECTIVES: 2", "CONFLICTS: 0"]
        elif n == 5:    # after the cup/fork pair: the voice moment, then reveal all units
            voice_show_off()
            for unit in range(3, args.units + 1):
                active_units.add(unit)
                L, R = arms[unit]; unit_choreography(L, R, fleet_scene.UNIT_X[unit], sim_t() + 0.7 * unit)
            rec.cam_target[:] = [135.0, -30.0, 4.8, 0.6, -0.10, 0.0]
            rec.fleet = [f"FLEET: {n_arms} MANIPULATORS", f"ACTIVE UNITS: {len(active_units)}",
                         f"OBJECTIVES: {len(active_units)}", "CONFLICTS: 0"]

    def apply(req):
        res = orig(req); after([req]); return res

    def apply_pair(reqs):
        out = orig_par(reqs); after(reqs); return out
    world.apply_transition, world.apply_transitions_parallel = apply, apply_pair

    t_wall = time.time()
    receipt = engine.run(goal)
    placed = receipt.metrics.get("resolved")

    # -- the tail: drone, rover, finale (physics keeps running; units keep working)
    def run_for(seconds: float) -> None:
        t_end = sim_t() + seconds
        while sim_t() < t_end:
            world._mujoco.mj_step(m, d)

    if drone_id >= 0:
        state["drone_t0"] = sim_t()
        rec.fleet.append("DRONE_01: ONLINE   CAPABILITY: AERIAL OBSERVATION")
        rec.notify("nobody asked for aviation.", 3)
        rec.follow = lambda: d.mocap_pos[drone_id] + np.array([0, 0, -0.1])
        rec.cam_target[:3] = [150.0, -20.0, 2.4]
        run_for(12.0)
        rec.cam_target[:3] = [140.0, -35.0, 3.2]
        run_for(8.0)
        rec.follow = None
        rec.cam_target[:] = [135.0, -30.0, 4.8, 0.6, -0.10, 0.0]
        run_for(8.0)
    if rover_id >= 0:
        state["rover_t0"] = sim_t()
        rec.fleet.append("MOBILE_01: ONLINE   CAPABILITY: TRANSPORT")
        rec.notify("MOBILE_01 delivering: 1 cup -> unit 1", 4)
        rec.follow = lambda: d.mocap_pos[rover_id] + np.array([0, 0, 0.3])
        rec.cam_target[:3] = [110.0, -22.0, 2.6]
        run_for(11.0)
        rec.follow = None
    rec.cam_target[:] = [140.0, -35.0, 8.0, 0.6, 0.2, 0.0]
    rec.hud = [f'run complete: "{goal}" -- resolved={placed}']
    rec.fleet = [f"FLEET: {n_arms} MANIPULATORS + DRONE_01 + MOBILE_01", f"ACTIVE UNITS: {len(active_units)}   OBJECTIVES: {len(active_units)}   CONFLICTS: 0",
                 "PERCEPTION: 6 cameras   VOICE: online", "", "                 OMNI", "        manipulators | drone | mobile", "             one objective space"]
    rec.notify("one orchestration layer: manipulation, transport, aerial observation, perception, voice", 8)
    run_for(max(0.0, args.tail - 39.0) + 10.0)
    print(rec.close(), "resolved", placed, f"({time.time() - t_wall:.0f}s wall)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
