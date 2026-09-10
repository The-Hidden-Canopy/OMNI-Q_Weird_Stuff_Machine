"""Flourish-envelope probe (OQ-026 measured pre-work).

Measures, from the pinned SO-ARM100 MJCF alone (no dependency on the
contact-handoff controller):

1. ``velocity_limits``  -- max achievable joint speeds and TCP linear speed
   under the model's own position actuators (kp=50, forcerange +/-3.5).
2. ``grip_force``       -- contact force vs. jaw angle closing on a static
   40 mm probe cylinder between the pads (clamp-force envelope).
3. ``shared_workspace`` -- bimanual reachable overlap for the dual-scene mount
   pair (left/right at x=+/-0.26, y=0.20, both facing -y): candidate handoff
   region in world coordinates.

Evidence conventions match probe_so101.py: schema_version / probe_revision /
probed_at, one JSON bundle under evidence/benchmark_results/, appended .jsonl
history. Failure-RATE trials remain OQ-026 proper and need the OQ-014 spin
primitive -- this probe supplies the bounds those trials must respect.

    .venv/Scripts/python integrations/intel/scripts/probe_flourish_envelope.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARM_XML = ROOT / "integrations" / "intel" / "assets" / "menagerie_so_arm100" / "so_arm100.xml"
ARM_ASSETS = ARM_XML.parent / "assets"
EVIDENCE_DIR = ROOT / "evidence" / "benchmark_results"

PROBE_REVISION = 1
HOME = (0.0, -1.57, 1.57, 1.57, -1.57, 0.0)
ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
# dual_so101_xml() mount poses (omni_q.intel_sim)
MOUNTS = {"left": (-0.26, 0.20, 0.0), "right": (0.26, 0.20, 0.0)}


def _load(mujoco, xml_text: str):
    assets = {f"assets/{p.name}": p.read_bytes() for p in ARM_ASSETS.glob("*.stl")}
    return mujoco.MjModel.from_xml_string(xml_text, assets=assets)


def velocity_limits(mujoco) -> dict:
    """Command each joint min->max and max->min; record peak |qvel| and the
    TCP linear speed peak over the whole sweep."""
    import numpy as np

    model = _load(mujoco, ARM_XML.read_text(encoding="utf-8"))
    data = mujoco.MjData(model)
    tcp = model.body("Fixed_Jaw").id
    rows = []
    for jid, name in enumerate(ARM_JOINTS):
        lo, hi = (float(v) for v in model.jnt_range[model.joint(name).id])
        for target in (hi, lo):
            data.qpos[:] = 0.0
            data.qpos[:] = list(HOME)
            data.ctrl[:] = 0.0
            data.ctrl[:] = list(HOME)
            mujoco.mj_forward(model, data)
            data.ctrl[jid] = target
            peak_qvel = 0.0
            peak_tcp = 0.0
            vel = np.zeros(6)
            for _ in range(600):  # 1.2 s sim
                mujoco.mj_step(model, data)
                peak_qvel = max(peak_qvel, abs(float(data.qvel[jid])))
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, tcp, vel, 0)
                peak_tcp = max(peak_tcp, float(np.linalg.norm(vel[3:])))
            rows.append({
                "joint": name,
                "sweep_to": "max" if target == hi else "min",
                "peak_joint_speed_rad_s": round(peak_qvel, 3),
                "peak_joint_speed_deg_s": round(peak_qvel * 57.2958, 1),
                "peak_tcp_speed_m_s": round(peak_tcp, 3),
            })
    return {"method": "position-actuator step command, 1.2 s sweep, model kp=50 forcerange+/-3.5",
            "rows": rows}


def grip_force(mujoco) -> dict:
    """Close the jaw on a static probe cylinder; record normal contact force."""
    import numpy as np

    xml = ARM_XML.read_text(encoding="utf-8")
    probe = (
        '<body name="probe" pos="-0.02 -0.24 0.125">'
        '<geom name="probe_geom" type="cylinder" size=".02 .03" '
        'rgba=".9 .2 .2 1" contype="1" conaffinity="1"/>'
        "</body>"
        '<geom name="probe_floor" type="plane" size="0 0 .05" pos="0 0 0"/>'
    )
    xml = xml.replace("</worldbody>", probe + "</worldbody>")
    model = _load(mujoco, xml)
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[:] = list(HOME)
    data.qpos[model.joint("Jaw").id] = 1.65  # pads open at home height
    data.ctrl[:] = 0.0
    data.ctrl[:] = list(HOME)
    data.ctrl[model.joint("Jaw").id] = 1.65
    mujoco.mj_forward(model, data)

    jaw = model.joint("Jaw").id
    lo = float(model.jnt_range[jaw][0])
    probe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe_geom")
    samples = []
    steps = 24
    for k in range(steps + 1):
        target = 1.65 + (lo - 1.65) * k / steps
        data.ctrl[jaw] = target
        for _ in range(60):
            mujoco.mj_step(model, data)
        force = 0.0
        ncontacts = 0
        f6 = np.zeros(6)
        for ci in range(data.ncon):
            con = data.contact[ci]
            if probe_id not in (int(con.geom1), int(con.geom2)):
                continue
            ncontacts += 1
            mujoco.mj_contactForce(model, data, ci, f6)
            force = max(force, abs(float(f6[0])))
        samples.append({
            "jaw_rad": round(float(data.qpos[jaw]), 4),
            "jaw_deg": round(float(data.qpos[jaw]) * 57.2958, 1),
            "pad_gap_m": None,
            "max_normal_force_n": round(force, 3),
            "n_pad_contacts": ncontacts,
        })
        if force > 25.0:  # stop bound well above expected clamp forces
            break
    return {
        "method": "static 40 mm probe cylinder between pads, jaw closed in 24 steps",
        "note": "controller stop bound, not a hardware-safe force calibration",
        "samples": samples,
    }


def shared_workspace(mujoco, samples: int, seed: int) -> dict:
    """FK-sweep each arm at its dual-scene mount; overlap of reachable TCP
    sets (2 cm grid) = candidate handoff region."""
    import numpy as np

    model = _load(mujoco, ARM_XML.read_text(encoding="utf-8"))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    jids = [model.joint(n).id for n in ARM_JOINTS]
    lo = model.jnt_range[jids, 0]
    hi = model.jnt_range[jids, 1]
    tcp = model.body("Fixed_Jaw").id
    grip_home = float(model.key_qpos[model.key("home").id][model.joint("Jaw").id])

    def cloud(origin):
        pts = np.empty((samples, 3))
        for i in range(samples):
            data.qpos[:] = 0.0
            for k, jid in enumerate(jids):
                data.qpos[jid] = rng.uniform(lo[k], hi[k])
            data.qpos[model.joint("Jaw").id] = grip_home
            mujoco.mj_forward(model, data)
            pts[i] = np.asarray(origin) + data.body(tcp).xpos
        return pts

    grids = {}
    boxes = {}
    for arm, origin in MOUNTS.items():
        pts = cloud(origin)
        grids[arm] = set(map(tuple, np.round(pts / 0.02).astype(int)))
        boxes[arm] = {
            "x": [round(float(pts[:, 0].min()), 3), round(float(pts[:, 0].max()), 3)],
            "y": [round(float(pts[:, 1].min()), 3), round(float(pts[:, 1].max()), 3)],
            "z": [round(float(pts[:, 2].min()), 3), round(float(pts[:, 2].max()), 3)],
        }
    overlap = grids["left"] & grids["right"]
    if overlap:
        cells = np.array(sorted(overlap)) * 0.02
        obox = {
            "x": [round(float(cells[:, 0].min()), 3), round(float(cells[:, 0].max()), 3)],
            "y": [round(float(cells[:, 1].min()), 3), round(float(cells[:, 1].max()), 3)],
            "z": [round(float(cells[:, 2].min()), 3), round(float(cells[:, 2].max()), 3)],
        }
    else:
        obox = None
    return {
        "method": f"{samples} random FK samples/arm, 2 cm grid overlap",
        "mounts": MOUNTS,
        "arm_envelope": boxes,
        "overlap_cell_count_2cm": len(overlap),
        "overlap_bbox_m": obox,
        "handoff_note": "overlap region is where a plate/cup can be passed without either arm re-mounting",
    }


def build_spec(samples: int, seed: int) -> dict:
    import mujoco

    return {
        "schema_version": 1,
        "probe_revision": PROBE_REVISION,
        "probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mujoco_version": mujoco.__version__,
        "sources": {"mjcf": "integrations/intel/assets/menagerie_so_arm100/so_arm100.xml"},
        "velocity_limits": velocity_limits(mujoco),
        "grip_force": grip_force(mujoco),
        "shared_workspace": shared_workspace(mujoco, samples, seed),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else (
        EVIDENCE_DIR / time.strftime("flourish_envelope_%Y-%m-%d", time.gmtime()))
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[flourish-probe] measuring velocity limits ...", flush=True)
    spec = build_spec(args.samples, args.seed)
    path = out_dir / "flourish_envelope.json"
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hist = EVIDENCE_DIR / "flourish_envelope_history.jsonl"
    with hist.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(spec, sort_keys=True) + "\n")
    sw = spec["shared_workspace"]
    print(f"[flourish-probe] wrote {path}", flush=True)
    print(f"[flourish-probe] overlap cells={sw['overlap_cell_count_2cm']} "
          f"bbox={sw['overlap_bbox_m']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
