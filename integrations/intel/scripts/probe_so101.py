"""
probe_so101.py
================
Loads the pinned SO-ARM100 MJCF (The Robot Studio lineage — the SO-101's
mechanical parent, see assets/menagerie_so_arm100/SOURCE.md) and measures the
arm facts Omni Q's Intel track needs: joints, limits, actuators, gripper
opening, wrist-roll travel, and an FK-sampled reachable workspace envelope.

Evidence conventions follow the hardware-probe pattern from
Semantically-Aware_ISR (schema_version / probe_revision / probed_at, JSON
report + appended .jsonl history under evidence/).

Portions derived from *The Hidden Canopy LLC* —
[`Semantically-Aware_ISR`](https://github.com/The-Hidden-Canopy/Semantically-Aware_ISR).
Used with permission.

This module never installs or downloads anything -- it only measures and
reports. Requires: mujoco (see integrations/intel/requirements.txt).

    .venv/Scripts/python integrations/intel/scripts/probe_so101.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XML = ROOT / "integrations" / "intel" / "assets" / "menagerie_so_arm100" / "so_arm100.xml"
EVIDENCE_DIR = ROOT / "evidence" / "benchmark_results"

PROBE_REVISION = 1

# XML joint name -> LeRobot SO-101 motor name (huggingface/lerobot
# src/lerobot/robots/so_follower/so_follower.py, pinned in the capability map).
LEROBOT_NAMES = {
    "Rotation": "shoulder_pan",
    "Pitch": "shoulder_lift",
    "Elbow": "elbow_flex",
    "Wrist_Pitch": "wrist_flex",
    "Wrist_Roll": "wrist_roll",
    "Jaw": "gripper",
}

# Joints that position the wrist center (gripper excluded from the sweep).
ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]

GRIPPER_PADS = ("fixed_jaw_pad_4", "moving_jaw_pad_4")


def _deg(rad: float) -> float:
    return round(rad * 180.0 / 3.141592653589793, 2)


def load_model(xml_path: Path):
    import mujoco

    if not xml_path.exists():
        raise SystemExit(f"[so101-probe] FATAL: MJCF not found: {xml_path}")
    return mujoco.MjModel.from_xml_path(str(xml_path))


def joint_table(model) -> list[dict]:
    rows = []
    for jid in range(model.njnt):
        name = model.joint(jid).name
        lo, hi = model.jnt_range[jid]
        rows.append({
            "xml_name": name,
            "lerobot_name": LEROBOT_NAMES.get(name),
            "axis": [round(float(v), 4) for v in model.jnt_axis[jid]],
            "range_rad": [round(float(lo), 4), round(float(hi), 4)],
            "range_deg": [_deg(lo), _deg(hi)],
            "range_source": "mjc_jnt_range (menagerie notes: approximate limits)",
        })
    return rows


def actuator_table(model) -> list[dict]:
    rows = []
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        lo, hi = model.actuator_ctrlrange[aid]
        rows.append({
            "name": model.actuator(aid).name,
            "joint": model.joint(jid).name,
            "ctrlrange_rad": [round(float(lo), 4), round(float(hi), 4)],
            "ctrlrange_deg": [_deg(lo), _deg(hi)],
            "gear": round(float(model.actuator_gear[aid, 0]), 2),
            "kp": round(float(model.actuator_gainprm[aid, 0]), 2),
            "forcerange": [round(float(v), 3) for v in model.actuator_forcerange[aid]],
        })
    return rows


def keyframes(model) -> dict[str, list[float]]:
    frames = {}
    for kid in range(model.nkey):
        qpos = model.key_qpos[kid][: model.nq]
        frames[model.key(kid).name] = [round(float(v), 4) for v in qpos]
    return frames


def fk_workspace(model, data, samples: int, table_height: float,
                 rng) -> dict:
    """Random FK sampling over the 5 arm joints; report the reachable
    envelope of the gripper mount (Fixed_Jaw body) and xy coverage at a
    table plane."""
    import mujoco
    import numpy as np

    jids = [model.joint(n).id for n in ARM_JOINTS]
    ranges = model.jnt_range[jids]
    lo, hi = ranges[:, 0], ranges[:, 1]
    tcp_id = model.body("Fixed_Jaw").id

    # Gripper parked at its home keyframe value so the TCP frame is stable.
    grip_home = float(model.key_qpos[model.key("home").id][model.joint("Jaw").id])

    pts = np.empty((samples, 3))
    for i in range(samples):
        q = rng.uniform(lo, hi)
        data.qpos[:] = 0.0
        for k, jid in enumerate(jids):
            data.qpos[jid] = q[k]
        data.qpos[model.joint("Jaw").id] = grip_home
        mujoco.mj_forward(model, data)
        pts[i] = data.body(tcp_id).xpos

    env = {
        "samples": samples,
        "tcp_body": "Fixed_Jaw",
        "x": [round(float(pts[:, 0].min()), 4), round(float(pts[:, 0].max()), 4)],
        "y": [round(float(pts[:, 1].min()), 4), round(float(pts[:, 1].max()), 4)],
        "z": [round(float(pts[:, 2].min()), 4), round(float(pts[:, 2].max()), 4)],
        "max_radial_xy": round(float(np.hypot(pts[:, 0], pts[:, 1]).max()), 4),
        "table_plane_z": table_height,
    }
    on_plane = pts[np.abs(pts[:, 2] - table_height) <= 0.01]
    env["table_plane"] = {
        "tolerance_m": 0.01,
        "samples_on_plane": int(on_plane.shape[0]),
        "coverage_fraction": round(float(on_plane.shape[0]) / samples, 4),
        "x": ([round(float(on_plane[:, 0].min()), 4), round(float(on_plane[:, 0].max()), 4)]
              if on_plane.shape[0] else None),
        "y": ([round(float(on_plane[:, 1].min()), 4), round(float(on_plane[:, 1].max()), 4)]
              if on_plane.shape[0] else None),
        "max_radial_xy": (round(float(np.hypot(on_plane[:, 0], on_plane[:, 1]).max()), 4)
                          if on_plane.shape[0] else None),
    }

    # Front workspace: the arm faces -y at the home keyframe (measured
    # TCP ~= [0, -0.239, 0.177]). The demo table lives in that half-space.
    front = pts[pts[:, 1] < -0.05]
    env["front_y_neg"] = {
        "definition": "TCP samples with y < -0.05 (arm faces -y at home keyframe)",
        "samples": int(front.shape[0]),
        "coverage_fraction": round(float(front.shape[0]) / samples, 4),
        "x": ([round(float(front[:, 0].min()), 4), round(float(front[:, 0].max()), 4)]
              if front.shape[0] else None),
        "y": ([round(float(front[:, 1].min()), 4), round(float(front[:, 1].max()), 4)]
              if front.shape[0] else None),
        "z": ([round(float(front[:, 2].min()), 4), round(float(front[:, 2].max()), 4)]
              if front.shape[0] else None),
        "max_radial_xy": (round(float(np.hypot(front[:, 0], front[:, 1]).max()), 4)
                          if front.shape[0] else None),
    }
    return env


def wrist_roll_analysis(model) -> dict:
    jid = model.joint("Wrist_Roll").id
    lo, hi = (float(v) for v in model.jnt_range[jid])
    travel = hi - lo
    import math
    return {
        "range_rad": [round(lo, 4), round(hi, 4)],
        "range_deg": [_deg(lo), _deg(hi)],
        "total_travel_rad": round(travel, 4),
        "total_travel_deg": _deg(travel),
        "full_turns_within_range": round(travel / (2 * math.pi), 3),
        "spin_implication": (
            "less than one full 360 deg turn in a single grasp -- rotations "
            "beyond ~%.0f deg require handoff/regrasp (feeds OQ-014)"
            % _deg(travel)
        ),
    }


def gripper_analysis(model, data) -> dict:
    import mujoco

    home = model.key("home").id
    data.qpos[:] = 0.0
    data.qpos[:] = model.key_qpos[home][: model.nq]
    gid = model.joint("Jaw").id
    lo, hi = (float(v) for v in model.jnt_range[gid])
    pad_a, pad_b = (model.geom(n).id for n in GRIPPER_PADS)

    openings = {}
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        data.qpos[gid] = lo + frac * (hi - lo)
        mujoco.mj_forward(model, data)
        gap = float(((data.geom(pad_a).xpos - data.geom(pad_b).xpos) ** 2).sum() ** 0.5)
        openings[f"{frac:.2f}"] = round(gap, 5)

    mujoco.mj_forward(model, data)
    return {
        "joint": "Jaw",
        "lerobot_name": "gripper",
        "lerobot_norm": "RANGE_0_100 (percent open); body joints use degrees",
        "range_rad": [round(lo, 4), round(hi, 4)],
        "range_deg": [_deg(lo), _deg(hi)],
        "pad_gap_by_fraction": openings,
        "max_opening_m": openings["1.00"],
        "note": "gap measured between finger-pad tip geoms %s" % (GRIPPER_PADS,),
    }


def build_spec(xml_path: Path, samples: int, table_height: float, seed: int) -> dict:
    import mujoco
    import numpy as np

    model = load_model(xml_path)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)

    return {
        "schema_version": 1,
        "probe_revision": PROBE_REVISION,
        "probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mujoco_version": mujoco.__version__,
        "model": {
            "name": "so_arm100 (menagerie trs_so_arm100) -- SO-101 mechanical lineage",
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "total_arm_mass_kg": round(
                float(sum(float(model.body(i).mass[0]) for i in range(model.nbody))), 4),
            "timestep_s": model.opt.timestep,
        },
        "sources": {
            "mjcf": "integrations/intel/assets/menagerie_so_arm100/so_arm100.xml",
            "mjcf_origin": "google-deepmind/mujoco_menagerie @ 8161bba264d7fa7c99ca301e91e7fb44737676ad",
            "lerobot_crosscheck": "huggingface/lerobot @ b6ec0060779550c0a157ae34feb89e0cf86012a8 (so_follower.py)",
        },
        "joints": joint_table(model),
        "actuators": actuator_table(model),
        "keyframes": keyframes(model),
        "cameras": [model.camera(i).name for i in range(model.ncam)],
        "sensors": [model.sensor(i).name for i in range(model.nsensor)],
        "workspace": fk_workspace(model, data, samples, table_height, rng),
        "wrist_roll": wrist_roll_analysis(model),
        "gripper": gripper_analysis(model, data),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe SO-101-class arm capabilities from MJCF.")
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--samples", type=int, default=40000)
    p.add_argument("--table-height", type=float, default=0.0,
                   help="z of the table plane relative to the arm base origin (m).")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", default=None,
                   help="default: evidence/benchmark_results/so101_capability_map_<date>")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else (
        EVIDENCE_DIR / time.strftime("so101_capability_map_%Y-%m-%d", time.gmtime()))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[so101-probe] loading {args.xml} ...", flush=True)
    spec = build_spec(Path(args.xml), args.samples, args.table_height, args.seed)

    spec_path = out_dir / "so101_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[so101-probe] wrote {spec_path}", flush=True)

    hist_path = EVIDENCE_DIR / "so101_probe_history.jsonl"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with hist_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(spec, sort_keys=True) + "\n")

    w = spec["workspace"]
    print(
        f"[so101-probe] joints={len(spec['joints'])} actuators={len(spec['actuators'])} "
        f"cameras={len(spec['cameras'])} sensors={len(spec['sensors'])}\n"
        f"[so101-probe] envelope x={w['x']} y={w['y']} z={w['z']} "
        f"max_radial={w['max_radial_xy']}m\n"
        f"[so101-probe] table@{args.table_height}m: {w['table_plane']['samples_on_plane']}/{w['samples']} "
        f"samples reachable\n"
        f"[so101-probe] wrist_roll travel={spec['wrist_roll']['total_travel_deg']}deg "
        f"gripper max_open={spec['gripper']['max_opening_m']}m",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
