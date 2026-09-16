from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from omni_q.demo import (  # noqa: E402
    scenario_constraint_change,
    scenario_normal,
    scenario_world_change,
)


SCENARIOS = {
    "All scenarios": None,
    "1. Normal": scenario_normal,
    "2. Constraint change": scenario_constraint_change,
    "3. World change / replan": scenario_world_change,
}

SIM_OBJECTS = ("cup_1", "plate_1", "fork_1", "spoon_1", "napkin_1")
SIM_CAMERAS = ("third_person", "table_overhead", "left_flank", "right_flank", "left_wrist", "right_wrist")


def run_demo(selection: str) -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        if selection == "All scenarios":
            scenario_normal()
            scenario_constraint_change()
            scenario_world_change()
            print("\nAll scenarios completed.\n")
        else:
            SCENARIOS[selection]()
    return output.getvalue()


def run_simulation(object_name: str, camera: str, seed: int) -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        gif_path = Path(temp_dir) / "simulation.gif"
        command = [
            sys.executable,
            str(REPO_ROOT / "integrations" / "intel" / "scripts" / "watch_sim.py"),
            "--record",
            str(gif_path),
            "--camera",
            camera,
            "--pick",
            object_name,
            "--move",
            "--seed",
            str(seed),
            "--every",
            "8",
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(SRC_ROOT)},
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not gif_path.exists():
            details = completed.stderr.strip() or completed.stdout.strip() or "Simulation failed."
            raise RuntimeError(details)
        return gif_path.read_bytes(), completed.stdout


st.set_page_config(page_title="OMNI-Q Demo", page_icon="O", layout="wide")
st.title("OMNI-Q: Observe -> Plan -> Act -> Verify")
st.caption("Mock demo and browser-rendered MuJoCo simulation")

tab_demo, tab_sim = st.tabs(["OMNI-Q demo", "Live sim view"])

with tab_demo:
    selection = st.selectbox("Scenario", list(SCENARIOS))
    if st.button("Run demo", type="primary"):
        with st.spinner("Running scenario..."):
            result = run_demo(selection)
        st.code(result, language="text")
    else:
        st.info("Choose a scenario and click Run demo.")

with tab_sim:
    st.write("Run a real MuJoCo table scene and view the camera recording here.")
    object_name = st.selectbox("Object", SIM_OBJECTS)
    camera = st.selectbox("Camera", SIM_CAMERAS)
    seed = st.number_input("Random seed", min_value=0, value=701, step=1)
    if st.button("Run live simulation", type="primary"):
        try:
            with st.spinner("Running MuJoCo simulation..."):
                gif_bytes, sim_output = run_simulation(object_name, camera, int(seed))
            st.image(gif_bytes)
            st.code(sim_output, language="text")
        except RuntimeError as error:
            st.error(str(error))
    else:
        st.info("Choose an object and camera, then run the simulation.")