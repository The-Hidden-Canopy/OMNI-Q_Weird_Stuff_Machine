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
VOICE_SCRIPT = REPO_ROOT / "integrations" / "speechmatics" / "scripts" / "run_voice_transport.py"
VOICE_REPLAY = REPO_ROOT / "integrations" / "speechmatics" / "samples" / "session_replay.jsonl"


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


def speechmatics_key() -> str:
    try:
        secret = st.secrets.get("SPEECHMATICS_API_KEY", "")
    except Exception:
        secret = ""
    return str(secret or os.environ.get("SPEECHMATICS_API_KEY", "")).strip()


def run_voice(source: Path, *, replay: bool, operator: str, language: str,
              api_key: str = "") -> str:
    command = [
        sys.executable,
        str(VOICE_SCRIPT),
        "--operator",
        operator,
        "--language",
        language,
        "--mock-engine",
    ]
    if replay:
        command.extend(["--replay", str(source)])
    else:
        command.extend(["--file", str(source), "--no-response"])
    environment = {**os.environ, "PYTHONPATH": str(SRC_ROOT)}
    if api_key:
        environment["SPEECHMATICS_API_KEY"] = api_key
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout.strip()
    if completed.returncode != 0:
        error = completed.stderr.strip() or output or "Speechmatics run failed."
        raise RuntimeError(error)
    return output


st.set_page_config(page_title="OMNI-Q Demo", page_icon="O", layout="wide")
st.title("OMNI-Q: Observe -> Plan -> Act -> Verify")
st.caption("Mock demo and browser-rendered MuJoCo simulation")

tab_demo, tab_sim, tab_voice = st.tabs(["OMNI-Q demo", "Live sim view", "Speechmatics voice"])

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

with tab_voice:
    st.write("Speech -> transcription -> authorized OMNI-Q graph mutation")
    operator = st.text_input("Authorized speaker label", value="S1")
    language = st.selectbox("Recognition language", ("en", "es", "fr"))

    st.subheader("Offline replay")
    st.caption("Uses the checked-in Speechmatics transcript sample; no key or network required.")
    if st.button("Run Speechmatics replay"):
        try:
            with st.spinner("Replaying Speechmatics session..."):
                voice_output = run_voice(
                    VOICE_REPLAY,
                    replay=True,
                    operator=operator,
                    language=language,
                )
            st.code(voice_output, language="text")
        except RuntimeError as error:
            st.error(str(error))

    st.subheader("Live WAV transcription")
    st.caption("Upload a 16-bit mono WAV. The API key is read from SPEECHMATICS_API_KEY and never shown.")
    audio_file = st.file_uploader("Speechmatics WAV file", type=["wav"])
    if st.button("Run live transcription"):
        api_key = speechmatics_key()
        if not api_key:
            st.error("Add SPEECHMATICS_API_KEY to Streamlit Secrets or your local environment first.")
        elif audio_file is None:
            st.error("Upload a WAV file first.")
        else:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                    handle.write(audio_file.getvalue())
                    audio_path = Path(handle.name)
                try:
                    with st.spinner("Streaming audio to Speechmatics..."):
                        voice_output = run_voice(
                            audio_path,
                            replay=False,
                            operator=operator,
                            language=language,
                            api_key=api_key,
                        )
                finally:
                    audio_path.unlink(missing_ok=True)
                st.code(voice_output, language="text")
            except RuntimeError as error:
                st.error(str(error))