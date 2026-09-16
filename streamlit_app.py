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


def run_profile_mission(profile_id: str, goal: str, constraint_kind: str,
                        constraint_value: str, justification: str) -> dict:
    from omni_q.events import EventBus
    from omni_q.profiles import build_profile_engine

    bus = EventBus()
    mission = build_profile_engine(profile_id, bus)
    if constraint_kind:
        value = constraint_value.strip() or None
        mission.add_constraint(
            constraint_kind,
            value,
            source="operator",
            justification=justification.strip() or "operator direction",
        )
    receipt = mission.run(goal)
    return {
        "profile": mission.profile.as_dict(),
        "events": [event.as_dict() for event in bus.log],
        "receipt": receipt.as_dict(),
    }


st.set_page_config(page_title="OMNI-Q Demo", page_icon="O", layout="wide")
st.title("OMNI-Q: Observe -> Plan -> Act -> Verify")
st.caption("Mock demo and browser-rendered MuJoCo simulation")

tab_demo, tab_control, tab_sim, tab_voice = st.tabs([
    "OMNI-Q demo",
    "Mission control",
    "Live sim view",
    "Speechmatics voice",
])

with tab_demo:
    selection = st.selectbox("Scenario", list(SCENARIOS))
    if st.button("Run demo", type="primary"):
        with st.spinner("Running scenario..."):
            result = run_demo(selection)
        st.code(result, language="text")
    else:
        st.info("Choose a scenario and click Run demo.")

with tab_control:
    from omni_q.profiles import list_profiles

    profiles = list_profiles()
    profile_options = {profile.profile_id: profile for profile in profiles}
    profile_id = st.selectbox("Venue / event profile", list(profile_options))
    profile = profile_options[profile_id]
    st.caption(f"{profile.venue} · {profile.guest_count} guests · {profile.required_count} required placements")
    goal = st.text_input("Operator objective", value="Set the table.", key="profile_goal")
    constraint_kind = st.selectbox(
        "Initial constraint",
        ("", "keep_local", "prefer_arm", "forbid_object"),
        format_func=lambda value: {
            "": "No additional constraint",
            "keep_local": "Keep inference on-device",
            "prefer_arm": "Prefer an arm",
            "forbid_object": "Do not touch an object",
        }[value],
        key="profile_constraint_kind",
    )
    constraint_value = st.text_input(
        "Constraint value",
        value="left" if constraint_kind == "prefer_arm" else "",
        help="Use an object ID for forbid_object, or left/right for prefer_arm.",
        key="profile_constraint_value",
    )
    justification = st.text_input("Why this constraint?", value="operator direction", key="profile_justification")
    if st.button("Start mission control", type="primary"):
        with st.spinner("Running profile mission..."):
            st.session_state["profile_run"] = run_profile_mission(
                profile_id,
                goal,
                constraint_kind,
                constraint_value,
                justification,
            )

    profile_run = st.session_state.get("profile_run")
    if profile_run:
        receipt = profile_run["receipt"]
        metrics = receipt.get("metrics", {})
        report = metrics.get("profile_report", {})
        st.subheader("Mission pulse")
        event_kinds = {event.get("kind", "") for event in profile_run["events"]}
        phases = {
            "observe": "observed" in event_kinds,
            "plan": bool(event_kinds & {"graph.compiled", "graph.recompiled"}),
            "act": "step.finished" in event_kinds,
            "verify": "verified" in event_kinds,
        }
        phase_columns = st.columns(4)
        for column, phase in zip(phase_columns, ("observe", "plan", "act", "verify")):
            column.metric(phase.title(), "done" if phases[phase] else "queued")
        metric_columns = st.columns(4)
        metric_columns[0].metric("Placements", report.get("final_compliance", "-"))
        metric_columns[1].metric("Replans", metrics.get("revisions", "-"))
        metric_columns[2].metric("Actions", metrics.get("steps_executed", "-"))
        metric_columns[3].metric("Status", report.get("status", "-"))

        with st.expander("Execution graph", expanded=True):
            st.json(receipt.get("plan", {}))
        with st.expander("Why OMNI-Q did that"):
            st.dataframe(profile_run["events"], use_container_width=True, hide_index=True)
        with st.expander("Run receipt"):
            st.json(receipt)

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
    st.caption("Record from your browser or upload a 16-bit mono WAV. The API key is read from SPEECHMATICS_API_KEY and never shown.")
    audio_file = st.file_uploader("Speechmatics WAV file", type=["wav"])
    microphone_file = st.audio_input("Record a voice command")
    if st.button("Run live transcription"):
        api_key = speechmatics_key()
        if not api_key:
            st.error("Add SPEECHMATICS_API_KEY to Streamlit Secrets or your local environment first.")
        elif microphone_file is None and audio_file is None:
            st.error("Record a voice command or upload a WAV file first.")
        else:
            selected_audio = microphone_file or audio_file
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                    handle.write(selected_audio.getvalue())
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