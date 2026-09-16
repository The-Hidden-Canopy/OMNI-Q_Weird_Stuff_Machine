from __future__ import annotations

import contextlib
import io
import sys
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


st.set_page_config(page_title="OMNI-Q Demo", page_icon="O", layout="wide")
st.title("OMNI-Q: Observe -> Plan -> Act -> Verify")
st.caption("Mock demo; no hardware required")

selection = st.selectbox("Scenario", list(SCENARIOS))

if st.button("Run demo", type="primary"):
    with st.spinner("Running scenario..."):
        result = run_demo(selection)
    st.code(result, language="text")
else:
    st.info("Choose a scenario and click Run demo.")