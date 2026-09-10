"""Run/session isolation for the SSE boundary."""

from __future__ import annotations

import pytest

from omni_q import build_mock_engine
from omni_q.contracts import ConstraintValidationError
from omni_q.events import validate_event_chain
from omni_q.sessions import SessionError, SessionManager


def test_constraints_are_scoped_to_one_queued_session_then_close_at_terminal_state():
    manager = SessionManager(build_mock_engine)
    first = manager.create()
    second = manager.create()

    manager.add_constraint(
        first.session_id,
        "forbid_object",
        "connector_2",
        justification="operator protects this component",
    )
    manager.start(first.session_id, "inspect and correct the workspace")
    assert first.done.wait(timeout=2)

    summary = manager.summary(first.session_id)
    assert summary["status"] == "finished"
    assert second.engine._pending_constraints == []
    assert first.engine._pending_constraints == []
    assert first.engine.world.state().objects["connector_2"].zone == "A"
    validate_event_chain(manager.events_after(first.session_id))

    with pytest.raises(SessionError, match="constraints are closed"):
        manager.add_constraint(
            first.session_id,
            "keep_local",
            None,
            justification="too late",
        )


def test_session_rejects_missing_constraint_justification_before_start():
    manager = SessionManager(build_mock_engine)
    session = manager.create()

    with pytest.raises(ConstraintValidationError, match="requires justification"):
        manager.add_constraint(session.session_id, "keep_local", None, justification="")


def test_unknown_session_cannot_receive_a_constraint():
    manager = SessionManager(build_mock_engine)
    with pytest.raises(SessionError, match="unknown session"):
        manager.add_constraint("not-a-session", "keep_local", None, justification="operator request")
