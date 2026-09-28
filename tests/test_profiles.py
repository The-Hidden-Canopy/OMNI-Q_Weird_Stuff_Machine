"""Executable TableOps profile contracts and report evidence."""

from __future__ import annotations

from dataclasses import replace

from omni_q.contracts import DataStatus
from omni_q.events import EventBus, validate_event_chain
from omni_q.profiles import ProfileWorld, build_profile_engine, load_profile


def test_formal_dinner_profile_compiles_the_desired_state_and_reports_symbolic_limits():
    profile = load_profile("formal_dinner_v3")

    assert profile.guest_count == 4
    assert profile.required_count == 24
    assert {item.cls for item in profile.place_setting.values()} == {
        "plate",
        "fork",
        "knife",
        "water_glass",
        "wine_glass",
        "napkin",
    }

    world = ProfileWorld(profile)
    initial = profile.diff(world.state())

    assert initial.required_count == 24
    assert initial.satisfied_count == 5
    assert len(initial.residuals) == 19
    assert initial.actionable_count == 19
    assert len(initial.metric_verification_gaps) == 3

    world._world._objects.pop("fork_1")
    missing = profile.diff(world.state())
    fork = next(residual for residual in missing.residuals if residual.object_id == "fork_1")
    assert fork.kind == "missing"
    assert fork.actionable is False

    stale_detection = replace(
        world._world._objects["plate_1"],
        status=DataStatus.STALE,
    )
    world._world._objects["plate_1"] = stale_detection
    stale = profile.diff(world.state())
    plate = next(residual for residual in stale.residuals if residual.object_id == "plate_1")
    assert plate.kind == "stale"
    assert plate.actionable is False


def test_profile_engine_reuses_governed_loop_and_seals_profile_report():
    mission = build_profile_engine("formal_dinner_v3", EventBus())

    receipt = mission.run("set the table")
    report = receipt.metrics["profile_report"]

    assert report["status"] == "COMPLETE"
    assert report["required_placements"] == 24
    assert report["first_pass_correct"] == 5
    assert report["self_corrected"] == 19
    assert report["unresolved"] == 0
    assert report["final_compliance"] == "24/24"
    assert report["verification_scope"] == "symbolic-zone"
    assert len(report["metric_verification_gaps"]) == 16
    assert "seat_1.plate: metric pose unavailable for 3mm tolerance" in report["metric_verification_gaps"]
    assert receipt.inputs["profile"]["profile"] == "formal_dinner_v3"
    assert receipt.inputs["profile_initial_diff"]["residual_count"] == 19
    validate_event_chain(mission.bus.log)
