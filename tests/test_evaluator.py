"""OQ-019 acceptance: the table-layout evaluator scores a finished setting.

Uses a mock place setting (the OQ-007 scene pack replaces the fixture later;
the evaluator's input shape is fixed). Units: metres, degrees.
"""

from __future__ import annotations

import json

from omni_q.evaluator import (
    LayoutReport,
    PlacedObject,
    PlacementSpec,
    evaluate_layout,
    load_layout_spec,
    yaw_error,
)

# A single place setting, scene frame = table centre at origin, arm faces -y.
# Stands in for the OQ-007 object pack's target layout.
SETTING_SPEC = load_layout_spec([
    {"object_id": "plate_1", "target_zone": "setting_1",
     "position": [0.0, -0.25, 0.02], "orientation_deg": 0.0},
    {"object_id": "cup_1", "target_zone": "setting_1",
     "position": [0.15, -0.18, 0.02]},
    {"object_id": "fork_1", "target_zone": "setting_1",
     "position": [-0.12, -0.22, 0.01], "orientation_deg": 90.0},
    {"object_id": "knife_1", "target_zone": "setting_1",
     "position": [0.12, -0.22, 0.01], "orientation_deg": -90.0},
    {"object_id": "napkin_1", "target_zone": "setting_1",
     "position": [-0.15, -0.30, 0.005]},
])


def placed_from_spec(specs=SETTING_SPEC, jitter_mm=0.0, jitter_deg=0.0) -> list[PlacedObject]:
    out = []
    for s in specs:
        jx = jitter_mm / 1000.0
        out.append(PlacedObject(
            object_id=s.object_id,
            cls=s.object_id.rsplit("_", 1)[0],
            position=tuple(p + jx for p in (s.position or (0.0, 0.0, 0.0))),
            orientation_deg=(s.orientation_deg or 0.0) + jitter_deg,
            zone=s.target_zone,
        ))
    return out


def test_perfect_setting_passes():
    report = evaluate_layout(placed_from_spec(), SETTING_SPEC)
    assert isinstance(report, LayoutReport)
    assert report.passed
    assert report.missing == ()
    assert report.n_failed == 0
    assert report.n_passed == len(SETTING_SPEC)
    assert report.max_pos_err_m is not None and report.max_pos_err_m < 1e-9


def test_positional_jitter_within_tolerance_passes():
    report = evaluate_layout(placed_from_spec(jitter_mm=5.0), SETTING_SPEC)
    assert report.passed
    assert report.max_pos_err_m <= 0.01


def test_positional_error_beyond_tolerance_fails():
    report = evaluate_layout(placed_from_spec(jitter_mm=20.0), SETTING_SPEC)
    assert not report.passed
    assert report.max_pos_err_m > 0.01
    bad = [v for v in report.objects if not v.passed]
    assert bad and all("position error" in r for v in bad for r in v.reasons)


def test_orientation_wraparound():
    # 2 deg off across the 0/360 boundary is 2 deg, not 358
    assert abs(yaw_error(358.0, 0.0)) == 2.0
    plates = [PlacedObject("plate_1", "plate", (0.0, -0.25, 0.02), orientation_deg=358.0,
                           zone="setting_1")]
    spec = [PlacementSpec("plate_1", target_zone="setting_1",
                          position=(0.0, -0.25, 0.02), orientation_deg=0.0)]
    assert evaluate_layout(plates, spec).passed
    plates = [PlacedObject("plate_1", "plate", (0.0, -0.25, 0.02), orientation_deg=30.0,
                           zone="setting_1")]
    report = evaluate_layout(plates, spec)
    assert not report.passed
    assert report.max_ori_err_deg == 30.0


def test_zone_mismatch_fails_even_at_perfect_pose():
    obj = [PlacedObject("plate_1", "plate", (0.0, -0.25, 0.02), 0.0, zone="tray")]
    report = evaluate_layout(obj, SETTING_SPEC[:1])
    assert not report.passed
    verdict = report.objects[0]
    assert verdict.zone_ok is False
    assert any("zone" in r for r in verdict.reasons)


def test_missing_object_fails_and_is_listed():
    setting = placed_from_spec()[:-1]  # napkin never made it
    report = evaluate_layout(setting, SETTING_SPEC)
    assert not report.passed
    assert report.missing == ("napkin_1",)


def test_extra_object_on_table_fails_by_default():
    setting = placed_from_spec() + [
        PlacedObject("screwdriver_9", "screwdriver", (0.3, -0.1, 0.01))]
    report = evaluate_layout(setting, SETTING_SPEC)
    assert not report.passed
    assert any("not part of the layout" in r
               for v in report.objects if v.object_id == "screwdriver_9"
               for r in v.reasons)
    # ... unless the caller allows extras
    assert evaluate_layout(setting, SETTING_SPEC, allow_extras=True).passed


def test_zone_only_spec_needs_no_pose():
    spec = [PlacementSpec("sleeve_1", target_zone="bin")]
    ok = evaluate_layout([PlacedObject("sleeve_1", "sleeve", (0.4, 0.1, 0.0), zone="bin")], spec)
    bad = evaluate_layout([PlacedObject("sleeve_1", "sleeve", (0.4, 0.1, 0.0), zone="A")], spec)
    assert ok.passed and not bad.passed


def test_metrics_aggregation():
    setting = placed_from_spec(jitter_mm=5.0)  # 5 mm/axis -> 8.7 mm euclidean, within tol
    report = evaluate_layout(setting, SETTING_SPEC)
    assert report.passed
    assert abs(report.mean_pos_err_m - report.max_pos_err_m) < 1e-9
    n = len(SETTING_SPEC)
    assert report.n_passed + report.n_failed == n


def test_report_is_json_serializable_for_receipts():
    report = evaluate_layout(placed_from_spec(jitter_mm=3.0), SETTING_SPEC)
    blob = json.dumps(report.as_dict(), sort_keys=True)
    assert json.loads(blob)["passed"] is True
