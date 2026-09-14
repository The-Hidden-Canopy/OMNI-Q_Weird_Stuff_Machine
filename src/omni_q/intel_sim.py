"""Intel Online dual-SO-101 MuJoCo adapters.

This module uses the pinned SO-ARM100 Menagerie MJCF as the documented
six-joint mechanical proxy for SO-101.  It creates one real MuJoCo scene with
two independently actuated arms, a table-setting object pack (plate, cup,
fork, spoon, napkin -- distinct masses/friction/collision per OQ-007), and a
passive slide-jointed drawer holding the cutlery, matching the brief's
"open the top drawer, retrieve spoons and forks" scenario.

PICK/MOVE/PLACE on tracked tableware (OQ-010) use a real weighted
damped-least-squares differential IK controller (``IntelTableWorld._ik_reach``)
driving the arm's position servos toward the object/target -- real joint
motion via mj_step, no teleport, no velocity override -- and a genuine
contact-driven grasp attempt (close the gripper and see what actually
happens), with success/failure grounded in the measured outcome (lift height
on PICK, final position error on PLACE) rather than an assumed label: a
failed grasp/placement reverts the WorldState change and reports the step as
failed, so the engine's normal replan loop actually retries instead of the
receipt silently claiming success.

**Current fidelity, measured, not assumed:** two IK paths exist. The
safe-transit motion (``_ik_reach``/``_move_to``/``_ik_track_line``) is
5-joint, weighted to discourage the proximal joints -- 5 joints solving a
3D position task is redundant, and the unweighted minimum-norm solution was
measured swinging the shoulder/forearm through tableware even for a small
vertical lift where that swing wasn't geometrically necessary. The final
pinch (``_ik_reach_pad``) instead pins wrist-roll to a fixed value and
solves only 4 joints, tracking the fingertip pad geom rather than the
gripper body origin -- ported from ``_ContactHandoffController``, which
proved this pattern 10/10 for one hand-tuned cup sequence elsewhere in this
file. Applied generally here, it measurably tightens position accuracy
(reach error dropped from ~0.04-0.05m to as low as ~0.01m for some
targets) but **does not reliably produce a held grasp**.

**Update, same investigation, one level deeper:** the "one roll scalar
doesn't generalize" finding above turned out to be two separate problems,
not one. (1) Tested whether the best roll per object correlates with the
arm's own shoulder bearing to the target (a real kinematic-compensation
hypothesis, not another magic number) -- linear fit against measured data
across all 5 tracked objects came back with large residuals (up to ~1.9
rad), i.e. **falsified**; roll alone doesn't explain the variance. (2) A
much bigger, independent bug: with wrist-roll fully freed (5-DOF, best
case), `fork_1`/`spoon_1`'s pad-tracking IK still didn't converge (~0.20m
error, vs 0.01m tol) *regardless of roll* -- because their scene position,
~0.64m from each arm's base, is physically outside the arm's own
independently measured reach envelope (~0.386m radius, see
`so101_capability_map.md`, OQ-003 -- cross-validated by this file's own IK
convergence sweep separately finding the same ~0.39-0.40m boundary). That
was a real scene-authoring bug (an unreachable target position), not a
physics-realism question, and has been fixed by repositioning them (see
the comment at their `tableware_pose(...)` call below) -- legitimate under
the no-simulation-cheating rule the same way any factually-wrong-parameter
correction is. Repositioning alone raised their reach convergence from
~0.20m to ~0.05m error but **still doesn't produce a held grasp** (lift
~0, same as the already-reachable objects) -- meaning even for objects
that were always in reach (`cup_1`, `plate_1`), the deeper, still-open
problem is that this controller's own position-tracking ceiling (~0.03-
0.05m even at best) simply isn't tight enough for a reliable pinch, not
orientation or reachability specifically. The honest next step is real
6-DOF pose-aware IK (position + full orientation, solved jointly, with
tighter damping/convergence tuning) or the contact-handoff scene's own
combined physics tuning -- which stays a deliberately separate, bounded
evidence track (see `integrations/intel/README.md`'s "Contact-handoff
evidence boundary"), not something to fold into this general path without
the same scrutiny. See `integrations/intel/README.md` for the fuller
writeup and `docs/oq-021-red-team-findings.md` for how this compounds with
downstream testability.

**Fourth update: real orientation-aware IK landed (a teammate's independent
work, merged same day), a real held grasp exists now, and a real bug was
found and fixed in the same merge.** ``_grasp_frame``/``_ik_reach_pad_pose``
solve position AND orientation jointly (``mj_jacGeom``'s rotational
Jacobian, not just position), with a bounded roll-candidate retry verified
against actual close+lift outcome, not just position error -- exactly the
"real 6-DOF pose-aware IK" the previous update called for. ``cup_1``
specifically was also resized to fit the gripper's actually-measured
envelope (a prior 64mm cup left no real margin against the pad gap) and its
friction/contact softness tuned to real ceramic/rubber-pad values. Net
result, honestly measured: ``world._do_pick(6, "cup_1")`` now returns a
genuine held grasp -- the first reliable success this investigation has
produced, achieved through real control improvement + a real geometry
correction, not through weakened physics.

**The bug**: the same merged commit also added ``contype``/``conaffinity``
overrides setting every non-fingertip-pad arm geom to ``0``/``0``. Under
MuJoCo's collision rule (contact requires ``(contype1 & conaffinity2) or
(contype2 & conaffinity1)`` nonzero), a geom with both at zero can never
collide with *anything* -- the entire arm mesh except the two pads could
pass through the table, the drawer, and every object with zero contact
resistance. That's not a control improvement, it's the simulation no
longer simulating the arm's own body, which is exactly the category of
change the no-simulation-cheating rule rules out. Found, flagged, and
reverted (see the comment at the per-arm geom loop below) before this
lands anywhere near a demo or a rubric claim -- the fix was verified not to
regress the new orientation-aware grasp: ``cup_1`` still holds with real
collision fully restored, because the underlying orientation-aware control
+ correctly-sized geometry was doing the real work, not the no-clip
exemption. ``plate_1``/``fork_1``/``spoon_1``/``napkin_1`` still don't
hold. Removing the exemption also cost real wall-clock time (more contact
resolution, more retry attempts on the objects that still fail) -- full
suite runtime moved from ~95s to ~6 minutes; an honest, accepted cost, not
something to optimize away by re-introducing the exemption.

Two ``cup_1``-specific tests (``test_failed_grasp_reverts_worldstate_
instead_of_claiming_success``, ``test_failed_grasp_restores_mujoco_state_
for_a_clean_retry``) now use an adversarial fixture (see
``_disable_gripper_collision`` in ``tests/test_intel_sim_primitives.py``)
that disables one arm's whole gripper body, forcing a real failure on
demand regardless of which object currently succeeds -- more robust than
switching to a specific object, since it stays correct even as more
objects start holding.

**The win is bigger than the aggregate 10-seed number shows.** Re-ran
``run_intel_table_evaluation_report`` after both fixes
(``evidence/benchmark_results/intel_table_eval_2026-09-10-v6/``): still
10/10 ``grasp_failure`` in the summary table, but reading a trial's raw
receipt directly shows ``pick_cup_1`` *and* ``move_cup_1`` both succeeding
-- a full, real pick-and-place, not a near-miss -- before the plan reaches
``fork_1``, which still doesn't converge and exhausts its retries. The
per-trial outcome field is a single first-blocking-failure label, so a
trial that completes one whole object's pick-and-place looks identical in
the summary to one that never succeeds at anything. Read the receipts, not
just the summary, when auditing partial progress -- that's exactly how
this was found.

Not addressed this pass: the *separate*, pre-existing, already-documented
``_ContactHandoffController`` scene (see "Contact-handoff evidence
boundary" below) has used a similar contype/conaffinity scheme since
before this session -- it predates this merge and stays a deliberately
bounded, separate evidence track, not folded into the main route either
way. Worth the team's attention on its own terms, just out of scope here.

**Fifth update: real orientation-aware IK landed (independent, further
work by the team) -- confirmed the local-minimum escape technique from
the "Third update" above still applies, but the payoff is too fragile to
wire in.** All four still-failing objects (``plate_1``/``napkin_1``/
``fork_1``/``spoon_1``) plateau at the same ~0.05-0.06m "pinch" position
error under the new ``_grasp_frame``/``_ik_reach_pad_pose`` solver --
the same signature as the pre-orientation-aware local-minimum trap.
Confirmed it's the same failure mode: perturbing the pre-descent Elbow/
Wrist_Pitch seed (same technique, same injection point -- after
``_move_to``'s transit, before the precision descent) also escapes it
here. A dense 81-point seed grid found a genuine held grasp for
``plate_1`` (0.0211m lift, just over the 0.02m threshold). But: (1) the
sweet spot is narrow -- a bounded, cheap 17-seed grid (the same list that
reliably found ``plate_1``'s hold under the old solver) only reached
0.0169m, short of the threshold; the winning point needed the dense
grid's finer resolution. (2) 0.0211m is barely over the line, not a
comfortable margin -- likely fragile to the same kind of small
perturbation (scene jitter, a different seed's RNG draw) that would be
present in the 10-seed randomized harness. (3) A dense-enough search to
find it reliably is expensive per attempt, multiplied across every
object on every retry. Given the cost (denser search, more wall-clock
time on top of the current ~6-minute suite) against a marginal, likely
non-robust payoff, **not wired into** ``_do_pick`` **this pass** -- a
judgment call, not a dead end: the underlying diagnosis (differential IK
local minima, not a hard reach/orientation limit) still stands and still
generalizes across solver versions, which is useful for whoever picks
this up next. The honest way to actually close this gap remains what the
"Third update" already said: real, non-gradient-trapped convergence
(multi-resolution/coarse-to-fine search, or a smarter initial guess than
a fixed HOME-derived transit configuration), not incrementally denser
random seed grids.

**One "coarse-to-fine" variant tested and ruled out**: an annealed damping
schedule for the final pinch descent (start with high damping -- a
smoother, more forgiving basin of attraction, standard Levenberg-Marquardt
practice for avoiding sharp local traps -- decaying to low damping for
tight final precision), in place of the fixed damping value
``_ik_reach_pad`` uses throughout. Tested 4 start/end damping pairs across
all 4 failing objects: **zero improvement in any case** -- same ~0.05m
plateau regardless of schedule. This rules out "wrong descent dynamics" as
the cause and reinforces the earlier finding (freeing wrist-roll from a
stuck state barely helped either): the trap is a genuinely different
attractor basin in *joint configuration space*, not a step-size or
damping-tuning problem -- only a different starting configuration (a real
seed, not a smoother path to the same one) escapes it. Coarse-to-fine
still means something different and untried: a genuine multi-resolution
*spatial* search (e.g. sampling several candidate transit configurations
via forward kinematics before committing to one, the way a real motion
planner would), not a smoother numerical schedule on top of the same
single starting point.

**Sixth update: a real, separate bug fix -- the revision budget was being
spent entirely on whichever object failed first, never reaching the
others.** ``IntelTablePlanner`` always regenerated every misplaced object
into each replanned graph in the same ``_OBJECT_ORDER`` priority, but
``engine.py`` only ever executes the single first-ready step before any
failure triggers another full replan -- so the first still-failing object
after ``cup_1`` (usually ``napkin_1``) consumed the entire
``max_revisions`` budget by itself, measured failing 6 times in a row
while ``plate_1``/``fork_1``/``spoon_1`` never got a single real attempt.
Fixed: ``IntelTablePlanner`` now tracks how many times each object has
actually been attempted (via a ``replan`` override identifying the one
object that was genuinely tried, not every object merely present in the
graph -- an earlier draft of this fix counted every present object
identically each time, which meant they all crossed the deprioritize
threshold in lockstep and the relative order never actually changed);
once an object exceeds ``_MAX_ATTEMPTS_BEFORE_DEPRIORITIZE``, it's pushed
after objects with fewer attempts, not dropped. **This does not change
whether a run resolves** -- every object still has to actually succeed
for that, and the 10-seed harness's aggregate outcome label is unchanged
(``evidence/benchmark_results/intel_table_eval_2026-09-11-v7/``, still
10/10 ``grasp_failure`` -- expected and checked, since that classifier
scans the whole receipt for any failure). What it does change: the same
budget now produces a genuinely richer, more representative attempt
record per run, and a live demo visibly tries different objects instead
of appearing to get stuck repeating the identical failed motion.

**Seventh update: surfaced the real per-object signal the coarse outcome
label was hiding, instead of leaving it as a "read the receipts by hand"
finding.** ``run_intel_table_evaluation_report`` now also computes
``_per_object_pick_place_outcomes`` per trial and aggregates a
``per_object_summary`` (``schema_version`` 2) -- how many of the 10
trials each object ever achieved a held grasp / a placed result, plus
the same breakdown on every individual receipt entry. Real result,
honestly measured (``evidence/benchmark_results/intel_table_eval_2026-
09-11-v8/``): ``cup_1`` held in **10/10** trials -- genuinely robust
across the harness's own ±3mm/±0.08rad randomized scene jitter, not a
one-off -- and placed in 9/10 (one trial's carry-and-release didn't
settle within tolerance, a real, minor, not-yet-investigated gap).
``plate_1`` held in 1/10, consistent with the "Fifth update" finding of
a narrow, fragile margin -- present, just not reliable. The coarse
``outcomes`` table itself is unchanged (still 10/10 ``grasp_failure``,
expected, unrelated to this change) -- ``_classify_intel_table_receipt``
scans the whole receipt for the first failure anywhere in it, so this
was never going to move it; what changed is that the real per-object
story is now in the report's own JSON, not something that requires
reading raw receipts to find.

**Eighth update: the "Fifth update" seed-bias escape is now wired into
``_do_pick`` for ``plate_1`` specifically, after re-checking the exact
worry that update raised.** That update found the escape but declined to
land it, reasoning that (a) landing it generally would mean a dense
81-point per-attempt search, too expensive across every object/retry, and
(b) the one measured win (0.0211m lift, just over the 0.02m threshold)
looked fragile enough to not survive the harness's own randomized scene
jitter. Both concerns dissolve for the form actually landed:
``OBJECT_GRASP_SEED_BIAS`` is a fixed, zero-search per-object constant
(currently just ``{"plate_1": (0.0, 0.2)}`` for elbow/wrist_pitch), not a
runtime grid search, so objection (a) doesn't apply. Objection (b) was
checked rather than assumed: run across 10 of the harness's own
``IntelSceneConfig(randomized=True)`` seeds (±3mm position / ±0.08rad yaw
jitter, the same bounds ``run_intel_table_evaluation_report`` uses),
``plate_1`` now holds **10/10**, not the 1/10 the "Seventh update"
measured under the old, unbiased attempt. ``cup_1`` and the still-failing
``fork_1``/``spoon_1``/``napkin_1`` are unaffected (verified directly,
not assumed) since the bias only applies when the object key is present
in the dict. Still not extended to those three: none showed a comparable
per-object win under any of the four escape mechanisms tried this
session (seed perturbation, annealed damping, approach-bearing variation,
full free-DOF search) -- this fix closes exactly the gap it was measured
against, nothing more.

**Ninth update: the Intel challenge hosts clarified what "10 seeds"
actually means, and every prior randomized bundle (v2 through v8) only
satisfied a narrow reading of it.** Their own words: 10 seeds can mean 10
different non-trivial variations of the environment -- lighting, object
location, input-prompt phrasing, object color/texture, etc. -- and which
axes/how many is left to the entrant, as long as it demonstrates real
robustness rather than repeating one trivial RNG draw. Every prior bundle
varied exactly one axis (``IntelSceneConfig``'s ±3mm/±0.08rad
position/yaw jitter) -- a real axis, but alone, risked reading as
trivial. ``IntelSceneConfig`` gained three more, all checked to be
physics-inert (rgba/light only, never touching contype/conaffinity/
friction/mass/solref/solimp): ``color_jitter`` (tableware rgba),
``light_diffuse_jitter``/``light_angle_jitter_rad`` (key-light intensity
and incidence angle). ``run_intel_table_evaluation_report`` gained a
fourth: each trial now uses a different real instruction phrasing
(``TABLE_SETTING_PHRASINGS``), not the same literal string 10 times --
each phrasing was checked against ``RulePlanner``'s keyword gate and a
live run producing an identical op sequence *before* being added, so this
axis can't silently degrade to the "unrecognised goal; observe only"
fallback and corrupt the evidence.

Real result, honestly measured across all four combined axes
(``evidence/benchmark_results/intel_table_eval_2026-09-12-v9/``):
``cup_1`` and ``plate_1`` both still hold **10/10** -- the seed-bias fix
from the "Eighth update" was only checked against position/yaw jitter
before; this is the first time it was checked against lighting/color
variation too, and it held. A new, real, previously-too-rare-to-see
finding surfaced by having ``plate_1`` succeed in every trial instead of
1/10: it holds 10/10 but is **placed in 0/10** -- every ``MOVE`` rejected
on ``"unsafe carry separation"`` (``relative_object_pad_distance_m`` ~
0.121m against ``LEGACY_MAX_CARRY_OFFSET_M = 0.100``). Root cause: the
carry-safety check measures pad-to-object-center distance against one
global bound, but ``plate_1`` is deliberately grasped at its **rim**
(``OBJECT_GRASP_OFFSET["plate_1"] = 0.078``, a ~95mm-radius plate can't
be pinched through its own middle), so a legitimately safe rim grasp is
inherently going to exceed a bound tuned around ``cup_1``'s centered
grasp. Not fixed this pass -- a safety-relevant threshold is a distinct
question from the local-minimum escape this update was built to verify,
and deserves its own focused look, not a same-commit patch. See
``evidence/benchmark_results/intel_table_eval_2026-09-12-v9/README.md``
for the full writeup and ``BACKLOG.md`` (OQ-010) for the follow-up flag.

This is still a proxy, not hardware evidence -- no vision-guided grasp point,
no force control. The separate OQ-010/OQ-011 contact adapter uses only MuJoCo
contact dynamics for a bounded ``cup_1`` handoff. It is a SO-ARM100
mechanical-proxy simulation, not evidence of perception, VLA control,
hardware, complete table setting, or concurrent execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .contracts import (
    Detection,
    DeviceSpec,
    ManipResult,
    MissionEnvelope,
    PlanDecision,
    PlanGraph,
    Pose,
    Step,
    TransitionRejected,
    TransitionRequest,
    TransitionResult,
    VerifyResult,
    WorldState,
    content_hash_of,
)
from .devices import DeviceRouter
from .engine import OmniQ
from .expressive import validate_expressive_command
from .fakes import FakeManipulator, FakeObserver, FakeRecorder, FakeVerifier, RulePlanner
from .world import MockWorld


ROOT = Path(__file__).resolve().parents[2]
ARM_XML = ROOT / "integrations" / "intel" / "assets" / "menagerie_so_arm100" / "so_arm100.xml"
ARM_ASSETS = ARM_XML.parent / "assets"
HOME = (0.0, -1.57, 1.57, 1.57, -1.57, 0.0)

# OQ-010 primitive targets. GRIPPER_OPEN/CLOSED drive the real IK grasp
# (IntelTableWorld._do_pick/_do_place); the rest are coarse fixed poses for
# ops without dedicated IK behaviour (OPEN/CLOSE/ROTATE/PRESENT/HANDOFF, and
# any PICK/MOVE/PLACE on an object this adapter doesn't track).
GRIPPER_OPEN = 1.5     # Jaw joint, rad -- near the SO-101 open end of its range
# The Jaw joint's physical stop is -0.174 rad. Commanding 0.0 left the gripper
# HALF CLOSED: the fingertip pads still 15.8 mm apart, so every object thinner
# than that was ungrippable by construction, whatever the IK did. That is the
# whole fork/spoon/napkin failure -- measured 2026-09-13:
#
#   cup_1    110 mm  > 15.8 -> held 10/10
#   plate_1   32 mm  > 15.8 -> held  9/10
#   fork_1     8 mm  < 15.8 -> held     0
#   spoon_1    8 mm  < 15.8 -> held     0
#   napkin_1   6 mm  < 15.8 -> held     0
#
# Closing to the real stop gives a 3.6 mm fingertip gap (the jaws close as a
# wedge; see so101_capability_map.md's corrected entry). Nothing is relaxed
# here -- this commands the joint its own MJCF range already allows.
GRIPPER_CLOSED = -0.174
LEGACY_MODELS_FLAG = os.environ.get("OMNIQ_LEGACY_MODELS", "0") not in {"", "0", "false", "no"}
# Real pull-out drawer trays on the flanks, opened by a pinch on the handle
# (2026-09-14). Opt-in: the pull itself is 6/6 but the cutlery picks out of
# the trays are still ~50%, and the default layout is the one measured at
# 47/50 and 50/50 placements. See docs/contact-honesty-2026-09-13.md.
REAL_DRAWERS = (not LEGACY_MODELS_FLAG) and os.environ.get("OMNIQ_REAL_DRAWERS", "0") not in {"", "0", "false", "no"}
DRAWER_OPEN = 0.12 if LEGACY_MODELS_FLAG else 0.05   # drawer_slide qpos, m -- matches its MJCF range max
DRAWER_CLOSED = 0.0
TRANSIT_HEIGHT = 0.28  # m -- above the table/drawer/tableware envelope, within reach (see _move_to)

# OQ-HAND-011 idle-slack flourishes -> a single joint swung off HOME and back
# (radians). MIRROR / FREEZE are handled separately in apply_transition. Joint
# order per arm: 0 base-yaw, 1 shoulder, 2 elbow, 3 wrist-pitch, 4 wrist-roll.
_FLOURISH_GESTURES: dict[str, tuple[int, float]] = {
    "SWAY": (0, 0.35), "WAVE": (4, 0.9), "SPIN_WRISTS": (4, 1.5),
    "BOUNCE": (2, -0.30), "BOW": (1, 0.40), "CROSS": (0, 0.5),
    "HIGH_FIVE": (3, -0.6), "CALL_AND_RESPONSE": (0, -0.4),
}


@dataclass(frozen=True)
class IntelSceneConfig:
    """Build-time scene perturbations for the legacy table-setting route.

    The randomized evaluation changes initial tableware pose, tableware
    color/texture (rgba only -- mass/friction/solref/solimp are untouched,
    so this never weakens contact physics), and the key light's intensity
    and angle in the generated MJCF. It never writes a free-joint pose
    during a transition, so a report still distinguishes scene
    initialization from scripted object ownership or placement.

    Per the Intel challenge hosts' own clarification on the "10 seeds"
    requirement (they explicitly list lighting, object location, prompt
    variation, and color/texture as acceptable non-trivial axes -- picking
    which and how many is left to the entrant): position/yaw jitter alone
    was too narrow a reading of that bar, since ±3mm/±0.08rad on one axis
    risks looking trivial even repeated 10 times. Color and lighting are
    added here as physics-inert, genuinely different-looking scene
    variations; instruction-phrasing variation (the fourth axis the hosts
    named) is handled separately, in run_intel_table_evaluation_report,
    since it isn't a build-time scene property.
    """

    seed: int = 0
    randomized: bool = False
    position_jitter_m: float = 0.003
    yaw_jitter_rad: float = 0.08
    color_jitter: float = 0.12
    light_diffuse_jitter: float = 0.30
    light_angle_jitter_rad: float = 0.35

    def __post_init__(self) -> None:
        if not math.isfinite(self.position_jitter_m) or not 0.0 <= self.position_jitter_m <= 0.010:
            raise ValueError("position_jitter_m must be between 0 and 0.010 m")
        if not math.isfinite(self.yaw_jitter_rad) or not 0.0 <= self.yaw_jitter_rad <= 0.25:
            raise ValueError("yaw_jitter_rad must be between 0 and 0.25 rad")
        if not math.isfinite(self.color_jitter) or not 0.0 <= self.color_jitter <= 0.40:
            raise ValueError("color_jitter must be between 0 and 0.40")
        if not math.isfinite(self.light_diffuse_jitter) or not 0.0 <= self.light_diffuse_jitter <= 0.80:
            raise ValueError("light_diffuse_jitter must be between 0 and 0.80")
        if not math.isfinite(self.light_angle_jitter_rad) or not 0.0 <= self.light_angle_jitter_rad <= 0.80:
            raise ValueError("light_angle_jitter_rad must be between 0 and 0.80 rad")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

# Half-height (m) of each tableware geom in dual_so101_xml() -- used to place
# the IK grasp/place target just above the object's actual TOP surface, not
# its centre. Targeting centre + a small fixed offset put the gripper target
# *inside* tall objects (the cup's half-height alone is 0.055m, more than
# the old fixed 0.012m clearance), which doesn't converge -- the position
# servo just pushes into the object's own volume, dragging it around instead
# of approaching it cleanly. Not derived from the model at runtime because
# _do_pick/_do_place need it before the arm ever gets there.
OBJECT_HALF_HEIGHT: dict[str, float] = (
    {"plate_1": 0.016, "cup_1": 0.055, "fork_1": 0.0035, "spoon_1": 0.0035, "napkin_1": 0.009}
    if LEGACY_MODELS_FLAG else
    # realistic models: origin -> top. plate: lip top; cup: rim; cutlery: grip segment top
    {"plate_1": 0.0275, "cup_1": 0.045, "fork_1": 0.0035, "spoon_1": 0.0035, "napkin_1": 0.009}
)
# Horizontal offset from an object's centre to the fixed pad target.  Large
# flat fixtures are grasped at their rim/edge; targeting their centre puts the
# pad inside the collision volume and drags them during the lift.
OBJECT_GRASP_OFFSET: dict[str, float] = {
    # cup_1: its RADIUS, so the fixed jaw lands on the cup's near wall. The
    # SO-101 has one fixed jaw and one moving jaw; with the fixed pad aimed at
    # the cup's centre (0.0) the moving jaw closed by shoving the cup sideways
    # across the table until it met the fixed jaw, and a 110 mm cup pushed at
    # its base tips over. Seen live by the operator on 2026-09-13 ("only one
    # claw is moving, the other stayed out"). Land the fixed jaw first; then
    # the moving jaw closes onto an object that cannot travel.
    "plate_1": 0.078, "cup_1": 0.022 if LEGACY_MODELS_FLAG else 0.030, "fork_1": 0.006, "spoon_1": 0.0,
    # The cloth's broad, thin footprint is more stable under a centred pinch;
    # the earlier rim offset let the moving jaw skim past it during lift.
    "napkin_1": 0.0,
}
# Small vertical calibration for thin proxies whose pad centre is not exactly
# coincident with the object's geometric centre.  This remains a target
# offset; the object freejoint is never written.
# Offset along the object's own +y axis (toward the head) from its origin to
# the grip point. The bent cutlery's origin is the middle of the handle, but
# the bowl/tines put the centre of mass 17 mm toward the head; pinched at the
# geometric middle the piece hangs bowl-down and pivots in a two-pad grip
# (measured 2026-09-14: spoon rolled 61 deg about its handle after the lift,
# then slipped out in the first 0.2 s of the carry). Grip at the CoM.
OBJECT_GRASP_ALONG: dict[str, float] = (
    {} if LEGACY_MODELS_FLAG else {"fork_1": 0.017, "spoon_1": 0.017}
)
# Objects picked by a SIDEWAYS pinch on a rim: the gripper comes in
# horizontally from outside, fixed jaw above the lip, moving jaw below.
# (lip_radius_m, lip_centre_z_from_origin_m, lip_half_thickness_m)
EDGE_PINCH: dict[str, tuple[float, float, float]] = (
    {} if LEGACY_MODELS_FLAG else {"plate_1": (0.0925, 0.025, 0.0025)}
)
# Jaw opening for the horizontal approach. Fully open (77 mm) would put the
# moving jaw ~50 mm below the table when the fixed jaw is on a lip 25 mm up;
# ~26 mm of gap clears a 5 mm lip with the moving jaw tip still above the table.
# Measured pad-to-pad at the tips: q=0 -> 17.8 mm, q=0.15 -> 29 mm. At 0.05
# the moving jaw's tip sits ~20 mm below the fixed pad: under a lip whose
# underside is 23 mm up, and still clear of the table.
EDGE_PINCH_APPROACH_JAW_RAD = 0.05
# Objects that need BOTH arms: measured 2026-09-14 (evidence/benchmark_results/
# realistic_models_2026-09-14/plate_single_arm_edge_pinch_tilts.png), one arm
# pinching the rim of the 200 mm plate lifts its near edge 21 mm and the
# plate hangs at 19 deg with the far rim on the table -- 0.2 N.m of plate
# torque against ~0.1 N.m of pinch. Two arms on opposite rims lift it flat.
BIMANUAL_OBJECTS: frozenset[str] = frozenset() if LEGACY_MODELS_FLAG else frozenset({"plate_1"})
BIMANUAL_MAX_TILT_RAD = 0.21  # 12 deg: a plate carried flat, not dragged
# Top-down picks that start from a scanned vertical-finger posture instead
# of HOME (see IntelTableWorld._topdown_seed_joints). Opt-in per object: the
# cutlery and napkin picks are 10/10 from HOME and are left alone.
TOPDOWN_SEED_OBJECTS: frozenset[str] = (frozenset() if LEGACY_MODELS_FLAG else
                                        (frozenset({"cup_1", "fork_1", "spoon_1"}) if REAL_DRAWERS else frozenset({"cup_1"})))
# Objects whose jaw is re-pinned at its stall angle (+ a bounded squeeze)
# instead of left to ramp toward fully closed. Measured 2026-09-14 while the
# arms started working in parallel: a held cup with the jaw still driving
# toward -0.174 creeps 0.69 -> 0.34 -> -0.17 and slides out of the pinch
# within ~10 s of holding -- sequential runs only "worked" because the MOVE
# followed the PICK immediately. The plate is NOT in this set (see
# _gripper_stall_hold).
STALL_HOLD_OBJECTS: frozenset[str] = frozenset({"cup_1"})
# Wall pinch for hollow objects (2026-09-14): the jaw descends only part-open
# so the moving fingertip lands INSIDE the rim, the fixed pad outside, and the
# close clamps the 3 mm wall between them -- a hold that does not depend on
# the finger being vertical. The diameter pinch it replaces held the cup on a
# tilted 45 deg contact and let it slide 35-45 mm down the pads while held.
# jaw angle for the descent, and how far below the rim the fingertips go.
WALL_PINCH_OBJECTS: dict[str, tuple[float, float]] = ({} if LEGACY_MODELS_FLAG else {"cup_1": (0.30, 0.030)})
OBJECT_GRASP_VERTICAL_OFFSET: dict[str, float] = {
    "spoon_1": 0.0,
    # cup: fingertips 30 mm below the rim (rim = centre + 45 mm)
    **({} if LEGACY_MODELS_FLAG else {"cup_1": 0.045 - 0.030}),
}
# Empirically found, not a general principle: this exact (elbow, wrist_pitch)
# joint bias, applied to qpos right after the transit approach and before the
# precision descent, escapes a confirmed differential-IK local minimum for
# plate_1's specific scene position. Verified deterministic and reproducible
# (three consecutive runs, identical result each time): err=0.0492,
# lift=0.0211, held=True. Four independent escape mechanisms were tried this
# session (this seed perturbation, an annealed damping schedule, approach-
# bearing variation, and a full free-DOF solve with a 4000-iteration budget)
# -- only seed perturbation ever escapes this class of trap, and only for the
# specific target found this way. Do not extend to other objects without
# equally rigorous per-object verification: fork_1/spoon_1/napkin_1 showed no
# comparable win under any of the four mechanisms this session.
OBJECT_GRASP_SEED_BIAS: dict[str, tuple[float, float]] = {
    "plate_1": (0.0, 0.2),
}
# Fingertip pad contact: sliding, torsional, rolling friction.
#
# Default is exactly the value that has been running, so this constant is a
# no-op rename until someone overrides it. It exists because the middle term is
# a live hypothesis: the SO-101 RL project that solved a cube grasp in the same
# simulator used torsional 0.05 (`friction="1 0.05 0.001"`), 2.5x ours, and
# torsional friction is precisely what stops a pinched object rotating out of
# the grip -- the failure mode of the thin, flat objects still unsolved here
# (fork_1/spoon_1/napkin_1). See
# docs/oq-010-external-sources-crosscheck-2026-09-13.md.
#
# Overridable by env var so a probe never has to edit physics in the working
# tree: OMNIQ_PAD_FRICTION="3.00 0.050 0.001"
PAD_FRICTION = os.environ.get("OMNIQ_PAD_FRICTION", "3.00 0.020 0.001")
# Pad contact compliance. The scene shipped with solref ".050 1" /
# solimp ".80 .95 .010" on the pads and the cup -- a "soft rubber" contact
# whose impedance only reaches 95% after 10 mm (the solimp width) of
# penetration. Measured over one seed-900 trial (2026-09-13): pads sank up
# to 22 mm into the plate, 16 mm into the cup, and the cup was pushed 14 mm
# into the table. A fork is 7 mm thick; a pad 10 mm deep is not gripping it,
# it is inside it. Real pad rubber compresses about a millimetre, so the
# default now saturates within ~1.5 mm (still compliant, no longer hollow).
# Overridable for probes: OMNIQ_PAD_SOLREF="0.05 1" OMNIQ_PAD_SOLIMP="0.8 0.95 0.01"
#
# Second finding, same day: MuJoCo's default (time-constant) solref scales
# contact stiffness with the mass of the body in contact, so a 0.18 kg plate
# at the default 20 ms is ~450 N/m -- the arm pressing on it sank it 24 mm
# into the table whatever the pads were set to. Direct stiffness
# (negative solref = "-k -b", N/m and N.s/m) is mass-independent: measured
# in isolation, 80 N of push penetrates 3.3 mm at -50000 -200 instead of
# 24 mm. Tableware, table and drawer get priority="1" so their parameters
# govern contact with the arm's default-parameter links (no averaging).
# Pads are the softer partner (rubber); pad<->object contacts average the
# two, both in direct mode.
#
# Status 2026-09-13 (end of day), measured on seeds 900-902 with the
# 32 mm-puck plate and solid cup still in the scene:
#   soft (shipped)         placed 9/15   pad->cup 16.5 mm, plate->table 22.5 mm
#   firm time-constant     placed 10/15  pad->cup  7.7 mm, plate->table 23.8 mm
#   rigid direct-stiffness placed  8/15  pad->cup  3.8 mm, but the plate rim
#       jams in the V between fixed pad and jaw body and the wedge pops it
#       off the table at 2.9 m/s (1.2-4.7 kN contact), and cup_1 goes 3 -> 0
#       because its grasp was resting on the penetration.
# So the DEFAULT is the firm time-constant contact (a measured improvement,
# no regression), and rigid contact is opt-in (OMNIQ_RIGID_CONTACT=1) as the
# physics the realistic models -- thin-lipped plate, hollow cup, bent-handle
# cutlery -- have to be built and tuned against. See
# docs/contact-honesty-2026-09-13.md.
RIGID = os.environ.get("OMNIQ_RIGID_CONTACT", "0") not in {"", "0", "false", "no"}
PAD_SOLREF = os.environ.get("OMNIQ_PAD_SOLREF", "-30000 -150" if RIGID else "0.02 1")
PAD_SOLIMP = os.environ.get("OMNIQ_PAD_SOLIMP", "0.90 0.95 0.001" if RIGID else "0.90 0.99 0.0015")
RIGID_CONTACT = ({"solref": os.environ.get("OMNIQ_RIGID_SOLREF", "-50000 -200"),
                  "solimp": "0.90 0.95 0.001", "priority": "1"} if RIGID else {})
NAPKIN_CONTACT = {**RIGID_CONTACT, "solref": "-8000 -60"} if RIGID else {}
# The isolated contact-handoff scene was tuned and gated (10/10) against the
# original compliant pads; it keeps them until it is re-tuned against the
# firm default (its deterministic gate failed on first try with them).
HANDOFF_PAD_SOLREF = os.environ.get("OMNIQ_HANDOFF_PAD_SOLREF", ".050 1")
HANDOFF_PAD_SOLIMP = os.environ.get("OMNIQ_HANDOFF_PAD_SOLIMP", ".80 .95 .010")

GRASP_CLEARANCE = 0.015  # m -- gap kept above an object's top surface before closing on it
# Descent stops when a jaw geom presses the target harder than this (0 =
# off). Opt-in: at 10 N it did not prevent the plate-rim wedge jam (the rim
# is already in the notch by the time the reading crosses the threshold)
# and it aborted good descents that brush the object -- seeds 900-902 went
# 10/15 -> 7/15 placed with it on. Kept as the hook for the edge-pinch
# plate grasp, where a force stop is the right primitive.
DESCEND_CONTACT_STOP_N = float(os.environ.get("OMNIQ_DESCEND_CONTACT_STOP_N", "0"))
# How far a grasped object is raised to prove it is held. This used to be
# clear_z - start_z = half_height + GRASP_CLEARANCE, which SHRINKS with the
# object: 70 mm for the cup, 31 mm for the plate, and 18.5 mm for a 7 mm fork
# -- below the 20 mm "held" threshold. A perfectly grasped fork could never
# register as held because the arm was never asked to lift it far enough.
# Measured 2026-09-13: fork lifting 12.8 mm of a possible 18.5 and counted as
# a failure. Verification height is now a floor independent of object size.
LIFT_VERIFY_MIN = 0.060  # m
# Legacy-path safety bounds.  These are controller stop bounds, not hardware
# force limits: the contact-handoff path owns the promotion-grade force gate.
LEGACY_MAX_CARRY_OFFSET_M = 0.100
LEGACY_MIN_JOINT_MARGIN_RAD = 0.020
LEGACY_MAX_CONTACT_FORCE_N = 250.0
LEGACY_MIN_SETTLE_SPEED_MPS = 0.080
LEGACY_MAX_SETTLE_DRIFT_M = 0.012

# Real MuJoCo (x, y, z) table-setting target per zone name, keyed by the same
# strings IntelTableWorld/IntelTablePlanner use as Detection target_zones.
# IK targets for _do_place -- unrelated to scheduler.DEFAULT_LAYOUT, which is
# an abstract reachability space, not a physical coordinate frame (see that
# dict's comment).
ZONE_POSITIONS: dict[str, tuple[float, float, float]] = (
    {
        "center": (0.00, -0.10, 0.02),        # plate
        "upper_right": (0.16, 0.02, 0.055),   # cup
        "left": (-0.16, -0.10, 0.015),        # fork
        "right": (0.16, -0.10, 0.015),        # spoon
        "lower_left": (-0.16, 0.05, 0.012),   # napkin
    } if LEGACY_MODELS_FLAG else {
        # z = the realistic model's resting centre height (measured after settle)
        "center": (0.00, -0.16, 0.012),       # plate on its foot ring; both near rims 0.35 m from their bases
        # A set place around the centred plate (rim reaches x +/-0.10,
        # y -0.06..-0.26): fork and spoon beside it, napkin outside the fork,
        # cup upper right. The plate is set first, so nothing is in either
        # arm's sideways corridor when it is picked.
        "upper_right": (0.12, 0.02, 0.046),   # hollow cup, 90 mm tall
        "left": (-0.16, -0.12, 0.010),        # fork, grip segment arched up
        "right": (0.16, -0.12, 0.010),        # spoon
        "lower_left": (-0.20, 0.00, 0.012) if os.environ.get("OMNIQ_REAL_DRAWERS", "0") not in {"", "0", "false", "no"}
                      else (0.0, 0.04, 0.012),      # napkin above the plate, reachable by BOTH arms (0.30 m each)
    }
)


class IntelSimulationUnavailable(RuntimeError):
    """Raised when the optional MuJoCo integration dependency is absent."""


def _mujoco():
    try:
        import mujoco
    except ImportError as exc:  # allow the pure-Python core to remain runnable
        raise IntelSimulationUnavailable(
            "MuJoCo is required for the Intel simulation; install integrations/intel/requirements.txt"
        ) from exc
    return mujoco


def _prefixed(element: ET.Element, prefix: str) -> ET.Element:
    """Copy a Menagerie arm subtree while preserving shared default/mesh names."""
    clone = copy.deepcopy(element)
    for node in clone.iter():
        if "name" in node.attrib:
            node.set("name", f"{prefix}_{node.attrib['name']}")
    return clone


def _body(name: str, pos: str, geom: dict[str, str], *, euler: str | None = None) -> ET.Element:
    attrs = {"name": name, "pos": pos}
    if euler is not None:
        attrs["euler"] = euler
    body = ET.Element("body", attrs)
    ET.SubElement(body, "freejoint", {"name": f"{name}_free"})
    ET.SubElement(body, "geom", {"name": name, **geom})
    return body


def _cutlery(name: str, pos: str, *, handle: dict[str, str], head: dict[str, str],
             handle_offset: str, head_offset: str, mass: str, friction: str, rgba: str,
             euler: str | None = None) -> ET.Element:
    """A piece of cutlery as a narrow HANDLE plus a wider HEAD.

    A single flat box was the previous model, and it was the reason the flat
    objects could not be picked: with the fingertip pads 8 mm tall and the slab
    8 mm thick, any horizontal offset lands the pads on top of the slab rather
    than beside it, and the jaws pinch a corner at an angle -- measured live,
    2 pads at 15.5 N, +9 mm of lift, then a slip. Real cutlery is not a slab.
    The handle is ~10 mm wide and the head ~25 mm, so the jaws close squarely
    on a handle a fraction of their travel wide, with clearance to the table.

    The body origin -- which is what the pick targets -- is the handle's
    centre. (Moving it to the neck was tried on 2026-09-13 to reduce the
    seesaw pivot seen on lift: it did not change the spoon and it broke the
    fork's grasp outright, so the mid-handle origin stays.) The head geom
    carries the ``<name>_head`` suffix; contact accounting matches on the
    prefix.
    """
    attrs = {"name": name, "pos": pos}
    if euler is not None:
        attrs["euler"] = euler
    body = ET.Element("body", attrs)
    ET.SubElement(body, "freejoint", {"name": f"{name}_free"})
    common = {"rgba": rgba, "friction": friction, **RIGID_CONTACT}
    ET.SubElement(body, "geom", {"name": name, "type": "box", "mass": mass,
                                 "pos": handle_offset, **handle, **common})
    ET.SubElement(body, "geom", {"name": f"{name}_head", "type": "box", "pos": head_offset,
                                 "mass": "0", **head, **common})
    return body


# Realistic tableware, built from primitives (2026-09-14). The blocky proxies
# (32 mm puck plate, solid cup, flat cutlery slabs) were the operator's call
# to replace and the physics agreed: under honest (rigid) contact the puck
# jams in the jaw notch and the solid cup's grasp collapses. Dimensions are
# real-world; every shape is still MuJoCo boxes/cylinders/ellipsoids so
# there are no mesh assets. The old proxies stay reachable with
# OMNIQ_LEGACY_MODELS=1 for A/B evidence only.
LEGACY_MODELS = LEGACY_MODELS_FLAG


def _free_body(name: str, pos: str, euler: str | None = None) -> ET.Element:
    attrs = {"name": name, "pos": pos}
    if euler is not None:
        attrs["euler"] = euler
    body = ET.Element("body", attrs)
    ET.SubElement(body, "freejoint", {"name": f"{name}_free"})
    return body


def _bent_cutlery(name: str, pos: str, *, kind: str, mass: str, friction: str, rgba: str,
                  euler: str | None = None) -> ET.Element:
    """Cutlery with a bent handle, the way it actually lies on a table.

    Body origin = centre of the mid-handle, the grip point. A real fork or
    spoon does not lie flat: the head and the handle tip touch the table and
    the middle of the handle arches ~7 mm above it. That arch is what makes
    the handle graspable at all with 8 mm-tall fingertip pads -- the flat
    slab had nothing to straddle (measured 2026-09-13). Overall 150 mm,
    handle 10 x 7 mm grip section (5 mm at the tip and neck).
    """
    body = _free_body(name, pos, euler)
    common = {"rgba": rgba, "friction": friction, **RIGID_CONTACT}
    # mid-handle, level, 50 mm: the grip
    # 10 x 7 mm grip section: the jaw's hard stop is a 5.6 mm tip gap, so a
    # 5 mm handle was barely touched (1-2 pads, single-digit newtons).
    ET.SubElement(body, "geom", {"name": name, "type": "box", "size": ".005 .025 .0035",
                                 "pos": "0 0 0", "mass": "0.012", **common})
    # handle tip, 40 mm sloping down to the table (drop 6.5 mm over 40 mm)
    ET.SubElement(body, "geom", {"name": f"{name}_tip", "type": "box", "size": ".005 .02 .0025",
                                 "pos": "0 -.044 -.0033", "euler": "0.162 0 0", "mass": "0.010", **common})
    # neck, 25 mm sloping down to the head
    ET.SubElement(body, "geom", {"name": f"{name}_neck", "type": "box", "size": ".004 .0125 .0025",
                                 "pos": "0 .036 -.0033", "euler": "-0.262 0 0", "mass": "0.005", **common})
    if kind == "fork":
        # four tines, 45 mm, on a 25 mm bridge, resting on the table
        ET.SubElement(body, "geom", {"name": f"{name}_head", "type": "box", "size": ".0125 .005 .0015",
                                     "pos": "0 .053 -.0075", "mass": "0.004", **common})
        for i, x in enumerate((-.0095, -.0032, .0032, .0095)):
            ET.SubElement(body, "geom", {"name": f"{name}_tine_{i}", "type": "box", "size": ".0015 .0225 .0015",
                                         "pos": f"{x} .0805 -.0075", "mass": "0.00225", **common})
    else:
        # spoon bowl: a flattened ellipsoid, 50 x 32 x 8 mm, resting on the table
        ET.SubElement(body, "geom", {"name": f"{name}_head", "type": "ellipsoid", "size": ".016 .025 .004",
                                     "pos": "0 .073 -.005", "mass": "0.013", **common})
    return body


def _lipped_plate(name: str, pos: str, *, rgba: str, mass: str, friction: str,
                  euler: str | None = None) -> ET.Element:
    """A rimmed soup plate: bowl floor on a foot ring, stepped wall, flat lip.

    Body origin = bowl-floor centre. Measured 2026-09-14: the SO-101's
    moving-jaw tip is ~12.5 mm thick and the fixed jaw ~24 mm, so nothing on
    this gripper fits under a flat dinner plate's lip (~8 mm above the
    table). A rimmed soup/pasta plate -- a real, common piece -- carries its
    lip well up off the table, and that lip is the feature a sideways
    pinch takes: the thin fixed fingertip slides under it, the moving jaw
    closes down on top. The jaw body behind the fingertip is 42 mm thick,
    so the lip underside must be >= ~33 mm up for it to clear the table:
    a rimmed soup bowl (real ones run 230 x 40 mm). 200 mm across, 5 mm
    lip at +25 mm, 25 mm bowl depth, 350 g.
    """
    body = _free_body(name, pos, euler)
    common = {"rgba": rgba, "friction": friction, **RIGID_CONTACT}
    ET.SubElement(body, "geom", {"name": name, "type": "cylinder", "size": ".060 .0025",
                                 "mass": str(float(mass) * 0.35), **common})
    n_foot, n_wall, n_lip = 12, 16, 16
    for i in range(n_foot):
        th = 2 * math.pi * i / n_foot  # the scene compiler is in radians
        r = 0.050
        ET.SubElement(body, "geom", {"name": f"{name}_foot_{i}", "type": "box", "size": ".003 .0135 .004",
                                     "pos": f"{r * math.cos(th):.4f} {r * math.sin(th):.4f} -.0065",
                                     "euler": f"0 0 {th:.4f}", "mass": str(float(mass) * 0.10 / n_foot), **common})
    # stepped wall: three rings climbing from the floor (r 60) to the lip (r 85)
    for ring, (r, z) in enumerate(((0.065, 0.004), (0.075, 0.0115), (0.085, 0.019))):
        for i in range(n_wall):
            th = 2 * math.pi * i / n_wall
            ET.SubElement(body, "geom", {"name": f"{name}_wall{ring}_{i}", "type": "box",
                                         "size": f".003 {r * math.tan(math.pi / n_wall) * 1.08:.4f} .0045",
                                         "pos": f"{r * math.cos(th):.4f} {r * math.sin(th):.4f} {z:.4f}",
                                         "euler": f"0 0 {th:.4f}", "mass": str(float(mass) * 0.30 / (3 * n_wall)), **common})
    for i in range(n_lip):
        th = 2 * math.pi * i / n_lip
        r = 0.0925  # lip spans r 85-100 mm, 5 mm thick, centre +25 mm (top +27.5)
        ET.SubElement(body, "geom", {"name": f"{name}_lip_{i}", "type": "box",
                                     "size": f".0075 {r * math.tan(math.pi / n_lip) * 1.08:.4f} .0025",
                                     "pos": f"{r * math.cos(th):.4f} {r * math.sin(th):.4f} .025",
                                     "euler": f"0 0 {th:.4f}", "mass": str(float(mass) * 0.25 / n_lip), **common})
    return body


def _hollow_cup(name: str, pos: str, *, rgba: str, mass: str, friction: str, contact: dict[str, str],
                euler: str | None = None) -> ET.Element:
    """A cup that is actually hollow: 16 wall segments on a bottom disc.

    Body origin = mid-height, like the old solid cylinder, so every existing
    height constant still means the same thing. 60 mm across, 90 mm tall,
    3 mm wall. A pinch across the diameter now closes on two thin walls; a
    wall pinch (one pad inside, one outside) becomes possible.
    """
    R, H, WALL, N = 0.030, 0.090, 0.003, 16
    body = _free_body(name, pos, euler)
    common = {"rgba": rgba, "friction": friction, **contact}
    ET.SubElement(body, "geom", {"name": name, "type": "cylinder", "size": f"{R:.4f} {WALL / 2:.4f}",
                                 "pos": f"0 0 {-H / 2 + WALL / 2:.4f}", "mass": str(float(mass) * 0.2), **common})
    rc = R - WALL / 2
    half_chord = rc * math.tan(math.pi / N) * 1.08  # slight overlap closes the seams
    for i in range(N):
        th = 2 * math.pi * i / N
        ET.SubElement(body, "geom", {"name": f"{name}_wall_{i}", "type": "box",
                                     "size": f"{WALL / 2:.4f} {half_chord:.4f} {H / 2:.4f}",
                                     "pos": f"{rc * math.cos(th):.4f} {rc * math.sin(th):.4f} 0",
                                     "euler": f"0 0 {th:.4f}",  # radians
                                     "mass": f"{float(mass) * 0.8 / N:.5f}", **common})
    return body


def _drawer(pos: str, name: str = "drawer") -> ET.Element:
    """A drawer on a slide joint (OQ-007), passive -- no actuator.

    Legacy: a solid block at the far edge, out of every arm's reach, whose
    ``OPEN`` is a symbolic write to the slide joint.

    Realistic (2026-09-14, the operator's call -- "the arms can reach
    everything including the drawer"): a pull-out tray with 12 mm walls and a
    handle bar on its front, one unit on each flank inside its arm's reach,
    the fork lying in the left one and the spoon in the right. ``OPEN`` is
    then a real pinch on the handle and a 50 mm pull along the slide
    (:meth:`IntelTableWorld._do_open_drawer`). Opens toward the arms (+y).
    """
    body = ET.Element("body", {"name": name, "pos": pos})
    ET.SubElement(body, "joint", {
        "name": f"{name}_slide", "type": "slide", "axis": "0 1 0",
        "range": "0 .12" if LEGACY_MODELS_FLAG else "0 .05", "limited": "true", "damping": "3",
    })
    wood = {"rgba": ".30 .19 .11 1", "friction": "1.20 .006 .0002", **RIGID_CONTACT}
    if not REAL_DRAWERS:
        ET.SubElement(body, "geom", {"name": name, "type": "box", "size": ".12 .045 .015", "mass": ".2", **wood})
        return body
    # shallow cutlery tray: 200 x 90 mm inside, bottom 8 mm, walls 6 mm tall
    # (12 mm walls blocked the open jaw -- 121 mm at the tips -- from
    # straddling a handle lying in a 90 mm-wide tray)
    ET.SubElement(body, "geom", {"name": name, "type": "box", "size": ".10 .045 .004", "pos": "0 0 .004", "mass": ".15", **wood})
    for i, (px, py, sx, sy) in enumerate(((0, .045, .10, .004), (0, -.045, .10, .004), (.10, 0, .004, .045), (-.10, 0, .004, .045))):
        ET.SubElement(body, "geom", {"name": f"{name}_wall_{i}", "type": "box", "size": f"{sx} {sy} .003",
                                     "pos": f"{px} {py} .011", "mass": ".02", **wood})
    # handle: a 60 x 8 x 10 mm bar standing off the front wall on a post, so
    # fingertip pads can close across its 8 mm thickness
    # handle clear of the wall top (z .024): post up from the wall, a thin
    # bracket out to the bar, bar centre at z .040 with 13 mm of free space
    # behind it for the inner pad (the bar level with the wall top jammed the
    # inner pad on the wall, 2026-09-14)
    ET.SubElement(body, "geom", {"name": f"{name}_handle_post", "type": "box", "size": ".004 .003 .012",
                                 "pos": "0 .048 .026", "mass": ".005", **wood})
    ET.SubElement(body, "geom", {"name": f"{name}_handle_bracket", "type": "box", "size": ".004 .010 .002",
                                 "pos": "0 .055 .042", "mass": ".003", **wood})
    ET.SubElement(body, "geom", {"name": f"{name}_handle", "type": "box", "size": ".030 .004 .005",
                                 "pos": "0 .064 .040", "mass": ".01", "rgba": ".75 .75 .78 1",
                                 "friction": "1.20 .006 .0002", **RIGID_CONTACT})
    return body


def dual_so101_xml(config: IntelSceneConfig | None = None) -> str:
    """Return a dual-arm, table-setting MJCF built from the pinned asset."""
    config = config or IntelSceneConfig()
    rng = random.Random(config.seed)

    def tableware_pose(base: tuple[float, float, float], yaw0: float = 0.0) -> tuple[str, str | None]:
        if not config.randomized:
            return "%.6f %.6f %.6f" % base, ("0 0 %.6f" % yaw0 if yaw0 else None)
        x, y, z = base
        x += rng.uniform(-config.position_jitter_m, config.position_jitter_m)
        y += rng.uniform(-config.position_jitter_m, config.position_jitter_m)
        yaw = yaw0 + rng.uniform(-config.yaw_jitter_rad, config.yaw_jitter_rad)
        return "%.6f %.6f %.6f" % (x, y, z), "0 0 %.6f" % yaw

    def tableware_rgba(base_rgba: str) -> str:
        # Visual-only perturbation: rgba has no effect on contype/
        # conaffinity/friction/mass/solref/solimp, so this can never weaken
        # contact physics -- it only changes what a camera/vision model
        # would see, which is the point (object color/texture variation).
        if not config.randomized:
            return base_rgba
        r, g, b, a = (float(v) for v in base_rgba.split())
        jittered = (
            min(1.0, max(0.0, r + rng.uniform(-config.color_jitter, config.color_jitter))),
            min(1.0, max(0.0, g + rng.uniform(-config.color_jitter, config.color_jitter))),
            min(1.0, max(0.0, b + rng.uniform(-config.color_jitter, config.color_jitter))),
            a,
        )
        return "%.4f %.4f %.4f %.4f" % jittered

    source = ET.parse(ARM_XML).getroot()
    root = ET.Element("mujoco", {"model": "omni_q_dual_so101_table"})
    for tag in ("compiler", "option", "asset", "default"):
        node = source.find(tag)
        if node is not None:
            root.append(copy.deepcopy(node))

    visual = ET.SubElement(root, "visual")
    # offwidth/offheight: offscreen buffer for 1280x720 demo recordings (render-only)
    ET.SubElement(visual, "global", {"azimuth": "125", "elevation": "-28", "offwidth": "1280", "offheight": "720"})
    worldbody = ET.SubElement(root, "worldbody")
    # A real, physics-inert "different lighting condition" axis: the light's
    # own base position/direction/intensity are fixed above, but a
    # randomized trial swings the incidence angle (a lower/higher, more
    # off-axis key light -- like a different time of day) and the diffuse
    # intensity (dimmer/brighter). Nothing here touches contype/
    # conaffinity/friction/mass, only what a camera/vision model sees.
    light_pos = (0.0, -0.3, 1.3)
    light_dir = (0.0, 0.0, -1.0)
    light_diffuse = 0.7
    if config.randomized:
        light_pos = (
            light_pos[0] + rng.uniform(-0.25, 0.25),
            light_pos[1] + rng.uniform(-0.2, 0.2),
            light_pos[2] + rng.uniform(-0.3, 0.15),
        )
        light_dir = (
            light_dir[0] + rng.uniform(-config.light_angle_jitter_rad, config.light_angle_jitter_rad),
            light_dir[1] + rng.uniform(-config.light_angle_jitter_rad, config.light_angle_jitter_rad),
            light_dir[2],
        )
        light_diffuse = min(1.0, max(0.15, light_diffuse + rng.uniform(
            -config.light_diffuse_jitter, config.light_diffuse_jitter,
        )))
    ET.SubElement(worldbody, "light", {
        "name": "key", "pos": "%.4f %.4f %.4f" % light_pos, "dir": "%.4f %.4f %.4f" % light_dir,
        "directional": "true", "diffuse": "%.4f %.4f %.4f" % (light_diffuse, light_diffuse, light_diffuse),
    })
    # No contype/conaffinity here (or anywhere else in this scene): plain
    # MuJoCo defaults collide with everything, which is what "no clipping
    # through the environment" requires. A prior version of this comment
    # justified an arm/table collision exemption as necessary to keep the
    # position-only IK from "jamming" against the table -- that concern is
    # real in principle, but the fix for it is a controller that doesn't
    # command infeasible poses, not physics that pretends the arm has no
    # body. Verified empirically: the real orientation-aware grasp (see
    # this file's module docstring) still works correctly with full
    # collision, no jamming observed.
    # The floor is 0.75 m below the tabletop (a table height). It used to be
    # the plane z=0, coplanar with the table's top face: the renderer
    # z-fought between them (torn table edges, moire in every recording) and
    # contacts were being booked against "floor" on the tabletop.
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "pos": "0 0 -.75", "size": "0 0 .05", "rgba": ".08 .12 .12 1",
    })
    # pos.z -0.05 puts the box's top face at world z = 0 -- a real surface
    # flush with the floor plane. The previous -0.055 sank the top to
    # z = -0.005 (below the floor), so nothing ever rested on the table:
    # every object fell through it to the floor plane. Tableware start
    # z-heights below are set to each geom's own half-height so they begin
    # resting on this surface instead of dropping onto it.
    ET.SubElement(worldbody, "geom", {
        "name": "table", "type": "box", "pos": "0 -0.10 -0.05", "size": ".42 .36 .05",
        **RIGID_CONTACT,
        "rgba": ".23 .14 .08 1", "friction": "1 .005 .0001",
    })
    ET.SubElement(worldbody, "camera", {"name": "third_person", "pos": "0 -1.15 .85", "euler": "1.05 0 0"})
    ET.SubElement(worldbody, "camera", {"name": "table_overhead", "pos": "0 -.10 1.20", "euler": "0 0 0"})
    # Flank cameras (2026-09-14), one over each side of the table looking in
    # at the cutlery. The challenge allows six cameras; with overhead,
    # third-person and the two wrist cameras these make six. They replace the
    # table-height grazing view: fused perception (vision.MultiCameraFusion)
    # missed the spoon and fork parked at x = +/-0.32 -- tiny from overhead,
    # foreshortened from the front -- and the grazing view contributed only a
    # misread. Under OMNIQ_LEGACY_MODELS=1 the grazing camera stays, since
    # the recorded cup evidence of 2026-09-13 was taken from it.
    if LEGACY_MODELS:
        ET.SubElement(worldbody, "camera", {"name": "table_grazing", "pos": "-.90 -.10 .035",
                                            "euler": "1.5708 -1.5708 0", "fovy": "42"})
    else:
        ET.SubElement(worldbody, "camera", {"name": "left_flank", "pos": "-.60 -.02 .42",
                                            "xyaxes": "-0.1322 -0.9912 0 0.8042 -0.1072 0.5846", "fovy": "50"})
        ET.SubElement(worldbody, "camera", {"name": "right_flank", "pos": ".60 -.02 .42",
                                            "xyaxes": "-0.1322 0.9912 0 -0.8042 -0.1072 0.5846", "fovy": "50"})

    base = source.find("./worldbody/body[@name='Base']")
    if base is None:  # static source validation, not a recoverable runtime state
        raise RuntimeError("pinned SO-ARM100 MJCF is missing Base")
    # The source model excludes self-collision between adjacent links (its
    # own <contact><exclude .../></contact>, e.g. Base/Rotation_Pitch)
    # because those meshes overlap by construction at the joint. Dropping
    # this (as an earlier version of this function did) doesn't just look
    # wrong -- with that self-collision active, rotating the shoulder joint
    # drives it straight into the excluded contact, and MuJoCo generates a
    # resisting force that saturates the actuator's forcerange, jamming the
    # joint anywhere off its rest angle (silently, since nothing before the
    # real IK controller ever commanded it away from home). Re-declare each
    # source exclude pair once per arm, prefixed to match _prefixed()'s
    # renaming.
    source_excludes = source.findall("./contact/exclude")
    contact = ET.SubElement(root, "contact")
    for arm, pos in (("left", "-.26 .20 .0"), ("right", ".26 .20 .0")):
        arm_body = _prefixed(base, arm)
        arm_body.set("pos", pos)
        # Real friction/contact-softness tuning for the fingertip pads only
        # (a rubber-ish pad genuinely does grip harder than bare mesh) --
        # NOT contype/conaffinity masking. An earlier version of this
        # function set contype=0/conaffinity=0 on every non-pad arm geom,
        # which under MuJoCo's collision rule ((c1&a2)|(c2&a1)) makes a geom
        # unable to collide with ANYTHING regardless of what else is in the
        # scene -- the whole arm except the two pads could pass through the
        # table, the drawer, and every object with zero contact resistance.
        # That's not a control improvement, it's the simulation not
        # simulating the arm's own body -- reverted per the no-cheating
        # rule: only the pads get a friction override, nothing gets a
        # collision exemption, and no contype/conaffinity is set anywhere
        # in this scene (plain MuJoCo defaults collide with everything).
        for geom in arm_body.iter("geom"):
            name = geom.attrib.get("name", "")
            if "jaw_pad_" in name:
                geom.set("friction", PAD_FRICTION)
                geom.set("solref", PAD_SOLREF)
                geom.set("solimp", PAD_SOLIMP)
                if RIGID:
                    geom.set("priority", "1")
        # Wrist camera, mounted on the fixed jaw looking down the finger at the
        # pinch. The challenge allows up to 6 cameras and this scene used 2.
        #
        # Motivated by measurement, not decoration: the pinch solve lands within
        # ~3.5 mm of its target (see _ik_reach_pad(track_tcp=True)) while fork
        # and spoon are 8 mm thick and the napkin 6 mm, so the error budget and
        # the object are the same size. That is the regime where a local visual
        # loop earns its keep, and it is what the SO-101 RL write-up used an
        # 84x84 wrist view for. Its author also found console coordinates
        # insufficient to see a grasp-frame bug at all -- the same bug this file
        # had -- and recommended looking at the grasp point directly.
        #
        # Purely additive: a camera has no collision geometry, no mass and no
        # contype, so it cannot change the physics this scene is judged on.
        # _prefixed() has already renamed every body, so the jaw is
        # "<arm>_Fixed_Jaw" here, not "Fixed_Jaw".
        jaw = arm_body.find(f".//body[@name='{arm}_Fixed_Jaw']")
        if jaw is not None:
            # Mount (2026-09-14): beside the fixed jaw on its +z side, at the
            # finger base, aimed down the finger at the fingertip pads. The
            # earlier mount (x=.045, looking -x) stared at the jaw's own body.
            # Three candidates were rendered mid-descent over the cup; this one
            # shows the pads and the object beneath them.
            ET.SubElement(jaw, "camera", {
                "name": f"{arm}_wrist",
                "pos": ".012 -.010 .048",
                "xyaxes": "0 -0.4706 0.8824 0.9983 -0.0518 -0.0276",
                "fovy": "70",
            })
        worldbody.append(arm_body)
        for exclude in source_excludes:
            ET.SubElement(contact, "exclude", {
                "body1": f"{arm}_{exclude.attrib['body1']}",
                "body2": f"{arm}_{exclude.attrib['body2']}",
            })

    if LEGACY_MODELS:
        worldbody.append(_drawer("0 -.40 .015"))  # half-height .015 -> base rests on the z=0 table top
    elif not REAL_DRAWERS:
        worldbody.append(_drawer("0 -.45 .015"))
    else:
        # one pull-out tray per flank, inside its arm's reach (handle ~0.26 m from the base)
        # 3 mm above the tabletop: a drawer rides in its housing, it is not
        # dragged across the table (a tray on the table stalled at 30 mm with
        # 300 N of finger load turning into sliding friction)
        worldbody.append(_drawer("-.32 -.10 .003", "drawer"))
        worldbody.append(_drawer(".32 -.10 .003", "drawer_right"))
    # Each z is the geom's own half-height: the object starts resting on the
    # table surface (top face at world z = 0) rather than hovering above a
    # surface that used to be below the floor. z is never jittered by
    # tableware_pose(), so these stay exact.
    # Realistic plate: further across the table from its arm. Scanned joint
    # space 2026-09-14: a horizontal finger with the fixed jaw on top exists
    # only with the fingertip 0.30-0.48 m from the base at lip height, so the
    # near rim must be at least 0.30 m out -- the old spot put it at 0.215.
    # Bimanual plate: centred between the arms, far side of the table, so
    # each arm's near rim is ~0.40 m from its base (band 0.30-0.48).
    # (y -0.13: the drawer opens toward the arms by 0.12 m and its open front
    # reaches y -0.275 with the drawer at the table edge -- at y -0.22 the
    # plate's rim sat in that path, 19 mm of drawer/plate penetration.)
    plate_pos, plate_euler = tableware_pose((-.13, -.08, .016) if LEGACY_MODELS else (0.0, -.22, .0105))
    # Cup: out of the right arm's corridor to the plate rim (it spawned on
    # that line and blocked the bimanual approach, 2026-09-14), still 0.30 m
    # from the right base.
    cup_pos, cup_euler = tableware_pose((.16, -.06, .050) if LEGACY_MODELS else (.06, .03, .045))
    # fork_1/spoon_1 used to sit at (+/-.04, -.40) -- co-located with the
    # drawer prop above. Measured (two independent ways: this file's own IK
    # convergence sweep, and OQ-003's separately-published reach probe in
    # so101_capability_map.md) that position is ~0.64m from each arm's base,
    # well outside the arm's real ~0.386m max reach radius -- physically
    # unreachable on top of anything about grasp orientation. Repositioned
    # to a spot verified in-reach (~0.25-0.27m radial, solid margin) and
    # collision-checked against plate_1/cup_1/napkin_1's own footprints.
    # The drawer body itself is left at its old position: it's a passive
    # prop the arm's controller never actually touches (OPEN only writes
    # its qpos directly, see IntelTableWorld.apply_transition), so moving it
    # doesn't change what's reachable -- it also means, worth noting
    # explicitly, that fork_1/spoon_1 were never kinematically attached to
    # the drawer's slide joint in the first place, so "opening" it was
    # always symbolic and didn't literally reveal these bodies either way.
    if LEGACY_MODELS:
        fork_pos, fork_euler = tableware_pose((-.32, -.05, .004))
        spoon_pos, spoon_euler = tableware_pose((.32, -.05, .004))
        napkin_pos, napkin_euler = tableware_pose((-.22, .02, .009))
    elif not REAL_DRAWERS:
        fork_pos, fork_euler = tableware_pose((-.32, -.05, .009))
        spoon_pos, spoon_euler = tableware_pose((.34, .02, .009))
        # napkin in the SHARED band between the arms (0.24 m from the left
        # base, 0.32 m from the right): the one left-side task the right arm
        # can take over when the left arm fails (2026-09-14, arm-failure
        # continuation demo). Everything else is single-arm by reach.
        napkin_pos, napkin_euler = tableware_pose((-.04, .10, .009))
    else:
        # cutlery lying inside its drawer tray (tray bottom top face at z .008),
        # along the tray's long axis; the napkin off the left unit's footprint
        fork_pos, fork_euler = tableware_pose((-.32, -.10, .017), yaw0=1.5708)
        spoon_pos, spoon_euler = tableware_pose((.32, -.10, .017), yaw0=-1.5708)
        napkin_pos, napkin_euler = tableware_pose((-.10, .10, .009))
    tableware = [
        _lipped_plate("plate_1", plate_pos, rgba=tableware_rgba(".93 .93 .91 1"),
                      mass=".20", friction="1.20 .006 .0002", euler=plate_euler),
        _hollow_cup("cup_1", cup_pos, rgba=tableware_rgba(".22 .58 .78 1"), mass=".10",
                    friction="3.00 .020 .001",
                    contact=(RIGID_CONTACT or {"solref": PAD_SOLREF, "solimp": PAD_SOLIMP}), euler=cup_euler),
        _bent_cutlery("fork_1", fork_pos, kind="fork", mass=".04", friction="1.20 .006 .0002",
                      rgba=tableware_rgba(".72 .73 .75 1"), euler=fork_euler),
        _bent_cutlery("spoon_1", spoon_pos, kind="spoon", mass=".04", friction="1.20 .006 .0002",
                      rgba=tableware_rgba(".72 .73 .75 1"), euler=spoon_euler),
    ]
    if LEGACY_MODELS:
        # (legacy plate note) Rim half-height .016 (32mm full thickness), not
        # the original .007 (14mm): the gripper's fully-closed pad gap was
        # then believed to be 21.3mm, so the puck was made thick enough to
        # pinch. It is kept only as the A/B baseline.
        tableware = [
            _body("plate_1", plate_pos, {
                "type": "cylinder", "size": ".095 .016", "rgba": tableware_rgba(".93 .93 .91 1"),
                "mass": ".18", "friction": "1.20 .006 .0002",  # ceramic
                **RIGID_CONTACT,
            }, euler=plate_euler),
            _body("cup_1", cup_pos, {
                # Calibrated to the measured SO-101 pad envelope: the previous
                # 64 mm / 120 g fixture exceeded the 27 mm open-pad gap and could
                # not distinguish a bad grasp from an impossible geometry.
                "type": "cylinder", "size": ".022 .050", "rgba": tableware_rgba(".22 .58 .78 1"),
                "mass": ".08", "friction": "3.00 .020 .001",
                **(RIGID_CONTACT or {"solref": PAD_SOLREF, "solimp": PAD_SOLIMP}),
            }, euler=cup_euler),
            # fork/spoon start inside the drawer -- retrieval is gated on OPEN, matching
            # the brief's scenario ("open the top drawer, retrieve spoons and forks").
            # Handle 10 mm wide x 100 mm x 7 mm; head 25 mm x 50 mm x 6 mm at the
            # +y end. Same 150 mm overall length and 40 g as the old slab.
            _cutlery("fork_1", fork_pos,
                     handle={"size": ".005 .05 .0035"}, handle_offset="0 0 0",
                     head={"size": ".0125 .025 .003"}, head_offset="0 .075 0",
                     mass=".04", friction="1.20 .006 .0002",
                     rgba=tableware_rgba(".72 .73 .75 1"), euler=fork_euler),
            _cutlery("spoon_1", spoon_pos,
                     handle={"size": ".005 .05 .0035"}, handle_offset="0 0 0",
                     head={"size": ".013 .02 .004"}, head_offset="0 .07 0",
                     mass=".04", friction="1.20 .006 .0002",
                     rgba=tableware_rgba(".72 .73 .75 1"), euler=spoon_euler),
        ]
    worldbody.extend(tableware)
    worldbody.extend([
        # A FOLDED napkin -- 70 x 50 x 18 mm -- as it sits on a set table, not
        # a 140 x 100 x 6 mm sheet laid flat. The flat sheet was unpickable by
        # construction: 8 mm-tall fingertip pads cannot straddle a 6 mm slab
        # with no narrow feature (2026-09-13: 76 evidence-driven retries, 22
        # with pad contact, zero lifts). Folding is what people actually do
        # with a napkin; the block it makes is thick enough to pinch and
        # narrow enough to close across. Same 10 g.
        _body("napkin_1", napkin_pos, {
            "type": "box", "size": ".035 .025 .009", "rgba": tableware_rgba(".90 .40 .38 1"),
            # Cloth genuinely grips more than metal cutlery or glazed
            # ceramic (higher real sliding-friction coefficient) -- a
            # uniform 1.20 across every material lost that distinction;
            # restoring it, not inventing it.
            "mass": ".02", "friction": "1.60 .006 .0002",
            # Cloth compresses: ~2.5 mm under 20 N, not ceramic-rigid.
            **NAPKIN_CONTACT,
        }, euler=napkin_euler),
    ])
    # A real, narrower fix for the same order-dependence bug the reverted
    # contype/conaffinity scheme above was also (over-broadly) solving: an
    # earlier-placed object could get physically shoved by a later PICK's
    # approach before its own governed transition, turning a valid grasp
    # into a scene-order-dependent failure. Excluding tableware-vs-
    # tableware contact specifically (not touching arm-vs-anything or
    # object-vs-table/pad collision at all) fixes that without exempting
    # the arm from the environment -- explicit named pairs, not a bitmask,
    # so the scope is exactly these five bodies and nothing else.
    _TABLEWARE = ("plate_1", "cup_1", "fork_1", "spoon_1", "napkin_1")
    for i, body1 in enumerate(_TABLEWARE):
        for body2 in _TABLEWARE[i + 1:]:
            ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})

    actuators = ET.SubElement(root, "actuator")
    source_actuators = source.find("actuator")
    if source_actuators is None:
        raise RuntimeError("pinned SO-ARM100 MJCF is missing actuators")
    for arm in ("left", "right"):
        for actuator in source_actuators:
            copied = copy.deepcopy(actuator)
            copied.set("name", f"{arm}_{actuator.attrib['name']}")
            copied.set("joint", f"{arm}_{actuator.attrib['joint']}")
            actuators.append(copied)
    return ET.tostring(root, encoding="unicode")


def load_dual_so101_model(config: IntelSceneConfig | None = None):
    """Load the dual-arm model without writing generated MJCF into the repo."""
    mujoco = _mujoco()
    assets = {
        f"assets/{path.name}": path.read_bytes()
        for path in ARM_ASSETS.glob("*.stl")
    }
    return mujoco.MjModel.from_xml_string(dual_so101_xml(config), assets=assets)


class IntelTableWorld(MockWorld):
    """Authoritative table world backed by a real MuJoCo model and timestep."""

    mode = "simulation-scripted-manipulation"

    def __init__(self, scene_config: IntelSceneConfig | None = None) -> None:
        self.scene_config = scene_config or IntelSceneConfig()
        # Distinct starting zones (OQ-007): a shared literal zone string for
        # every object collapses the scheduler's workspace-conflict check
        # into full serialization regardless of arm (see
        # integrations/intel/README.md). fork_1/spoon_1 sharing "drawer" is
        # the one intentional exception -- they really do start in the same
        # physical drawer cavity (see dual_so101_xml()).
        super().__init__([
            Detection("plate_1", "plate", "tray_plate", "center"),
            Detection("cup_1", "cup", "tray_cup", "upper_right"),
            Detection("fork_1", "fork", "drawer", "left"),
            Detection("spoon_1", "spoon", "drawer", "right"),
            Detection("napkin_1", "napkin", "tray_napkin", "lower_left"),
            # The drawer itself, so OPEN/CLOSE(object="drawer") passes
            # MockWorld.apply_transition's object-registration check.
            # zone == target_zone: it's a fixture, never "misplaced", so
            # RulePlanner never tries to PICK/MOVE it like tableware.
            Detection("drawer", "fixture", "closed", "closed"),
        ] + ([Detection("drawer_right", "fixture", "closed", "closed")] if REAL_DRAWERS else []))
        mujoco = _mujoco()
        self.model = load_dual_so101_model(self.scene_config)
        self.data = mujoco.MjData(self.model)
        self._mujoco = mujoco
        self._controller_steps = 0
        self.bimanual_objects = BIMANUAL_OBJECTS   # the engine never pairs these with another arm's step
        drawer_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        self._drawer_qpos_adr = self.model.jnt_qposadr[drawer_joint_id]
        self._drawer_slide_adr = {"drawer": self._drawer_qpos_adr}
        if REAL_DRAWERS:
            jr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_right_slide")
            self._drawer_slide_adr["drawer_right"] = self.model.jnt_qposadr[jr]
        # TCP reference per arm -- the same body the capability-map probe
        # (integrations/intel/scripts/probe_so101.py) already uses.
        self._tcp_body = {
            0: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_Fixed_Jaw"),
            6: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_Fixed_Jaw"),
        }
        # Fingertip pad geom (same one the proven _ContactHandoffController
        # tracks) for the final precision pinch -- more accurate than the
        # Fixed_Jaw body origin used for safe-transit motion above.
        self._pad_geom = {
            0: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_fixed_jaw_pad_4"),
            6: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_fixed_jaw_pad_4"),
        }
        # (qpos address, dof/qvel address) per tableware freejoint -- used by
        # the IK grasp/place below; a freejoint is 7 qpos (xyz + quat) but
        # only 6 dof (linvel + angvel), so the two addresses are not
        # interchangeable.
        self._object_joints: dict[str, tuple[int, int]] = {}
        for obj_id in ("plate_1", "cup_1", "fork_1", "spoon_1", "napkin_1"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj_id}_free")
            self._object_joints[obj_id] = (self.model.jnt_qposadr[jid], self.model.jnt_dofadr[jid])
        for index, value in enumerate(HOME * 2):
            self.data.ctrl[index] = value
        mujoco.mj_forward(self.model, self.data)
        # mj_forward only computes kinematics for the CURRENT qpos (still all
        # zeros -- MjData starts zeroed); it does not integrate toward ctrl.
        # Actually settle the arms into HOME before anything else touches
        # this world, otherwise every qpos-based read (including the IK
        # solver's "current configuration") starts from a pose the arm was
        # never really in.
        mujoco.mj_step(self.model, self.data, nstep=400)
        self._controller_steps += 400

    def state(self) -> WorldState:
        """The base symbolic ``WorldState`` with a real metric ``pose`` stamped
        onto every tracked tableware Detection, read live from the MuJoCo
        free-joint qpos (OQ-009). The zone strings stay the authority for
        planning; the pose is the metric truth underneath, for the evaluator,
        the ontology, and receipts. Untracked fixtures (the drawer) keep
        ``pose=None``.
        """
        st = super().state()
        located = {}
        for oid, det in st.objects.items():
            if oid in self._object_joints:
                adr, _ = self._object_joints[oid]
                x, y, z = (float(v) for v in self.data.qpos[adr:adr + 3])
                located[oid] = replace(det, pose=Pose(x, y, z, self._object_yaw(oid)))
            else:
                located[oid] = det
        return replace(st, objects=located)

    def apply_transition(self, request: TransitionRequest) -> TransitionResult:
        op = request.op
        obj = request.args.get("object")
        if op == "EXPRESS":
            # The provider repeats validation because a direct TransitionRequest
            # can bypass the policy generator.  Generic expression is free-space
            # only; contact primitives remain a separate controller gate.
            validate_expressive_command(request.args)
        if op == "PICK" and obj in self._object_joints and request.actor:
            # One object per gripper (2026-09-14). With the camera observer
            # the observation carries no ownership, and the scheduler sent
            # PICK cup to the right arm while it still held the spoon; the
            # spoon's MOVE then failed "unsafe carry separation" and so did
            # the cup's. A held object is a physical fact the world knows
            # regardless of what perception reported: refuse, without moving
            # anything, so the planner sends the MOVE first.
            held = [o for o, owner in self._ownership.items() if owner == request.actor and o != obj]
            if held:
                return TransitionResult(
                    request.step_id, False, self.revision,
                    {"grasp": "refused", "held": False, "lift_height_m": 0.0,
                     "reason": f"{request.actor} is already holding {held[0]}"},
                )
        physics_snapshot = None
        if op == "OPEN" and obj in getattr(self, "_drawer_slide_adr", {}):
            physics_snapshot = (
                self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(),
                float(self.data.time), self._controller_steps,
            )
        if op in {"PICK", "MOVE", "PLACE"} and obj in self._object_joints:
            # A failed real attempt must not leave the next governed retry
            # starting from a disturbed arm/object configuration. WorldState
            # rollback alone is insufficient because MuJoCo has already
            # integrated joint and free-body dynamics.
            physics_snapshot = (
                self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(),
                float(self.data.time), self._controller_steps,
            )
        # captured before super() mutates WorldState, so a failed real grasp
        # can be reverted below rather than leaving a scripted "success" that
        # the physics never actually delivered.
        prev_zone = self._objects[obj].zone if obj in self._objects else None
        prev_owner = self._ownership.get(obj) if obj in self._ownership else None

        result = super().apply_transition(request)
        arm_offset = 6 if request.actor and "right" in request.actor else 0
        grasp_info: dict[str, Any] = {}
        expressive_info: dict[str, Any] = {}

        if op == "PICK" and obj in self._object_joints:
            grasp_info = self._do_pick(arm_offset, obj)
            if not grasp_info["held"]:
                self._ownership[obj] = prev_owner  # revert the scripted grasp claim
                # per-arm: the other arm may be holding something (2026-09-14)
                self._restore_physics(physics_snapshot, arm_offset=arm_offset, obj=obj)
                result = replace(result, ok=False)
        elif op in {"MOVE", "PLACE"} and obj in self._object_joints:
            grasp_info = self._do_place(arm_offset, obj, request.args.get("to"))
            if not grasp_info["placed"]:
                if prev_zone is not None:
                    self._objects[obj] = replace(self._objects[obj], zone=prev_zone)
                # A failed placement is NOT rewound (2026-09-14). The gripper
                # has already opened and retracted; the object is on the table
                # wherever it landed. Restoring the pre-MOVE physics re-created
                # a phantom "still holding it" state, and the scheduler then
                # sent the same arm -- jaw closed on the napkin -- to pick the
                # plate, which failed, which retried... (seed 900 trace). The
                # honest state is: object released, arm free, ownership
                # cleared so the planner picks it up again from where it is.
                self._ownership[obj] = None
                result = replace(result, ok=False)
        elif op == "EXPRESS":
            expressive_info = self._execute_expressive(arm_offset, request.args)
            if not expressive_info["ok"]:
                result = replace(result, ok=False)
        else:
            # OQ-010 fallback pose for ops with no dedicated IK behaviour, or
            # a PICK/MOVE/PLACE on an object this adapter doesn't track
            # (e.g. the "drawer" fixture): a fixed coarse target, same as
            # before real grasping existed for tableware.
            target = list(HOME)
            arm_command = True
            if op in {"PICK", "MOVE"}:
                target[2], target[3], target[5] = 1.20, 0.85, GRIPPER_CLOSED
            elif op == "PLACE":
                target[2], target[3], target[5] = 1.20, 0.85, GRIPPER_OPEN
            elif op == "OPEN" and obj in self._drawer_slide_adr:
                arm_command = False
                if not REAL_DRAWERS:
                    # Symbolic write to the slide joint -- the block sits at
                    # the far edge, out of every arm's reach.
                    self.data.qpos[self._drawer_qpos_adr] = DRAWER_OPEN
                else:
                    # Realistic: pinch the handle and pull (2026-09-14).
                    grasp_info = self._do_open_drawer(arm_offset, obj)
                    if not grasp_info["opened"]:
                        self._restore_physics(physics_snapshot)
                        result = replace(result, ok=False)
            elif op == "CLOSE" and obj in self._drawer_slide_adr:
                arm_command = False
                self.data.qpos[self._drawer_slide_adr[obj]] = DRAWER_CLOSED
            elif op == "OPEN":
                target[5] = GRIPPER_OPEN
            elif op == "CLOSE":
                target[5] = GRIPPER_CLOSED
            elif op == "ROTATE":
                target[4] = -0.8 if arm_offset else 0.8
            elif op == "PRESENT":
                target[1], target[3], target[5] = -0.8, 0.3, GRIPPER_CLOSED
            elif op == "HANDOFF":
                target[4] = 1.20 if arm_offset else -1.20
            elif op in _FLOURISH_GESTURES or op == "FREEZE":
                # OQ-HAND-011: non-contact idle-slack flourishes -- a free-space
                # arm gesture, no object, no grasp. Drive an offset-from-HOME
                # pose, then return to HOME, so the arm ends where it started.
                arm_command = False
                if op == "MIRROR":
                    other = 0 if arm_offset else 6
                    mirrored = [float(v) for v in self.data.qpos[other:other + 6]]
                    mirrored[0] = -mirrored[0]          # base yaw
                    mirrored[4] = -mirrored[4]          # wrist roll
                    swings = [mirrored, list(HOME)]
                elif op == "FREEZE":
                    swings = [list(HOME)]               # hold the home pose
                else:
                    joint, delta = _FLOURISH_GESTURES[op]
                    swings = []
                    for off in (delta, -delta, 0.0):
                        pose = list(HOME)
                        pose[joint] = HOME[joint] + off
                        swings.append(pose)
                for pose in swings:
                    for i, v in enumerate(pose):
                        self.data.ctrl[arm_offset + i] = v
                    self._mujoco.mj_step(self.model, self.data, nstep=18)
                    self._controller_steps += 18
            if arm_command:
                for index, value in enumerate(target, start=arm_offset):
                    self.data.ctrl[index] = value
            self._mujoco.mj_step(self.model, self.data, nstep=20)
            self._controller_steps += 20

        detail = {
            **result.detail,
            **grasp_info,
            **expressive_info,
            "sim_time": round(float(self.data.time), 4),
            "controller_steps": self._controller_steps,
            "simulation_mode": self.mode,
            "physics_note": (
                "PICK/MOVE/PLACE on tracked tableware use real IK + a "
                "contact-driven grasp (no teleport, no velocity override); "
                "everything else is still a coarse scripted pose"
            ),
        }
        return replace(result, detail=detail)

    def _execute_expressive(self, arm_offset: int, args: dict[str, Any]) -> dict[str, Any]:
        """Run one bounded generic free-space motion in the MuJoCo proxy.

        This is a provider implementation of the EXPRESS contract, not
        evidence of learned or hardware motor intelligence.  The trajectory
        starts and ends at the current pose and is shaped by the proposal's
        primitive parameters rather than a named gesture table.
        """
        primitive = str(args["primitive"]).upper()
        duration_ms = int(args["duration_ms"])
        axis = int(args.get("axis", 0))
        amplitude = float(args.get("amplitude", 0.0))
        phase = math.radians(float(args.get("phase_deg", 0.0)))
        cycles = float(args.get("cycles", 1.0))
        current = [float(v) for v in self.data.qpos[arm_offset:arm_offset + 6]]
        steps = max(1, min(180, int(math.ceil(duration_ms / 8.0))))
        initial_contacts = int(self.data.ncon)

        # A generic primitive selects a trajectory family.  No semantic
        # gesture is encoded here; a policy can vary axis, phase, amplitude,
        # cycles, arm, and timing on every proposal.
        for index in range(1, steps + 1):
            progress = index / steps
            envelope = math.sin(math.pi * progress)
            wave = math.sin((2.0 * math.pi * cycles * progress) + phase)
            pose = list(current)
            if primitive != "PAUSE":
                pose[axis] += amplitude * envelope * wave
            if primitive in {"CHANGE_SPEED", "CHANGE_AMPLITUDE"}:
                pose[axis] = current[axis] + (amplitude * 0.5) * envelope * wave
            for joint_index, value in enumerate(pose):
                self.data.ctrl[arm_offset + joint_index] = value
            self._mujoco.mj_step(self.model, self.data, nstep=4)
            self._controller_steps += 4

        # Return to the proposal's starting pose before yielding control back
        # to the task graph.  A contact appearing during free-space expression
        # is a failed provider action, never a semantic success.
        for joint_index, value in enumerate(current):
            self.data.ctrl[arm_offset + joint_index] = value
        self._mujoco.mj_step(self.model, self.data, nstep=8)
        self._controller_steps += 8
        contact_detected = int(self.data.ncon) > initial_contacts
        return {
            "ok": not contact_detected,
            "expressive": True,
            "primitive": primitive,
            "duration_ms": duration_ms,
            "axis": axis,
            "contact_detected": contact_detected,
            "return_error_rad": round(max(
                abs(float(self.data.qpos[arm_offset + i]) - current[i])
                for i in range(6)
            ), 6),
            "simulation_mode": self.mode,
        }

    def fail_arm(self, arm_offset: int, *, reason: str = "servo communication lost") -> dict[str, Any]:
        """Make an arm physically stop responding (2026-09-14): its six
        actuator commands are frozen at their current values on every physics
        step from now on, as a servo bus with no host would hold its last
        target. Nothing about the world state is edited -- the arm is simply
        a fixture that no longer moves. Whatever the planner sends it will
        fail on the physics, which is the point of the failure demo.
        Returns the frozen command so the receipt can quote it."""
        frozen = self.data.ctrl[arm_offset:arm_offset + 6].copy()
        real = self._mujoco
        self._failed_arms = getattr(self, "_failed_arms", {})
        self._failed_arms[arm_offset] = {"reason": reason, "frozen_ctrl": [round(float(v), 4) for v in frozen],
                                         "at_controller_step": self._controller_steps}
        world = self

        class FrozenArm:
            def __getattr__(self, name):
                return getattr(real, name)

            def mj_step(self, m, d, nstep=1):
                for _ in range(nstep):
                    for a, info in world._failed_arms.items():
                        d.ctrl[a:a + 6] = info["frozen_ctrl"]
                    real.mj_step(m, d)
        self._mujoco = FrozenArm()
        return dict(self._failed_arms[arm_offset])

    def _set_pick_track(self, arm_offset: int, value: bool) -> None:
        """Per-arm (2026-09-14): two arms run their primitives at the same
        time now, and a shared flag let the cup's deep grasp on one arm switch
        the other arm's fingertip tracking mid-pick."""
        by_arm = getattr(self, "_pick_track_tcp_by_arm", None)
        if by_arm is None:
            by_arm = self._pick_track_tcp_by_arm = {}
        by_arm[arm_offset] = value

    # -- simultaneous two-arm execution (2026-09-14) ----------------------
    #
    # The engine executes one revision-bound transition per control boundary.
    # To move both arms at once on independent objects it submits a PAIR
    # under one revision check; each primitive runs in its own thread and
    # every physics step is shared: a primitive's mj_step waits until all
    # active primitives have asked for a step, then one real step advances the
    # world for everyone. Each thread writes only its own arm's six actuator
    # commands, so the two controllers never touch each other's state; the
    # physics -- contacts between the two arms included -- is the same single
    # simulation as before.

    class _CoopStepper:
        def __init__(self, real, workers: int) -> None:
            import threading

            self._real = real
            self._cv = threading.Condition()
            self._active = workers
            self._waiting = 0
            self._generation = 0
            self.steps = 0

        def __getattr__(self, name):
            return getattr(self._real, name)

        def mj_step(self, m, d, nstep: int = 1) -> None:
            for _ in range(int(nstep)):
                with self._cv:
                    self._waiting += 1
                    gen = self._generation
                    if self._waiting >= self._active:
                        self._advance(m, d)
                    else:
                        while gen == self._generation and self._active > 1:
                            self._cv.wait(0.5)
                        if gen == self._generation:      # partner finished: step alone
                            self._advance(m, d)

        def _advance(self, m, d) -> None:
            self._real.mj_step(m, d)
            self.steps += 1
            self._waiting = 0
            self._generation += 1
            self._cv.notify_all()

        def finish(self) -> None:
            with self._cv:
                self._active -= 1
                if self._waiting and self._waiting >= self._active > 0:
                    # the partner is parked at the barrier waiting for us
                    self._waiting = 0
                    self._generation += 1
                    self._cv.notify_all()

    def apply_transitions_parallel(self, requests: list[TransitionRequest]) -> list[TransitionResult]:
        """Run two manipulation transitions on different arms at the same time.

        One revision check for the pair; physical primitives run concurrently
        under a cooperative stepper; then each request's symbolic transition is
        applied in order (the second against the revision the first produced).
        A failed primitive is reported like the sequential path: PICK reverts
        the ownership claim; MOVE/PLACE leaves the object where it landed and
        clears ownership. Physics is never rewound here -- the other arm's
        real motion cannot be undone.
        """
        import threading
        from dataclasses import replace as _replace

        if not requests:
            return []
        for request in requests:
            if request.expected_revision != self.revision:
                raise TransitionRejected(
                    f"{request.step_id}: expected revision {request.expected_revision}, "
                    f"current revision is {self.revision}"
                )
        arms = [6 if (r.actor and "right" in r.actor) else 0 for r in requests]
        if len(set(arms)) != len(arms):
            raise TransitionRejected("parallel transitions must use different arms")

        def physical(request: TransitionRequest, arm_offset: int) -> dict[str, Any]:
            op, obj = request.op, request.args.get("object")
            if op == "PICK" and obj in self._object_joints:
                if request.actor:
                    held = [o for o, owner in self._ownership.items() if owner == request.actor and o != obj]
                    if held:
                        return {"grasp": "refused", "held": False, "lift_height_m": 0.0,
                                "reason": f"{request.actor} is already holding {held[0]}"}
                return self._do_pick(arm_offset, obj)
            if op in {"MOVE", "PLACE"} and obj in self._object_joints:
                return self._do_place(arm_offset, obj, request.args.get("to"))
            if op == "OPEN" and obj in getattr(self, "_drawer_slide_adr", {}):
                if REAL_DRAWERS:
                    return self._do_open_drawer(arm_offset, obj)
                self.data.qpos[self._drawer_slide_adr[obj]] = DRAWER_OPEN
                return {"opened": True, "symbolic": True}
            if op == "CLOSE" and obj in getattr(self, "_drawer_slide_adr", {}):
                self.data.qpos[self._drawer_slide_adr[obj]] = DRAWER_CLOSED
                return {"closed": True, "symbolic": True}
            return {"noop": True}

        real = self._mujoco
        stepper = IntelTableWorld._CoopStepper(real, len(requests))
        self._mujoco = stepper
        results: dict[int, dict[str, Any]] = {}
        errors: dict[int, BaseException] = {}

        def worker(i: int) -> None:
            try:
                results[i] = physical(requests[i], arms[i])
            except BaseException as exc:  # noqa: BLE001 - surface, never hang the partner
                errors[i] = exc
                results[i] = {"held": False, "placed": False, "opened": False, "reason": f"{type(exc).__name__}: {exc}"}
            finally:
                stepper.finish()

        threads = [threading.Thread(target=worker, args=(i,), name=f"arm-{arms[i]}") for i in range(len(requests))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self._mujoco = real
        self._parallel_steps = stepper.steps

        out: list[TransitionResult] = []
        for i, request in enumerate(requests):
            info = results[i]
            op, obj = request.op, request.args.get("object")
            prev_zone = self._objects[obj].zone if obj in self._objects else None
            prev_owner = self._ownership.get(obj)
            symbolic = super().apply_transition(_replace(request, expected_revision=self.revision))
            ok = symbolic.ok
            if op == "PICK" and obj in self._object_joints:
                if not info.get("held"):
                    self._ownership[obj] = prev_owner
                    ok = False
            elif op in {"MOVE", "PLACE"} and obj in self._object_joints:
                if not info.get("placed"):
                    if prev_zone is not None:
                        self._objects[obj] = _replace(self._objects[obj], zone=prev_zone)
                    self._ownership[obj] = None
                    ok = False
            elif op == "OPEN" and not info.get("opened", True):
                ok = False
            detail = {**symbolic.detail, **info, "parallel_with": [r.step_id for r in requests if r is not request],
                      "parallel_physics_steps": stepper.steps}
            out.append(TransitionResult(request.step_id, ok, self.revision, detail))
        return out

    def _restore_physics(self, snapshot, *, arm_offset: int | None = None, obj: str | None = None) -> None:
        """Restore a pre-attempt MuJoCo state after a rejected transition.

        With ``arm_offset`` (and optionally ``obj``) only that arm's six joints
        and commands and that object's free joint are restored -- the
        per-attempt retry rollback (2026-09-14). The whole-world form rewound
        the OTHER arm too: while both arms picked at once, the fork's retry
        put the right arm's joints and jaw back to before it had lifted the
        cup, and the cup was on the table when its MOVE began.
        """
        if snapshot is None:
            return
        qpos, qvel, ctrl, sim_time, controller_steps = snapshot
        if arm_offset is None:
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel
            self.data.ctrl[:] = ctrl
            self.data.time = sim_time
            self._controller_steps = controller_steps
        else:
            a = arm_offset
            self.data.qpos[a:a + 6] = qpos[a:a + 6]
            self.data.qvel[a:a + 6] = qvel[a:a + 6]
            self.data.ctrl[a:a + 6] = ctrl[a:a + 6]
            if obj in self._object_joints:
                qa, da = self._object_joints[obj]
                self.data.qpos[qa:qa + 7] = qpos[qa:qa + 7]
                self.data.qvel[da:da + 6] = qvel[da:da + 6]
        self._mujoco.mj_forward(self.model, self.data)

    # -- OQ-010: real IK + contact grasp for tracked tableware ----------

    # Weighted (damped, weighted-pseudoinverse) IK joint costs: heavily
    # discourage the proximal joints (shoulder Rotation, then Pitch) relative
    # to the distal ones (Elbow, Wrist_Pitch, Wrist_Roll). The 5-joint chain
    # is kinematically redundant for a 3D position task, and the *unweighted*
    # minimum-norm solution is free to route error reduction through
    # whichever joint the local Jacobian favours -- measured swinging the
    # shoulder/upper-arm through a wide arc and knocking tableware aside even
    # for a small, purely vertical TCP motion where that swing wasn't
    # geometrically necessary. Proximal joints move a much larger swept
    # volume per radian (long lever arm back to the base) than distal ones,
    # so biasing the redundant solution toward the distal joints keeps the
    # arm's own body closer to the path the *end effector* traces instead of
    # ranging far from it.
    _IK_JOINT_WEIGHTS = (80.0, 25.0, 1.0, 1.0, 1.0)

    def _ik_reach(
        self, arm_offset: int, target_pos, *, iters: int = 500,
        max_dq: float = 0.05, sub_steps: int = 3, tol: float = 0.012, weighted: bool = True,
    ) -> float:
        """Damped-least-squares differential IK on this arm's first 5 joints
        (excluding the gripper) driving the ``Fixed_Jaw`` body toward
        ``target_pos``. Each iteration nudges the position servo targets by a
        small joint-space step and lets ``mj_step`` actually move the arm --
        real joint motion, not a jump. Returns the final position error (m);
        callers decide what error counts as "close enough".

        ``weighted=True`` (default) uses ``_IK_JOINT_WEIGHTS`` to discourage
        the proximal joints -- needed near the table, where the unweighted
        minimum-norm solution was measured swinging the shoulder/upper-arm
        through the tableware even for small vertical motions where that
        swing wasn't geometrically necessary. It also converges much slower,
        so callers that are already at a safe height and only need
        horizontal reach (nothing nearby to hit) should pass
        ``weighted=False``."""
        import numpy as np

        mujoco = self._mujoco
        tcp_id = self._tcp_body[arm_offset]
        jacp = np.zeros((3, self.model.nv))
        lo = self.model.jnt_range[arm_offset:arm_offset + 5, 0]
        hi = self.model.jnt_range[arm_offset:arm_offset + 5, 1]
        weights = self._IK_JOINT_WEIGHTS if weighted else (1.0, 1.0, 1.0, 1.0, 1.0)
        w_inv = np.diag([1.0 / w for w in weights])
        target = np.asarray(target_pos, dtype=float)
        err_norm = float("inf")
        for _ in range(iters):
            mujoco.mj_jacBody(self.model, self.data, jacp, None, tcp_id)
            jac = jacp[:, arm_offset:arm_offset + 5]
            err = target - self.data.body(tcp_id).xpos
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                break
            damping = 0.06
            jac_w = jac @ w_inv
            dq = w_inv @ jac_w.T @ np.linalg.solve(
                jac_w @ jac_w.T + damping * damping * np.eye(3), err)
            dq = np.clip(dq, -max_dq, max_dq)
            q_now = self.data.qpos[arm_offset:arm_offset + 5]
            self.data.ctrl[arm_offset:arm_offset + 5] = np.clip(q_now + dq, lo, hi)
            mujoco.mj_step(self.model, self.data, nstep=sub_steps)
            self._controller_steps += sub_steps
        return err_norm

    # Proven fix, ported from _ContactHandoffController (10/10 for its one
    # hand-tuned cup sequence): pin wrist-roll to a fixed per-arm value
    # instead of leaving it in the redundant position-only solve, and track
    # the fingertip pad geom instead of the Fixed_Jaw body origin. This is
    # what _ik_reach's 5-joint weighted solve above doesn't do -- it avoids
    # the table (safe transit), but never controlled jaw *orientation*, which
    # is why the general grasp path converges on position (~1cm) without
    # reliably pinching anything. Values are the same ones proven in the
    # contact-handoff scene; not yet re-tuned per object geometry here.
    _GRASP_WRIST_ROLL = {0: 1.65, 6: -1.65}

    def _fingertip_pinch_offset(self, arm_offset: int) -> float:
        """Height to add so the *fingertip* lands at the grasp height.

        The IK tracks ``fixed_jaw_pad_4``, which sits ~43 mm up the finger from
        the fingertip (measured: tip 0.0076 m, pad_4 0.0510 m at a stalled
        pinch). The grasp height, though, is computed from the object's own
        centre. Demanding the object's centre height *of pad_4* therefore asks
        the fingertip to go that same 43 mm lower -- about 4 cm below the table
        for a fork lying flat, which is unsatisfiable, so the solver stalls with
        the tip grazing the object instead of straddling it.

        That single frame mismatch explains the whole success pattern measured
        on 2026-09-13: ``cup_1`` (centre 0.050 m, the only object taller than
        the offset) held 10/10, ``plate_1`` (0.016 m, rim-grasped) 9/12, and
        ``fork_1`` / ``spoon_1`` / ``napkin_1`` (0.003-0.004 m) never once.

        **Superseded and no longer called.** Shifting the *target* by a
        pose-dependent world-z offset was the naive version of this fix and
        regressed ``cup_1`` 10/10 -> 0/10. The right mechanism is to track the
        fingertip midpoint directly (``_ik_reach_pad(track_tcp=True)``), which
        needs no offset at all. Kept only so the measurement is not repeated.
        """
        if os.environ.get("OMNIQ_FINGERTIP_PINCH", "") not in ("1", "true", "True"):
            return 0.0
        tracked = self._pad_geom.get(arm_offset)
        if tracked is None:
            return 0.0
        prefix = "left_" if arm_offset == 0 else "right_"
        tip = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}fixed_jaw_pad_1")
        if tip < 0:
            return 0.0
        offset = float(self.data.geom_xpos[tracked][2] - self.data.geom_xpos[tip][2])
        return max(0.0, offset)

    #: Objects at least this tall are grasped DEEP in the fingers (pad_4
    #: tracking, all four pads along the object) rather than at the fingertips.
    #: A two-fingertip pinch on a 110 mm cup is a line contact on a cylinder:
    #: the cup swings like a pendulum during the carry and is set down ~45 deg
    #: tilted -- watched directly on the table_grazing camera, seed 701,
    #: 2026-09-13. Fingertips for cutlery, wrapped fingers for a cup, which is
    #: what a person does. The threshold is the finger span from tip to pad_4.
    DEEP_GRASP_MIN_HEIGHT = 0.045  # m

    def _track_tcp_for(self, obj: str | None) -> bool:
        """Fingertip tracking for thin objects, deep grasp for tall ones."""
        if not self._fingertip_pinch:
            return False
        if obj is None:
            return True
        if obj in WALL_PINCH_OBJECTS:
            return True   # the fingertips are what go inside the rim
        height = 2.0 * OBJECT_HALF_HEIGHT.get(obj, 0.01)
        return height < self.DEEP_GRASP_MIN_HEIGHT

    @property
    def _fingertip_pinch(self) -> bool:
        """Track the fingertip midpoint (not pad_4) for every pick IK solve."""
        # Default ON since 2026-09-13. Ten randomized seeds, held per object:
        #   pad_4 tracking:     cup 10  plate 9   fork 0   spoon 0
        #   fingertip tracking: cup 10  plate 10  fork 10  spoon 4
        # Better or equal on every object. Set OMNIQ_FINGERTIP_PINCH=0 to
        # reproduce the old pad_4 behaviour for comparison.
        return os.environ.get("OMNIQ_FINGERTIP_PINCH", "1") not in ("0", "false", "False")

    def _tcp_geoms(self, arm_offset: int) -> tuple[int, int] | None:
        """The two fingertip pads whose midpoint is the real TCP.

        The vendored menagerie SO-ARM100 defines **no sites at all**
        (``model.nsite == 0``), which is why this file improvised by tracking a
        pad geom -- and picked ``fixed_jaw_pad_4``, 43 mm up the finger. The
        SO-101 RL write-up the hosts circulated hit both halves of this: its
        model *does* carry a ``gripperframe`` (fingertips) and a ``graspframe``
        (further back), and targeting the wrong one made the gripper
        "keep overshooting the cube"; separately, "one finger is fixed, one
        moves", so the contact point is the midpoint, not the fixed finger.
        """
        prefix = "left_" if arm_offset == 0 else "right_"
        ids = []
        for name in (f"{prefix}fixed_jaw_pad_1", f"{prefix}moving_jaw_pad_1"):
            gid = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                return None
            ids.append(gid)
        return ids[0], ids[1]

    def _ik_reach_pad(
        self, arm_offset: int, target_pos, *, iters: int = 300, max_dq: float = 0.04, tol: float = 0.01,
        roll: float | None = None, track_tcp: bool | None = None,
        stop_on_contact: tuple[str, float] | None = None, geom_id: int | None = None,
    ) -> float:
        """4-DOF (Rotation/Pitch/Elbow/Wrist_Pitch) IK tracking the fixed-jaw
        pad geom toward ``target_pos`` with wrist-roll pinned to
        ``_GRASP_WRIST_ROLL[arm_offset]`` for the whole solve -- see the
        class comment above. Returns the final pad position error (m)."""
        import numpy as np

        mujoco = self._mujoco
        pad_id = self._pad_geom[arm_offset] if geom_id is None else geom_id
        # track_tcp drives the *midpoint of the two fingertips* instead of the
        # fixed jaw's fourth pad. Position and Jacobian are both averaged, so
        # the solve stays consistent rather than steering one point while
        # measuring another.
        # None -> the world-level default (OMNIQ_FINGERTIP_PINCH). Every IK
        # call in the pick sequence must agree on the tracked point: switching
        # only the final pinch to the fingertip made the reference jump ~43 mm
        # along the finger mid-sequence, which showed up as an 88 mm XY error
        # the 180-iteration pinch solve could not close.
        if track_tcp is None:
            track_tcp = getattr(self, "_pick_track_tcp_by_arm", {}).get(arm_offset)
            if track_tcp is None:
                track_tcp = self._fingertip_pinch
        tcp = self._tcp_geoms(arm_offset) if track_tcp else None
        roll = self._GRASP_WRIST_ROLL[arm_offset] if roll is None else float(roll)
        jacp = np.zeros((3, self.model.nv))
        jacp_b = np.zeros((3, self.model.nv))
        lo = self.model.jnt_range[arm_offset:arm_offset + 4, 0]
        hi = self.model.jnt_range[arm_offset:arm_offset + 4, 1]
        target = np.asarray(target_pos, dtype=float)
        err_norm = float("inf")
        for _ in range(iters):
            self.data.ctrl[arm_offset + 4] = roll  # re-pin every iteration; the servo can drift under load
            if tcp is None:
                mujoco.mj_jacGeom(self.model, self.data, jacp, None, pad_id)
                jac = jacp[:, arm_offset:arm_offset + 4]
                current = self.data.geom_xpos[pad_id]
            else:
                mujoco.mj_jacGeom(self.model, self.data, jacp, None, tcp[0])
                mujoco.mj_jacGeom(self.model, self.data, jacp_b, None, tcp[1])
                jac = 0.5 * (jacp[:, arm_offset:arm_offset + 4]
                             + jacp_b[:, arm_offset:arm_offset + 4])
                current = 0.5 * (self.data.geom_xpos[tcp[0]]
                                 + self.data.geom_xpos[tcp[1]])
            err = target - current
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                break
            damping = 0.04
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * damping * np.eye(3), err)
            dq = np.clip(dq, -max_dq, max_dq)
            q_now = self.data.qpos[arm_offset:arm_offset + 4]
            self.data.ctrl[arm_offset:arm_offset + 4] = np.clip(q_now + dq, lo, hi)
            mujoco.mj_step(self.model, self.data, nstep=3)
            self._controller_steps += 3
            if stop_on_contact is not None:
                # Force-guarded descent. Measured 2026-09-13 (seed 900, plate):
                # the open gripper landed its fixed pad on the plate rim at
                # 19 N and the solve kept driving toward a target below the
                # rim; the rim jammed into the V between pad and jaw body --
                # a wedge, which turned ~1 N.m of wrist servo into 1.2-4.7 kN
                # of contact force and popped the plate off the table at
                # 2.9 m/s. A controller with a force reading stops pushing
                # at first firm contact; this is that reading.
                obj, limit = stop_on_contact
                force = self._jaw_object_force(arm_offset, obj)
                if force > limit:
                    self._descend_contact_stop = round(force, 2)
                    # Hold the pose reached, don't keep leaning on the command.
                    self.data.ctrl[arm_offset:arm_offset + 4] = self.data.qpos[arm_offset:arm_offset + 4]
                    break
        return err_norm

    def _jaw_object_force(self, arm_offset: int, obj: str) -> float:
        """Largest contact force (N) between any geom of this arm's jaw --
        pads, fixed jaw body, moving jaw body -- and any geom of ``obj``.
        Read from the contact buffer; the pads-only reading in
        ``_set_gripper`` misses a rim jammed against the jaw body."""
        import numpy as np

        mujoco = self._mujoco
        prefix = "left_" if arm_offset == 0 else "right_"
        jaw_bodies = {f"{prefix}Fixed_Jaw", f"{prefix}Moving_Jaw"}
        f = np.zeros(6)
        best = 0.0
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[con.geom1]) or ""
            b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[con.geom2]) or ""
            if {b1, b2} & jaw_bodies and obj in (b1, b2):
                mujoco.mj_contactForce(self.model, self.data, c, f)
                best = max(best, float(np.linalg.norm(f[:3])))
        return best

    def _object_yaw(self, obj: str) -> float:
        """Read tableware yaw from its live freejoint pose.

        This is an observation, not a pose command. Randomized scene yaw is
        therefore respected without mutating the object to make a grasp
        easier.
        """
        qpos_adr, _ = self._object_joints[obj]
        qw, qx, qy, qz = (float(value) for value in self.data.qpos[qpos_adr + 3:qpos_adr + 7])
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def _grasp_frame(self, arm_offset: int, obj: str) -> tuple[Any, float]:
        """Return an object-aware fingertip orientation and roll hint."""
        import numpy as np

        yaw = self._object_yaw(obj)
        if obj in {"fork_1", "spoon_1"}:
            # A box-section handle is the same grasp rotated 180 deg. Feeding
            # the raw yaw through unchanged flipped the wrist-roll hint by pi
            # whenever scene jitter left the piece pointing the other way, and
            # the jaw then closed on nothing (harness seeds 903/904, 2026-09-13:
            # roll +1.65 on an arm whose base roll is -1.65, zero contacts).
            # Fold the symmetry so yaw and yaw+pi describe one grasp.
            yaw = math.atan2(math.sin(2.0 * yaw), math.cos(2.0 * yaw)) / 2.0
            opening_angle = yaw
        elif obj == "napkin_1":
            opening_angle = yaw + math.pi / 2.0
        else:
            opening_angle = 0.0
        if arm_offset == 6:
            opening_angle += math.pi
        opening = np.array([math.cos(opening_angle), math.sin(opening_angle), 0.0])
        pad_x = -opening  # fixed-pad normal; moving pad lies in -local-x
        pad_y = np.array([0.0, 0.0, 1.0])
        pad_z = np.cross(pad_x, pad_y)
        target_rotation = np.column_stack((pad_x, pad_y, pad_z))
        base_roll = self._GRASP_WRIST_ROLL[arm_offset]
        if obj == "napkin_1":
            # Measured, not derived (2026-09-13): with the arm's base roll the
            # napkin's first attempt never made contact, and every successful
            # pick was rescued by the retry's MIRRORED roll on the third or
            # fourth try (lifts 79-84 mm once it landed). Closing across the
            # folded block's short side wants the jaw the other way round.
            base_roll = -base_roll
        # A parallel pinch is the same pinch after a half-turn of the wrist.
        # Cutlery lying at ~90 deg (in its drawer tray) produced a hint near
        # +/-pi that the clip below turned into a jaw 90 deg off -- closing
        # ALONG the handle (2026-09-14). Of the half-turn-equivalent hints,
        # take the one closest to the arm's base roll that the joint can
        # reach; the proven picks (yaw ~0, hint = base roll) are unchanged.
        candidates = [base_roll - yaw + k * math.pi for k in (-1, 0, 1)]
        # "reachable" = what the clip below used to absorb (base roll +/- a
        # small yaw jitter overshoots 1.62 by a few hundredths and is clipped,
        # as before); only a near-pi hint is replaced by its half-turn twin
        reachable = [c for c in candidates if abs(c) <= 2.2]
        roll_hint = min(reachable or candidates, key=lambda c: abs(c - base_roll))
        return target_rotation, float(np.clip(roll_hint, -1.62, 1.62))

    def _ik_reach_pad_pose(
        self, arm_offset: int, target_pos, target_rotation, *, roll_hint: float,
        iters: int = 360, max_dq: float = 0.04, position_tol: float = 0.012,
        orientation_tol: float = 0.18, stop_on_contact: tuple[str, float] | None = None,
    ) -> dict[str, Any]:
        """Bounded 6D damped-least-squares solve for a pad pose.

        The residual combines metres and radians with an explicit scale. It
        uses only arm joints, respects a 30 mrad joint-limit margin, and never
        writes an object's freejoint.
        """
        import numpy as np

        mujoco = self._mujoco
        pad_id = self._pad_geom[arm_offset]
        target = np.asarray(target_pos, dtype=float)
        target_rotation = np.asarray(target_rotation, dtype=float).reshape(3, 3)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        lo = self.model.jnt_range[arm_offset:arm_offset + 5, 0] + 0.03
        hi = self.model.jnt_range[arm_offset:arm_offset + 5, 1] - 0.03
        w_inv = np.diag([1.0 / 80.0, 1.0 / 25.0, 1.0, 1.0, 0.5])
        orientation_scale = 0.10  # metres-equivalent per radian
        self.data.ctrl[arm_offset + 4] = roll_hint
        position_error = float("inf")
        orientation_error = float("inf")
        for _ in range(iters):
            mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, pad_id)
            current_rotation = self.data.geom_xmat[pad_id].reshape(3, 3)
            position_residual = target - self.data.geom_xpos[pad_id]
            orientation_residual = 0.5 * (
                np.cross(current_rotation[:, 0], target_rotation[:, 0])
                + np.cross(current_rotation[:, 1], target_rotation[:, 1])
                + np.cross(current_rotation[:, 2], target_rotation[:, 2])
            )
            position_error = float(np.linalg.norm(position_residual))
            orientation_error = float(np.linalg.norm(orientation_residual))
            if position_error <= position_tol and orientation_error <= orientation_tol:
                break
            residual = np.concatenate((position_residual, orientation_residual * orientation_scale))
            jacobian = np.vstack((
                jacp[:, arm_offset:arm_offset + 5],
                jacr[:, arm_offset:arm_offset + 5] * orientation_scale,
            ))
            damped = jacobian @ w_inv @ jacobian.T + 0.06 * 0.06 * np.eye(6)
            delta = w_inv @ jacobian.T @ np.linalg.solve(damped, residual)
            delta = np.clip(delta, -max_dq, max_dq)
            q_now = self.data.qpos[arm_offset:arm_offset + 5]
            self.data.ctrl[arm_offset:arm_offset + 5] = np.clip(q_now + delta, lo, hi)
            mujoco.mj_step(self.model, self.data, nstep=3)
            self._controller_steps += 3
            if stop_on_contact is not None:
                # Same force guard as _ik_reach_pad: this solve is the first
                # to reach grasp height, and it was the one leaning on the
                # plate rim for 1.4 s before the guarded final descent ran.
                obj, limit = stop_on_contact
                force = self._jaw_object_force(arm_offset, obj)
                if force > limit:
                    self._descend_contact_stop = round(force, 2)
                    self.data.ctrl[arm_offset:arm_offset + 5] = self.data.qpos[arm_offset:arm_offset + 5]
                    break
        self.data.ctrl[arm_offset + 4] = float(np.clip(self.data.ctrl[arm_offset + 4], lo[4], hi[4]))
        return {
            "position_error_m": round(position_error, 6),
            "orientation_error_rad": round(orientation_error, 6),
            "orientation_satisfied": bool(position_error <= position_tol and orientation_error <= orientation_tol),
        }

    #: Squeeze held past the stall angle once the jaw is blocked, in rad. With
    #: the Jaw actuator's kp=50 this is ~2 N of grip -- deliberate, bounded, and
    #: well inside its +/-3.5 N forcerange.
    GRIPPER_STALL_SQUEEZE_RAD = 0.04

    @property
    def _gripper_stall_hold(self) -> bool:
        """Re-pin the jaw at a bounded squeeze once it stalls (OMNIQ_GRIPPER_STALL_HOLD=1).

        Detection and evidence are always on; only the *behaviour* is opt-in,
        and for a measured reason. With the hold enabled, plate_1 goes from
        held 9/10 to 0/10: its jaw stalls at ~1.0 rad on the rim's top face and
        the grasp only ever succeeded because the servo, left to ramp to its
        force limit for 180 steps, shoved the pad over the rim -- precisely the
        "force the finger through the object" behaviour the hosts warned about.
        That grasp needs a real rim-approach fix, not a bounded servo pretending
        it never happened. Until then the demo keeps its plate, and the receipt
        now records that the jaw stalled and how hard it pushed.
        """
        return os.environ.get("OMNIQ_GRIPPER_STALL_HOLD", "") in ("1", "true", "True")

    def _set_gripper(self, arm_offset: int, value: float, *, settle_steps: int = 15,
                     obj: str | None = None) -> dict[str, Any]:
        """Command the jaw and return what the actuator actually did.

        Closing is stall-aware. Commanding a fixed angle and stepping blind --
        what this used to do -- has a failure the challenge hosts called out
        directly: if a rigid object blocks the jaw short of the target, the
        position servo ramps toward its force limit trying to close *through*
        it. So while closing, the jaw is watched; the moment it stops moving
        while still short of the target it is declared stalled on something,
        and the command is re-pinned to the stall angle minus a small squeeze.
        The grip is then held at a bounded, known force instead of the
        actuator's limit.

        The stall angle is also the cheapest grasp sensor there is: a jaw that
        reaches the commanded closed angle closed on *nothing*. That is what a
        real gripper's position feedback reports, and it is returned here
        alongside the MuJoCo contact buffer (every active pad/object
        intersection and its force) so a grasp is judged on evidence rather
        than on the command having been issued.
        """
        import numpy as np

        mujoco = self._mujoco
        jaw = arm_offset + 5                       # ctrl, qpos and qvel index alike (hinge, arm block first)
        start = float(self.data.qpos[jaw])
        closing = value < start
        self.data.ctrl[jaw] = value
        stalled = False
        stall_angle: float | None = None
        # A stall is SUSTAINED stillness, not a momentary bounce. On first
        # contact with a rim the jaw briefly stops and then keeps closing;
        # calling that a stall pinned the jaw wide open on plate_1 and lost a
        # grasp that used to succeed. Conversely a soft contact keeps the jaw
        # creeping, so a single-step velocity test never fires on cup_1. So:
        # short of target, and total travel over the last window under 5 mrad.
        window = 25
        history: list[float] = []
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
            self._controller_steps += 1
            q = float(self.data.qpos[jaw])
            history.append(q)
            if (closing and not stalled
                    and abs(q - value) > 0.02                 # still short of the target
                    and len(history) > window
                    and abs(q - history[-window]) < 0.005):   # ... and stopped for a while
                stalled, stall_angle = True, q
                if self._gripper_stall_hold or obj in STALL_HOLD_OBJECTS:
                    self.data.ctrl[jaw] = q - self.GRIPPER_STALL_SQUEEZE_RAD
        actual = float(self.data.qpos[jaw])
        if closing and obj in STALL_HOLD_OBJECTS and not stalled and abs(actual - value) > 0.02:
            # The jaw never "stopped" -- it was creeping on the object the whole
            # settle -- so the stall rule above did not fire. Pin the command
            # at the achieved angle plus the bounded squeeze anyway: a servo
            # left driving toward fully-closed walked the cup out of the pinch
            # within ~10 s (2026-09-14).
            self.data.ctrl[jaw] = actual - self.GRIPPER_STALL_SQUEEZE_RAD
            for _ in range(40):
                mujoco.mj_step(self.model, self.data)
                self._controller_steps += 1
            actual = float(self.data.qpos[jaw])

        # Contact buffer: what is the jaw actually touching, and how hard?
        prefix = "left_" if arm_offset == 0 else "right_"
        pads_touching: set[str] = set()
        max_force = 0.0
        if obj is not None:
            gname = lambda i: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            f = np.zeros(6)
            for c in range(self.data.ncon):
                con = self.data.contact[c]
                g1, g2 = gname(con.geom1), gname(con.geom2)
                pad = g1 if (g1.startswith(prefix) and "jaw_pad" in g1) else (
                    g2 if (g2.startswith(prefix) and "jaw_pad" in g2) else None)
                other = g2 if pad == g1 else g1
                if pad is None or not other.startswith(obj):
                    continue
                mujoco.mj_contactForce(self.model, self.data, c, f)
                mag = float(np.linalg.norm(f[:3]))
                if mag > 0.05:
                    pads_touching.add(pad)
                    max_force = max(max_force, mag)
        return {
            "jaw_commanded_rad": round(value, 4),
            "jaw_actual_rad": round(actual, 4),
            "jaw_stalled": stalled,
            "jaw_stall_angle_rad": None if stall_angle is None else round(stall_angle, 4),
            "pad_contacts": sorted(pads_touching),
            "contact_pad_count": len(pads_touching),
            "max_contact_force_n": round(max_force, 3),
        }

    def _workspace_safety(self, arm_offset: int, obj: str | None = None, target_xy=None) -> dict[str, Any]:
        """Return bounded safety observables for a legacy physical primitive.

        Full MuJoCo geom collisions remain enabled in the legacy scene; this
        position controller still has no predictive whole-arm planner.  The
        guard therefore fails closed on the measurable hazards it can prove:
        joint-limit approach, excessive object-to-pad separation, excessive
        contact force, and an arm entering a target occupied by the other pad.
        """
        import numpy as np

        lo = self.model.jnt_range[arm_offset:arm_offset + 5, 0]
        hi = self.model.jnt_range[arm_offset:arm_offset + 5, 1]
        q = self.data.qpos[arm_offset:arm_offset + 5]
        joint_margin = float(np.min(np.minimum(q - lo, hi - q)))
        if joint_margin < LEGACY_MIN_JOINT_MARGIN_RAD:
            return {
                "safe": False,
                "reason": "joint-limit proximity",
                "min_joint_margin_rad": round(joint_margin, 6),
            }

        relative_distance = 0.0
        if obj is not None and obj in self._object_joints:
            qpos_adr, _ = self._object_joints[obj]
            # Measure from the point the grasp actually tracks. Under fingertip
            # tracking the object sits at the fingertip midpoint, ~43 mm down
            # the finger from pad_4; measuring from pad_4 read a 0.138 m
            # "separation" on a correctly held cup and rejected every carry
            # (2026-09-13, seed 701). The guard's job is to catch an object
            # that has left the grip, so it must use the grip's own reference.
            tcp = self._tcp_geoms(arm_offset) if self._track_tcp_for(obj) else None
            if tcp is not None:
                reference = 0.5 * (self.data.geom_xpos[tcp[0]] + self.data.geom_xpos[tcp[1]])
            else:
                reference = self.data.geom_xpos[self._pad_geom[arm_offset]]
            relative_distance = float(np.linalg.norm(
                self.data.qpos[qpos_adr:qpos_adr + 3] - reference
            ))
            if relative_distance > LEGACY_MAX_CARRY_OFFSET_M:
                return {
                    "safe": False,
                    "reason": "unsafe carry separation",
                    "relative_object_pad_distance_m": round(relative_distance, 6),
                    "min_joint_margin_rad": round(joint_margin, 6),
                }

        max_contact_force = 0.0
        force = np.zeros(6)
        for contact_id in range(self.data.ncon):
            self._mujoco.mj_contactForce(self.model, self.data, contact_id, force)
            max_contact_force = max(max_contact_force, float(np.linalg.norm(force[:3])))
        if max_contact_force > LEGACY_MAX_CONTACT_FORCE_N:
            return {
                "safe": False,
                "reason": "collision force bound exceeded",
                "max_contact_force_n": round(max_contact_force, 6),
                "min_joint_margin_rad": round(joint_margin, 6),
            }

        other_offset = 6 if arm_offset == 0 else 0
        other_pad = self.data.geom_xpos[self._pad_geom[other_offset]]
        target_clearance = None
        if target_xy is not None:
            target_xy = np.asarray(target_xy, dtype=float)
            target_clearance = float(np.linalg.norm(other_pad[:2] - target_xy[:2]))
            if target_clearance < 0.060 and float(other_pad[2]) < 0.14:
                return {
                    "safe": False,
                    "reason": "unsafe shared-workspace entry",
                    "other_pad_clearance_m": round(target_clearance, 6),
                    "min_joint_margin_rad": round(joint_margin, 6),
                }

        return {
            "safe": True,
            "reason": None,
            "min_joint_margin_rad": round(joint_margin, 6),
            "relative_object_pad_distance_m": round(relative_distance, 6),
            "max_contact_force_n": round(max_contact_force, 6),
            "other_pad_clearance_m": None if target_clearance is None else round(target_clearance, 6),
        }

    def _settle_released_object(self, obj: str) -> dict[str, Any]:
        """Advance a bounded settle interval and observe table support/speed."""
        import numpy as np

        qpos_adr, dof_adr = self._object_joints[obj]
        position_before = self.data.qpos[qpos_adr:qpos_adr + 3].copy()
        self._mujoco.mj_step(self.model, self.data, nstep=80)
        self._controller_steps += 80
        velocity = float(np.linalg.norm(self.data.qvel[dof_adr:dof_adr + 3]))
        position_after = self.data.qpos[qpos_adr:qpos_adr + 3].copy()
        drift = float(np.linalg.norm(position_after - position_before))
        support = False
        # Body-level: the realistic plate stands on its foot ring and the
        # cutlery on its tip and head, so the geom that carries the object's
        # name may never touch the table itself.
        object_body = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, obj)
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            b1 = int(self.model.geom_bodyid[int(contact.geom1)])
            b2 = int(self.model.geom_bodyid[int(contact.geom2)])
            if object_body not in (b1, b2):
                continue
            other_geom = int(contact.geom2 if b1 == object_body else contact.geom1)
            other_name = self._mujoco.mj_id2name(
                self.model, self._mujoco.mjtObj.mjOBJ_GEOM, other_geom,
            ) or ""
            if other_name in {"table", "floor"}:
                support = True
                break
        return {
            "settle_steps": 80,
            "settle_speed_mps": round(velocity, 6),
            "settle_drift_m": round(drift, 6),
            "table_supported": support,
            "settled": bool(
                support
                and velocity <= LEGACY_MIN_SETTLE_SPEED_MPS
                and drift <= LEGACY_MAX_SETTLE_DRIFT_M
            ),
            "position": [round(float(value), 6) for value in position_after],
        }

    def _ik_track_line(
        self, arm_offset: int, start_xyz, end_xyz, *,
        step: float = 0.03, iters_per_waypoint: int = 180, weighted: bool = True,
    ) -> float:
        """Follow a straight Cartesian line in small (~3cm) waypoints rather
        than one direct IK reach to a distant target -- keeps each solve
        close to its start, which keeps the path close to the line, on top
        of whatever ``weighted`` already buys (see ``_ik_reach``)."""
        import numpy as np

        start = np.asarray(start_xyz, dtype=float)
        end = np.asarray(end_xyz, dtype=float)
        n = max(1, int(np.linalg.norm(end - start) / step))
        err = 0.0
        for i in range(1, n + 1):
            waypoint = start + (end - start) * (i / n)
            err = self._ik_reach(arm_offset, waypoint, iters=iters_per_waypoint, tol=0.012, weighted=weighted)
        return err

    def _move_to(self, arm_offset: int, target_xyz, *, transit_z: float = TRANSIT_HEIGHT) -> float:
        """Up-over-down waypoint motion, each leg tracked in small steps
        (see ``_ik_track_line``): straight up to a safe transit height,
        translate horizontally at that height, then descend. The vertical
        legs stay weighted (near the table/drawer/other tableware); the
        horizontal leg at the already-safe transit height doesn't need it --
        measured no disturbance either way up there, and it converges far
        faster unweighted (proximal joints do most large lateral reaches).
        Not a full collision-aware planner -- it avoids the table, nothing
        else."""
        x, y, z = target_xyz
        cur = self.data.body(self._tcp_body[arm_offset]).xpos.copy()
        self._ik_track_line(arm_offset, cur, (cur[0], cur[1], transit_z))
        cur = self.data.body(self._tcp_body[arm_offset]).xpos.copy()
        self._ik_track_line(arm_offset, cur, (x, y, transit_z), weighted=False)
        cur = self.data.body(self._tcp_body[arm_offset]).xpos.copy()
        return self._ik_track_line(arm_offset, cur, (x, y, z))

    def _go_home(self, arm_offset: int, *, steps: int = 150) -> None:
        """Return one arm to its verified HOME posture, jaw command unchanged.

        Every pick used to start from wherever the previous action left the
        arm. _move_to is an up-over-down waypoint move, so the *fingertip*
        arrives at the right place -- but the joint posture behind it (elbow,
        wrist) is inherited, and the descent then settles in a different IK
        basin. Measured 2026-09-13: spoon_1 held 10/10 in isolation and 3/10
        in the harness, where the right arm had just done the cup; the
        failing picks showed reach error 0.047 vs 0.032 and zero lift.
        Starting each manipulation from the same ready pose is ordinary robot
        practice and makes the harness behave like the isolated case.
        Interpolated, not slammed, so nothing gets flung.
        """
        import numpy as np

        target = np.array(HOME, dtype=float)
        # A posture reset, not a release: keep whatever jaw command is in
        # force. A failed MOVE restores physics to a snapshot in which the arm
        # is still holding the previous object; opening the jaw here dropped
        # that object at a random point along the homing path (measured
        # 2026-09-13: fork_1 10/10 -> 6/10 after a failed plate MOVE preceded
        # it). The pick opens the gripper explicitly, right after this.
        target[5] = float(self.data.ctrl[arm_offset + 5])
        start = self.data.ctrl[arm_offset:arm_offset + 6].copy()
        for k in range(1, steps + 1):
            self.data.ctrl[arm_offset:arm_offset + 6] = start + (target - start) * (k / steps)
            self._mujoco.mj_step(self.model, self.data)
        self._mujoco.mj_step(self.model, self.data, nstep=40)
        self._controller_steps += steps + 40
        # Wait for arrival, don't assume it. A place can leave the wrist roll
        # near its +2.79 limit (spoon at the right zone: 2.69 rad); the swing
        # back to HOME is over 4 rad and the servo is still travelling when
        # the fixed schedule ends, so the next pick started from the wrong
        # posture (cup after spoon: reach error 66 mm vs 38 mm, slipped at
        # 9 mm; measured 2026-09-14). Bounded: up to 0.8 s more.
        import numpy as np
        # Arrival = at the target AND slow. Position alone declared "arrived"
        # the instant a 4.3 rad roll swing crossed HOME at speed, and momentum
        # carried the wrist 1.25 rad past it into its hard limit during the
        # next settle (seed 905 spoon place, 2026-09-14: "joint-limit
        # proximity" on an arm that was commanded to HOME).
        for _ in range(600):
            err = np.abs(self.data.qpos[arm_offset:arm_offset + 5] - target[:5]).max()
            speed = np.abs(self.data.qvel[arm_offset:arm_offset + 5]).max()
            if err < 0.02 and speed < 0.3:
                break
            self._mujoco.mj_step(self.model, self.data)
            self._controller_steps += 1

    def _edge_seed_joints(self, arm_offset: int, tip_target, outward_xy) -> Any:
        """Joint configuration (5 arm joints) whose forward kinematics puts the
        fixed-jaw tip pad nearest ``tip_target`` with the finger horizontal,
        pointing along ``outward_xy``, fixed pad closing downward. A coarse
        forward-kinematic scan of pitch/elbow/wrist-pitch at roll 0, with the
        base joint aimed at the target; the 6D damped-least-squares solve then
        starts from a configuration on the right branch instead of from HOME
        (from HOME it converged 0.88 rad off every time, 2026-09-14).
        """
        import numpy as np
        import itertools

        mujoco = self._mujoco
        m = self.model
        prefix = "left_" if arm_offset == 0 else "right_"
        base = self.data.xpos[m.body(f"{prefix}Base").id].copy()
        pad1 = m.geom(f"{prefix}fixed_jaw_pad_1").id
        pad4 = self._pad_geom[arm_offset]
        cache = getattr(self, "_edge_scan_cache", None)
        if cache is None or arm_offset not in cache:
            scratch = mujoco.MjData(m)
            rng = m.jnt_range[arm_offset:arm_offset + 5]
            grid = [np.linspace(lo, hi, 33) for lo, hi in rng[1:4]]
            rows = []
            for pitch, elbow, wp in itertools.product(*grid):
                scratch.qpos[:] = self.data.qpos
                scratch.qpos[arm_offset:arm_offset + 5] = [0.0, pitch, elbow, wp, 0.0]
                mujoco.mj_kinematics(m, scratch)
                R = scratch.geom_xmat[pad4].reshape(3, 3)
                if abs(R[2, 1]) < 0.12 and R[2, 0] < -0.95:
                    tip = scratch.geom_xpos[pad1] - base
                    rows.append((pitch, elbow, wp, float(np.hypot(tip[0], tip[1])), float(tip[2]),
                                 float(np.arctan2(tip[1], tip[0]))))
            cache = getattr(self, "_edge_scan_cache", None) or {}
            cache[arm_offset] = np.array(rows)
            self._edge_scan_cache = cache
        rows = cache[arm_offset]
        tgt = np.asarray(tip_target, dtype=float)
        rel = tgt - base
        want_r, want_z = float(np.hypot(rel[0], rel[1])), float(rel[2])
        heading = float(np.arctan2(rel[1], rel[0]))
        err = np.hypot(rows[:, 3] - want_r, rows[:, 4] - want_z)
        best = rows[int(np.argmin(err))]
        # base joint: the scan was done at base=0, whose tip heading is best[5]
        base_q = heading - best[5]
        base_q = float(np.arctan2(np.sin(base_q), np.cos(base_q)))
        lo, hi = m.jnt_range[arm_offset]
        return np.array([float(np.clip(base_q, lo, hi)), best[0], best[1], best[2], 0.0]), float(err.min())

    def _topdown_seed_joints(self, arm_offset: int, tip_target, roll: float) -> Any:
        """Joint configuration whose forward kinematics puts the fixed-jaw tip
        pad nearest ``tip_target`` with the finger VERTICAL (pointing down).
        The top-down pinch solve for the cup settled in a tilted local
        minimum from HOME (fingertips 30-80 mm apart in height at closure,
        measured 2026-09-14) and lifted by leaning the fixed jaw on the wall;
        started from a posture that is already vertical above the target,
        the pose solve stays vertical."""
        import numpy as np
        import itertools

        mujoco = self._mujoco
        m = self.model
        prefix = "left_" if arm_offset == 0 else "right_"
        base = self.data.xpos[m.body(f"{prefix}Base").id].copy()
        pad1 = m.geom(f"{prefix}fixed_jaw_pad_1").id
        pad4 = self._pad_geom[arm_offset]
        cache = getattr(self, "_topdown_scan_cache", None) or {}
        if arm_offset not in cache:
            scratch = mujoco.MjData(m)
            rng = m.jnt_range[arm_offset:arm_offset + 5]
            grid = [np.linspace(lo, hi, 33) for lo, hi in rng[1:4]]
            rows = []
            for pitch, elbow, wp in itertools.product(*grid):
                scratch.qpos[:] = self.data.qpos
                scratch.qpos[arm_offset:arm_offset + 5] = [0.0, pitch, elbow, wp, roll]
                mujoco.mj_kinematics(m, scratch)
                R = scratch.geom_xmat[pad4].reshape(3, 3)
                if R[2, 1] > 0.95:   # pad frame y (tip->wrist) points up: finger vertical, tip down
                    tip = scratch.geom_xpos[pad1] - base
                    rows.append((pitch, elbow, wp, float(np.hypot(tip[0], tip[1])), float(tip[2]),
                                 float(np.arctan2(tip[1], tip[0]))))
            cache[arm_offset] = np.array(rows)
            self._topdown_scan_cache = cache
        rows = cache[arm_offset]
        if len(rows) == 0:
            return None, float("inf")
        rel = np.asarray(tip_target, dtype=float) - base
        want_r, want_z = float(np.hypot(rel[0], rel[1])), float(rel[2])
        heading = float(np.arctan2(rel[1], rel[0]))
        err = np.hypot(rows[:, 3] - want_r, rows[:, 4] - want_z)
        best = rows[int(np.argmin(err))]
        base_q = heading - best[5]
        base_q = float(np.arctan2(np.sin(base_q), np.cos(base_q)))
        lo, hi = m.jnt_range[arm_offset]
        return np.array([float(np.clip(base_q, lo, hi)), best[0], best[1], best[2], roll]), float(err.min())

    def _drive_joints(self, arm_offset: int, target_q, *, steps: int = 200) -> None:
        """Interpolate the arm's five joint commands to ``target_q`` (jaw
        command untouched) and let the servos follow -- like _go_home."""
        import numpy as np

        start = self.data.ctrl[arm_offset:arm_offset + 5].copy()
        target = np.asarray(target_q, dtype=float)
        for k in range(1, steps + 1):
            self.data.ctrl[arm_offset:arm_offset + 5] = start + (target - start) * (k / steps)
            self._mujoco.mj_step(self.model, self.data)
            self._controller_steps += 1

    def _edge_geometry(self, arm_offset: int, obj: str) -> dict[str, Any]:
        """Rim point facing this arm's base, the tip-pad frame for a sideways
        pinch, and the tip targets (under-lip grip, 6 cm outside, 6 cm up)."""
        import numpy as np

        lip_r, lip_dz, lip_half = EDGE_PINCH[obj]
        qpos_adr, _ = self._object_joints[obj]
        start_z = float(self.data.qpos[qpos_adr + 2])
        centre = self.data.qpos[qpos_adr:qpos_adr + 2].copy()
        prefix = "left" if arm_offset == 0 else "right"
        base_xy = self.data.xpos[self.model.body(f"{prefix}_Base").id][:2].copy()
        u = base_xy - centre
        u = u / (np.linalg.norm(u) + 1e-9)
        grip = np.array([centre[0] + u[0] * lip_r, centre[1] + u[1] * lip_r, start_z + lip_dz])
        pad_y = np.array([u[0], u[1], 0.0])
        tip_grip = grip - np.array([0.0, 0.0, lip_half + 0.005])   # fixed tip pad under the lip
        tip_approach = tip_grip + pad_y * 0.06
        return {
            "grip": grip, "pad_y": pad_y, "tip_grip": tip_grip, "tip_approach": tip_approach,
            "tip_high": tip_approach + np.array([0.0, 0.0, 0.06]),
            "tip_pad": self.model.geom(f"{prefix}_fixed_jaw_pad_1").id,
            "start_z": start_z, "qpos_adr": qpos_adr,
        }

    def _edge_waypoint(self, arm_offset: int, tip_pad: int, tip_target, pad_y, iters: int) -> tuple[float, float]:
        """Joint-space plan from the forward-kinematic scan (finger horizontal,
        roll 0), then a short position-only refinement of the fixed TIP pad
        with roll pinned. The scan gets the branch and the posture right; the
        refinement takes out its ~2 cm heading error without letting the
        finger pitch into the table."""
        q, scan_err = self._edge_seed_joints(arm_offset, tip_target, pad_y[:2])
        self._drive_joints(arm_offset, q)
        err = self._ik_reach_pad(arm_offset, tip_target, iters=iters, roll=0.0,
                                 track_tcp=False, geom_id=tip_pad, tol=0.004, max_dq=0.02)
        return err, scan_err

    def _edge_approach_and_pinch(self, arm_offset: int, obj: str) -> dict[str, Any]:
        """One arm: home, part-open, high -> outside the rim -> slide the fixed
        tip under the lip, close. Returns the errors and the grasp sensor."""
        import numpy as np

        g = self._edge_geometry(arm_offset, obj)
        self._go_home(arm_offset)
        self._set_gripper(arm_offset, EDGE_PINCH_APPROACH_JAW_RAD)
        high_err, seed_err = self._edge_waypoint(arm_offset, g["tip_pad"], g["tip_high"], g["pad_y"], 120)
        approach_err, _ = self._edge_waypoint(arm_offset, g["tip_pad"], g["tip_approach"], g["pad_y"], 160)
        slide_err, _ = self._edge_waypoint(arm_offset, g["tip_pad"], g["tip_grip"], g["pad_y"], 200)
        sensor = self._set_gripper(arm_offset, GRIPPER_CLOSED, settle_steps=180, obj=obj)
        return {
            "geometry": g, "seed_scan_error_m": round(seed_err, 4),
            "approach_error_m": round(float(approach_err), 6), "reach_error_m": round(float(slide_err), 6),
            "grasp_sensor": sensor, "grip_point": [round(float(v), 4) for v in g["grip"]],
        }

    def _drive_joints_both(self, targets: dict[int, Any], *, steps: int = 300) -> None:
        """Interpolate several arms' five joint commands together, one physics
        step per increment, so two arms holding one object move in lockstep."""
        import numpy as np

        starts = {a: self.data.ctrl[a:a + 5].copy() for a in targets}
        ends = {a: np.asarray(t, dtype=float) for a, t in targets.items()}
        for k in range(1, steps + 1):
            for a in targets:
                self.data.ctrl[a:a + 5] = starts[a] + (ends[a] - starts[a]) * (k / steps)
            self._mujoco.mj_step(self.model, self.data)
            self._controller_steps += 1

    def _object_tilt(self, obj: str) -> float:
        """Angle (rad) between the object's +z and world up."""
        import numpy as np

        qpos_adr, _ = self._object_joints[obj]
        qw, qx, qy, qz = (float(v) for v in self.data.qpos[qpos_adr + 3:qpos_adr + 7])
        zz = 1.0 - 2.0 * (qx * qx + qy * qy)   # R[2,2]
        return float(np.arccos(np.clip(zz, -1.0, 1.0)))

    def _do_open_drawer(self, arm_offset: int, drawer: str) -> dict[str, Any]:
        """Pull a drawer tray open by its handle (2026-09-14): top-down pinch
        across the 8 mm handle bar (the same pinch the cutlery uses), then a
        straight pull along the slide axis (+y, toward the arm) by the
        drawer's travel, release, withdraw. Verified by the slide joint
        reading, the same observable the symbolic OPEN was verified by."""
        import numpy as np

        mujoco = self._mujoco
        handle = self.model.geom(f"{drawer}_handle").id
        adr = self._drawer_slide_adr[drawer]
        travel = float(self.model.jnt_range[self.model.joint(f"{drawer}_slide").id][1])
        start = float(self.data.qpos[adr])
        hpos = self.data.geom_xpos[handle].copy()
        # pads close along y (across the bar's 8 mm); finger vertical
        opening = np.array([0.0, -1.0, 0.0]) if arm_offset == 0 else np.array([0.0, 1.0, 0.0])
        pad_x = -opening
        pad_y = np.array([0.0, 0.0, 1.0])
        target_rotation = np.column_stack((pad_x, pad_y, np.cross(pad_x, pad_y)))
        # the bar lies along x, like cutlery at yaw +/-90 deg: base roll minus
        # +pi/2 on the left, minus -pi/2 on the mirrored right arm
        yaw = math.pi / 2 if arm_offset == 0 else -math.pi / 2
        roll_hint = float(np.clip(self._GRASP_WRIST_ROLL[arm_offset] - yaw, -1.62, 1.62))
        clear_z = hpos[2] + 0.005 + GRASP_CLEARANCE + 0.02
        self._go_home(arm_offset)
        self._set_pick_track(arm_offset, True)
        self._set_gripper(arm_offset, GRIPPER_OPEN)
        # start from a scanned vertical-finger posture above the handle (the
        # plain solve arrived tilted and landed 36 mm along the bar)
        seed_q, _ = self._topdown_seed_joints(arm_offset, (hpos[0], hpos[1], clear_z + 0.03), roll_hint)
        if seed_q is not None:
            self._drive_joints(arm_offset, seed_q, steps=250)
        self._move_to(arm_offset, (hpos[0], hpos[1], clear_z))
        self._ik_reach_pad(arm_offset, (hpos[0], hpos[1], clear_z), iters=200, roll=roll_hint)
        target_rotation = self.data.geom_xmat[self._pad_geom[arm_offset]].reshape(3, 3).copy()
        self._ik_reach_pad_pose(arm_offset, (hpos[0], hpos[1], clear_z), target_rotation, roll_hint=roll_hint, iters=200)
        pinch_err = self._ik_reach_pad(arm_offset, (hpos[0], hpos[1], hpos[2]), iters=200, roll=roll_hint)
        sensor = self._set_gripper(arm_offset, GRIPPER_CLOSED, settle_steps=150, obj=drawer)
        # pull: track the fingertips +y in 10 mm steps
        tip_now = None
        pull_err = 0.0
        for k in range(1, int(round(travel / 0.01)) + 2):
            goal = np.array([hpos[0], hpos[1] + min(travel + 0.01, 0.01 * k), hpos[2]])
            pull_err = self._ik_reach_pad(arm_offset, goal, iters=60, roll=roll_hint, max_dq=0.02)
        for _ in range(60):
            mujoco.mj_step(self.model, self.data)
            self._controller_steps += 1
        opened_by = float(self.data.qpos[adr]) - start
        self._set_gripper(arm_offset, GRIPPER_OPEN, settle_steps=60)
        self._ik_reach_pad(arm_offset, (hpos[0], hpos[1] + travel, hpos[2] + 0.06), iters=120, roll=roll_hint)
        self._go_home(arm_offset)
        final = float(self.data.qpos[adr])
        opened = abs(final - DRAWER_OPEN) <= 0.008
        return {
            "grasp": "drawer_pull", "drawer": drawer, "opened": opened,
            "slide_m": round(final, 4), "opened_by_m": round(opened_by, 4),
            "pinch_error_m": round(float(pinch_err), 4), "pull_error_m": round(float(pull_err), 4),
            "grasp_sensor": sensor,
            "reason": None if opened else f"drawer pulled {opened_by * 1000:.0f} mm of {travel * 1000:.0f}",
        }

    def _do_pick_edge(self, arm_offset: int, obj: str) -> dict[str, Any]:
        """Sideways rim pinch, one arm (2026-09-14): for tableware whose
        graspable feature is a lip standing off the table. The gripper comes
        in horizontally from outside the rim, finger axis pointing back toward
        the arm, jaw partly open; the thin fixed fingertip slides under the
        lip, the moving jaw closes down on top; then lift. Same receipt shape
        as the top-down pick. Objects in BIMANUAL_OBJECTS never get here.
        """
        import numpy as np

        attempt_snapshot = (
            self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(),
            float(self.data.time), self._controller_steps,
        )
        g = self._edge_geometry(arm_offset, obj)
        safety = self._workspace_safety(arm_offset, target_xy=g["grip"][:2])
        if not safety["safe"]:
            return {"grasp": "edge", "reach_error_m": None, "lift_height_m": 0.0, "held": False,
                    "reason": safety["reason"], "safety": {"approach": safety}}
        self._set_pick_track(arm_offset, False)
        pinch = self._edge_approach_and_pinch(arm_offset, obj)
        lift_target = g["tip_grip"] + np.array([0.0, 0.0, LIFT_VERIFY_MIN + 0.02])
        lift_q, lift_scan_err = self._edge_seed_joints(arm_offset, lift_target, g["pad_y"][:2])
        self._drive_joints(arm_offset, lift_q, steps=300)
        lift_err = self._ik_reach_pad(arm_offset, lift_target, iters=120, roll=0.0, track_tcp=False,
                                      geom_id=g["tip_pad"], tol=0.006, max_dq=0.02)
        lift = float(self.data.qpos[g["qpos_adr"] + 2]) - g["start_z"]
        held = lift >= LIFT_VERIFY_MIN
        result = {
            "grasp": "edge", "lift_error_m": round(float(lift_err), 6), "lift_height_m": round(lift, 4),
            "tilt_rad": round(self._object_tilt(obj), 4), "held": held, "retries": [],
            "reason": None if held else "edge pinch did not lift", "safety": {"approach": safety},
            **{k: v for k, v in pinch.items() if k != "geometry"},
        }
        if not held:
            self._set_gripper(arm_offset, GRIPPER_OPEN, settle_steps=40)
            self._restore_physics(attempt_snapshot, arm_offset=arm_offset, obj=obj)
        return result

    def _do_pick_bimanual(self, obj: str) -> dict[str, Any]:
        """Two-arm rim pinch and cooperative lift (2026-09-14). Left and right
        each take the rim point facing their own base with the sideways
        pinch, then both arms drive to their lifted postures in lockstep so
        the plate rises flat. Held = it rose LIFT_VERIFY_MIN and stayed
        within BIMANUAL_MAX_TILT_RAD of level. Both grippers keep holding
        until _do_place_bimanual releases them together.
        """
        import numpy as np

        attempt_snapshot = (
            self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(),
            float(self.data.time), self._controller_steps,
        )
        per_arm: dict[str, Any] = {}
        for arm_offset in (0, 6):
            g = self._edge_geometry(arm_offset, obj)
            safety = self._workspace_safety(arm_offset, target_xy=g["grip"][:2])
            if not safety["safe"]:
                return {"grasp": "bimanual_edge", "reach_error_m": None, "lift_height_m": 0.0, "held": False,
                        "reason": f"{'left' if arm_offset == 0 else 'right'}: {safety['reason']}",
                        "safety": {"approach": safety}, "arms": per_arm}
            self._set_pick_track(arm_offset, False)
            pinch = self._edge_approach_and_pinch(arm_offset, obj)
            per_arm["left" if arm_offset == 0 else "right"] = {k: v for k, v in pinch.items() if k != "geometry"}
            per_arm["left" if arm_offset == 0 else "right"]["_geom"] = pinch["geometry"]
        start_z = per_arm["left"]["_geom"]["start_z"]
        qpos_adr = per_arm["left"]["_geom"]["qpos_adr"]
        # Cooperative lift: both arms driven together to their scanned
        # postures LIFT_VERIFY_MIN + 20 mm up. (Tracking both tips straight up
        # with the 4-DOF IK was tried 2026-09-14 and does not climb from this
        # near-extended posture -- 0 mm on four seeds; the joint-space drive
        # lifts 66-71 mm; a shorter, slower drive measured worse: 17-55 mm.)
        lift_targets = {}
        for name, arm_offset in (("left", 0), ("right", 6)):
            g = per_arm[name]["_geom"]
            lift_tip = g["tip_grip"] + np.array([0.0, 0.0, LIFT_VERIFY_MIN + 0.02])
            q, scan_err = self._edge_seed_joints(arm_offset, lift_tip, g["pad_y"][:2])
            lift_targets[arm_offset] = q
            per_arm[name]["lift_scan_error_m"] = round(scan_err, 4)
        self._drive_joints_both(lift_targets, steps=400)
        for _ in range(60):
            self._mujoco.mj_step(self.model, self.data)
            self._controller_steps += 1
        lift = float(self.data.qpos[qpos_adr + 2]) - start_z
        tilt = self._object_tilt(obj)
        held = lift >= LIFT_VERIFY_MIN and tilt <= BIMANUAL_MAX_TILT_RAD
        pads = sum(per_arm[n]["grasp_sensor"]["contact_pad_count"] for n in ("left", "right"))
        for n in ("left", "right"):
            per_arm[n].pop("_geom", None)
        result = {
            "grasp": "bimanual_edge", "arms": per_arm,
            "reach_error_m": max(per_arm[n]["reach_error_m"] for n in ("left", "right")),
            "lift_height_m": round(lift, 4), "tilt_rad": round(tilt, 4), "held": held,
            "grasp_sensor": {"contact_pad_count": pads,
                             "max_contact_force_n": max(per_arm[n]["grasp_sensor"]["max_contact_force_n"] for n in ("left", "right")),
                             "jaw_commanded_rad": GRIPPER_CLOSED,
                             "jaw_actual_rad": {n: per_arm[n]["grasp_sensor"]["jaw_actual_rad"] for n in ("left", "right")}},
            "retries": [],
            "reason": None if held else ("plate tilted in the pinch" if lift >= LIFT_VERIFY_MIN else "bimanual pinch did not lift"),
        }
        if not held:
            self._set_gripper(0, GRIPPER_OPEN, settle_steps=40)
            self._set_gripper(6, GRIPPER_OPEN, settle_steps=40)
            self._restore_physics(attempt_snapshot)
        return result

    def _do_place_bimanual(self, obj: str, to_zone: str | None) -> dict[str, Any]:
        """Carry the two-arm-held object level to ``to_zone`` and set it down:
        both fingertips are tracked in lockstep along a straight line (the
        plate is rigid, so the tips must translate identically), lowered to
        the resting height, released together, and both arms retract."""
        import numpy as np

        target = ZONE_POSITIONS.get(to_zone)
        if target is None:
            return {"grasp": "bimanual_edge", "placed": False, "reason": f"unknown zone {to_zone!r}"}
        qpos_adr, _ = self._object_joints[obj]
        lip_r, lip_dz, lip_half = EDGE_PINCH[obj]
        centre_now = self.data.qpos[qpos_adr:qpos_adr + 3].copy()
        tips = {a: self.model.geom(f"{'left' if a == 0 else 'right'}_fixed_jaw_pad_1").id for a in (0, 6)}
        tip_now = {a: self.data.geom_xpos[tips[a]].copy() for a in (0, 6)}
        # plate centre path: horizontal at the current height, then straight down
        rest_centre_z = float(target[2])
        goal_high = np.array([target[0], target[1], centre_now[2]])
        goal_low = np.array([target[0], target[1], rest_centre_z])

        # Lockstep IK carry: both fingertips tracked along the straight line,
        # 12 waypoints out and 6 down. Measured 2026-09-14 over two 10-trial
        # harness runs: plate placed 19/20 with this. Two alternatives were
        # tried the same day and were worse -- a scan-planned joint-space
        # drive (tips arrive, plate follows 2 cm of 6: it slips in the
        # pinches while the arms' arcs diverge) and 1 cm scan-planned
        # increments (both arms pull the plate toward their own bases at
        # every re-plan). The tips lag the line here (carry error ~60 mm)
        # but the plate stays with them.
        def track_both(delta_total, n_steps, iters):
            errs = []
            for k in range(1, n_steps + 1):
                d = delta_total * (k / n_steps)
                for a in (0, 6):
                    errs.append(self._ik_reach_pad(a, tip_now[a] + d, iters=iters, roll=0.0, track_tcp=False,
                                                   geom_id=tips[a], tol=0.003, max_dq=0.015))
            return max(errs[-2:]) if errs else 0.0

        carry_err = track_both(goal_high - centre_now, 12, 14)
        lower_err = track_both(goal_low - centre_now, 6, 14)
        self._set_gripper(0, GRIPPER_OPEN, settle_steps=30)
        self._set_gripper(6, GRIPPER_OPEN, settle_steps=30)
        # retract both straight up and out
        for a in (0, 6):
            self._ik_reach_pad(a, self.data.geom_xpos[tips[a]] + np.array([0.0, 0.0, 0.05]), iters=80, roll=0.0,
                               track_tcp=False, geom_id=tips[a], tol=0.005, max_dq=0.02)
        # Both arms back to the ready pose: the extended sideways posture
        # sits near a wrist-pitch limit, and every later step on either arm
        # failed its joint-margin safety check from there (engine trace,
        # seeds 900-903: fork MOVE "joint-limit proximity" right after the
        # plate). The jaws are open; nothing is held.
        self._go_home(0)
        self._go_home(6)
        settle = self._settle_released_object(obj)
        final_xy = self.data.qpos[qpos_adr:qpos_adr + 2].copy()
        offset = float(np.linalg.norm(final_xy - np.asarray(target[:2])))
        tilt = self._object_tilt(obj)
        safety_after = {"safe": True, "reason": None}
        for a in (0, 6):
            sa = self._workspace_safety(a, obj=None)
            if not sa["safe"]:
                safety_after = sa
        placed = bool(offset < 0.06 and settle["settled"] and tilt <= BIMANUAL_MAX_TILT_RAD and safety_after["safe"])
        return {
            "grasp": "bimanual_edge", "reach_error_m": round(float(lower_err), 6),
            "carry_error_m": round(float(carry_err), 6),
            "placement_error_m": round(offset, 4), "tilt_rad": round(tilt, 4), "placed": placed,
            "settle": settle, "safety": {"after": safety_after},
            "reason": None if placed else (safety_after["reason"] or (
                "placement offset" if offset >= 0.06 else ("tilted" if tilt > BIMANUAL_MAX_TILT_RAD else "unstable placement"))),
        }

    def _do_pick(self, arm_offset: int, obj: str) -> dict[str, Any]:
        """Approach from above (via a safe transit height, unweighted-vs-table
        motion), then switch to the fixed-wrist-roll pad-tracking solve for
        the actual pinch (see ``_ik_reach_pad``) -- close the gripper on the
        real object, lift, and report whether it's actually being carried
        (measured height gain), not just whether the motion finished."""
        import numpy as np

        if obj in BIMANUAL_OBJECTS:
            return self._do_pick_bimanual(obj)
        if obj in EDGE_PINCH:
            return self._do_pick_edge(arm_offset, obj)
        target_rotation, roll_hint = self._grasp_frame(arm_offset, obj)
        attempt_snapshot = (
            self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(),
            float(self.data.time), self._controller_steps,
        )
        qpos_adr, _ = self._object_joints[obj]
        start_z = float(self.data.qpos[qpos_adr + 2])
        xy = self.data.qpos[qpos_adr:qpos_adr + 2].copy()
        half_h = OBJECT_HALF_HEIGHT.get(obj, 0.01)
        clear_z = start_z + half_h + GRASP_CLEARANCE  # just above the object's real top
        opening_xy = -target_rotation[:2, 0]
        grasp_offset = OBJECT_GRASP_OFFSET.get(obj, 0.0)
        grasp_xy = xy - opening_xy * grasp_offset
        along_offset = OBJECT_GRASP_ALONG.get(obj, 0.0)
        if along_offset:
            # Along the body's +y (toward the head), from the live yaw: grip
            # at the centre of mass, not the handle's geometric middle.
            raw_yaw = self._object_yaw(obj)
            grasp_xy = grasp_xy + np.array([-math.sin(raw_yaw), math.cos(raw_yaw)]) * along_offset
        approach_safety = self._workspace_safety(
            arm_offset, target_xy=grasp_xy,
        )
        if not approach_safety["safe"]:
            return {
                "grasp": "contact",
                "reach_error_m": None,
                "lift_height_m": 0.0,
                "held": False,
                "reason": approach_safety["reason"],
                "safety": {"approach": approach_safety},
            }

        self._go_home(arm_offset)          # same ready pose for every pick; see _go_home
        self._set_pick_track(arm_offset, self._track_tcp_for(obj))
        open_to = WALL_PINCH_OBJECTS[obj][0] if obj in WALL_PINCH_OBJECTS else GRIPPER_OPEN
        self._set_gripper(arm_offset, open_to)
        if obj in TOPDOWN_SEED_OBJECTS:
            # Vertical-finger posture above the grasp point, from the scan,
            # before the up-over-down move (see _topdown_seed_joints).
            seed_q, _ = self._topdown_seed_joints(
                arm_offset, (grasp_xy[0], grasp_xy[1], clear_z + 0.03), roll_hint)
            if seed_q is not None:
                self._drive_joints(arm_offset, seed_q, steps=250)
        self._move_to(arm_offset, (grasp_xy[0], grasp_xy[1], clear_z))

        # Object-specific escape from a confirmed differential-IK local
        # minimum (see OBJECT_GRASP_SEED_BIAS's comment above). This is a
        # fixed, zero-search-cost nudge -- not the dense per-attempt seed
        # grid the "Fifth update" module note declined to wire in -- so that
        # cost objection doesn't apply here; only the fragility one might,
        # and that's checked empirically below rather than assumed.
        elbow_bias, wrist_pitch_bias = OBJECT_GRASP_SEED_BIAS.get(obj, (0.0, 0.0))
        if elbow_bias or wrist_pitch_bias:
            self.data.qpos[arm_offset + 2] += elbow_bias
            self.data.qpos[arm_offset + 3] += wrist_pitch_bias
            self.data.ctrl[arm_offset:arm_offset + 6] = self.data.qpos[arm_offset:arm_offset + 6]
            self._mujoco.mj_forward(self.model, self.data)

        # First locate the actual pad with the bounded object-aware roll hint,
        # then hold that measured, reachable frame while descending. Asking
        # the small five-joint chain for an arbitrary world orientation would
        # over-constrain the proxy; the measured frame preserves the physical
        # wrist convention while still making yaw a live input.
        self._ik_reach_pad(
            arm_offset, (grasp_xy[0], grasp_xy[1], clear_z),
            iters=220, roll=roll_hint,
        )
        target_rotation = self.data.geom_xmat[self._pad_geom[arm_offset]].reshape(3, 3).copy()
        clear_pose = self._ik_reach_pad_pose(
            arm_offset, (grasp_xy[0], grasp_xy[1], clear_z), target_rotation,
            roll_hint=roll_hint, iters=220,
        )
        xy_now = self.data.qpos[qpos_adr:qpos_adr + 2].copy()
        z_now = float(self.data.qpos[qpos_adr + 2])
        grasp_z = z_now + OBJECT_GRASP_VERTICAL_OFFSET.get(obj, 0.0)
        grasp_xy = xy_now - opening_xy * grasp_offset
        self._descend_contact_stop = None
        pinch_pose = self._ik_reach_pad_pose(
            arm_offset, (grasp_xy[0], grasp_xy[1], grasp_z), target_rotation,
            roll_hint=roll_hint, iters=80, stop_on_contact=((obj, DESCEND_CONTACT_STOP_N) if DESCEND_CONTACT_STOP_N > 0 else None),
        )
        # The final vertical approach is position-dominant: the reachable
        # orientation frame is already established above, and this 4-DOF
        # roll-pinned solve avoids twisting a pad into the object while the
        # jaws are entering contact.
        # Track the FINGERTIP midpoint for the final descent, not
        # fixed_jaw_pad_4. The jaws close as a wedge -- 3.6 mm at the tip,
        # 19.3 mm at pad_4 -- so a thin object can only ever be pinched at the
        # tip, and pad_4 sits ~43 mm further up the finger besides. Opt-in
        # while the downstream constants are re-tuned around it.
        # pinch_pose above is the first solve to reach grasp height; keep its
        # contact-stop reading if it fired, and guard this final descent too.
        pinch_error = self._ik_reach_pad(
            arm_offset, (grasp_xy[0], grasp_xy[1], grasp_z),
            iters=180, roll=roll_hint, stop_on_contact=((obj, DESCEND_CONTACT_STOP_N) if DESCEND_CONTACT_STOP_N > 0 else None),
        )

        grasp_sensor = self._set_gripper(arm_offset, GRIPPER_CLOSED, settle_steps=180, obj=obj)  # stall-aware; let the grip settle
        grasp_sensor["descend_stopped_by_contact_n"] = self._descend_contact_stop
        lift_rise = max(clear_z - start_z, LIFT_VERIFY_MIN)
        lift_error = self._ik_reach_pad(
            arm_offset, (xy_now[0], xy_now[1], grasp_z + lift_rise),
            iters=280, roll=roll_hint,
        )

        lifted_z = float(self.data.qpos[qpos_adr + 2])
        lift = lifted_z - start_z
        attempts_log: list[dict[str, Any]] = []
        lift_pose: dict[str, Any] = {"position_error_m": round(lift_error, 6), "retry_reason": "first_attempt"}
        if lift <= 0.02:
            # A bounded orientation search is still physical: each candidate
            # starts from the exact pre-attempt simulator state and is judged
            # by lift height.  This handles mirrored jaw conventions and
            # small object-yaw errors without writing the object freejoint.
            # Evidence-driven retry. The previous attempt's grasp_sensor says
            # *what kind* of failure it was, and each kind wants a different
            # correction -- the fixed roll list this replaces tried the same
            # four wrist rolls whether the pads had touched nothing or had
            # gripped and slipped. Every retry re-reads the object's live pose
            # (it may have been nudged) and is bounded to four attempts.
            #   no contact      -> fingertips stopped above the object: descend
            #                      deeper, same roll
            #   contact, no lift-> gripped but pivoted/slipped: regrasp toward
            #                      the head (balance point), squeeze deeper
            #   stalled wide    -> came down ON the object: back off and shift
            #   otherwise       -> roll alternatives (mirrored jaw, +/-0.25)
            raw_yaw = self._object_yaw(obj)
            along = np.array([-math.sin(raw_yaw), math.cos(raw_yaw)])  # +y of the body: toward the head
            base_roll = self._GRASP_WRIST_ROLL[arm_offset]
            roll_alternatives = [base_roll, -base_roll, roll_hint - 0.25, roll_hint + 0.25]
            attempts_log: list[dict[str, Any]] = []
            tried: list[tuple[float, float, float]] = []
            last_sensor = grasp_sensor
            last_lift = lift
            for _attempt in range(4):
                pads = int(last_sensor.get("contact_pad_count", 0))
                stalled_wide = bool(last_sensor.get("jaw_stalled")) and (
                    last_sensor.get("jaw_stall_angle_rad") or 0.0) > 0.5
                if stalled_wide:
                    reason, roll_c, dz, shift = "stalled_on_top", roll_hint, +0.006, 0.015
                elif pads == 0:
                    reason, roll_c, dz, shift = "no_contact_descend", roll_hint, -0.004, 0.0
                elif last_lift < 0.02:
                    reason, roll_c, dz, shift = "slipped_regrasp_toward_head", roll_hint, -0.002, 0.02
                else:
                    reason, roll_c, dz, shift = "roll_alternative", roll_hint, 0.0, 0.0
                # If this exact correction was already tried, fall through the
                # roll alternatives instead of repeating it.
                key = (round(roll_c, 3), round(dz, 4), round(shift, 4))
                if key in tried:
                    reason = "roll_alternative"
                    remaining = [r for r in roll_alternatives
                                 if all(abs(float(np.clip(r, -1.65, 1.65)) - t[0]) > 1e-5 for t in tried)]
                    if not remaining:
                        break
                    roll_c, dz, shift = remaining[0], 0.0, 0.0
                    key = (round(roll_c, 3), round(dz, 4), round(shift, 4))
                roll_c = float(np.clip(roll_c, -1.65, 1.65))
                tried.append(key)

                self._restore_physics(attempt_snapshot, arm_offset=arm_offset, obj=obj)
                self._set_gripper(arm_offset, open_to)
                self._move_to(arm_offset, (grasp_xy[0], grasp_xy[1], clear_z))
                if elbow_bias or wrist_pitch_bias:
                    self.data.qpos[arm_offset + 2] += elbow_bias
                    self.data.qpos[arm_offset + 3] += wrist_pitch_bias
                    self.data.ctrl[arm_offset:arm_offset + 6] = self.data.qpos[arm_offset:arm_offset + 6]
                    self._mujoco.mj_forward(self.model, self.data)
                self._ik_reach_pad(
                    arm_offset, (grasp_xy[0], grasp_xy[1], clear_z),
                    iters=220, roll=roll_c,
                )
                target_rotation = self.data.geom_xmat[self._pad_geom[arm_offset]].reshape(3, 3).copy()
                self._ik_reach_pad_pose(
                    arm_offset, (xy[0], xy[1], clear_z), target_rotation,
                    roll_hint=roll_c, iters=120,
                )
                xy_retry = self.data.qpos[qpos_adr:qpos_adr + 2].copy()  # re-observe: it may have moved
                z_retry = float(self.data.qpos[qpos_adr + 2])
                grasp_z_retry = z_retry + OBJECT_GRASP_VERTICAL_OFFSET.get(obj, 0.0) + dz
                grasp_xy_retry = xy_retry - opening_xy * grasp_offset + along * shift
                self._descend_contact_stop = None
                self._ik_reach_pad(
                    arm_offset, (grasp_xy_retry[0], grasp_xy_retry[1], grasp_z_retry),
                    iters=180, roll=roll_c, stop_on_contact=((obj, DESCEND_CONTACT_STOP_N) if DESCEND_CONTACT_STOP_N > 0 else None),
                )
                retry_sensor = self._set_gripper(arm_offset, GRIPPER_CLOSED, settle_steps=180, obj=obj)
                retry_sensor["descend_stopped_by_contact_n"] = self._descend_contact_stop
                self._ik_reach_pad(
                    arm_offset,
                    (grasp_xy_retry[0], grasp_xy_retry[1], grasp_z_retry + lift_rise),
                    iters=280, roll=roll_c,
                )
                retry_lift = float(self.data.qpos[qpos_adr + 2]) - start_z
                attempts_log.append({
                    "reason": reason, "roll_rad": round(roll_c, 4), "dz_m": dz,
                    "shift_toward_head_m": shift, "lift_m": round(retry_lift, 4),
                    "contact_pad_count": retry_sensor["contact_pad_count"],
                    "max_contact_force_n": retry_sensor["max_contact_force_n"],
                })
                last_sensor, last_lift = retry_sensor, retry_lift
                if retry_lift > lift:
                    grasp_sensor = retry_sensor  # evidence from the attempt that actually won
                    lift = retry_lift
                    lift_pose = {"position_error_m": round(lift_error, 6), "retry_roll_rad": round(roll_c, 6),
                                 "retry_reason": reason}
                    roll_hint = roll_c
                if lift > 0.02:
                    break
        return {
            "grasp": "contact", "reach_error_m": round(pinch_error, 6),
            "lift_height_m": round(lift, 4), "held": lift > 0.02,
            # What the actuator and the contact buffer reported at the close:
            # jaw stall angle (a jaw that reached its commanded closed angle
            # closed on nothing) and every pad/object contact with its force.
            # This is the producer side of skills/verification/grasp.py's
            # GraspEvidence -- contact_pad_count and max_contact_force_n map
            # straight across.
            "grasp_sensor": grasp_sensor,
            # Every retry, with the evidence that chose it. This is the
            # primitive updating its own plan from observation rather than
            # cycling a fixed list; the engine-level replan (OQ-018) sits
            # above it.
            "retries": attempts_log if lift <= 0.02 or attempts_log else [],
            "orientation": {
                "clearance": clear_pose,
                "pinch": pinch_pose,
                "lift": lift_pose,
                "roll_hint_rad": round(roll_hint, 6),
            },
            "safety": {"approach": approach_safety},
        }

    def _do_place(self, arm_offset: int, obj: str, to_zone: str | None) -> dict[str, Any]:
        """Carry (still gripping -- no teleport) to the target zone's real
        table position, release, and report whether it actually ended up
        there, not just whether the arm reached the coordinate.

        No ``_move_to`` call here: by the time PLACE runs, _do_pick's lift
        already left the arm elevated at a safe height with wrist-roll
        pinned (the planner always sequences PICK before MOVE/PLACE for the
        same object, and a failed PICK never reaches here). Switching back
        to the body-tracked, roll-unpinned solve mid-carry would let the
        pinch orientation drift and likely drop the object -- everything
        here stays on ``_ik_reach_pad``, horizontal first at the current
        height, then descend onto the target."""
        import numpy as np

        if obj in BIMANUAL_OBJECTS:
            return self._do_place_bimanual(obj, to_zone)
        target = ZONE_POSITIONS.get(to_zone)
        if target is None:
            return {"grasp": "contact", "placed": False, "reason": f"unknown zone {to_zone!r}"}
        target_arr = np.asarray(target)  # z here is the object's intended resting *centre* height
        qpos_adr, _ = self._object_joints[obj]
        half_h = OBJECT_HALF_HEIGHT.get(obj, 0.01)
        clear = np.array([0.0, 0.0, half_h + GRASP_CLEARANCE])
        # Preserve the measured grasp frame while carrying; re-solving toward
        # an arbitrary world orientation here can twist a live contact and
        # turn a valid hold into a drop.
        target_rotation = self.data.geom_xmat[self._pad_geom[arm_offset]].reshape(3, 3).copy()
        roll_hint = float(self.data.qpos[arm_offset + 4])

        pad_position = self.data.geom_xpos[self._pad_geom[arm_offset]].copy()
        object_position = self.data.qpos[qpos_adr:qpos_adr + 3].copy()
        carry_offset = object_position - pad_position
        safety_before = self._workspace_safety(
            arm_offset, obj=obj, target_xy=target_arr,
        )
        if not safety_before["safe"]:
            return {
                "grasp": "contact", "placed": False,
                "reason": safety_before["reason"],
                "safety": {"before": safety_before},
            }

        cur_pad_z = float(self.data.geom_xpos[self._pad_geom[arm_offset]][2])
        # A broad, thin plate must be released above its rim. Driving a pad
        # target through the plate centre makes contact solver impulses push
        # the plate sideways before the jaws open; the other fixtures can be
        # released at their centre-height target.
        release_target = target_arr.copy()
        if obj == "plate_1":
            release_target[2] += half_h + GRASP_CLEARANCE
        # Carry contact is not a weld: the object can settle a few centimetres
        # away from the pad while the arm moves.  Command the pad to the
        # measured object-relative offset, so the object—not the pad—arrives
        # at the requested zone.  This is an observed correction, not a write
        # to the object's freejoint pose.
        pad_release_target = release_target - carry_offset
        self._ik_reach_pad_pose(
            arm_offset, (pad_release_target[0], pad_release_target[1], cur_pad_z),
            target_rotation, roll_hint=roll_hint, iters=240,
        )
        place_pose = self._ik_reach_pad_pose(
            arm_offset, pad_release_target, target_rotation,
            roll_hint=roll_hint, iters=300,
        )
        # let the carried object stop swinging before the release
        for _ in range(80):
            self._mujoco.mj_step(self.model, self.data)
            self._controller_steps += 1
        place_error = float(place_pose["position_error_m"])
        # Wall pinch: opening fully from INSIDE the cup swings the moving
        # finger across to the far inner wall (121 mm open vs 54 mm bore) and
        # shoves the cup 60-120 mm (2026-09-14). Open only to the descent
        # angle, lift clear of the rim, then open fully.
        release_open = WALL_PINCH_OBJECTS[obj][0] if obj in WALL_PINCH_OBJECTS else GRIPPER_OPEN
        self._set_gripper(arm_offset, release_open, settle_steps=40)  # let it drop/settle
        retract_pose = self._ik_reach_pad_pose(
            arm_offset, pad_release_target + clear, target_rotation,
            roll_hint=roll_hint, iters=240,
        )  # retract straight up
        lift_error = float(retract_pose["position_error_m"])
        if release_open != GRIPPER_OPEN:
            self._set_gripper(arm_offset, GRIPPER_OPEN, settle_steps=30)   # now clear of the rim

        # Withdraw to the ready pose after the release (2026-09-14): the arm
        # used to stay retracted right above the object it had just set down,
        # and the overhead camera could not see the napkin under the left
        # gripper -- the camera end-to-end reported "not verified" for a
        # placement that was fine. Moving out of the cameras' way before
        # verification is what a robot does; the next pick homes anyway.
        self._go_home(arm_offset)
        settle = self._settle_released_object(obj)

        final_xy = self.data.qpos[qpos_adr:qpos_adr + 2].copy()
        offset = float(np.linalg.norm(final_xy - target_arr[:2]))
        safety_after = self._workspace_safety(arm_offset, obj=None)
        placed = bool(offset < 0.06 and settle["settled"] and safety_after["safe"])
        return {
            "grasp": "contact", "reach_error_m": round(place_error, 6),
            "placement_error_m": round(offset, 4), "placed": placed,
            "settle": settle,
            "safety": {"before": safety_before, "after": safety_after},
            "reason": None if placed else (
                safety_after["reason"] or "unstable placement"
            ),
            "orientation": {
                "place": {**place_pose},
                "retract": {"position_error_m": round(lift_error, 6)},
                "roll_hint_rad": round(roll_hint, 6),
            },
        }

    def simulation_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "nq": self.model.nq,
            "nv": self.model.nv,
            "nu": self.model.nu,
            "cameras": self.model.ncam,
            "time": float(self.data.time),
            "controller_steps": self._controller_steps,
            "drawer_qpos": round(float(self.data.qpos[self._drawer_qpos_adr]), 4),
        }


class IntelTablePlanner(RulePlanner):
    """Deterministic role assignment for the first dual-arm table-setting slice."""

    _RIGHT_OBJECTS = {"cup_1", "spoon_1"}
    # The left-arm napkin approach otherwise sweeps through the napkin while
    # retrieving the fork from the drawer.  Keep the same governed plan shape,
    # but reserve the shared left workspace in a measured, deterministic order.
    # Plate first (2026-09-14): it is the two-arm object and both arms'
    # sideways approaches run down the middle of the table; anything placed
    # in those corridors before it -- the cup at upper_right, the napkin at
    # lower_left -- blocked the plate pick in every engine trial while the
    # same pick was 5/5 in isolation. Setting the plate first is also how a
    # table is actually set. Legacy proxies keep the measured old order.
    _OBJECT_ORDER = (("cup_1", "napkin_1", "plate_1", "fork_1", "spoon_1") if LEGACY_MODELS_FLAG
                     else ("plate_1", "cup_1", "fork_1", "spoon_1", "napkin_1"))
    # objects that physically start inside the drawer (dual_so101_xml) -- their
    # PICK depends on an OPEN step first, matching the brief's literal
    # scenario ("open the top drawer, retrieve spoons and forks").
    _DRAWER_OBJECTS = {"fork_1", "spoon_1"}
    # A persistently-failing object used to exhaust the engine's whole
    # max_revisions budget by itself: the engine replans on ANY failure,
    # always re-plans every misplaced object, and this planner always put
    # them back in the same _OBJECT_ORDER -- so the first object that
    # can't grasp blocks every object behind it from ever being attempted
    # at all, even ones that would have succeeded. Track how many replans
    # each object has already appeared in; once one exceeds
    # _MAX_ATTEMPTS_BEFORE_DEPRIORITIZE, push it after objects with fewer
    # attempts (still eventually retried if the budget allows, just not
    # blocking). This doesn't change whether a run resolves -- every
    # object still needs to actually succeed for that -- but it means the
    # objects that CAN succeed actually get a real attempt within the same
    # budget, instead of the budget being spent entirely on whichever
    # object happens to be stuck first.
    _MAX_ATTEMPTS_BEFORE_DEPRIORITIZE = 3

    def __init__(self) -> None:
        super().__init__()
        self._attempt_counts: dict[str, int] = {}

    def plan(self, goal, world):
        graph = super().plan(goal, world)

        # Which drawer holds which piece. Legacy: one block, both pieces
        # nominally in it. Realistic: a tray per flank, fork left, spoon right,
        # each opened by its own arm with a real pull.
        drawer_of = ({"fork_1": "drawer", "spoon_1": "drawer_right"} if REAL_DRAWERS
                     else {"fork_1": "drawer", "spoon_1": "drawer"})
        needed = []
        for s in graph.steps:
            if s.op == "PICK" and s.args.get("object") in drawer_of:
                d = drawer_of[s.args["object"]]
                if d not in needed:
                    needed.append(d)
        for d in needed:
            step_id = "open_drawer" if d == "drawer" else f"open_{d}"
            open_drawer = Step(
                step_id, "manipulate", "OPEN", args={"object": d},
                rationale="cutlery starts in the drawer; open it before retrieving it",
            )
            open_drawer.arm = "right" if d == "drawer_right" else "left"
            graph.steps.insert(0, open_drawer)
            for s in graph.steps:
                if s.op == "PICK" and drawer_of.get(s.args.get("object")) == d:
                    s.deps = tuple(sorted(set(s.deps) | {open_drawer.id}))

        for step in graph.steps:
            object_id = step.args.get("object")
            if object_id in self._RIGHT_OBJECTS or object_id == "drawer_right":
                step.arm = "right"
            elif object_id:
                step.arm = "left"

        # Authority (2026-09-14): an operator "prefer_arm" constraint -- the
        # NLU's reading of "don't use the left arm anymore" / "stop using the
        # left arm" -- overrides the default assignment above. The engine
        # queues it at any time and recompiles at the next control boundary
        # (engine.run -> _apply_constraints -> _recompile), so this is where
        # a mid-run change of authority lands. Honestly: steps the remaining
        # arm can reach are reassigned to it; steps it cannot (the other
        # side's objects, and the two-arm plate) are removed from the plan
        # and listed in ``self.last_authority_report`` so the receipt shows
        # what the objective lost, rather than pretending or silently using
        # the forbidden arm. The scheduler returns an explicit ``step.arm``
        # before it consults prefer_arm, so this has to happen here.
        prefer = next((c.value for c in getattr(world, "constraints", ()) if c.kind == "prefer_arm"), None)
        self.last_authority_report = None
        if prefer in ("left", "right"):
            forbidden = "right" if prefer == "left" else "left"
            reassigned, pruned = [], []
            keep = []
            for step in graph.steps:
                object_id = step.args.get("object")
                if object_id in ("drawer", "drawer_right") and step.arm == forbidden:
                    if not REAL_DRAWERS:
                        step.arm = prefer  # symbolic open: either arm
                    else:
                        pruned.append((step.id, object_id, f"{forbidden} arm's drawer; withdrawn"))
                        continue
                if not object_id or object_id in ("drawer", "drawer_right") or step.arm != forbidden:
                    keep.append(step)
                    continue
                if object_id in BIMANUAL_OBJECTS:
                    pruned.append((step.id, object_id, f"needs both arms; {forbidden} arm withdrawn"))
                    continue
                if self._arm_can_reach(prefer, object_id, world):
                    step.arm = prefer
                    step.rationale = (step.rationale + "; " if step.rationale else "") +                         f"reassigned to {prefer}: operator withdrew the {forbidden} arm"
                    reassigned.append((step.id, object_id))
                    keep.append(step)
                else:
                    pruned.append((step.id, object_id, f"out of the {prefer} arm's reach"))
            pruned_ids = {pid for pid, _, _ in pruned}
            for step in keep:
                step.deps = tuple(d for d in step.deps if d not in pruned_ids)
            # The drawer only needs opening for cutlery that is still planned.
            for d, sid in (("drawer", "open_drawer"), ("drawer_right", "open_drawer_right")):
                still = any(st.op == "PICK" and drawer_of.get(st.args.get("object")) == d for st in keep)
                if not still:
                    keep = [st for st in keep if st.id != sid]
                    for step in keep:
                        step.deps = tuple(x for x in step.deps if x != sid)
            graph.steps = keep
            self.last_authority_report = {
                "prefer_arm": prefer, "withdrawn_arm": forbidden,
                "reassigned": reassigned, "pruned": pruned,
            }

        # RulePlanner sorts object ids lexically.  Reorder only tracked
        # tableware pairs; fixture and terminal steps keep their existing
        # positions and dependencies.  This is scheduling, not ownership or
        # object-state mutation.
        tracked = set(self._OBJECT_ORDER)
        tableware_steps = [
            step for step in graph.steps
            if step.args.get("object") in tracked
        ]
        other_steps = [
            step for step in graph.steps
            if step.args.get("object") not in tracked
        ]
        def sort_key(object_id: str) -> tuple[bool, int, int]:
            attempts = self._attempt_counts.get(object_id, 0)
            return (
                attempts > self._MAX_ATTEMPTS_BEFORE_DEPRIORITIZE,
                attempts,
                self._OBJECT_ORDER.index(object_id),
            )

        priority_order = sorted(self._OBJECT_ORDER, key=sort_key)
        ordered_tableware = []
        for object_id in priority_order:
            ordered_tableware.extend(
                step for step in tableware_steps
                if step.args.get("object") == object_id
            )
        leading = [step for step in other_steps if step.op != "VERIFY"]
        terminal = [step for step in other_steps if step.op == "VERIFY"]
        graph.steps = leading + ordered_tableware + terminal
        return graph

    # Top-down reach of one arm, measured on the realistic layout: picks
    # succeed out to ~0.33 m from the base and fail at 0.345 (cup, 2026-09-14).
    _ARM_REACH_M = 0.33
    _ARM_BASE_XY = {"left": (-0.26, 0.20), "right": (0.26, 0.20)}

    def _arm_can_reach(self, arm: str, object_id: str, world) -> bool:
        """Can ``arm`` pick this object where it is now AND set it down at its
        target zone? Both ends of the chain must be inside the measured reach."""
        base = self._ARM_BASE_XY[arm]
        det = world.objects.get(object_id) if hasattr(world, "objects") else None
        pose = getattr(det, "pose", None)
        target = ZONE_POSITIONS.get(getattr(det, "target_zone", None) or "")
        points = []
        if pose is not None:
            points.append((pose.x, pose.y))
        if target is not None:
            points.append((target[0], target[1]))
        if not points:
            return False
        return all(math.hypot(x - base[0], y - base[1]) <= self._ARM_REACH_M for x, y in points)

    def replan(self, current, world, reason):
        """Count the one object that was actually just attempted, before
        generating the fresh plan _MAX_ATTEMPTS_BEFORE_DEPRIORITIZE reads.

        Every object still misplaced appears in every generated graph
        (RulePlanner.plan includes all of them), but the engine only ever
        executes the single first-ready step before a failure triggers
        another replan (engine.py's _next_step + immediate recompile on
        any failure) -- so incrementing every present object's count
        (an earlier version of this method did, via plan() itself) counts
        objects that were never actually tried, and since every misplaced
        object's count then rises in lockstep, the relative order (and
        thus which object gets deprioritized first) never actually
        changes. The one that was genuinely attempted is the first
        tableware PICK in `current`'s own order that's still misplaced in
        the fresh `world` -- steps ahead of it in that same prior graph
        either already succeeded (no longer misplaced) or were never
        reached at all."""
        tracked = set(self._OBJECT_ORDER)
        still_misplaced = {det.object_id for det in world.misplaced()}
        # Only a failure-triggered replan means the head object was
        # attempted. A replan for a queued operator constraint ("constraint
        # change") happens BEFORE the next step runs; counting that step as
        # an attempt deprioritised the cup behind the spoon the moment the
        # operator withdrew the left arm (2026-09-14), for a pick that had
        # never been tried.
        if "constraint" not in (reason or "").lower():
            for step in current.steps:
                if step.op != "PICK":
                    continue
                object_id = step.args.get("object")
                if object_id in tracked and object_id in still_misplaced:
                    self._attempt_counts[object_id] = self._attempt_counts.get(object_id, 0) + 1
                    break
        return super().replan(current, world, reason)


class IntelTableObserver(FakeObserver):
    """Simulation-grounded observation reference; it is never labelled camera live."""

    def observe(self, world):
        observation = super().observe(world)
        return replace(
            observation,
            raw_ref=f"mujoco://dual-so101/state/{world.frame:06d}",
        )


class IntelTableVerifier(FakeVerifier):
    """Verify scripted fixture primitives against the MuJoCo state.

    The legacy table-setting route is intentionally still scripted, but a
    scripted command must not turn into a fabricated postcondition.  Drawer
    OPEN/CLOSE is the first fixture primitive with a concrete simulator
    observable; other operations retain the existing object/terminal checks.
    """

    _DRAWER_TOLERANCE_M = 1e-3

    def __init__(self, world: IntelTableWorld) -> None:
        self.world = world

    def check(self, step: Step, observation: Any) -> VerifyResult:
        if step.op in {"OPEN", "CLOSE"} and step.args.get("object") in getattr(self.world, "_drawer_slide_adr", {"drawer": None}):
            expected = DRAWER_OPEN if step.op == "OPEN" else DRAWER_CLOSED
            adr = self.world._drawer_slide_adr.get(step.args.get("object"), self.world._drawer_qpos_adr)
            observed = float(self.world.data.qpos[adr])
            # a real pull lands within a few mm of the travel stop; the symbolic
            # write is exact
            tol = 0.008 if REAL_DRAWERS else self._DRAWER_TOLERANCE_M
            ok = abs(observed - expected) <= tol
            return VerifyResult(
                ok=ok,
                expected={"drawer_qpos": expected, "tolerance_m": self._DRAWER_TOLERANCE_M},
                observed={"drawer_qpos": round(observed, 6)},
                mismatch=() if ok else ("drawer_qpos",),
            )
        return super().check(step, observation)


def intel_devices() -> DeviceRouter:
    return DeviceRouter([
        DeviceSpec("intel.perception.sim", ("perception", "verify"), local=True),
        DeviceSpec("intel.cpu.sim", ("reasoning",), local=True),
        DeviceSpec("intel.left_arm", ("arm",), local=True),
        DeviceSpec("intel.right_arm", ("arm",), local=True),
    ])


def build_intel_sim_engine(
    scene_config: IntelSceneConfig | None = None,
    *,
    recorder: Any | None = None,
    envelope: MissionEnvelope | None = None,
    bus: Any | None = None,
) -> OmniQ:
    """Construct the explicitly labelled Intel simulation path."""
    scene_config = scene_config or IntelSceneConfig()
    world = IntelTableWorld(scene_config)
    # Include randomized build-time scene identity in the governed run
    # envelope; otherwise every seed would hash to the same run_id because
    # WorldState intentionally contains semantic zones, not raw MuJoCo pose.
    if envelope is None and scene_config.randomized:
        envelope = MissionEnvelope(mission_id=f"intel-table-scene-{scene_config.seed}")
    return OmniQ(
        world=world,
        observer=IntelTableObserver(),
        planner=IntelTablePlanner(),
        manipulator=FakeManipulator(world),
        verifier=IntelTableVerifier(world),
        device=intel_devices(),
        recorder=recorder or FakeRecorder(),
        envelope=envelope,
        bus=bus,
    )


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    """Persist one report artifact without silently overwriting evidence."""
    if path.exists():
        raise FileExistsError(f"evidence already exists at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _classify_intel_table_receipt(receipt: Any) -> str:
    """Classify only outcomes evidenced by a legacy run receipt."""
    if receipt.metrics.get("resolved"):
        return "success"
    failed = [action for action in receipt.actions if action.get("state") == "failed"]
    details = [action.get("result") or {} for action in failed]
    text = json.dumps(details, sort_keys=True, default=str).lower()
    if "timeout" in text:
        return "timeout"
    if "collision" in text:
        return "collision"
    if any(detail.get("held") is False for detail in details):
        return "grasp_failure"
    if any(detail.get("placed") is False for detail in details):
        return "placement_failure"
    if failed:
        return "transition_failure"
    return "unresolved"


def _per_object_pick_place_outcomes(receipt: Any) -> dict[str, dict[str, bool]]:
    """Per-object real outcome within one run, independent of the coarse
    receipt-wide classification.

    ``_classify_intel_table_receipt`` records only the first failure type
    found anywhere in the whole receipt, so a trial where cup_1 genuinely
    completes a real pick-and-place and a later object then fails still
    classifies identically to a trial where nothing ever succeeds --
    real progress was invisible in the aggregate report even though it was
    always present in the raw per-trial receipts (see
    evidence/benchmark_results/intel_table_eval_2026-09-10-v6/README.md,
    where this was first found by reading a receipt directly). This walks
    every PICK/MOVE action and records, per object, whether *any* attempt
    at it this run ever achieved a held grasp / a placed result -- object
    identity comes from the step id (``pick_<object>``/``move_<object>``,
    set by RulePlanner.plan), not the result payload, since a rejected or
    early-failed attempt may not carry object-specific result fields."""
    outcomes: dict[str, dict[str, bool]] = {}
    for action in receipt.actions:
        step_id = action.get("step", "")
        result = action.get("result") or {}
        if action.get("op") == "PICK" and step_id.startswith("pick_"):
            object_id = step_id.removeprefix("pick_")
            entry = outcomes.setdefault(object_id, {"held": False, "placed": False})
            if result.get("held") is True:
                entry["held"] = True
        elif action.get("op") == "MOVE" and step_id.startswith("move_"):
            object_id = step_id.removeprefix("move_")
            entry = outcomes.setdefault(object_id, {"held": False, "placed": False})
            if result.get("placed") is True:
                entry["placed"] = True
    return outcomes


# Ten distinct, real phrasings of the same table-setting instruction, used
# by run_intel_table_evaluation_report as its prompt-variation axis (one of
# the four the Intel challenge hosts named as acceptable non-trivial "10
# seed" variation -- lighting, object location, prompt phrasing, object
# color/texture -- picking how many/which is left to the entrant). Each was
# checked against RulePlanner's own keyword gate before being added here
# (all contain "set", so none silently degrade to the unrecognised-goal
# "observe only" fallback, which would corrupt the evidence by making a
# vocabulary gap look like a grasp-robustness failure instead).
TABLE_SETTING_PHRASINGS: tuple[str, ...] = (
    "set the table",
    "please set the table for dinner",
    "set up the table now",
    "can you set the table",
    "time to set the table",
    "set the table for the meal",
    "go ahead and set the table",
    "set the dinner table",
    "set the table, thanks",
    "could you please go ahead and set the table for us",
)


def run_intel_table_evaluation_report(
    root: str | Path,
    *,
    trials: int = 10,
    seed: int = 100,
) -> dict[str, Any]:
    """Run bounded randomized legacy table-setting trials with retained receipts.

    This is an exploratory controller/scene report.  It is deliberately not a
    promotion gate: the general 3-DOF grasp path remains low-success, and the
    report preserves those failures instead of converting them into a score.

    Each trial combines four independent variation axes, per the Intel
    challenge hosts' own clarification that "10 seeds" means 10 non-trivial
    environment variations, not 10 draws of one narrow RNG: object position/
    yaw jitter, tableware color jitter, key-light intensity/angle jitter (all
    three via IntelSceneConfig, physics-inert -- see its docstring), and
    instruction phrasing (via TABLE_SETTING_PHRASINGS, cycled by trial index
    so a run with >10 trials repeats the cycle rather than indexing out of
    range).
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    outcomes = {
        "success": 0,
        "grasp_failure": 0,
        "placement_failure": 0,
        "timeout": 0,
        "collision": 0,
        "transition_failure": 0,
        "unresolved": 0,
    }
    per_object_holds: dict[str, int] = {}
    per_object_places: dict[str, int] = {}
    entries: list[dict[str, Any]] = []
    for index in range(trials):
        trial_seed = seed + index
        goal = TABLE_SETTING_PHRASINGS[index % len(TABLE_SETTING_PHRASINGS)]
        scene_config = IntelSceneConfig(seed=trial_seed, randomized=True)
        receipt = build_intel_sim_engine(scene_config).run(goal)
        if receipt.content_hash != content_hash_of(receipt.as_dict()):
            raise RuntimeError(f"receipt hash mismatch for trial seed {trial_seed}")
        outcome = _classify_intel_table_receipt(receipt)
        outcomes[outcome] += 1
        per_object = _per_object_pick_place_outcomes(receipt)
        for object_id, result in per_object.items():
            if result["held"]:
                per_object_holds[object_id] = per_object_holds.get(object_id, 0) + 1
            if result["placed"]:
                per_object_places[object_id] = per_object_places.get(object_id, 0) + 1
        receipt_name = f"trial-{index:02d}-seed-{trial_seed}.json"
        _write_json_once(root_path / receipt_name, receipt.as_dict())
        entries.append({
            "trial": index,
            "seed": trial_seed,
            "goal": goal,
            "scene": scene_config.as_dict(),
            "outcome": outcome,
            "resolved": bool(receipt.metrics.get("resolved")),
            "per_object": per_object,
            "run_id": receipt.run_id,
            "content_hash": receipt.content_hash,
            "receipt": receipt_name,
        })
    report = {
        "schema_version": 3,
        "mode": IntelTableWorld.mode,
        "kind": "exploratory randomized legacy table-setting report; not a promotion claim",
        # schema_version 3 (was 2): each trial now also varies instruction
        # phrasing (`goal`, see TABLE_SETTING_PHRASINGS) and IntelSceneConfig
        # gained color_jitter/light_diffuse_jitter/light_angle_jitter_rad
        # alongside the pre-existing position/yaw jitter -- see this
        # function's docstring for why (the hosts' "10 seeds" clarification).
        "trials": trials,
        "seed_start": seed,
        "outcomes": outcomes,
        # Real per-object signal the coarse `outcomes` label above can't
        # show: how many of these `trials` had at least one held grasp /
        # placed result for each object, regardless of what else failed in
        # the same run. See _per_object_pick_place_outcomes.
        "per_object_summary": {
            "held_in_trials": per_object_holds,
            "placed_in_trials": per_object_places,
        },
        "receipts": entries,
    }
    _write_json_once(root_path / "report.json", report)
    return report


# ---------------------------------------------------------------------------
# OQ-010/OQ-011: bounded contact-physics cup handoff
# ---------------------------------------------------------------------------

CONTACT_HANDOFF_MODE = "simulation-contact-handoff"
CONTACT_HANDOFF_SCHEMA_VERSION = 1
_CUP_GEOM = "cup_1_contact"
_TABLE_GEOM = "handoff_table"
_PAD_MARKER = "jaw_pad_"


class ContactHandoffRejected(RuntimeError):
    """A physics or evidence boundary rejected a proposed handoff."""


@dataclass(frozen=True)
class ContactHandoffConfig:
    """Bounded controller and scene parameters for one reproducible trial.

    The randomized mode intentionally keeps perturbations narrow.  It is a
    robustness characterization of this particular proxy scene, not a
    perception, policy, or hardware claim.
    """

    seed: int = 19
    randomized: bool = False
    position_jitter_m: float = 0.002
    orientation_jitter_rad: float = 0.025
    # Wider variation axes (2026-09-13), all drawn from ``seed`` when
    # ``randomized``: where the cup is handed over, and how the receiving
    # arm starts. With only the 2 mm build-time jitter above, twenty trials
    # were twenty near-identical trajectories; a success rate over those is
    # a repeatability number, not a robustness one.
    shared_point_jitter_m: float = 0.0     # handoff location, +/- in x and y
    receiver_pose_jitter_rad: float = 0.0  # right-arm start joints 0-3, +/-
    ik_damping: float = 0.001
    ik_iterations: int = 700
    ik_tolerance_m: float = 0.002
    interpolation_segments: int = 8
    motion_steps_per_segment: int = 75
    motion_timeout_steps: int = 120
    # 0.6 s at the pinned MuJoCo timestep; enough for a perturbed cup to
    # dissipate post-release sliding before stable-placement admission.
    settle_steps: int = 300
    joint_limit_margin_rad: float = 0.03
    # Measured against the proxy's compliant finger pads.  This remains a
    # controller stop bound, not a hardware-safe force calibration.
    max_contact_force_n: float = 80.0
    max_relative_cup_distance_m: float = 0.105
    shared_workspace_x_limit_m: float = 0.18
    gripper_open_rad: float = 1.65
    # The proxy cup is 52 mm across; this leaves a compliant, contact-rich
    # clamp instead of driving the 21 mm hard-stop through the object.
    gripper_closed_rad: float = 0.35

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContactHandoffReceipt:
    """Tamper-evident, controller-level evidence for one contact trial."""

    schema_version: int
    mode: str
    mjcf_sha256: str
    controller: dict[str, Any]
    seed: int
    deterministic: bool
    randomized: bool
    source_state_revision: int
    phase_timings: tuple[dict[str, Any], ...]
    contact_transitions: tuple[dict[str, Any], ...]
    state_samples: tuple[dict[str, Any], ...]
    final_cup_pose: dict[str, Any]
    final_owner: str | None
    final_stable: bool
    success: bool
    failure_reason: str | None
    content_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "mjcf_sha256": self.mjcf_sha256,
            "controller": self.controller,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "randomized": self.randomized,
            "source_state_revision": self.source_state_revision,
            "phase_timings": list(self.phase_timings),
            "contact_transitions": list(self.contact_transitions),
            "state_samples": list(self.state_samples),
            "final_cup_pose": self.final_cup_pose,
            "final_owner": self.final_owner,
            "final_stable": self.final_stable,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "content_hash": self.content_hash,
        }


def _contact_receipt_hash(payload: dict[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "content_hash"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_contact_handoff_receipt(receipt: ContactHandoffReceipt | dict[str, Any]) -> None:
    """Fail closed when a controller receipt has been altered or relabelled."""
    payload = receipt.as_dict() if isinstance(receipt, ContactHandoffReceipt) else dict(receipt)
    if payload.get("schema_version") != CONTACT_HANDOFF_SCHEMA_VERSION:
        raise ContactHandoffRejected("unsupported contact-handoff receipt schema")
    if payload.get("mode") != CONTACT_HANDOFF_MODE:
        raise ContactHandoffRejected("receipt is not contact-handoff evidence")
    if not payload.get("mjcf_sha256"):
        raise ContactHandoffRejected("receipt is missing an MJCF hash")
    if payload.get("content_hash") != _contact_receipt_hash(payload):
        raise ContactHandoffRejected("contact-handoff receipt content hash mismatch")


def write_contact_handoff_receipt(receipt: ContactHandoffReceipt, path: str | Path) -> Path:
    """Atomically persist one immutable controller receipt at a caller-owned path."""
    verify_contact_handoff_receipt(receipt)
    target = Path(path)
    if target.exists():
        raise ContactHandoffRejected(
            "refusing to overwrite existing contact-handoff receipt"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    encoded = json.dumps(receipt.as_dict(), sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def _contact_scene_cup_pose(config: ContactHandoffConfig) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Build-time perturbation only; runtime never writes the free-joint pose."""
    # Keep the cup clear of the parked left finger pads at simulator reset.
    # It is then approached dynamically from the front; this avoids treating
    # initial interpenetration as a grasp or as a contact-force success.
    position = [-0.260, -0.150, 0.050]
    orientation = [0.0, 0.0, 0.0]
    if config.randomized:
        rng = random.Random(config.seed)
        position[0] += rng.uniform(-config.position_jitter_m, config.position_jitter_m)
        position[1] += rng.uniform(-config.position_jitter_m, config.position_jitter_m)
        orientation = [
            rng.uniform(-config.orientation_jitter_rad, config.orientation_jitter_rad),
            rng.uniform(-config.orientation_jitter_rad, config.orientation_jitter_rad),
            rng.uniform(-config.orientation_jitter_rad, config.orientation_jitter_rad),
        ]
    return tuple(position), tuple(orientation)


def _configure_contact_arm(arm_body: ET.Element) -> None:
    """Keep only named finger-pad contacts active in the contact test scene.

    The group mask deliberately permits cup↔pad and cup↔table contacts while
    suppressing pad↔pad and unmodelled mesh contacts.  That makes the receipt's
    contact ownership calculation auditable rather than a side effect of mesh
    tessellation.
    """
    for geom in arm_body.iter("geom"):
        name = geom.attrib.get("name", "")
        if _PAD_MARKER in name:
            geom.set("contype", "4")
            geom.set("conaffinity", "0")
            geom.set("group", "3")
            geom.set("friction", PAD_FRICTION)
            geom.set("solref", HANDOFF_PAD_SOLREF)
            geom.set("solimp", HANDOFF_PAD_SOLIMP)
        else:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def contact_handoff_xml(config: ContactHandoffConfig | None = None) -> str:
    """Return the isolated, contact-physics MJCF for the ``cup_1`` handoff.

    This is intentionally not an edit to :func:`dual_so101_xml`: the legacy
    table-setting route must remain a visibly distinct scripted route.
    """
    config = config or ContactHandoffConfig()
    cup_pos, cup_euler = _contact_scene_cup_pose(config)
    source = ET.parse(ARM_XML).getroot()
    root = ET.Element("mujoco", {"model": "omni_q_dual_so101_contact_handoff"})
    for tag in ("compiler", "option", "asset", "default"):
        node = source.find(tag)
        if node is not None:
            root.append(copy.deepcopy(node))

    visual = ET.SubElement(root, "visual")
    # offwidth/offheight: offscreen buffer for 1280x720 demo recordings (render-only)
    ET.SubElement(visual, "global", {"azimuth": "125", "elevation": "-28", "offwidth": "1280", "offheight": "720"})
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {
        "name": "key", "pos": "0 -0.25 1.2", "dir": "0 0 -1", "directional": "true",
    })
    ET.SubElement(worldbody, "geom", {
        "name": _TABLE_GEOM,
        "type": "box",
        "pos": "0 -0.10 -0.05",
        "size": ".42 .36 .05",
        "rgba": ".23 .14 .08 1",
        "friction": "1.20 0.006 0.0002",
        "contype": "2",
        "conaffinity": "16",
    })
    # Keep failed trials bounded when a cup is knocked beyond the table edge.
    # This is containment only: the controller still fails closed on a carried
    # cup below the drop bound, and final success requires cup↔table contact.
    ET.SubElement(worldbody, "geom", {
        "name": "handoff_floor",
        "type": "plane",
        "pos": "0 0 -0.12",
        "size": "0 0 .05",
        "rgba": ".08 .12 .12 1",
        "friction": "0.80 0.005 0.0001",
        "contype": "2",
        "conaffinity": "16",
    })
    ET.SubElement(worldbody, "camera", {
        "name": "handoff_third_person", "pos": "0 -1.15 .85", "euler": "1.05 0 0",
    })

    base = source.find("./worldbody/body[@name='Base']")
    if base is None:
        raise RuntimeError("pinned SO-ARM100 MJCF is missing Base")
    for arm, pos in (("left", "-.26 .20 .0"), ("right", ".26 .20 .0")):
        arm_body = _prefixed(base, arm)
        arm_body.set("pos", pos)
        _configure_contact_arm(arm_body)
        worldbody.append(arm_body)

    cup = ET.SubElement(worldbody, "body", {
        "name": "cup_1",
        "pos": "%.6f %.6f %.6f" % cup_pos,
        "euler": "%.6f %.6f %.6f" % cup_euler,
    })
    ET.SubElement(cup, "freejoint", {"name": "cup_1_free"})
    ET.SubElement(cup, "geom", {
        "name": _CUP_GEOM,
        "type": "cylinder",
        "size": ".022 .050",
        "mass": ".010",
        "rgba": ".22 .58 .78 1",
        "friction": PAD_FRICTION,
        "solref": HANDOFF_PAD_SOLREF,
        "solimp": HANDOFF_PAD_SOLIMP,
        "contype": "16",
        "conaffinity": "4",
        "group": "1",
    })

    actuators = ET.SubElement(root, "actuator")
    source_actuators = source.find("actuator")
    if source_actuators is None:
        raise RuntimeError("pinned SO-ARM100 MJCF is missing actuators")
    for arm in ("left", "right"):
        for actuator in source_actuators:
            copied = copy.deepcopy(actuator)
            copied.set("name", f"{arm}_{actuator.attrib['name']}")
            copied.set("joint", f"{arm}_{actuator.attrib['joint']}")
            actuators.append(copied)
    return ET.tostring(root, encoding="unicode")


def load_contact_handoff_model(config: ContactHandoffConfig | None = None):
    """Load the generated contact scene without creating a generated MJCF file."""
    mujoco = _mujoco()
    assets = {f"assets/{path.name}": path.read_bytes() for path in ARM_ASSETS.glob("*.stl")}
    return mujoco.MjModel.from_xml_string(contact_handoff_xml(config), assets=assets)


class IntelContactHandoffWorld(MockWorld):
    """Authoritative world for one atomic, verified physical handoff.

    The physical controller has no authority to alter ``WorldState``.  It can
    only attach a proposed contact receipt; this world admits the represented
    handoff after validating the receipt, org scope, and state revision.
    """

    mode = CONTACT_HANDOFF_MODE

    def __init__(self, config: ContactHandoffConfig | None = None) -> None:
        self.config = config or ContactHandoffConfig()
        super().__init__([
            Detection("cup_1", "cup", "left_table", "right_table"),
        ])
        self._mujoco = _mujoco()
        self.model_xml = contact_handoff_xml(self.config)
        assets = {f"assets/{path.name}": path.read_bytes() for path in ARM_ASSETS.glob("*.stl")}
        self.model = self._mujoco.MjModel.from_xml_string(self.model_xml, assets=assets)
        self.data = self._mujoco.MjData(self.model)
        self.mjcf_sha256 = "sha256:" + hashlib.sha256(self.model_xml.encode("utf-8")).hexdigest()
        self.pending_contact_receipt: ContactHandoffReceipt | None = None
        self.last_admitted_receipt: ContactHandoffReceipt | None = None
        self.contact_events: list[dict[str, Any]] = []

        # Initializing only the arm joints is a normal simulator reset, not a
        # cup pose write.  The free-joint state remains the pose compiled from
        # ``contact_handoff_xml`` (including any randomized build-time offset).
        self.data.qpos[:12] = list(HOME) * 2
        self.data.ctrl[:12] = list(HOME) * 2
        if self.config.randomized and self.config.receiver_pose_jitter_rad > 0:
            # The receiving (right) arm does not start from the same posture
            # every trial. A joint-space start offset, commanded through the
            # servos like any other pose -- the arm settles there itself.
            rng = random.Random(self.config.seed * 7919 + 11)
            margin = self.config.joint_limit_margin_rad + 0.02
            for j in range(6, 10):
                offset = rng.uniform(-self.config.receiver_pose_jitter_rad, self.config.receiver_pose_jitter_rad)
                lo, hi = self.model.jnt_range[j]
                # inside the controller's own joint-limit margin: a start
                # posture the guard would reject is not a trial, it is a
                # misconfiguration (3 of 20 wide-variation trials, 2026-09-13)
                self.data.qpos[j] = self.data.ctrl[j] = float(min(max(HOME[j - 6] + offset, lo + margin), hi - margin))
        self.data.qpos[5] = self.config.gripper_open_rad
        self.data.qpos[11] = self.config.gripper_open_rad
        self.data.ctrl[5] = self.config.gripper_open_rad
        self.data.ctrl[11] = self.config.gripper_open_rad
        self._mujoco.mj_forward(self.model, self.data)

    def apply_transition(self, request: TransitionRequest) -> TransitionResult:
        if request.expected_revision != self.revision:
            raise TransitionRejected(
                f"{request.step_id}: expected revision {request.expected_revision}, current revision is {self.revision}"
            )
        if request.org_id != self.org_id:
            raise TransitionRejected(
                f"{request.step_id}: request org {request.org_id} does not match world org {self.org_id}"
            )
        if request.op != "HANDOFF" or request.args.get("object") != "cup_1":
            raise TransitionRejected(f"{request.step_id}: contact world only admits HANDOFF of cup_1")
        if request.actor != "intel.left_arm" or request.args.get("to_actor") != "intel.right_arm":
            raise TransitionRejected(f"{request.step_id}: contact handoff requires left giver and right receiver")
        receipt = self.pending_contact_receipt
        if receipt is None:
            raise TransitionRejected(f"{request.step_id}: no contact receipt was proposed")
        try:
            verify_contact_handoff_receipt(receipt)
        except ContactHandoffRejected as exc:
            raise TransitionRejected(f"{request.step_id}: {exc}") from exc
        if not receipt.success or not receipt.final_stable or receipt.final_owner is not None:
            raise TransitionRejected(f"{request.step_id}: contact handoff did not reach a stable released cup")
        if receipt.source_state_revision != self.revision:
            raise TransitionRejected(f"{request.step_id}: contact evidence has a stale state revision")

        self._objects["cup_1"] = replace(self._objects["cup_1"], zone="right_table")
        self._ownership["cup_1"] = None
        self.revision += 1
        self.last_admitted_receipt = receipt
        self.pending_contact_receipt = None
        event = {
            "kind": "contact_handoff.admitted",
            "state_revision": self.revision,
            "receipt_hash": receipt.content_hash,
            "mjcf_sha256": receipt.mjcf_sha256,
        }
        self.contact_events.append(event)
        return TransitionResult(
            step_id=request.step_id,
            ok=True,
            state_revision=self.revision,
            detail={
                "handed_off": "cup_1",
                "to_actor": "intel.right_arm",
                "simulation_mode": self.mode,
                "contact_receipt": receipt.as_dict(),
                "domain_event": event,
            },
        )

    def move_object(self, obj_id: str, zone: str) -> None:
        """Reject an unaudited background mutation in the contact world.

        A real external disturbance must enter through an integration that
        records its causal domain event first; inheriting ``MockWorld``'s
        convenience mutator here would silently bypass that boundary.
        """
        raise TransitionRejected(
            f"background mutation of {obj_id} requires a recorded contact domain event"
        )

    def simulation_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "nq": self.model.nq,
            "nv": self.model.nv,
            "nu": self.model.nu,
            "time": float(self.data.time),
            "mjcf_sha256": self.mjcf_sha256,
            "admitted_contact_events": len(self.contact_events),
        }


class _ContactHandoffController:
    """Sequential, contact-verified primitive controller for a single cup.

    Every arm target is calculated with damped least squares and is sent as an
    interpolated position-actuator command.  It intentionally contains no
    equality/weld attachment, no mocap body, and no write to the cup freejoint.
    """

    def __init__(self, world: IntelContactHandoffWorld) -> None:
        self.world = world
        self.model = world.model
        self.data = world.data
        self.mujoco = world._mujoco
        self.config = world.config
        try:
            import numpy as np
        except ImportError as exc:  # MuJoCo itself requires numpy, but stay explicit
            raise IntelSimulationUnavailable("NumPy is required by the MuJoCo handoff controller") from exc
        self.np = np
        self.ik_data = self.mujoco.MjData(self.model)
        self.phase_timings: list[dict[str, Any]] = []
        self.recoveries: list[dict[str, Any]] = []
        self.contact_transitions: list[dict[str, Any]] = []
        self.state_samples: list[dict[str, Any]] = []
        self._last_contact_signature: tuple[str, ...] = ()
        self._carry_required = False
        self._phase = "initial"

    def run(self, source_state_revision: int) -> ContactHandoffReceipt:
        failure: str | None = None
        final_stable = False
        try:
            self._sample("initial")
            self._run_phase("initial_settle", lambda: self._advance(self.config.settle_steps, "initial_settle"))

            cup0 = self._cup_position()
            left_grasp = self._grasp_target("left", cup0, wrist_roll=1.65)
            self._run_phase("left_approach", lambda: self._move_to(
                "left", self._raised_target("left", left_grasp, 0.075), 1.65, "left_approach",
            ))
            self._run_phase("left_descend", lambda: self._move_to("left", left_grasp, 1.65, "left_descend"))
            self._run_phase("left_grasp", lambda: self._command_gripper("left", self.config.gripper_closed_rad, "left_grasp"))
            self._require_owner("left", "left_grasp")
            self._carry_required = True

            # The shared target was selected inside both arm workspaces during
            # the measured OQ-003 proxy sweep.  A tighter operator-configured
            # bound rejects entry before either arm crosses it.
            left_shared = self.np.array([-0.040, -0.110, 0.134])
            if self.config.randomized and self.config.shared_point_jitter_m > 0:
                rng = random.Random(self.config.seed * 7919 + 23)
                left_shared = left_shared + self.np.array([
                    rng.uniform(-self.config.shared_point_jitter_m, self.config.shared_point_jitter_m),
                    rng.uniform(-self.config.shared_point_jitter_m, self.config.shared_point_jitter_m),
                    0.0,
                ])
            self.handoff_point = [round(float(v), 4) for v in left_shared]
            if abs(float(left_shared[0])) > self.config.shared_workspace_x_limit_m:
                raise ContactHandoffRejected("unsafe shared-workspace entry")
            self._run_phase("left_lift_transfer", lambda: self._move_to(
                "left", left_shared, 1.65, "left_lift_transfer",
            ))
            self._require_owner("left", "left_lift_transfer")

            def right_reach(tag: str) -> None:
                cup_now = self._cup_position()
                grasp = self._grasp_target("right", cup_now, wrist_roll=-1.65)
                self._run_phase(f"right_approach{tag}", lambda: self._move_to(
                    "right", self._raised_target("right", grasp, 0.035), -1.65, f"right_approach{tag}",
                ))
                self._run_phase(f"right_descend{tag}", lambda: self._move_to("right", grasp, -1.65, f"right_descend{tag}"))

            try:
                right_reach("")
            except ContactHandoffRejected as exc:
                if "right inverse-kinematics" not in str(exc):
                    raise
                # Recovery (2026-09-14): the receiver could not reach the cup
                # where the giver presented it (8 of 20 wide-variation trials
                # timed out here). The giver still owns the cup; bring it to
                # the default shared point -- inside both arms' measured
                # workspace -- re-observe, and let the receiver try once more.
                self.recoveries.append({"phase": "right_approach", "reason": str(exc),
                                        "action": "giver re-presents cup at the default shared point"})
                self._require_owner("left", "recovery_check")
                default_shared = self.np.array([-0.040, -0.110, 0.134])
                self._run_phase("left_represent", lambda: self._move_to(
                    "left", default_shared, 1.65, "left_represent",
                ))
                self._require_owner("left", "left_represent")
                self.handoff_point = [round(float(v), 4) for v in default_shared]
                right_reach("_retry")
            self._run_phase("right_grasp", lambda: self._command_gripper("right", self.config.gripper_closed_rad, "right_grasp"))
            self._require_owner("dual", "dual_contact_confirmation")

            def release_left() -> None:
                self._command_gripper("left", self.config.gripper_open_rad, "left_release")
                # Opening alone can leave a compliant pad touching the cup;
                # withdraw the giver before admitting right-only ownership.
                self._move_to("left", self.np.array([-0.115, -0.075, 0.180]), 1.65, "left_release")

            self._run_phase("left_release", release_left)
            self._require_owner("right", "left_release")

            self._run_phase("right_retreat", lambda: self._move_to(
                "right", self.np.array([0.115, -0.105, 0.145]), -1.65, "right_retreat",
            ))
            self._require_owner("right", "right_retreat")
            self._carry_required = False
            # Place by the CUP's height, not a fixed pad height: with a
            # perturbed grasp the cup sits at a different height in the jaw,
            # and a fixed pad target of 0.050 left it hanging when the jaw
            # pre-opened -- it dropped onto an edge and rolled (5 of 20
            # wide-variation trials ended with the cup on its side at
            # z=0.021, 2026-09-14). Lower until the cup's resting centre
            # height is reached, whatever the grasp offset.
            fixed_pad, _, _ = self._arm_ids("right")
            grasp_dz = float(self.data.geom_xpos[fixed_pad][2] - self._cup_position()[2])
            place_z = 0.050 + grasp_dz + 0.002
            self._run_phase("right_place", lambda: self._move_to(
                "right", self.np.array([0.145, -0.085, place_z]), -1.65, "right_place",
            ))
            def release_right() -> None:
                # Open to a clearance gap before lifting the pads away from
                # the cup. Fully opening while the compliant pads are still
                # pressed against a tilted, perturbed cup can impart a large
                # lateral impulse.
                preopen = self.config.gripper_closed_rad + 0.45
                self._command_gripper("right", preopen, "right_release_preopen")
                self._move_to("right", self.np.array([0.145, -0.085, 0.090]), -1.65, "right_release_lift")
                self._command_gripper("right", self.config.gripper_open_rad, "right_release")
                self._move_to("right", self.np.array([0.190, -0.060, 0.155]), -1.65, "right_release")

            self._run_phase("right_release", release_right)
            self._run_phase("final_settle", lambda: self._advance(self.config.settle_steps, "final_settle"))
            if self._ownership() is not None:
                raise ContactHandoffRejected("cup remains held after the requested release")
            final_stable = self._cup_is_stable_on_table()
            if not final_stable:
                raise ContactHandoffRejected("cup did not reach stable final placement")
        except ContactHandoffRejected as exc:
            failure = str(exc)
        finally:
            self._sample("final")

        receipt = ContactHandoffReceipt(
            schema_version=CONTACT_HANDOFF_SCHEMA_VERSION,
            mode=CONTACT_HANDOFF_MODE,
            mjcf_sha256=self.world.mjcf_sha256,
            controller={**self.config.as_dict(), "handoff_point": getattr(self, "handoff_point", None),
                        "recoveries": list(self.recoveries)},
            seed=self.config.seed,
            deterministic=not self.config.randomized,
            randomized=self.config.randomized,
            source_state_revision=source_state_revision,
            phase_timings=tuple(self.phase_timings),
            contact_transitions=tuple(self.contact_transitions),
            state_samples=tuple(self.state_samples),
            final_cup_pose=self._cup_pose(),
            final_owner=self._ownership(),
            final_stable=final_stable,
            success=failure is None and final_stable,
            failure_reason=failure,
        )
        payload = receipt.as_dict()
        return ContactHandoffReceipt(**{**payload, "content_hash": _contact_receipt_hash(payload)})

    # -- bounded motion -------------------------------------------------
    def _arm_slice(self, arm: str) -> slice:
        return slice(0, 6) if arm == "left" else slice(6, 12)

    def _arm_ids(self, arm: str) -> tuple[int, int, int]:
        offset = 0 if arm == "left" else 6
        return (
            self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, f"{arm}_fixed_jaw_pad_4"),
            self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, f"{arm}_moving_jaw_pad_4"),
            offset,
        )

    def _solve_ik(self, arm: str, target: Any, wrist_roll: float) -> Any:
        fixed_id, _, offset = self._arm_ids(arm)
        q = self.data.qpos[offset:offset + 5].copy()
        q[4] = wrist_roll
        desired = self.np.asarray(target, dtype=float)
        joint_low = self.model.jnt_range[offset:offset + 4, 0] + self.config.joint_limit_margin_rad
        joint_high = self.model.jnt_range[offset:offset + 4, 1] - self.config.joint_limit_margin_rad
        dofs = self.np.arange(offset, offset + 4)
        for _ in range(self.config.ik_iterations):
            self.ik_data.qpos[:] = self.data.qpos
            self.ik_data.qpos[offset:offset + 5] = q
            self.ik_data.qpos[offset + 5] = self.config.gripper_open_rad
            self.mujoco.mj_forward(self.model, self.ik_data)
            error = desired - self.ik_data.geom_xpos[fixed_id]
            if float(self.np.linalg.norm(error)) <= self.config.ik_tolerance_m:
                return q
            jacobian = self.np.zeros((3, self.model.nv))
            self.mujoco.mj_jacGeom(self.model, self.ik_data, jacobian, None, fixed_id)
            active = jacobian[:, dofs]
            damped = active @ active.T + (self.config.ik_damping ** 2) * self.np.eye(3)
            delta = active.T @ self.np.linalg.solve(damped, error)
            q[:4] = self.np.clip(q[:4] + self.np.clip(delta, -0.075, 0.075), joint_low, joint_high)
        raise ContactHandoffRejected(f"controller timeout: {arm} inverse-kinematics target not reached")

    def _grasp_target(self, arm: str, cup_position: Any, wrist_roll: float) -> Any:
        """Choose a fixed-pad target so open pads bracket the observed cup."""
        desired_cup = self.np.asarray(cup_position, dtype=float)
        if arm == "right":
            # The mirrored SO-ARM100 jaw opens mainly along +y at this wrist
            # roll.  A small front offset puts the cup inside the closed gap;
            # solving the exact open-gap midpoint would ask the right arm for
            # a near-limit pose and would be less robust to table settling.
            target = desired_cup + self.np.array([0.006, -0.012, 0.0])
            self._solve_ik(arm, target, wrist_roll)
            return target
        target = desired_cup.copy()
        for _ in range(4):
            q = self._solve_ik(arm, target, wrist_roll)
            fixed_id, moving_id, offset = self._arm_ids(arm)
            self.ik_data.qpos[:] = self.data.qpos
            self.ik_data.qpos[offset:offset + 5] = q
            self.ik_data.qpos[offset + 5] = self.config.gripper_open_rad
            self.mujoco.mj_forward(self.model, self.ik_data)
            opening = self.ik_data.geom_xpos[moving_id] - self.ik_data.geom_xpos[fixed_id]
            next_target = desired_cup - 0.5 * opening
            if float(self.np.linalg.norm(next_target - target)) < self.config.ik_tolerance_m:
                return next_target
            target = next_target
        return target

    def _raised_target(self, arm: str, grasp_target: Any, dz: float) -> Any:
        target = self.np.asarray(grasp_target, dtype=float).copy()
        target[2] += dz
        return target

    def _move_to(self, arm: str, target: Any, wrist_roll: float, phase: str) -> None:
        if self.config.motion_timeout_steps <= 0:
            raise ContactHandoffRejected(f"controller timeout: {phase} has no allowed control steps")
        q = self._solve_ik(arm, target, wrist_roll)
        arm_slice = self._arm_slice(arm)
        start = self.data.ctrl[arm_slice].copy()
        destination = self.np.concatenate((q, [self.data.ctrl[arm_slice][-1]]))
        for segment in range(1, self.config.interpolation_segments + 1):
            fraction = segment / self.config.interpolation_segments
            self.data.ctrl[arm_slice] = start + fraction * (destination - start)
            self._advance(self.config.motion_steps_per_segment, phase)
        fixed_id, _, _ = self._arm_ids(arm)
        error = float(self.np.linalg.norm(self.data.geom_xpos[fixed_id] - self.np.asarray(target)))
        if error > 0.035:
            raise ContactHandoffRejected(f"controller timeout: {phase} target error {error:.3f} m")

    def _command_gripper(self, arm: str, target: float, phase: str) -> None:
        arm_slice = self._arm_slice(arm)
        start = float(self.data.ctrl[arm_slice][-1])
        for segment in range(1, self.config.interpolation_segments + 1):
            self.data.ctrl[arm_slice][-1] = start + (target - start) * segment / self.config.interpolation_segments
            self._advance(max(1, self.config.motion_steps_per_segment // 2), phase)

    def _advance(self, steps: int, phase: str) -> None:
        for step in range(steps):
            self.mujoco.mj_step(self.model, self.data)
            if step % 10 == 0:
                self._record_contact_transition(phase)
                self._guard(phase)
        self._record_contact_transition(phase)

    def _guard(self, phase: str) -> None:
        for index in range(12):
            joint_id = int(self.model.actuator_trnid[index, 0])
            low, high = self.model.jnt_range[joint_id]
            value = self.data.qpos[self.model.jnt_qposadr[joint_id]]
            if min(value - low, high - value) < self.config.joint_limit_margin_rad:
                raise ContactHandoffRejected(f"joint limit approached during {phase}")
        contacts = self._contact_snapshot()
        if contacts["max_gripper_force_n"] > self.config.max_contact_force_n:
            raise ContactHandoffRejected(f"unsafe collision during {phase}")
        if self._carry_required and self._cup_position()[2] < 0.020:
            raise ContactHandoffRejected(f"cup dropped during {phase}")

    # -- contact/state verification ------------------------------------
    def _contact_snapshot(self) -> dict[str, Any]:
        cup_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, _CUP_GEOM)
        table_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, _TABLE_GEOM)
        by_arm = {"left": 0, "right": 0}
        labels: list[str] = []
        max_force = 0.0
        cup_on_table = False
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            first, second = int(contact.geom1), int(contact.geom2)
            if cup_id not in {first, second}:
                continue
            other = second if first == cup_id else first
            other_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, other) or "unnamed"
            force = self.np.zeros(6)
            self.mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            if other == table_id:
                cup_on_table = True
                labels.append("cup:table")
                continue
            if other_name.startswith("left_") and _PAD_MARKER in other_name:
                by_arm["left"] += 1
                max_force = max(max_force, abs(float(force[0])))
                labels.append(f"cup:{other_name}")
            elif other_name.startswith("right_") and _PAD_MARKER in other_name:
                by_arm["right"] += 1
                max_force = max(max_force, abs(float(force[0])))
                labels.append(f"cup:{other_name}")
            else:
                labels.append(f"cup:unexpected:{other_name}")
                max_force = max(max_force, abs(float(force[0])))
        return {
            "by_arm": by_arm,
            "labels": tuple(sorted(set(labels))),
            "max_gripper_force_n": max_force,
            "cup_on_table": cup_on_table,
        }

    def _ownership(self) -> str | None:
        contacts = self._contact_snapshot()["by_arm"]
        cup = self._cup_position()
        candidates: list[str] = []
        for arm in ("left", "right"):
            fixed_id, moving_id, _ = self._arm_ids(arm)
            center = 0.5 * (self.data.geom_xpos[fixed_id] + self.data.geom_xpos[moving_id])
            if contacts[arm] >= 1 and float(self.np.linalg.norm(cup - center)) <= self.config.max_relative_cup_distance_m:
                candidates.append(arm)
        if candidates == ["left"]:
            return "left"
        if candidates == ["right"]:
            return "right"
        if candidates == ["left", "right"]:
            return "dual"
        return None

    def _require_owner(self, expected: str, phase: str) -> None:
        actual = self._ownership()
        if actual == expected:
            return
        if actual is None:
            raise ContactHandoffRejected(f"missing contact during {phase}")
        raise ContactHandoffRejected(f"ambiguous ownership during {phase}: observed {actual}, expected {expected}")

    def _cup_position(self) -> Any:
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "cup_1")
        return self.data.xpos[body_id].copy()

    def _cup_pose(self) -> dict[str, Any]:
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "cup_1")
        return {
            "position": [round(float(value), 6) for value in self.data.xpos[body_id]],
            "quaternion": [round(float(value), 6) for value in self.data.xquat[body_id]],
            "linear_velocity": [round(float(value), 6) for value in self._cup_velocity()],
        }

    def _cup_velocity(self) -> Any:
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "cup_1")
        velocity = self.np.zeros(6)
        self.mujoco.mj_objectVelocity(self.model, self.data, self.mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
        return velocity[3:]

    def _cup_is_stable_on_table(self) -> bool:
        contacts = self._contact_snapshot()
        velocity = self._cup_velocity()
        # Upright, too (2026-09-14): the cup is 22 mm in radius, so lying on
        # its side it sits at z = 0.021 and, once it stops rolling, passed
        # every check here. Six of the 20 wide-variation "successes" ended at
        # 90 deg. A handoff that leaves the cup on its side did not succeed.
        pose = self._cup_pose()
        qw, qx, qy, qz = (float(v) for v in pose["quaternion"])
        upright = (1.0 - 2.0 * (qx * qx + qy * qy)) > 0.985   # R[2,2]: within ~10 deg
        return bool(
            contacts["cup_on_table"]
            and float(self.np.linalg.norm(velocity)) < 0.025
            and self._cup_position()[2] > 0.020
            and upright
        )

    # -- receipt trace --------------------------------------------------
    def _run_phase(self, name: str, operation: Any) -> None:
        self._phase = name
        start = float(self.data.time)
        operation()
        self.phase_timings.append({
            "phase": name,
            "start_s": round(start, 6),
            "end_s": round(float(self.data.time), 6),
            "duration_s": round(float(self.data.time) - start, 6),
        })
        self._sample(name)

    def _record_contact_transition(self, phase: str) -> None:
        signature = self._contact_snapshot()["labels"]
        if signature == self._last_contact_signature:
            return
        self.contact_transitions.append({
            "phase": phase,
            "sim_time_s": round(float(self.data.time), 6),
            "from": list(self._last_contact_signature),
            "to": list(signature),
        })
        self._last_contact_signature = signature

    def _sample(self, phase: str) -> None:
        contacts = self._contact_snapshot()
        limits: list[float] = []
        for index in range(12):
            joint_id = int(self.model.actuator_trnid[index, 0])
            low, high = self.model.jnt_range[joint_id]
            value = self.data.qpos[self.model.jnt_qposadr[joint_id]]
            limits.append(float(min(value - low, high - value)))
        self.state_samples.append({
            "phase": phase,
            "sim_time_s": round(float(self.data.time), 6),
            "cup": self._cup_pose(),
            "ownership": self._ownership(),
            "contacts": list(contacts["labels"]),
            "contact_counts": dict(contacts["by_arm"]),
            "cup_on_table": contacts["cup_on_table"],
            "max_gripper_force_n": round(float(contacts["max_gripper_force_n"]), 6),
            "min_joint_margin_rad": round(min(limits), 6),
        })


class IntelContactHandoffManipulator:
    """Physical proposal provider; final world authority remains external."""

    def __init__(self, world: IntelContactHandoffWorld) -> None:
        self.world = world
        self._attempted = False
        self._attempt_receipt: ContactHandoffReceipt | None = None

    def supports(self, op: str) -> bool:
        return op == "HANDOFF"

    def execute(self, step: Step, world: Any) -> ManipResult:
        if step.op != "HANDOFF" or step.args.get("object") != "cup_1":
            return ManipResult(step.id, False, {"error": "contact controller only supports HANDOFF cup_1"})
        # MuJoCo state is mutable even when authoritative WorldState is not.
        # Re-entering a failed controller from OmniQ's generic revision loop
        # would therefore retry on a dirty scene and make the receipt's causal
        # history ambiguous. A caller that wants another physical attempt must
        # construct a fresh contact world/engine.
        if self._attempted:
            detail: dict[str, Any] = {
                "error": "contact handoff is one-shot; retry requires a fresh contact world",
            }
            if self._attempt_receipt is not None:
                detail["contact_handoff"] = self._attempt_receipt.as_dict()
            return ManipResult(step.id, False, detail)
        self._attempted = True
        receipt = _ContactHandoffController(self.world).run(source_state_revision=world.revision)
        self._attempt_receipt = receipt
        self.world.pending_contact_receipt = receipt
        return ManipResult(step.id, receipt.success, {"contact_handoff": receipt.as_dict()})


class IntelContactHandoffPlanner:
    """One narrow proposal: no AI planner may widen this controller's scope."""

    def __init__(self, world: IntelContactHandoffWorld | None = None) -> None:
        self.world = world
        self.last_decision: PlanDecision | None = None

    def plan(self, goal: str, world: Any) -> PlanGraph:
        cup = world.objects.get("cup_1")
        if cup is None or not cup.authoritative:
            reason = "cup_1 lacks a live authoritative observation"
            graph = PlanGraph(goal=goal)
            self.last_decision = PlanDecision(
                goal=goal,
                revision=graph.revision,
                selected_ops=(),
                candidates_considered=1 if cup is not None else 0,
                candidates_feasible=0,
                governing_constraints=("live_observation",),
                rejected={"cup_1": reason},
                state_hash="contact-handoff",
                reason=reason,
            )
            return graph
        if cup.zone == cup.target_zone:
            reason = "cup_1 is already at its target zone; no second handoff is authorized"
            graph = PlanGraph(goal=goal)
            self.last_decision = PlanDecision(
                goal=goal,
                revision=graph.revision,
                selected_ops=(),
                candidates_considered=1,
                candidates_feasible=0,
                governing_constraints=("single_handoff", "live_observation"),
                rejected={"cup_1": reason},
                state_hash="contact-handoff",
                reason=reason,
            )
            return graph
        if self.world is not None:
            prior = self.world.pending_contact_receipt or self.world.last_admitted_receipt
            if prior is not None:
                reason = "contact handoff was already attempted; retry requires a fresh contact world"
                graph = PlanGraph(goal=goal)
                self.last_decision = PlanDecision(
                    goal=goal,
                    revision=graph.revision,
                    selected_ops=(),
                    candidates_considered=1,
                    candidates_feasible=0,
                    governing_constraints=("single_handoff", "contact_receipt"),
                    rejected={"cup_1": reason},
                    state_hash="contact-handoff",
                    reason=reason,
                )
                return graph
        step = Step(
            "contact_handoff_cup_1",
            "manipulate",
            "HANDOFF",
            args={"object": "cup_1", "to_actor": "intel.right_arm"},
            arm="left",
            rationale="bounded physical transfer of cup_1; authority waits for contact receipt verification",
        )
        graph = PlanGraph(goal=goal, steps=[step, Step(
            "verify_contact_handoff",
            "verify",
            "VERIFY",
            deps=(step.id,),
            rationale="admit success only after the world validates the physics receipt",
        )])
        self.last_decision = PlanDecision(
            goal=goal,
            revision=graph.revision,
            selected_ops=("HANDOFF", "VERIFY"),
            candidates_considered=1,
            candidates_feasible=1,
            governing_constraints=("contact_receipt", "state_revision", "org_scope"),
            state_hash="contact-handoff",
            reason="one fixed cup handoff is within this evidence slice",
        )
        return graph

    def replan(self, current: PlanGraph, world: Any, reason: str) -> PlanGraph:
        fresh = self.plan(current.goal, world)
        fresh.revision = current.revision + 1
        if self.last_decision is not None:
            self.last_decision.revision = fresh.revision
            self.last_decision.reason = f"replan: {reason}"
        return fresh


class IntelContactHandoffObserver(FakeObserver):
    def observe(self, world: Any):
        observation = super().observe(world)
        return replace(observation, raw_ref=f"mujoco://dual-so101/contact-state/{world.frame:06d}")


class IntelContactHandoffVerifier:
    """Verification reads the admitted receipt, never a controller self-claim."""

    def __init__(self, world: IntelContactHandoffWorld) -> None:
        self.world = world

    def check(self, step: Step, observation: Any) -> VerifyResult:
        receipt = self.world.last_admitted_receipt
        ok = bool(
            receipt is not None
            and receipt.success
            and receipt.final_stable
            and receipt.final_owner is None
            and self.world.state().objects["cup_1"].zone == "right_table"
        )
        return VerifyResult(
            ok=ok,
            expected={"cup_1": "right_table", "contact_receipt": "stable_released"},
            observed={
                "cup_1": self.world.state().objects["cup_1"].zone,
                "receipt_hash": receipt.content_hash if receipt else None,
                "final_stable": receipt.final_stable if receipt else False,
            },
            mismatch=() if ok else ("contact-handoff-verification",),
        )


def build_intel_contact_handoff_engine(
    config: ContactHandoffConfig | None = None,
    *,
    recorder: Any | None = None,
) -> OmniQ:
    """Construct the separate governed contact-handoff path."""
    world = IntelContactHandoffWorld(config)
    return OmniQ(
        world=world,
        observer=IntelContactHandoffObserver(),
        planner=IntelContactHandoffPlanner(world),
        manipulator=IntelContactHandoffManipulator(world),
        verifier=IntelContactHandoffVerifier(world),
        device=intel_devices(),
        recorder=recorder or FakeRecorder(),
    )


def run_contact_handoff(
    config: ContactHandoffConfig | None = None,
    *,
    receipt_path: str | Path | None = None,
) -> ContactHandoffReceipt:
    """Execute one governed physical proposal and optionally retain its receipt."""
    engine = build_intel_contact_handoff_engine(config)
    engine.run("transfer cup_1 from left arm to right arm")
    receipt = engine.world.last_admitted_receipt
    if receipt is None:
        pending = engine.world.pending_contact_receipt
        if pending is None:
            raise ContactHandoffRejected("controller returned no handoff receipt")
        receipt = pending
    if receipt_path is not None:
        write_contact_handoff_receipt(receipt, receipt_path)
    return receipt


def run_randomized_contact_handoff_report(
    root: str | Path,
    *,
    trials: int = 20,
    seed: int = 700,
    variation: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Retain every bounded randomized trial; report outcomes without promotion.

    ``variation`` overrides ContactHandoffConfig jitter fields, e.g.
    ``{"position_jitter_m": 0.02, "shared_point_jitter_m": 0.03,
    "receiver_pose_jitter_rad": 0.15}`` for the wide-variation sweep.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    outcomes = {"success": 0, "drop": 0, "timeout": 0, "collision": 0, "ambiguity": 0, "other_failure": 0}
    for index in range(trials):
        trial_seed = seed + index
        receipt = run_contact_handoff(
            ContactHandoffConfig(seed=trial_seed, randomized=True, **(variation or {})),
            receipt_path=root_path / f"trial-{index:02d}-seed-{trial_seed}.json",
        )
        if receipt.success:
            category = "success"
        else:
            reason = (receipt.failure_reason or "").lower()
            category = next((name for name in ("drop", "timeout", "collision", "ambiguity") if name in reason), "other_failure")
        outcomes[category] += 1
        receipts.append({
            "trial": index,
            "seed": trial_seed,
            "success": receipt.success,
            "failure_reason": receipt.failure_reason,
            # A phase that raises is never appended, so the failure sits
            # right after the last completed phase.
            "failed_after_phase": (receipt.phase_timings[-1]["phase"] if receipt.phase_timings else "initial")
                                  if not receipt.success else None,
            "handoff_point": receipt.controller.get("handoff_point"),
            "phases_completed": len(receipt.phase_timings),
            "final_owner": receipt.final_owner,
            "receipt": f"trial-{index:02d}-seed-{trial_seed}.json",
        })
    report = {
        "schema_version": CONTACT_HANDOFF_SCHEMA_VERSION,
        "mode": CONTACT_HANDOFF_MODE,
        "kind": "exploratory robustness report; not a promotion claim",
        "trials": trials,
        "seed_start": seed,
        "variation": variation or {},
        "outcomes": outcomes,
        "receipts": receipts,
    }
    report_path = root_path / "report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    return report
