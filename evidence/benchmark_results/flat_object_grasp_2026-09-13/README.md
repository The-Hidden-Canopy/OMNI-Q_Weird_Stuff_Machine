# Flat-object grasp: fork 0/10 → 10/10 (2026-09-13)

Ten randomized trials (`run_intel_table_evaluation_report`, seed 900), same
scene jitter, same harness. Only the manipulation changed.

| object | held before | held after | placed before | placed after |
| --- | --- | --- | --- | --- |
| `cup_1` | 10 | 10 | 8 | 7 |
| `plate_1` | 9 | **10** | 0 | 0 |
| **`fork_1`** | **0** | **10** | 0 | **6** |
| **`spoon_1`** | **0** | **4** | 0 | 2 |
| `napkin_1` | 0 | 0 | 0 | 0 |

The trial-level label is still `grasp_failure: 10` because every trial includes
the napkin, which is not yet solved; the per-object rows are the result.

## What changed, and what each piece contributed

Four defects, each found by measurement and each necessary. The full chain of
evidence is in [`../pad_friction_probe_2026-09-13/README.md`](../pad_friction_probe_2026-09-13/README.md).

1. **The gripper was commanded only half closed.** `GRIPPER_CLOSED` was 0.0;
   the Jaw joint's stop is −0.174. At 0.0 the fingertip pads stay 15.8 mm
   apart, so anything thinner was ungrippable by construction. That floor
   predicted every prior outcome exactly (cup 110 mm ✓, plate 32 mm ✓, fork
   8 mm ✗, spoon 8 mm ✗, napkin 6 mm ✗). Now closes to 3.6 mm.
2. **The pinch tracked the wrong pad.** `_do_pick` drove `fixed_jaw_pad_4`,
   43 mm up the finger, to the object's centre height — asking the fingertip
   to go 4 cm below the table for a fork. The jaws also close as a wedge
   (2.6 mm at the tip, 19.3 mm at pad_4), so a thin object can only be pinched
   at the tip regardless. `_ik_reach_pad(track_tcp=True)` now drives the
   midpoint of the two fingertip pads, averaging position *and* Jacobian, on
   every IK call in the pick (switching only the final pinch made the tracked
   point jump 43 mm mid-sequence: an 88 mm XY miss).
3. **The fork was a slab.** A 24 × 8 mm box with 8 mm-tall pads pinches a
   corner at an angle and slips (measured: 2 pads, 15.5 N, +9 mm, drop). Real
   cutlery has a ~10 mm handle and a ~25 mm head; `_cutlery()` models fork and
   spoon that way, body origin at the handle so the freejoint pose *is* the
   grasp point. Same 150 mm length and 40 g.
4. **The lift height shrank with the object.** Lift was
   `half_height + 15 mm` — 18.5 mm for a 7 mm fork, *below* the 20 mm "held"
   threshold. A perfectly grasped fork could never register as held. Now a
   40 mm floor (`LIFT_VERIFY_MIN`).

Plus the sensor that made the last two visible: `_set_gripper` now reports the
jaw's actual angle, whether it stalled, and every pad/object contact with its
force from MuJoCo's contact buffer — the two things the challenge hosts pointed
at.

## Still open

- **`napkin_1`** (6 mm cloth, no handle) is unsolved. It has no narrow feature
  to pinch and is the genuine flat-object case; it likely needs the two-arm
  tip-and-grasp manoeuvre (`OQ-010-BIMANUAL`) or a different model (a folded
  napkin has a ridge).
- **`spoon_1`** holds 4/10 — the head is heavier than the fork's and the grasp
  is off-centre; a grasp offset toward the balance point is the obvious next
  probe.
- **Placement** lags holding across the board; that is `_do_place`, not
  addressed here.
- **`plate_1`** still lifts by force-ramping the servo against a stall on the
  rim's top face (~15 N) rather than by closing on the rim. It works (10/10)
  and the receipt now says how; it is not yet a legitimate pinch.

Nothing here weakens the simulation: no collision exemptions, no teleports,
the gripper commands its own joint range, and the cutlery is more realistic
than the slab it replaces, not less.

## Update: evidence-driven retry — spoon 7/10 → 10/10

Same ten seeds, `after_evidence_retry.json`:

| object | held | placed |
| --- | --- | --- |
| `cup_1` | 10 | 9 |
| `plate_1` | 10 | — |
| `fork_1` | 10 | 9 |
| **`spoon_1`** | **10** | **10** |
| `napkin_1` | 0 | 0 |

The pick already retried on failure, but blindly — the same four wrist rolls
whether the pads had touched nothing or had gripped and slipped. The retry
now reads the previous attempt's `grasp_sensor` and chooses a correction for
*that* failure, re-observing the object's live pose each time (it may have been
nudged), bounded to four attempts:

| evidence | correction |
| --- | --- |
| no pad contact | fingertips stopped above the object → descend 4 mm deeper, same roll |
| contact but no lift | gripped and pivoted/slipped → regrasp 20 mm toward the head (balance point), 2 mm deeper |
| stalled at a wide jaw angle | came down *on* the object → back off 6 mm, shift 15 mm |
| otherwise | mirrored jaw / ±0.25 rad roll alternatives |

Across the run the spoon fired 4 × `no_contact_descend` and 5 ×
`slipped_regrasp_toward_head`; five of its ten picks were won by a retry. Every
retry and the evidence that chose it is in the receipt under `retries`, and
`orientation.lift.retry_reason` names the winning attempt.

This is the primitive updating its own plan from observation. The engine-level
replan (OQ-018: verify after every step, replan on mismatch) sits above it and
is unchanged. Note the trial-level label is *still* `grasp_failure: 10`,
because every trial includes the napkin — the label is all-or-nothing and now
hides four objects at 10/10.

**The napkin fired 76 retries and won none**: it gets pad contact
(`slipped_regrasp`, 22×) but never lifts. A 6 mm rigid cloth slab with no
narrow feature cannot be pinched by 8 mm fingertip pads from above. That is the
next model change.
