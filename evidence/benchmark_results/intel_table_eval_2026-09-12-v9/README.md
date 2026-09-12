# Legacy Intel table-setting randomized report (v9)

Same protocol as v2/v3/v5/v6/v7/v8: `run_intel_table_evaluation_report(...,
trials=10, seed=100)`, `simulation-scripted-manipulation` route, one
canonical hash-checked receipt per trial. First report generated with
`schema_version: 3`, which widens what "10 seeds" actually varies.

## Why this bundle exists

The Intel challenge hosts clarified the "10 seeds" requirement directly:
10 seeds can mean 10 different **non-trivial** variations of the
environment -- lighting, object location, input-prompt phrasing, object
color/texture, etc. -- picking which axes and how many is left to the
entrant, as long as it demonstrates the policy is robust to real
environment change, not just re-running the same scene with a coin flip.

Every prior bundle (v2 through v8) only varied one axis:
`IntelSceneConfig`'s object position/yaw jitter, and a narrow one at that
(±3mm / ±0.08rad). That's a real axis ("object location"), but alone it
risked reading as trivial even repeated 10 times -- a judge could
reasonably ask whether 10 draws of the same tiny RNG demonstrate
robustness to anything beyond floating-point noise.

This bundle combines four independent, verified axes per trial, matching
all four categories the hosts explicitly named:

1. **Object location** -- the existing position/yaw jitter (unchanged).
2. **Object color/texture** -- `IntelSceneConfig.color_jitter` perturbs
   each tableware body's `rgba` only. Mass/friction/solref/solimp are
   never touched, so this can't weaken contact physics -- it only changes
   what a camera/vision model would see.
3. **Lighting condition** -- `IntelSceneConfig.light_diffuse_jitter` /
   `light_angle_jitter_rad` perturb the scene's key light's intensity and
   incidence angle (a dimmer/brighter, more off-axis light -- like a
   different time of day). Also physics-inert.
4. **Input-prompt phrasing** -- each trial uses a different real phrasing
   of the same instruction (`TABLE_SETTING_PHRASINGS` in `intel_sim.py`:
   "set the table", "please set the table for dinner", "could you please
   go ahead and set the table for us", etc.), not the literal string "set
   the table" ten times. Each phrasing was checked against `RulePlanner`'s
   keyword gate *before* being added, and against a live run producing an
   identical op sequence to the baseline phrasing -- so this axis
   genuinely exercises language-front-end robustness without silently
   degrading to the "unrecognised goal; observe only" fallback, which
   would have corrupted the evidence by making a vocabulary gap look like
   a grasp-robustness failure.

See `IntelSceneConfig`'s and `run_intel_table_evaluation_report`'s
docstrings in `src/omni_q/intel_sim.py` for the implementation.

## Result

| Outcome | Count |
| --- | ---: |
| Success | 0 |
| Grasp failure | 10 |
| Placement failure | 0 |
| Timeout | 0 |
| Collision | 0 |
| Transition failure | 0 |
| Unresolved | 0 |

| Object | Held (of 10) | Placed (of 10) |
| --- | ---: | ---: |
| `cup_1` | **10/10** | 9/10 |
| `plate_1` | **10/10** | 0/10 |
| `napkin_1` / `fork_1` / `spoon_1` | 0/10 | 0/10 |

`cup_1` and `plate_1` both hold in every trial under this richer,
four-axis variation -- not just the narrow position/yaw jitter v8 used.
`plate_1`'s jump from 1/10 (v8) to 10/10 is the landed
`OBJECT_GRASP_SEED_BIAS` fix (see `intel_sim.py`'s module docstring,
"Eighth update"); this bundle is the first time that fix was checked
against lighting/color variation as well as position jitter, and it held.

## A new, real, more-visible finding: `plate_1` holds but never places

`plate_1` held in all 10 trials here (previously only 1/10 in v8, too
rare to have been a repeatable signal) but was placed in **0/10** --
every single `MOVE` was rejected with `"unsafe carry separation"`,
`relative_object_pad_distance_m` measured around 0.121-0.122m against
`LEGACY_MAX_CARRY_OFFSET_M = 0.100`.

This is not new behavior introduced by the seed-bias fix or this bundle
-- v8's one `plate_1` hold also never placed, for the same reason -- it
was simply too rare (1/10) to read as a repeatable pattern until this fix
made it visible in every trial. Root cause, read directly from a trial
receipt: `plate_1` is deliberately grasped at its **rim**
(`OBJECT_GRASP_OFFSET["plate_1"] = 0.078`, a ~95mm-radius plate), not its
center, because a flat 190mm plate can't be pinched through its own
middle. `_workspace_safety`'s carry-separation check measures the
distance from the pad to the object's freejoint anchor (its geometric
center) and rejects anything past a single global 0.100m bound --
un-aware that a legitimately safe rim grasp on an object this size is
*inherently* going to put the pad 0.08m+ from center before any reach
error is even added. `cup_1` never hits this because it's grasped
centered (`OBJECT_GRASP_OFFSET["cup_1"] = 0.0`).

This looks like a genuine candidate for the same category of fix as the
`OBJECT_GRASP_OFFSET`/`OBJECT_HALF_HEIGHT` constants already in this
file: a per-object-aware bound (e.g. accounting for the object's own
known grasp offset/radius) rather than one global figure tuned around a
small, centered grasp. Not fixed in this bundle -- it's a distinct
safety-bound question from the local-minimum escape this bundle was
built to verify, and a safety-relevant threshold deserves its own
focused look rather than a same-commit patch. Flagged in `BACKLOG.md`
(OQ-010) for follow-up.

## What this bundle does not claim

Not a promotion claim, same as every prior bundle. The coarse `outcomes`
table is still 10/10 `grasp_failure` and was never expected to move --
`_classify_intel_table_receipt` scans the whole receipt for the first
failure found anywhere in it, so `napkin_1`/`fork_1`/`spoon_1` still
failing (unchanged, no mechanism tried this session closed that gap)
means every trial still classifies as a failure at the coarse level. Kept
alongside v2 through v8, not replacing them, so this progression stays
auditable.
