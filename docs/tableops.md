# OMNI-Q TableOps

## Product

**OMNI-Q TableOps** is the enterprise expansion of the OMNI physical-host
platform: an adaptive bimanual system for precision setup, reset, inspection,
and reconfiguration of hospitality spaces.

The primary product framing for the demo is **OMNI HOME / OMNI CHEF**: a
physical host for a luxury homeowner, entertainer, yacht, private club, or
personal chef. It is useful, configurable, expressive, and able to work beside
a human chef without turning guest-facing hospitality into an automation
problem. TableOps applies the same core to higher-volume hospitality
operations.

Table setting is the benchmark because it makes the technical requirements
visible: exact relationships, fragile objects, arbitrary starting positions,
human interruptions, verification, and recovery. The product is the larger
operation of transforming a real room from its current state into a specified
venue standard.

The initial demo buyer hypothesis is a luxury household or chef-driven home.
The enterprise buyer hypothesis for TableOps is a high-throughput formal
hospitality operator: casino resorts, convention hotels, banquet operators,
wedding venues, cruise dining, and premium restaurants. In both cases, the
system is intended to provide another set of hands for repetitive physical work,
not to replace guest-facing hospitality, the personal chef, or service recovery.

## Operational model

```text
CURRENT ROOM STATE
        +
VENUE / EVENT PROFILE
        ↓
OMNI-Q CAPABILITY GRAPH
        ↓
SETUP / RESET / INSPECT / RECONFIGURE
        ↓
SELF-VERIFICATION
        ↓
PASS, SELF-CORRECT, OR ESCALATE
```

An event profile is configuration, not a hard-coded animation:

```yaml
profile: formal_dinner_v3
venue: example_banquet_room
guest_count: 4
place_setting:
  plate: {relationship: seat_datum, tolerance_mm: 3}
  fork: {relationship: plate_left, spacing_mm: 22, tolerance_mm: 3}
  knife: {relationship: plate_right, blade: inward, tolerance_mm: 3}
  water_glass: {enabled: true, relationship: plate_upper_right}
  wine_glass: {enabled: true, relationship: water_glass}
  napkin: {style: folded, tolerance_mm: 5}
constraints:
  preserve_guest_customizations: true
  human_zone: protected
```

The profile format is a product contract for the next evaluator/configuration
slice. It is not evidence that the current mock or Intel path already meets
these tolerances.

## Demo story

1. Start with a deliberately imperfect formal setting.
2. OMNI-Q observes the current state and executes the available placement graph.
3. The evaluator reports placement error, missing items, and first-pass status.
4. Move a glass or rotate a plate; OMNI-Q detects the residual and replans or
   reports the limitation of the current provider.
5. Have two speakers issue competing instructions. Voice claims are attributed,
   references can be grounded by pointing, and authority is resolved before a
   mutation is queued.
6. Give OMNI-Q a bounded expressive window. It may use only the explicitly
   granted time, region, arms, and generic motion primitives, then must return
   ready for the next commitment.

The expressive window is a demonstration of bounded scheduling slack, not an
idle-time promise that should be counted as throughput until measured.

## Business expansion

```text
precision table setup
        ↓
table reset and self-inspection
        ↓
banquet changeover: current state → desired state
        ↓
inventory observations and exception reporting
        ↓
room preparation and hospitality physical operations
```

Changeover is the strategic expansion: remove lunchware, preserve explicitly
customized settings, add the next event's glassware, and verify the resulting
room. Inventory counts and room-scale throughput are roadmap capabilities until
the live perception and multi-cell orchestration layers exist.

## Pilot measurement plan

Do not claim savings before a site supplies its baseline. A pilot should compare
the existing workflow with one OMNI-Q cell using the same venue profile and
representative event mix:

| Measure | Evidence to retain |
| --- | --- |
| settings per hour | timestamped run receipts and denominators |
| labor minutes per setting | customer baseline plus observed intervention time |
| first-pass compliance | evaluator result before correction |
| correction rate | residual events and replan count |
| changeover time | current-state and desired-state run timestamps |
| breakage / excess-force events | verified incident records, including zero only when measured |
| human interventions | attributed intervention events and reasons |
| exception recovery | moved object, occlusion, missing item, and stale-observation cases |

The output should be a measured delta for that operation, not a universal ROI
claim.

## Capability boundary

| Capability | Current status | Evidence boundary |
| --- | --- | --- |
| bimanual graph planning and routing | prototype | mock and bounded Intel simulation |
| post-action verification | prototype | existing evaluator/receipt paths |
| generic expressive slack | prototype | bounded simulation; not hardware evidence |
| attributed multi-speaker claims | prototype | provider-neutral adapter and adversarial tests |
| authorized voice mutation | prototype | existing governed `RuntimeMutator` path |
| live Speechmatics transport | pending | no provider credentials or latency receipt |
| live multi-camera fusion | pending | no production camera-fusion claim |
| servo load/current contact sensing | pending | no force/torque or hardware claim |
| room-scale inventory/changeover | pending | requires live perception and orchestration |

The product story is intentionally ahead of the benchmark, but never ahead of
the evidence: the table task demonstrates the control loop; the pilot measures
whether TableOps creates operational value.
