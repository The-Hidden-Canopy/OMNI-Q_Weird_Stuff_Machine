# Evidence lag audit (2026-09-13)

**Evidence lag** = the gap between when a piece of evidence was produced and the
world state it is finally acted against. This repo has the vocabulary, the
timestamps and the revision counters to detect it everywhere. It enforces it in
three places, ignores it in three more, and has one enum member that no producer
has ever set.

Nothing here is a hypothetical. The numbers come from this session's own
measured receipts and from the repo's own recorded incidents.

## Where lag *is* gated

| Site | Mechanism | Strictness |
| --- | --- | --- |
| `contracts.py:283-293` `TransitionRequest.expected_revision` | request carries the revision it was built against | exact match; "keeps a stale plan from silently overwriting newer observations" |
| `intel_sim.py:2363` | `receipt.source_state_revision != self.revision` → `TransitionRejected` | exact match |
| `expressive.py:262` | expressive window "stale against current world revision" | exact match |
| `omni_planner.py:33-52` | three stale-proposal classes rejected before they reach the world | behavioural, not revision-based |
| `engine.py:246`, `fleet_runtime.py:241`, `scheduler.py:991` | never execute a stale graph | fail-closed |

The `omni_planner` case is the best-documented: a model "proposes from its own
belief … which may lag the world by a step or more", and one measured run spent
**6 of 7 revisions** on a single repeated stale PICK before the check existed.

## Where lag is *not* gated

### 1. The voice path has no revision awareness at all

`grep -n revision src/omni_q/voice.py` returns **nothing**. `ingest_final`
accepts `world: WorldState | None` and uses it for three things — org matching,
reference resolution, authority — and never asks how old the speech is relative
to that world.

That was defensible when voice was untimed. It is not any more, because this
session measured the lag end to end:

| Session | final latency p50 | max |
| --- | --- | --- |
| first mic run | 1066 ms | 1116 ms |
| halting speech | 1187 ms | 3129 ms |
| post-aggregation | 3006 ms | 3183 ms |
| enhanced model | 1031 ms | 1154 ms |
| latest | 3613 ms | 3735 ms |

**Up to 3.7 seconds** from the operator finishing a word to the claim reaching
`RuntimeMutator` — and part of that is a *deliberate* hold introduced by
`UtteranceAggregator` and `IntentAccumulator`. We made the lag larger on
purpose, for good reasons, and never taught the boundary to account for it.

Why it matters is narrower than it first looks, and worth stating precisely:

- **Constraints are fine.** "Don't use the left arm" uttered 3 s ago is still a
  valid instruction now. Blanket rejection on lag would be wrong.
- **References are not.** `ReferenceResolver.resolve(...)` binds "that", "this",
  "the other one" against `world.objects` **at commit time, not utterance
  time**. If the referent moved zone, changed ownership, or was completed during
  those seconds, the phrase silently resolves against a world the speaker was
  not looking at. The resolver is careful to leave ambiguous references
  unresolved — but it has no way to know the world moved under it.

### 2. Ontology trust has no age term

```python
@property
def trusted(self) -> bool:
    return not self.conflict and self.type_conf >= 0.6
```

`Entity` tracks `first_seen`, `last_seen` and `missed` (`ontology.py:125-127`),
and `to_detection` stamps `verified_frame=e.last_seen`. But `trusted` consults
**neither** `last_seen` nor `missed`. An entity seen once at 0.9 confidence and
not observed since is still `DataStatus.LIVE`, whatever the frame counter says.

### 3. `DataStatus.STALE` is dead

```
src/omni_q/contracts.py:41      STALE = "stale"      # definition
src/omni_q/__init__.py:128      "STALE" exported
```

Ten `DataStatus` references across `src/`. Every producer sets `LIVE` or
`FALLBACK` (`frame_observer.py:299`, `ontology.py:340,351`). **No code path
anywhere assigns `STALE`.** The docstring above it says the planner "must
re-`Observe` or lower its commitment before acting on anything that is not
`LIVE`" — a rule that currently cannot fire, because nothing ever leaves `LIVE`
for age.

## What the two mechanisms are, and why mixing them up matters

The repo actually has *two* different ideas both called staleness:

1. **Revision equality** — "this evidence was produced at world revision N and
   the world is now at N; proceed." Exact, fail-closed, correct for
   *authoritative state transitions* (contact receipts, graph mutation).
2. **Age** — "this evidence is K frames / milliseconds old; trust it less."
   Graded, appropriate for *perception* and for *instructions*.

Only (1) is implemented. (2) has all its inputs recorded and none of its logic
written. The risk of conflating them is real: applying revision-equality to
voice would reject almost every spoken constraint, because the world advances
during the 1–3.7 s it takes to hear one. That would look like "we handled
staleness" while actually breaking the feature.

## Recommended, in order

1. **Stamp, don't gate, first.** Record `world.revision` on `SpeechClaim` at
   ingest and carry it through `VoiceDispatchResult`. Costs nothing, breaks
   nothing, and makes the lag auditable in the event log — which is how every
   other finding in this session got found.
2. **Then gate references only.** When a reference resolves against a world
   whose revision advanced after the utterance began, mark the
   `ReferenceClaim` unresolved rather than binding it. This is the one case
   where lag can produce a wrong action rather than a late one.
3. **Give `trusted` an age term**, or delete `last_seen`/`missed` as decoration.
   Either is defensible; the current state — tracking age and ignoring it — is
   the one that misleads.
4. **Assign `STALE` or remove it.** A status that documents a rule the code
   cannot enforce is worse than no status.

## Status — implemented 2026-09-13

All four landed, each default-off or additive so no existing behaviour changes
unless a caller opts in.

**1. Claims are stamped.** `SpeechClaim` gained `world_revision` (the revision
it was acted against), `observed_revision` (what the speaker was looking at),
and a `revision_lag` property. All three appear in `as_dict()`, so the lag is
visible in the event log — which is how every other defect in this session was
found.

**2. Stale references are unresolved, constraints are not touched.**
`ingest_final` takes an optional `observed_revision`. When supplied and the
world has advanced since, a reference that *did* resolve is downgraded to
`UNRESOLVED` with `stale_by_N_revisions` evidence, and
`voice.reference.stale` is published. A constraint uttered 39 revisions ago
still commits — verified by test. Omitting the argument reproduces previous
behaviour exactly.

**3. Ontology trust can consider age.** `Ontology(stale_after_frames=N)` reports
`DataStatus.STALE` for an entity not refreshed within `N` frames, in both
`world_state()` and `observation()`. Age is checked **before** confidence: an
observation the world has moved past is unreliable however confident the
detector was. Default `None` keeps age tracked-and-ignored, so nothing changes
for existing callers until someone picks a horizon.

**4. `DataStatus.STALE` is no longer dead** — item 3 is the first code path in
the repo that assigns it.

Tests: `test_voice.py` gains four (stamping, stale reference, constraint
survives lag, opt-in default); `test_ontology.py` gains three (goes stale, age
before confidence, off by default).

**5. The transport now supplies it.** `UtteranceAggregator(world_revision=...)`
samples the revision when an utterance's **first** fragment is buffered — as
close as this layer gets to "what the speaker was looking at when they started"
— and passes it at flush via `VoiceSink.deliver(..., observed_revision=N)`.
The CLI supplies `engine.world.state().revision` under `--mock-engine`; without
a live engine there is no provider and **no revision is invented**, which is
tested explicitly. The revision that matters is the one at utterance start, not
whichever happens to be current when the last fragment lands: also tested, since
that is the easy thing to get subtly wrong.

The chain is therefore complete end to end: world revision at utterance start →
aggregator → sink → `ingest_final` → `SpeechClaim.revision_lag` → stale
references unresolved, constraints untouched.
