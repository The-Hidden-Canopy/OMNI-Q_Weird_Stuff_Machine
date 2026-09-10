"""The ontology layer — claim fusion, IS-A, conflict, deltas, attention."""

from __future__ import annotations

from omni_q.contracts import Observe
from omni_q.ontology import (
    Claim,
    Ontology,
    OntologyObserver,
    canonical,
    compatible,
    stub_swarm,
    wakes,
)
from omni_q.world import MockWorld


def _box(cx, cy, s=0.08):
    b = s / 2
    return (cx - b, cy - b, cx + b, cy + b)


# -- IS-A hierarchy -----------------------------------------------------


def test_isa_hierarchy():
    assert canonical("mug") == "cup" and canonical("saucer") == "plate"
    assert canonical("butter_knife") == "knife" and canonical("hand") == "human"
    assert compatible("mug", "cup") and compatible("cup", "wineglass")
    assert not compatible("fork", "knife")


# -- fusion across specialist detectors ------------------------------


def test_two_detectors_disagree_on_vocab_but_fuse_to_one_entity():
    o = Ontology()
    d = o.ingest([
        Claim("yolo_objects", "cup", 0.91, _box(0.44, 0.44)),
        Claim("yolo_tools", "mug", 0.70, _box(0.445, 0.445)),
    ], 1)
    assert [x.kind for x in d] == ["entity.appeared"]          # one entity, not two
    (e,) = o.entities.values()
    assert e.canonical_type == "cup" and not e.conflict
    assert e.sources == {"yolo_objects", "yolo_tools"}
    assert e.type_conf == 1.0                                  # IS-A votes reinforce


def test_incompatible_types_on_the_same_box_become_a_conflict():
    o = Ontology()
    o.ingest([
        Claim("yolo_objects", "fork", 0.94, _box(0.62, 0.62)),
        Claim("yolo_tools", "knife", 0.61, _box(0.622, 0.622)),
    ], 1)
    (e,) = o.entities.values()
    assert e.conflict is True
    assert set(e.votes) == {"fork", "knife"} and e.type == "fork"
    # a conflicted entity is not trusted -> OMNI-Q sees FALLBACK, won't manipulate
    ws = o.world_state(1)
    assert ws.objects[e.id].status.value == "fallback"
    assert o.snapshot()["conflicts"] == [e.id]


def test_distinct_objects_far_apart_stay_separate():
    o = Ontology()
    o.ingest([
        Claim("yolo_objects", "fork", 0.9, _box(0.20, 0.5)),
        Claim("yolo_objects", "fork", 0.9, _box(0.80, 0.5)),
    ], 1)
    assert len(o.entities) == 2


def test_stable_ids_and_lost_delta():
    o = Ontology(max_missed=1)
    o.ingest([Claim("yolo_objects", "plate", 0.9, _box(0.4, 0.4))], 1)
    (eid,) = list(o.entities)
    o.ingest([Claim("yolo_objects", "plate", 0.9, _box(0.42, 0.4))], 2)
    assert list(o.entities) == [eid]                            # same id, drifted
    o.ingest([], 3)
    d = o.ingest([], 4)
    assert any(x.kind == "entity.lost" and x.entity == eid for x in d)


# -- relations + workspace conflict --------------------------------


def test_hand_in_the_arm_zone_raises_and_clears_a_workspace_conflict():
    o = Ontology()
    # an arm entity and, later, a hand on top of it
    o.ingest([Claim("yolo_tools", "arm", 0.99, _box(0.50, 0.50, s=0.12))], 1)
    d = o.ingest([
        Claim("yolo_tools", "arm", 0.99, _box(0.50, 0.50, s=0.12)),
        Claim("yolo_human", "hand", 0.85, _box(0.51, 0.51)),
    ], 2)
    assert any(x.kind == "workspace.conflict" for x in d)
    assert o.snapshot()["workspace_conflict"] is True
    # hand leaves
    d2 = o.ingest([Claim("yolo_tools", "arm", 0.99, _box(0.50, 0.50, s=0.12))], 3)
    assert any(x.kind == "workspace.clear" for x in d2)


def test_near_relation_between_hand_and_object():
    o = Ontology()
    o.ingest([
        Claim("yolo_objects", "cup", 0.9, _box(0.50, 0.50)),
        Claim("yolo_human", "hand", 0.9, _box(0.56, 0.50)),
    ], 1)
    preds = {(r.predicate) for r in o.relations}
    assert "near" in preds or "intersects" in preds


# -- attention filter ----------------------------------------------


def test_wakes_filter():
    from omni_q.ontology import Delta
    assert wakes(Delta("workspace.conflict", "workspace"))
    assert wakes(Delta("entity.appeared", "cup_1"))
    assert not wakes(Delta("relation.added", "cup_1"))


# -- OntologyObserver drop-in ------------------------------------


def test_ontology_observer_satisfies_observe_and_runs_the_engine():
    from omni_q import build_mock_engine

    engine = build_mock_engine()
    obs = OntologyObserver(stub_swarm(engine.world))
    assert isinstance(obs, Observe)
    engine.observer = obs
    receipt = engine.run("inspect and correct the workspace")
    assert "steps_executed" in receipt.metrics
    # the swarm populated the ontology
    assert obs.ontology.entities


def test_observer_wakes_on_a_hand_entering():
    world = MockWorld.sample()
    obs = OntologyObserver(stub_swarm(world))
    obs.observe(world.state())
    assert not obs.wake or all(d.kind == "entity.appeared" for d in obs.last_deltas)
    world.hand_xy = (0.5, 0.5)                                  # a hand appears
    obs.observe(world.state())
    assert obs.wake
    assert any(d.kind == "entity.appeared" and "human" in d.entity for d in obs.last_deltas)


def test_claims_carry_provenance():
    world = MockWorld.sample()
    obs = OntologyObserver(stub_swarm(world))
    obs.observe(world.state())
    for e in obs.ontology.entities.values():
        assert e.sources                                       # every entity traces to a model
