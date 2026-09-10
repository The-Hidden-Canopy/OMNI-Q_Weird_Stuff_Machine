"""The ontology layer — a living scene graph between perception and OMNI-Q.

    detectors (cheap specialists) -> Claims -> Ontology (fusion authority)
        -> WorldState + Deltas -> OMNI-Q reacts

A detector never mutates reality. It emits :class:`Claim`s — *hypotheses* with
provenance (which model, which camera, when, how sure). The :class:`Ontology`
reconciles claims from many specialists into authoritative :class:`Entity`s:

- temporal association (YOLO-A frame N and YOLO-B frame N+1 are ``entity_17``);
- an IS-A hierarchy so ``mug`` and ``cup`` fuse without either detector
  winning a vocabulary fight;
- **preserved disagreement** — ``fork (.94)`` vs ``knife (.61)`` ->
  ``entity.type='fork', conflict=True``; OMNI-Q sees the doubt and can decline
  to manipulate / ask for another observation;
- derived **relations** (``hand near left_arm``, ``cup missing_from setting_2``)
  and workspace-conflict detection.

The ontology also acts as an **attention filter**: it emits :class:`Delta`s and
:func:`wakes` says which ones are worth waking the expensive reasoner for. The
perception swarm can run at MXFP2/MXFP4 all day; OMNI-Q only fires on a
meaningful state change. See ``docs/ontology.md``.

:class:`OntologyObserver` implements the ``Observe`` contract, so it drops into
``OmniQ`` exactly where ``FakeObserver`` / ``FrameObserver`` did.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .contracts import DataStatus, Detection, Observation, WorldState
from .frame_observer import Detection2D, grid_zone_map, iou

# ---------------------------------------------------------------------------
# IS-A hierarchy — detector vocab differences don't reach OMNI-Q
# ---------------------------------------------------------------------------

# child -> parent
_ISA: dict[str, str] = {
    "mug": "cup", "coffee_cup": "cup", "wineglass": "cup", "wine_glass": "cup",
    "goblet": "cup", "tumbler": "cup",
    "saucer": "plate", "platter": "plate", "dish": "plate",
    "butter_knife": "knife", "table_knife": "knife", "kitchen_knife": "knife",
    "soupspoon": "spoon", "teaspoon": "spoon", "tablespoon": "spoon",
    "serviette": "napkin", "place_mat": "napkin",
    # people (a robot arm is NOT a person -> "arm" keeps its own type)
    "hand": "human", "person": "human", "finger": "human", "wrist": "human",
}


def canonical(t: str) -> str:
    seen = set()
    while t in _ISA and t not in seen:
        seen.add(t)
        t = _ISA[t]
    return t


def compatible(a: str, b: str) -> bool:
    """True if the two type strings could name the same thing (share a
    canonical root, or one IS-A the other)."""
    return canonical(a) == canonical(b) or a == b


# ---------------------------------------------------------------------------
# value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    source_model: str
    entity_type: str
    confidence: float
    geometry: tuple[float, float, float, float]      # normalised xyxy
    device: str = "cam0"
    ts: float = field(default_factory=time.time)
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.geometry
        return (x0 + x1) / 2, (y0 + y1) / 2


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str      # near | intersects | reachable_by | missing_from | pointing_toward
    obj: str
    conf: float = 1.0
    source: str = "ontology"

    def as_dict(self) -> dict[str, Any]:
        return {"subject": self.subject, "predicate": self.predicate,
                "object": self.obj, "conf": self.conf, "source": self.source}


@dataclass
class Entity:
    id: str
    type: str
    type_conf: float
    geometry: tuple[float, float, float, float]
    zone: str
    conflict: bool = False
    attrs: dict[str, Any] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)
    votes: dict[str, float] = field(default_factory=dict)   # raw type -> summed conf
    first_seen: int = 0
    last_seen: int = 0
    missed: int = 0

    @property
    def canonical_type(self) -> str:
        return canonical(self.type)

    @property
    def trusted(self) -> bool:
        return not self.conflict and self.type_conf >= 0.6

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "type": self.type, "canonical": self.canonical_type,
            "type_conf": round(self.type_conf, 3), "conflict": self.conflict,
            "zone": self.zone, "attrs": self.attrs, "sources": sorted(self.sources),
            "votes": {k: round(v, 3) for k, v in self.votes.items()},
            "first_seen": self.first_seen, "last_seen": self.last_seen,
        }


@dataclass(frozen=True)
class Delta:
    kind: str            # entity.appeared | entity.moved | entity.type_conflict
                         # | entity.lost | relation.added | workspace.conflict
                         # | workspace.clear
    entity: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "entity": self.entity, "detail": self.detail}


# deltas worth waking the expensive reasoner for
_WAKE = {"entity.appeared", "entity.lost", "entity.moved",
         "entity.type_conflict", "workspace.conflict", "workspace.clear"}


def wakes(delta: Delta) -> bool:
    return delta.kind in _WAKE


# ---------------------------------------------------------------------------
# the fusion authority
# ---------------------------------------------------------------------------

RelationRule = Callable[["Ontology"], "list[Relation]"]


class Ontology:
    def __init__(
        self,
        *,
        zone_map: Callable[[float, float], str] = grid_zone_map,
        iou_thresh: float = 0.3,
        strong_iou: float = 0.5,     # overlap this high == same object even if types clash
        center_gate: float = 0.15,
        conflict_conf: float = 0.45,
        max_missed: int = 3,
        relation_rules: list[RelationRule] | None = None,
    ) -> None:
        self.zone_map = zone_map
        self.iou_thresh = iou_thresh
        self.strong_iou = strong_iou
        self.center_gate = center_gate
        self.conflict_conf = conflict_conf
        self.max_missed = max_missed
        self.relation_rules = relation_rules or [near_rule, workspace_conflict_rule]

        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self.deltas: list[Delta] = []
        self._counts: dict[str, int] = {}
        self._ws_conflict = False

    # -- ingest --------------------------------------------------------
    def ingest(self, claims: list[Claim], frame: int) -> list[Delta]:
        out: list[Delta] = []
        # one claim per source model per entity per frame (lets DIFFERENT
        # detectors fuse onto the same entity, but de-dupes a single detector)
        frame_sources: dict[str, set[str]] = {}
        prev_zone = {e.id: e.zone for e in self.entities.values()}
        prev_conflict = {e.id: e.conflict for e in self.entities.values()}
        seen_ids: set[str] = set()

        for c in sorted(claims, key=lambda c: -c.confidence):
            ct = canonical(c.entity_type)
            ent = self._associate(c, ct, frame_sources)
            if ent is None:
                ent = self._mint(c, ct, frame)
                out.append(Delta("entity.appeared", ent.id,
                                 {"type": ent.type, "zone": ent.zone,
                                  "source": c.source_model}))
            else:
                self._fuse(ent, c, frame)
            frame_sources.setdefault(ent.id, set()).add(c.source_model)
            seen_ids.add(ent.id)

        # age out / lose
        for eid, e in list(self.entities.items()):
            if eid in seen_ids:
                e.missed = 0
                continue
            e.missed += 1
            if e.missed > self.max_missed:
                del self.entities[eid]
                out.append(Delta("entity.lost", eid, {"type": e.type}))

        # moved / new conflict
        for e in self.entities.values():
            if e.id in prev_zone and prev_zone[e.id] != e.zone:
                out.append(Delta("entity.moved", e.id,
                                 {"from": prev_zone[e.id], "to": e.zone}))
            if e.conflict and not prev_conflict.get(e.id, False):
                out.append(Delta("entity.type_conflict", e.id,
                                 {"votes": {k: round(v, 3) for k, v in e.votes.items()}}))

        # relations + workspace conflict
        self.relations = []
        for rule in self.relation_rules:
            self.relations.extend(rule(self))
        out.extend(self._workspace_delta())

        self.deltas.extend(out)
        return out

    def _associate(self, c: Claim, ct: str,
                   frame_sources: dict[str, set[str]]) -> Entity | None:
        def used(e: Entity) -> bool:
            return c.source_model in frame_sources.get(e.id, set())

        # spatial-first: a big overlap means the SAME physical thing even if a
        # different specialist labelled it differently (-> a recorded conflict).
        best, best_score = None, self.iou_thresh
        for e in self.entities.values():
            if used(e):
                continue
            v = iou(e.geometry, c.geometry)
            gate = self.iou_thresh if compatible(e.canonical_type, ct) else self.strong_iou
            if v >= gate and v >= best_score:
                best, best_score = e, v
        if best is not None:
            return best
        # centre fallback — compatible types only (don't merge distinct objects)
        best, best_d = None, self.center_gate
        ex, ey = c.center
        for e in self.entities.values():
            if used(e) or not compatible(e.canonical_type, ct):
                continue
            gx = (e.geometry[0] + e.geometry[2]) / 2
            gy = (e.geometry[1] + e.geometry[3]) / 2
            d = ((gx - ex) ** 2 + (gy - ey) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = e, d
        return best

    def _mint(self, c: Claim, ct: str, frame: int) -> Entity:
        # A detector that already knows the authoritative id/zone (e.g. a sim
        # bridge) passes it in attrs; a real camera detector does not, and the
        # ontology mints its own id + derives the zone from geometry.
        wid = c.attrs.get("world_id")
        eid = wid if wid and wid not in self.entities else None
        if eid is None:
            self._counts[ct] = self._counts.get(ct, 0) + 1
            eid = f"{ct}_{self._counts[ct]}"
        e = Entity(id=eid, type=c.entity_type, type_conf=c.confidence,
                   geometry=c.geometry,
                   zone=c.attrs.get("world_zone") or self.zone_map(*c.center),
                   attrs={k: v for k, v in c.attrs.items()
                          if k not in {"world_id", "world_zone"}},
                   sources={c.source_model},
                   votes={c.entity_type: c.confidence},
                   first_seen=frame, last_seen=frame)
        self.entities[eid] = e
        return e

    def _fuse(self, e: Entity, c: Claim, frame: int) -> None:
        e.geometry = c.geometry
        e.zone = c.attrs.get("world_zone") or self.zone_map(*c.center)
        e.last_seen = frame
        e.sources.add(c.source_model)
        e.attrs.update(c.attrs)
        e.votes[c.entity_type] = e.votes.get(c.entity_type, 0.0) + c.confidence
        self._recompute_type(e)

    def _recompute_type(self, e: Entity) -> None:
        # bucket raw votes by canonical type: IS-A-compatible detectors reinforce
        buckets: dict[str, float] = {}
        for raw, v in e.votes.items():
            buckets[canonical(raw)] = buckets.get(canonical(raw), 0.0) + v
        total = sum(buckets.values()) or 1.0
        win = max(buckets, key=buckets.get)
        # display the most specific raw label that rolls up to the winning bucket
        e.type = max((r for r in e.votes if canonical(r) == win),
                     key=lambda r: e.votes[r])
        e.type_conf = buckets[win] / total
        strong = [b for b, v in buckets.items() if v >= self.conflict_conf]
        e.conflict = len(strong) >= 2

    # -- projection to the rest of the stack -------------------------
    def world_state(self, frame: int, *, goal: str | None = None,
                    targets: dict[str, str] | None = None,
                    org_id: str = "local-demo") -> WorldState:
        targets = targets or {}
        objs: dict[str, Detection] = {}
        for e in self.entities.values():
            if e.canonical_type == "human":
                continue                                   # people aren't task objects
            status = DataStatus.LIVE if e.trusted else DataStatus.FALLBACK
            objs[e.id] = Detection(
                object_id=e.id, cls=e.canonical_type, zone=e.zone,
                target_zone=targets.get(e.id, e.zone),
                conf=round(e.type_conf, 3), status=status, verified_frame=e.last_seen,
            )
        return WorldState(frame=frame, objects=objs, goal=goal, org_id=org_id)

    def observation(self, frame: int) -> Observation:
        dets = tuple(
            Detection(e.id, e.canonical_type, e.zone, e.zone, round(e.type_conf, 3),
                      DataStatus.LIVE if e.trusted else DataStatus.FALLBACK, e.last_seen)
            for e in sorted(self.entities.values(), key=lambda e: e.id)
            if e.canonical_type != "human"
        )
        return Observation(frame=frame, detections=dets,
                           workspace_clear=not self._ws_conflict,
                           raw_ref=f"ontology://frame/{frame:06d}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "entities": [e.as_dict() for e in sorted(self.entities.values(), key=lambda e: e.id)],
            "relations": [r.as_dict() for r in self.relations],
            "workspace_conflict": self._ws_conflict,
            "conflicts": [e.id for e in self.entities.values() if e.conflict],
        }

    # -- workspace conflict ----------------------------------------
    def _workspace_delta(self) -> list[Delta]:
        conflict = any(r.predicate == "intersects"
                       and canonical(self._type_of(r.subject)) == "human"
                       and self._type_of(r.obj) in {"arm", "left_arm", "right_arm", "arm_zone"}
                       for r in self.relations)
        # also treat an explicit human/hand near an arm zone as a conflict
        conflict = conflict or any(
            r.predicate in {"near", "intersects"}
            and canonical(self._type_of(r.subject)) == "human"
            and "arm" in r.obj
            for r in self.relations
        )
        out: list[Delta] = []
        if conflict and not self._ws_conflict:
            out.append(Delta("workspace.conflict", "workspace",
                             {"relations": [r.as_dict() for r in self.relations
                                            if canonical(self._type_of(r.subject)) == "human"]}))
        elif not conflict and self._ws_conflict:
            out.append(Delta("workspace.clear", "workspace", {}))
        self._ws_conflict = conflict
        return out

    def _type_of(self, eid: str) -> str:
        e = self.entities.get(eid)
        return e.type if e else eid

    def live(self) -> list[Entity]:
        """Entities observed on the most recent frame (relations are about
        *now*, not memory)."""
        return [e for e in self.entities.values() if e.missed == 0]


# ---------------------------------------------------------------------------
# relation rules
# ---------------------------------------------------------------------------


def near_rule(ont: Ontology, radius: float = 0.14) -> list[Relation]:
    ents = ont.live()
    out: list[Relation] = []
    for i, a in enumerate(ents):
        acx = (a.geometry[0] + a.geometry[2]) / 2
        acy = (a.geometry[1] + a.geometry[3]) / 2
        for b in ents[i + 1:]:
            bcx = (b.geometry[0] + b.geometry[2]) / 2
            bcy = (b.geometry[1] + b.geometry[3]) / 2
            d = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
            if iou(a.geometry, b.geometry) > 0:
                out.append(Relation(a.id, "intersects", b.id))
            elif d <= radius:
                out.append(Relation(a.id, "near", b.id, conf=round(1 - d / radius, 2)))
    return out


def workspace_conflict_rule(ont: Ontology) -> list[Relation]:
    """A human/hand entity in the same zone as an arm entity -> intersects."""
    humans = [e for e in ont.live() if canonical(e.type) == "human"]
    arms = [e for e in ont.live() if "arm" in e.type or e.type == "robot"]
    out: list[Relation] = []
    for h in humans:
        for a in arms:
            if h.zone == a.zone or iou(h.geometry, a.geometry) > 0:
                out.append(Relation(h.id, "intersects", a.id, conf=0.9))
    return out


# ---------------------------------------------------------------------------
# Observe provider — the perception swarm
# ---------------------------------------------------------------------------

Detector = Callable[[Any], "list[Claim]"]


def claims_from_detections(source: str, dets: "list[Detection2D]",
                           device: str = "cam0") -> list[Claim]:
    """Lift a plain YOLO-style detector's output into provenance-carrying Claims."""
    return [Claim(source_model=source, entity_type=d.cls, confidence=d.conf,
                  geometry=d.xyxy, device=device) for d in dets]


class OntologyObserver:
    """``Observe`` provider: run a swarm of claim-producing detectors, fuse
    into the ontology, project a ``WorldState``/``Observation``. Drop-in for
    ``FrameObserver``."""

    def __init__(
        self,
        detectors: dict[str, Detector],
        *,
        frame_source: Callable[[WorldState], Any] | None = None,
        ontology: Ontology | None = None,
        targets: Callable[[str, str], str] | None = None,
    ) -> None:
        self.detectors = detectors
        self.frame_source = frame_source or (lambda w: None)
        self.ontology = ontology or Ontology()
        self.targets = targets
        self.last_deltas: list[Delta] = []

    def observe(self, world: WorldState) -> Observation:
        frame_img = self.frame_source(world)
        claims: list[Claim] = []
        for name, det in self.detectors.items():
            for c in det(frame_img):
                claims.append(c if c.source_model else
                              Claim(name, c.entity_type, c.confidence, c.geometry,
                                    c.device, c.ts, c.attrs))
        self.last_deltas = self.ontology.ingest(claims, world.frame)

        tgt = {}
        if self.targets:
            for e in self.ontology.entities.values():
                tgt[e.id] = self.targets(e.id, canonical(e.type))
        for e in self.ontology.entities.values():
            known = world.objects.get(e.id)
            if known is not None:
                tgt.setdefault(e.id, known.target_zone)

        # authoritative WorldState projected from the ontology (target zones
        # carried through, conflicted entities marked FALLBACK)
        self._world_state = self.ontology.world_state(
            world.frame, goal=world.goal, targets=tgt, org_id=world.org_id)
        dets = tuple(sorted(self._world_state.objects.values(),
                            key=lambda d: d.object_id))
        return Observation(
            frame=world.frame, detections=dets,
            workspace_clear=not self.ontology._ws_conflict,
            raw_ref=f"ontology://frame/{world.frame:06d}",
        )

    @property
    def wake(self) -> bool:
        return any(wakes(d) for d in self.last_deltas)


# ---------------------------------------------------------------------------
# a stub perception swarm (no models) — for tests / the demo
# ---------------------------------------------------------------------------


def stub_swarm(world_ref: Any, *, box: float = 0.08, arms: bool = True) -> dict[str, Detector]:
    """Cheap model-free specialists fed from the world: an object detector, a
    robot-state detector (both arms), and a human/hand detector (emits a hand
    at ``world.hand_xy`` when set). Swap each entry for real MXFP2 YOLO
    inference and nothing else changes.
    """
    pos: dict[str, tuple[float, float]] = {}
    _ARMS = {"left_arm": (0.30, 0.50), "right_arm": (0.70, 0.50)}

    def _place(oid: str) -> tuple[float, float]:
        if oid not in pos:
            h = abs(hash(oid))
            pos[oid] = (0.12 + (h % 76) / 100.0, 0.12 + (h // 97 % 76) / 100.0)
        return pos[oid]

    def _state():
        return world_ref.state() if hasattr(world_ref, "state") else world_ref

    def objects(_frame: Any) -> list[Claim]:
        st = _state()
        out: list[Claim] = []
        b = box / 2
        for oid, d in st.objects.items():
            cx, cy = _place(oid)
            out.append(Claim("yolo_objects", d.cls, float(d.conf),
                             (cx - b, cy - b, cx + b, cy + b), device="cam_table",
                             attrs={"world_id": oid, "world_zone": d.zone}))
        return out

    def robot(_frame: Any) -> list[Claim]:
        if not arms:
            return []
        b = 0.12 / 2
        return [Claim("yolo_robot", name, 0.99,
                      (x - b, y - b, x + b, y + b), device="cam_table")
                for name, (x, y) in _ARMS.items()]

    def humans(_frame: Any) -> list[Claim]:
        hand = getattr(world_ref, "hand_xy", None)   # set to (x, y) to simulate a hand
        if hand is None:
            return []
        cx, cy = hand
        b = box / 2
        return [Claim("yolo_human", "hand", 0.86,
                      (cx - b, cy - b, cx + b, cy + b), device="cam_overhead")]

    swarm: dict[str, Detector] = {"yolo_objects": objects, "yolo_human": humans}
    if arms:
        swarm["yolo_robot"] = robot
    return swarm
