"""Load an exported planner-advisor checkpoint through OMNI-Q's real reasoner path and plan.

This is the integration gate, not a unit test: the same code path
``demo_intel_reasoner`` takes with ``OMNIQ_OMNI_REASONER=omni`` --
``OmniReferenceReasoner`` -> vendored ``load_omni_reference`` (digest, architecture
hash, execution revision, precision, checkpoint revision, state boundary all
verified) -> ``OmniInference.generate`` -> ``OmniPlanner`` parse + validate.

Usage:
    PYTHONPATH=src .venv/Scripts/python reasoner/roundtrip_check.py \
        --checkpoint models/omni_planner_smoke_latest.pt \
        --receipt models/omni_planner_smoke_latest_receipt.json --n 3 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reasoner"))

from omni_q.omni_planner import OmniPlanner  # noqa: E402
from omni_q.omni_reasoner import OmniReferenceReasoner  # noqa: E402
from train_planner_reasoner import plan_signature, world_from_meta  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--data", default="reasoner/data/planner_pairs.jsonl")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    heldout = [r for r in rows if r["split"] == "eval"][: args.n]

    reasoner = OmniReferenceReasoner(args.checkpoint, args.receipt, device=args.device, max_new_tokens=args.max_new)
    planner = OmniPlanner(reasoner)
    t0 = time.time()
    result = reasoner.reason("roundtrip-0", heldout[0]["prompt"])
    print(f"backend={result.backend} artifact={result.artifact_identity.get('checkpoint_sha256','?')[:12]} "
          f"first turn {time.time()-t0:.1f}s, {len(result.token_ids)} tokens")

    ok = 0
    for i, row in enumerate(heldout):
        t0 = time.time()
        res = reasoner.reason(f"roundtrip-{i}", row["prompt"])
        world = world_from_meta(row["meta"])
        proposals, rationale = planner._parse(res.text)
        steps, rejected = planner._validate(proposals, world)
        manip = [s for s in steps if s.contract == "manipulate"]
        exact = plan_signature(res.text) == plan_signature(row["target"])
        ok += bool(manip)
        print(f"--- held-out {i} ({time.time()-t0:.1f}s): proposed {len(proposals)}, accepted {len(manip)}, "
              f"rejected {len(rejected)}, exact={exact}")
        print("model :", repr(res.text[:240]))
        print("oracle:", repr(row["target"][:240]))
        if rejected:
            print("rejections:", rejected)
    print(f"usable plans: {ok}/{len(heldout)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
