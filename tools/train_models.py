"""
Train Chakranetra's own models and write them to models/*.json.

    python tools/train_models.py                # real data if there is enough,
                                                # synthetic bootstrap if not
    python tools/train_models.py --no-bootstrap # refuse to fall back
    python tools/train_models.py --db other.db

Prints the held-out metrics for each head and, loudly, where the training
data came from. A model trained on the synthetic corpus is fit to demonstrate
the machinery and nothing else — see roadlens/ml/bootstrap.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roadlens.config import get_config              # noqa: E402
from roadlens.ml.registry import MODEL_DIR, ModelRegistry   # noqa: E402
from roadlens.predictive import PredictiveEngine    # noqa: E402
from roadlens.tickets import TicketStore            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Chakranetra's cost, degradation and repair-failure models")
    ap.add_argument("--db", default=None, help="ticket database (default: config.yaml db_path)")
    ap.add_argument("--out", default=MODEL_DIR, help=f"output directory (default: {MODEL_DIR})")
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="fail rather than train on the synthetic corpus")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    db = args.db or get_config().db_path or "roadlens.db"
    store = TicketStore(db)
    predictive = PredictiveEngine(db)

    registry = ModelRegistry.train_from_store(
        store, predictive, random_state=args.seed,
        allow_bootstrap=not args.no_bootstrap,
    )
    registry.save(args.out)

    source = registry.provenance.get("training_data")
    print()
    print("=" * 68)
    if source == "synthetic_bootstrap":
        print("  TRAINED ON SYNTHETIC DATA — demonstration only")
        print("  These models fit roadlens/ml/bootstrap.py's simulated city.")
        print("  They say nothing about any real city's repair costs.")
        print("  Record real costs with POST /api/tickets/{id}/cost, then")
        print("  re-run this script to train on them instead.")
    else:
        print("  TRAINED ON RECORDED MUNICIPAL DATA")
        for k in ("tickets", "labelled_costs", "growth_pairs", "repair_outcomes"):
            if k in registry.provenance:
                print(f"    {k:<18} {registry.provenance[k]}")
    print("=" * 68)
    print(json.dumps(registry.metrics(), indent=2))
    print(f"\nwrote {args.out}/")

    # A rejected or cold-start model is not a crash — the rules engine still
    # serves — but it should not pass silently in CI either.
    statuses = {k: v.get("status") for k, v in registry.metrics().items()}
    if any(s and s.startswith("rejected") for s in statuses.values()):
        print("\nWARNING: one or more models did not beat their baseline and "
              "were not installed:", statuses, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
