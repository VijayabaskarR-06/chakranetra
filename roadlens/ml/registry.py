"""
Chakranetra — Model Registry
============================
Trains, saves, loads and serves the models in `roadlens.ml.models`.

The rule this file enforces is the one that keeps the whole feature honest:
**real data wins, and when there isn't any, the demo says so.** Training reads
the ticket store first. If the city has recorded enough real repair costs,
sightings and repair outcomes, those are what the models learn from and the
provenance reads `observed`. If not, the synthetic bootstrap corpus is used
so the console has something to show, and the provenance reads
`synthetic_bootstrap` — a label that is written into the model file, returned
by every API response, and rendered as a badge on the dashboard. There is no
configuration that makes synthetic data look real.

Models are plain JSON in `models/`, small enough to serve to a browser, which
is what lets `tools/generate_ml_js.py` ship the identical cost model
client-side. They are written compactly rather than pretty-printed: a
300-tree ensemble indented one space per level triples in size for no
readable benefit, since nobody reads a tree ensemble by eye.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .bootstrap import seed as synthetic_seed
from .models import (
    MIN_COST_ROWS,
    MIN_DEGRADATION_PAIRS,
    MIN_FAILURE_ROWS,
    BudgetForecast,
    CostModel,
    DegradationModel,
    RepairFailureModel,
)
from ..logger import get_logger

logger = get_logger("ml.registry")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.environ.get("ROADLENS_MODEL_DIR", os.path.join(ROOT, "models"))

FILES = {"cost": "cost.json", "degradation": "degradation.json",
         "failure": "failure.json"}
MANIFEST = "manifest.json"


class ModelRegistry:
    """The trained models plus where they came from."""

    def __init__(self, cost: CostModel | None = None,
                 degradation: DegradationModel | None = None,
                 failure: RepairFailureModel | None = None,
                 provenance: dict | None = None):
        self.cost = cost or CostModel(metrics={"status": "not_loaded"})
        self.degradation = degradation or DegradationModel(metrics={"status": "not_loaded"})
        self.failure = failure or RepairFailureModel(metrics={"status": "not_loaded"})
        self.provenance = provenance or {"training_data": "none"}

    @property
    def budget(self) -> BudgetForecast:
        return BudgetForecast(self.cost, self.failure)

    # -- training -----------------------------------------------------------

    @classmethod
    def train(cls, tickets: list[dict], observations: list[dict],
              recurrence_records: list[dict], provenance: dict | None = None,
              monitoring_window_days: int = 90,
              random_state: int = 0, now: datetime | None = None) -> "ModelRegistry":
        cost = CostModel.train(tickets, random_state=random_state)
        degradation = DegradationModel.train(observations, tickets,
                                             random_state=random_state)
        failure = RepairFailureModel.train(recurrence_records, tickets,
                                           monitoring_window_days=monitoring_window_days,
                                           random_state=random_state, now=now)
        prov = dict(provenance or {"training_data": "observed"})
        prov["trained_at"] = (now or datetime.now(timezone.utc)).isoformat()
        return cls(cost, degradation, failure, prov)

    @classmethod
    def train_from_store(cls, store, predictive=None, random_state: int = 0,
                         now: datetime | None = None,
                         allow_bootstrap: bool = True) -> "ModelRegistry":
        """Train from the live database, falling back to synthetic data.

        The fallback is per-registry, not per-model, deliberately. A registry
        with a real cost model and a synthetic failure model would need every
        downstream number to carry its own provenance, and the composed budget
        forecast would carry two. One decision, one label, one thing to
        explain to a municipal officer.
        """
        tickets = store.list()
        observations = store.observations()
        records = []
        if predictive is not None:
            records = [dict(r) for r in predictive.conn.execute(
                "SELECT * FROM recurrence_records")]

        labelled = sum(1 for t in tickets
                       if t.get("actual_cost_inr") and t["actual_cost_inr"] > 0)
        pairs = len(DegradationModel.build_pairs(observations,
                                                 {t["id"]: t for t in tickets}))
        uncensored = len(RepairFailureModel.build_rows(
            records, {t["id"]: t for t in tickets}, now=now))

        enough = (labelled >= MIN_COST_ROWS
                  or pairs >= MIN_DEGRADATION_PAIRS
                  or uncensored >= MIN_FAILURE_ROWS)

        if enough or not allow_bootstrap:
            logger.info("Training on recorded municipal data",
                        labelled_costs=labelled, growth_pairs=pairs,
                        repair_outcomes=uncensored)
            return cls.train(tickets, observations, records,
                             provenance={"training_data": "observed",
                                         "tickets": len(tickets),
                                         "labelled_costs": labelled,
                                         "growth_pairs": pairs,
                                         "repair_outcomes": uncensored},
                             random_state=random_state, now=now)

        logger.warning(
            "Not enough recorded data to train; using the synthetic bootstrap "
            "corpus. Predictions are labelled synthetic_bootstrap until real "
            "repair costs are recorded via POST /api/tickets/{id}/cost.",
            labelled_costs=labelled, growth_pairs=pairs, repair_outcomes=uncensored,
        )
        corpus = synthetic_seed()
        return cls.train(corpus["tickets"], corpus["observations"],
                         corpus["recurrence_records"],
                         provenance={**corpus["provenance"],
                                     "observed_labelled_costs": labelled,
                                     "observed_growth_pairs": pairs,
                                     "observed_repair_outcomes": uncensored},
                         random_state=random_state, now=now)

    # -- persistence --------------------------------------------------------

    def save(self, model_dir: str | None = None) -> str:
        model_dir = model_dir or MODEL_DIR
        os.makedirs(model_dir, exist_ok=True)
        for name, model in (("cost", self.cost), ("degradation", self.degradation),
                            ("failure", self.failure)):
            with open(os.path.join(model_dir, FILES[name]), "w", encoding="utf-8") as fh:
                json.dump(model.to_dict(), fh, separators=(",", ":"))
        with open(os.path.join(model_dir, MANIFEST), "w", encoding="utf-8") as fh:
            json.dump({"provenance": self.provenance, "metrics": self.metrics()},
                      fh, indent=1)
        logger.info("Models saved", model_dir=model_dir,
                    training_data=self.provenance.get("training_data"))
        return model_dir

    @classmethod
    def load(cls, model_dir: str | None = None) -> "ModelRegistry | None":
        model_dir = model_dir or MODEL_DIR
        manifest_path = os.path.join(model_dir, MANIFEST)
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
            def _read(name, klass):
                with open(os.path.join(model_dir, FILES[name]), encoding="utf-8") as fh:
                    return klass.from_dict(json.load(fh))
            return cls(_read("cost", CostModel),
                       _read("degradation", DegradationModel),
                       _read("failure", RepairFailureModel),
                       manifest.get("provenance", {}))
        except (OSError, ValueError, KeyError) as exc:
            # A model file from an older format version, or a half-written
            # directory, must not take the API down — the rules engine is
            # always a working fallback.
            logger.warning("Could not load saved models; falling back to rules",
                           error=str(exc), model_dir=model_dir)
            return None

    # -- reporting ----------------------------------------------------------

    def metrics(self) -> dict:
        return {
            "cost": self.cost.metrics,
            "degradation": self.degradation.metrics,
            "failure": self.failure.metrics,
        }

    def status(self) -> dict:
        """What the dashboard badge and `GET /api/ml/status` report."""
        return {
            "provenance": self.provenance,
            "is_synthetic": self.provenance.get("training_data") == "synthetic_bootstrap",
            "models": {
                "cost": {"trained": self.cost.trained, **self.cost.metrics},
                "degradation": {"trained": self.degradation.trained, **self.degradation.metrics},
                "failure": {"trained": self.failure.trained, **self.failure.metrics},
            },
        }


_registry: ModelRegistry | None = None


def get_registry(store=None, predictive=None) -> ModelRegistry:
    """Process-wide registry: saved models if present, trained on demand if not."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry.load()
    if _registry is None and store is not None:
        _registry = ModelRegistry.train_from_store(store, predictive)
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def reset_registry() -> None:
    global _registry
    _registry = None
