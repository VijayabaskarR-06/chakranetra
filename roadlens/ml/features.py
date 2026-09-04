"""
Chakranetra — Feature Extraction
================================
One place that turns a ticket row into numbers, so the training script, the
API, and the generated JavaScript all see identical features. Drift between
"features at training time" and "features at serving time" is the classic way
a model that scored well offline quietly rots in production; the only defence
is that there be exactly one implementation, which is this file.

A `FeatureSpec` is data, not code: a list of numeric fields and a list of
categorical fields with their fixed vocabularies. It serialises into the
model JSON, which means the browser builds its feature vector from the same
spec the model was trained under, and a vocabulary that changes invalidates
the saved model loudly instead of silently shifting column meanings.

Categoricals are one-hot encoded rather than label-encoded. Label encoding
would let a tree split "road_class < 2", inventing an order between
`residential` and `highway` that does not exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# Vocabularies are fixed and explicit. An unseen value maps to the trailing
# "__other__" column instead of raising, because a city adding a new defect
# type should degrade the prediction, not take the API down.
OTHER = "__other__"

DEFECT_TYPES = ["pothole", "crack", "manhole", "zebra_crossing", "footpath"]

# Road class drives repair cost more than anything except size: a highway
# patch needs traffic management, night working and a heavier mix, and bills
# several times what the same hole costs on a residential lane.
ROAD_CLASSES = ["highway", "arterial", "collector", "residential"]
DEFAULT_ROAD_CLASS = "arterial"

# Indian monsoon months (SW monsoon, Jun-Sep, plus the NE monsoon's Oct-Nov
# over the peninsula). Water ingress is the dominant cause of both new
# potholes and failed patches, so season is a real predictor, not a filler
# feature.
MONSOON_MONTHS = {6, 7, 8, 9, 10, 11}


def parse_ts(value, fallback: datetime | None = None) -> datetime:
    """Parse a stored ISO timestamp into an aware UTC datetime."""
    fallback = fallback or datetime.now(timezone.utc)
    if not value:
        return fallback
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return fallback
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class FeatureSpec:
    """The contract between training and serving."""
    numeric: list[str] = field(default_factory=list)
    categorical: dict[str, list[str]] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        """Column names of the produced matrix, in order."""
        out = list(self.numeric)
        for name, vocab in self.categorical.items():
            out.extend(f"{name}={v}" for v in list(vocab) + [OTHER])
        return out

    @property
    def n_features(self) -> int:
        return len(self.names)

    def vector(self, row: dict) -> np.ndarray:
        """Encode one prepared feature dict into a vector."""
        values = [_finite(row.get(n, 0.0)) for n in self.numeric]
        for name, vocab in self.categorical.items():
            value = row.get(name)
            onehot = [0.0] * (len(vocab) + 1)
            try:
                onehot[list(vocab).index(value)] = 1.0
            except ValueError:
                onehot[-1] = 1.0          # unseen category
            values.extend(onehot)
        return np.asarray(values, dtype=np.float64)

    def matrix(self, rows: list[dict]) -> np.ndarray:
        if not rows:
            return np.zeros((0, self.n_features), dtype=np.float64)
        return np.vstack([self.vector(r) for r in rows])

    def to_dict(self) -> dict:
        return {"numeric": list(self.numeric),
                "categorical": {k: list(v) for k, v in self.categorical.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSpec":
        return cls(numeric=list(data["numeric"]),
                   categorical={k: list(v) for k, v in data["categorical"].items()})


def _finite(value) -> float:
    """Coerce anything to a finite float.

    A None from a nullable SQLite column becoming NaN would poison every
    split comparison it touches (NaN < t is False, so the row silently always
    goes right), so nulls become 0.0 here and the caller's defaults handle
    real meaning.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


# ---------------------------------------------------------------------------
# The specs
# ---------------------------------------------------------------------------

COST_SPEC = FeatureSpec(
    numeric=[
        "area_ratio",
        "confidence",
        "sightings",
        "severity_level",
        "priority_score",
        "recurrence_count",
        "is_monsoon",
        "log_rules_cost",
    ],
    categorical={
        "defect_type": DEFECT_TYPES,
        "road_class": ROAD_CLASSES,
    },
)

DEGRADATION_SPEC = FeatureSpec(
    numeric=[
        "area_ratio",
        "confidence",
        "severity_level",
        "is_monsoon",
        "interval_days",
        "recurrence_count",
    ],
    categorical={
        "defect_type": DEFECT_TYPES,
        "road_class": ROAD_CLASSES,
    },
)

FAILURE_SPEC = FeatureSpec(
    numeric=[
        "area_ratio",
        "severity_level",
        "priority_score",
        "repair_month_is_monsoon",
        "prior_recurrences",
    ],
    categorical={
        "defect_type": DEFECT_TYPES,
        "road_class": ROAD_CLASSES,
        "crew": [],          # filled from the training data; see registry.py
    },
)


# ---------------------------------------------------------------------------
# Row preparation — SQLite row -> feature dict
# ---------------------------------------------------------------------------

def cost_row(ticket: dict, rules_cost: float | None = None) -> dict:
    """Feature dict for the cost model.

    `log_rules_cost` is included as a feature *and* used as the model's base
    score. As a base score it makes an untrained model reproduce the rules
    engine exactly; as a feature it additionally lets the trees learn where
    the rule is systematically wrong (say, that it under-prices highway work)
    rather than only how far off it is on average.
    """
    cost = float(rules_cost if rules_cost is not None
                 else ticket.get("est_cost_inr") or 1.0)
    created = parse_ts(ticket.get("created_at"))
    return {
        "area_ratio": ticket.get("area_ratio"),
        "confidence": ticket.get("confidence"),
        "sightings": min(_finite(ticket.get("sightings") or 1), 20),
        "severity_level": ticket.get("severity_level"),
        "priority_score": ticket.get("priority_score"),
        "recurrence_count": ticket.get("recurrence_count") or 0,
        "is_monsoon": 1.0 if created.month in MONSOON_MONTHS else 0.0,
        "log_rules_cost": math.log(max(cost, 1.0)),
        "defect_type": ticket.get("defect_type"),
        "road_class": ticket.get("road_class") or DEFAULT_ROAD_CLASS,
    }


def degradation_row(earlier: dict, interval_days: float,
                    road_class: str | None = None,
                    defect_type: str | None = None) -> dict:
    """Feature dict for one observation-pair of a growing defect."""
    observed = parse_ts(earlier.get("observed_at"))
    return {
        "area_ratio": earlier.get("area_ratio"),
        "confidence": earlier.get("confidence"),
        "severity_level": earlier.get("severity_level"),
        "is_monsoon": 1.0 if observed.month in MONSOON_MONTHS else 0.0,
        "interval_days": max(float(interval_days), 0.0),
        "recurrence_count": earlier.get("recurrence_count") or 0,
        "defect_type": defect_type or earlier.get("defect_type"),
        "road_class": road_class or earlier.get("road_class") or DEFAULT_ROAD_CLASS,
    }


def failure_row(record: dict) -> dict:
    """Feature dict for the repair-failure classifier."""
    repaired = parse_ts(record.get("first_fixed_at"))
    return {
        "area_ratio": record.get("area_ratio"),
        "severity_level": record.get("severity_level"),
        "priority_score": record.get("priority_score"),
        "repair_month_is_monsoon": 1.0 if repaired.month in MONSOON_MONTHS else 0.0,
        "prior_recurrences": record.get("prior_recurrences") or 0,
        "defect_type": record.get("defect_type"),
        "road_class": record.get("road_class") or DEFAULT_ROAD_CLASS,
        "crew": record.get("assigned_crew"),
    }
