"""
Chakranetra — Learned Models
============================
Four heads, all built on `roadlens.ml.gbt`:

  CostModel            what a repair will actually cost, in rupees
  DegradationModel     how many days until a defect reaches the next severity band
  RepairFailureModel   the probability a repair will fail and the defect return
  BudgetForecast       ward-level 30/60/90-day spend, composed from the above

Three principles run through all of them, and they are the reason this file
is longer than a model file usually needs to be.

**Cold start is a first-class state, not an error.** `severity.py` says, in
as many words, that rules deciding public spending must be explainable — and
it is right. So none of these models replaces the rules engine; each one
*corrects* it, starting from zero correction. Below `MIN_TRAINING_ROWS`
labelled examples every model reports `source: "rules"` and returns the
rules-engine answer unchanged. A city that has entered no invoices sees
exactly today's behaviour, plus an honest note that the model is not trained.

**A model that cannot beat the rule does not get to serve.** Each `train()`
scores itself against the rules baseline on held-out data and refuses to
install itself if it is not better. A learned model that is worse than the
arithmetic it replaced is the single most expensive failure mode here, and it
is silent unless something checks.

**Uncertainty is reported, not hidden.** Point estimates for money invite
false confidence, so the cost model carries split-conformal prediction
intervals: given exchangeable data, the interval covers the true cost at the
requested rate, with no assumption about the shape of the error distribution.
`tests/test_ml_models.py` measures that coverage empirically.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from .features import (
    COST_SPEC,
    DEGRADATION_SPEC,
    FAILURE_SPEC,
    FeatureSpec,
    cost_row,
    degradation_row,
    failure_row,
    parse_ts,
)
from .gbt import GradientBoostedTrees
from ..logger import get_logger

logger = get_logger("ml")

# Below this many labelled rows a fit is noise. The number is deliberately
# conservative: with 19 one-hot features, fewer than a few dozen examples
# per model produces trees that memorise individual tickets.
MIN_COST_ROWS = 40
MIN_DEGRADATION_PAIRS = 30
MIN_FAILURE_ROWS = 40

# Fraction of the training data held back for conformal calibration and for
# the beats-the-baseline check. Never used to fit.
CALIBRATION_FRACTION = 0.3
MIN_CALIBRATION_ROWS = 12


def _severity_thresholds() -> list[tuple]:
    from ..severity import _severity_settings
    levels, _, _ = _severity_settings()
    return levels


def _rules_cost(ticket: dict) -> float:
    """The rules-engine estimate for a ticket, recomputed rather than trusted.

    A stored `est_cost_inr` may predate a change to config.yaml's severity
    bands; recomputing keeps the model's base score consistent with the rule
    the city is running today.
    """
    from ..severity import assess
    a = assess(
        defect_type=ticket.get("defect_type") or "pothole",
        area_ratio=float(ticket.get("area_ratio") or 0.0),
        confidence=float(ticket.get("confidence") or 0.0),
        sightings=int(ticket.get("sightings") or 1),
    )
    return float(a.est_cost_inr)


def _split(n: int, random_state: int):
    """Deterministic train/calibration split of `n` rows."""
    rng = np.random.default_rng(random_state)
    order = rng.permutation(n)
    n_cal = max(MIN_CALIBRATION_ROWS, int(round(n * CALIBRATION_FRACTION)))
    n_cal = min(n_cal, n // 2)          # never calibrate on more than half
    return order[n_cal:], order[:n_cal]


def _conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile with the finite-sample correction.

    The plain empirical (1-alpha) quantile under-covers on small calibration
    sets. The rank ceil((n+1)(1-alpha)) is what gives the distribution-free
    guarantee P(covered) >= 1-alpha for exchangeable data.
    """
    n = residuals.size
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")             # too few points to promise this level
    return float(np.sort(residuals)[k - 1])


# ---------------------------------------------------------------------------
# 1. Cost
# ---------------------------------------------------------------------------

class CostModel:
    """Predicts the rupee cost of a repair, as a correction to the rule.

    The model works in log space: it predicts `log(actual) - log(rules)`, so
    an untrained model predicts a correction of zero and returns the rules
    estimate exactly. Log space also means the prediction cannot go negative
    and the conformal interval is multiplicative — plus-or-minus a
    *percentage*, which is how repair cost error actually behaves. A fixed
    plus-or-minus rupees band would be absurdly wide on a small crack and
    absurdly tight on a highway cavity.
    """

    kind = "cost"

    def __init__(self, model: GradientBoostedTrees | None = None,
                 spec: FeatureSpec | None = None,
                 conformal_q: float | None = None,
                 alpha: float = 0.1,
                 metrics: dict | None = None):
        self.model = model
        self.spec = spec or COST_SPEC
        self.conformal_q = conformal_q
        self.alpha = float(alpha)
        self.metrics = metrics or {}

    @property
    def trained(self) -> bool:
        return self.model is not None

    # -- training -----------------------------------------------------------

    @classmethod
    def train(cls, tickets: list[dict], alpha: float = 0.1,
              random_state: int = 0, **gbt_kwargs) -> "CostModel":
        labelled = [
            t for t in tickets
            if t.get("actual_cost_inr") and float(t["actual_cost_inr"]) > 0
        ]
        if len(labelled) < MIN_COST_ROWS:
            logger.info("Cost model not trained: too few labelled repairs",
                        labelled=len(labelled), required=MIN_COST_ROWS)
            return cls(alpha=alpha, metrics={
                "status": "cold_start",
                "labelled_rows": len(labelled),
                "required_rows": MIN_COST_ROWS,
            })

        spec = COST_SPEC
        rules = np.array([_rules_cost(t) for t in labelled], dtype=np.float64)
        X = spec.matrix([cost_row(t, r) for t, r in zip(labelled, rules)])
        y = np.log(np.array([float(t["actual_cost_inr"]) for t in labelled]))
        base = np.log(np.maximum(rules, 1.0))

        fit_idx, cal_idx = _split(len(labelled), random_state)

        params = dict(loss="squared", n_estimators=300, learning_rate=0.05,
                      max_depth=3, min_samples_leaf=5, lambda_=1.0,
                      subsample=0.8, random_state=random_state)
        params.update(gbt_kwargs)
        gbt = GradientBoostedTrees(**params).fit(
            X[fit_idx], y[fit_idx], base_score=base[fit_idx]
        )

        # Held-out comparison against the rule the model would replace.
        pred_log = gbt.decision_function(X[cal_idx], base_score=base[cal_idx])
        model_mae = float(np.mean(np.abs(np.exp(pred_log) - np.exp(y[cal_idx]))))
        rules_mae = float(np.mean(np.abs(rules[cal_idx] - np.exp(y[cal_idx]))))

        if not (model_mae < rules_mae):
            logger.warning("Cost model rejected: does not beat the rules baseline",
                           model_mae_inr=round(model_mae), rules_mae_inr=round(rules_mae))
            return cls(alpha=alpha, metrics={
                "status": "rejected_not_better_than_rules",
                "labelled_rows": len(labelled),
                "model_mae_inr": round(model_mae),
                "rules_mae_inr": round(rules_mae),
            })

        residuals = np.abs(y[cal_idx] - pred_log)
        q = _conformal_quantile(residuals, alpha)

        actual = np.exp(y[cal_idx])
        mape = float(np.mean(np.abs(np.exp(pred_log) - actual) / actual))
        metrics = {
            "status": "trained",
            "labelled_rows": len(labelled),
            "fit_rows": int(fit_idx.size),
            "calibration_rows": int(cal_idx.size),
            "model_mae_inr": round(model_mae),
            "rules_mae_inr": round(rules_mae),
            "mae_improvement_pct": round(100.0 * (1 - model_mae / rules_mae), 1),
            "mape": round(mape, 4),
            "interval_level": round(1 - alpha, 3),
            "interval_half_width_pct": round(100.0 * (math.exp(q) - 1), 1),
            "trees": len(gbt.trees),
        }
        logger.info("Cost model trained", **{k: metrics[k] for k in
                    ("labelled_rows", "model_mae_inr", "rules_mae_inr", "mape")})
        return cls(model=gbt, spec=spec, conformal_q=q, alpha=alpha, metrics=metrics)

    # -- prediction ---------------------------------------------------------

    def predict(self, ticket: dict) -> dict:
        rules = _rules_cost(ticket)
        if not self.trained:
            return {
                "predicted_inr": int(round(rules)),
                "low_inr": None,
                "high_inr": None,
                "rules_inr": int(round(rules)),
                "source": "rules",
                "note": self.metrics.get("status", "cold_start"),
            }

        x = self.spec.vector(cost_row(ticket, rules))
        base = math.log(max(rules, 1.0))
        pred_log = float(self.model.decision_function(x.reshape(1, -1),
                                                      base_score=base)[0])
        q = self.conformal_q if self.conformal_q is not None else float("inf")
        low = math.exp(pred_log - q) if math.isfinite(q) else None
        high = math.exp(pred_log + q) if math.isfinite(q) else None
        return {
            "predicted_inr": int(round(math.exp(pred_log))),
            "low_inr": int(round(low)) if low is not None else None,
            "high_inr": int(round(high)) if high is not None else None,
            "rules_inr": int(round(rules)),
            "delta_vs_rules_pct": round(100.0 * (math.exp(pred_log) / rules - 1), 1),
            "interval_level": round(1 - self.alpha, 3),
            "source": "model",
        }

    def explain(self, ticket: dict, top_k: int = 6) -> dict:
        """Why this number — as multiplicative effects on the rules estimate.

        One-hot columns are summed back into the field they came from. A tree
        that splits on `road_class=residential < 0.5` is testing "is this
        *not* a residential road", so attributing that credit to
        `road_class=residential` on a highway ticket is technically accurate
        and reads as nonsense. The officer's question is what the road class
        did to the price, and that is one number: the sum over the group.
        """
        rules = _rules_cost(ticket)
        if not self.trained:
            return {"source": "rules", "rules_inr": int(round(rules)), "factors": []}

        x = self.spec.vector(cost_row(ticket, rules))
        base = math.log(max(rules, 1.0))
        ex = self.model.explain(x, base_score=base)
        names = self.spec.names

        grouped: dict[str, float] = {}
        values: dict[str, object] = {}
        row = cost_row(ticket, rules)
        for i, contribution in enumerate(ex["contributions"]):
            field = names[i].split("=", 1)[0]
            grouped[field] = grouped.get(field, 0.0) + float(contribution)
            values.setdefault(field, row.get(field))

        pairs = sorted(((f, c) for f, c in grouped.items() if abs(c) > 1e-12),
                       key=lambda p: abs(p[1]), reverse=True)
        return {
            "source": "model",
            "rules_inr": int(round(rules)),
            "predicted_inr": int(round(math.exp(ex["raw_prediction"]))),
            "factors": [
                {
                    "feature": field,
                    "value": values.get(field),
                    "effect_pct": round(100.0 * (math.exp(c) - 1), 1),
                    "direction": "increases" if c > 0 else "decreases",
                }
                for field, c in pairs[:top_k]
            ],
        }

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "spec": self.spec.to_dict(),
            "alpha": self.alpha,
            "conformal_q": self.conformal_q,
            "metrics": self.metrics,
            "gbt": self.model.to_dict() if self.model else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CostModel":
        gbt = GradientBoostedTrees.from_dict(data["gbt"]) if data.get("gbt") else None
        return cls(model=gbt, spec=FeatureSpec.from_dict(data["spec"]),
                   conformal_q=data.get("conformal_q"),
                   alpha=float(data.get("alpha", 0.1)),
                   metrics=data.get("metrics", {}))


# ---------------------------------------------------------------------------
# 2. Degradation
# ---------------------------------------------------------------------------

class DegradationModel:
    """Predicts how fast a defect is growing, and so when it turns critical.

    The target is the exponential growth constant `k` in
    `area(t) = area(0) * exp(k*t)`, fit from consecutive sightings of the same
    defect: `k = ln(area2/area1) / dt`. Exponential rather than linear because
    a pothole grows by losing its edges — the bigger the hole, the more edge
    there is to lose, and water ingress accelerates as the hole deepens.

    Working in the ratio also cancels the camera-distance bias that makes raw
    `area_ratio` noisy: a defect photographed from twice as far has a smaller
    area in both frames, and the ratio is unaffected.
    """

    kind = "degradation"

    # Growth slower than this is indistinguishable from measurement noise, and
    # dividing by it produces "this pothole becomes critical in 400 years",
    # which is not a forecast worth showing anyone.
    MIN_MEANINGFUL_K = 1e-4

    def __init__(self, model: GradientBoostedTrees | None = None,
                 spec: FeatureSpec | None = None,
                 conformal_q: float | None = None,
                 alpha: float = 0.2,
                 metrics: dict | None = None):
        self.model = model
        self.spec = spec or DEGRADATION_SPEC
        self.conformal_q = conformal_q
        self.alpha = float(alpha)
        self.metrics = metrics or {}

    @property
    def trained(self) -> bool:
        return self.model is not None

    @staticmethod
    def build_pairs(observations: list[dict], tickets_by_id: dict) -> list[dict]:
        """Turn a sighting history into (features, growth-rate) training rows."""
        by_ticket: dict[str, list[dict]] = {}
        for o in observations:
            by_ticket.setdefault(o["ticket_id"], []).append(o)

        pairs = []
        for ticket_id, obs in by_ticket.items():
            obs = sorted(obs, key=lambda o: parse_ts(o["observed_at"]))
            ticket = tickets_by_id.get(ticket_id, {})
            for a, b in zip(obs, obs[1:]):
                dt_days = (parse_ts(b["observed_at"]) - parse_ts(a["observed_at"])).total_seconds() / 86400.0
                a1, a2 = float(a["area_ratio"] or 0.0), float(b["area_ratio"] or 0.0)
                # Two sightings on the same day carry no rate information, and
                # a zero area makes the log ratio undefined.
                if dt_days <= 0.5 or a1 <= 0.0 or a2 <= 0.0:
                    continue
                k = math.log(a2 / a1) / dt_days
                row = degradation_row(
                    {**a, "recurrence_count": ticket.get("recurrence_count")},
                    interval_days=dt_days,
                    road_class=ticket.get("road_class"),
                    defect_type=ticket.get("defect_type"),
                )
                pairs.append({"features": row, "k": k, "ticket_id": ticket_id})
        return pairs

    @classmethod
    def train(cls, observations: list[dict], tickets: list[dict],
              alpha: float = 0.2, random_state: int = 0, **gbt_kwargs) -> "DegradationModel":
        tickets_by_id = {t["id"]: t for t in tickets}
        pairs = cls.build_pairs(observations, tickets_by_id)

        if len(pairs) < MIN_DEGRADATION_PAIRS:
            logger.info("Degradation model not trained: too few observation pairs",
                        pairs=len(pairs), required=MIN_DEGRADATION_PAIRS)
            return cls(alpha=alpha, metrics={
                "status": "cold_start",
                "observation_pairs": len(pairs),
                "required_pairs": MIN_DEGRADATION_PAIRS,
            })

        spec = DEGRADATION_SPEC
        X = spec.matrix([p["features"] for p in pairs])
        y = np.array([p["k"] for p in pairs], dtype=np.float64)

        fit_idx, cal_idx = _split(len(pairs), random_state)
        params = dict(loss="squared", n_estimators=250, learning_rate=0.05,
                      max_depth=3, min_samples_leaf=5, subsample=0.8,
                      random_state=random_state)
        params.update(gbt_kwargs)
        gbt = GradientBoostedTrees(**params).fit(X[fit_idx], y[fit_idx])

        pred = gbt.predict(X[cal_idx])
        # The baseline any rate model must beat is "assume every defect grows
        # at the fleet average rate" — the constant predictor.
        constant = float(np.mean(y[fit_idx]))
        model_mae = float(np.mean(np.abs(pred - y[cal_idx])))
        constant_mae = float(np.mean(np.abs(constant - y[cal_idx])))

        if not (model_mae < constant_mae):
            logger.warning("Degradation model rejected: no better than the mean growth rate",
                           model_mae=model_mae, constant_mae=constant_mae)
            return cls(alpha=alpha, metrics={
                "status": "rejected_not_better_than_mean",
                "observation_pairs": len(pairs),
                "model_mae": round(model_mae, 6),
                "constant_mae": round(constant_mae, 6),
            })

        q = _conformal_quantile(np.abs(y[cal_idx] - pred), alpha)
        metrics = {
            "status": "trained",
            "observation_pairs": len(pairs),
            "fit_rows": int(fit_idx.size),
            "calibration_rows": int(cal_idx.size),
            "model_mae": round(model_mae, 6),
            "constant_mae": round(constant_mae, 6),
            "mae_improvement_pct": round(100.0 * (1 - model_mae / constant_mae), 1),
            "median_growth_pct_per_day": round(100.0 * (math.exp(float(np.median(y))) - 1), 3),
            "interval_level": round(1 - alpha, 3),
            "trees": len(gbt.trees),
        }
        logger.info("Degradation model trained", **{k: metrics[k] for k in
                    ("observation_pairs", "model_mae", "constant_mae")})
        return cls(model=gbt, spec=spec, conformal_q=q, alpha=alpha, metrics=metrics)

    # -- prediction ---------------------------------------------------------

    def growth_rate(self, ticket: dict, interval_days: float = 30.0) -> float | None:
        if not self.trained:
            return None
        row = degradation_row(
            {
                "area_ratio": ticket.get("area_ratio"),
                "confidence": ticket.get("confidence"),
                "severity_level": ticket.get("severity_level"),
                "observed_at": ticket.get("created_at"),
                "recurrence_count": ticket.get("recurrence_count"),
            },
            interval_days=interval_days,
            road_class=ticket.get("road_class"),
            defect_type=ticket.get("defect_type"),
        )
        return float(self.model.predict(self.spec.vector(row).reshape(1, -1))[0])

    def forecast(self, ticket: dict) -> dict:
        """Days until this defect reaches the next severity band."""
        area = float(ticket.get("area_ratio") or 0.0)
        level = int(ticket.get("severity_level") or 1)
        thresholds = _severity_thresholds()

        # The next band up is the smallest threshold strictly above this
        # defect's current size. Level 4 is the top band: nothing to forecast.
        higher = sorted(t[0] for t in thresholds if t[0] > area)
        if not higher or level >= max(t[1] for t in thresholds):
            return {"source": "model" if self.trained else "rules",
                    "status": "already_at_top_band",
                    "days_to_next_band": None,
                    "current_severity_level": level}

        target = higher[0]
        next_level = min((t[1] for t in thresholds if t[0] == target), default=level + 1)

        if not self.trained:
            return {"source": "rules", "status": self.metrics.get("status", "cold_start"),
                    "days_to_next_band": None, "next_severity_level": next_level,
                    "current_severity_level": level}

        k = self.growth_rate(ticket)
        if k is None or k <= self.MIN_MEANINGFUL_K or area <= 0.0:
            return {"source": "model", "status": "stable",
                    "days_to_next_band": None, "next_severity_level": next_level,
                    "current_severity_level": level,
                    "growth_pct_per_day": round(100.0 * (math.exp(k or 0.0) - 1), 3)}

        span = math.log(target / area)
        days = span / k

        # A faster growth rate means fewer days, so the interval on k inverts
        # into the interval on days: the upper rate gives the earlier date.
        q = self.conformal_q or 0.0
        k_hi, k_lo = k + q, k - q
        earliest = span / k_hi if k_hi > self.MIN_MEANINGFUL_K else None
        latest = span / k_lo if k_lo > self.MIN_MEANINGFUL_K else None

        return {
            "source": "model",
            "status": "forecast",
            "current_severity_level": level,
            "next_severity_level": next_level,
            "next_band_area_ratio": target,
            "growth_pct_per_day": round(100.0 * (math.exp(k) - 1), 3),
            "days_to_next_band": round(days, 1),
            "days_earliest": round(earliest, 1) if earliest else None,
            "days_latest": round(latest, 1) if latest else None,
            "interval_level": round(1 - self.alpha, 3),
        }

    def to_dict(self) -> dict:
        return {"kind": self.kind, "spec": self.spec.to_dict(), "alpha": self.alpha,
                "conformal_q": self.conformal_q, "metrics": self.metrics,
                "gbt": self.model.to_dict() if self.model else None}

    @classmethod
    def from_dict(cls, data: dict) -> "DegradationModel":
        gbt = GradientBoostedTrees.from_dict(data["gbt"]) if data.get("gbt") else None
        return cls(model=gbt, spec=FeatureSpec.from_dict(data["spec"]),
                   conformal_q=data.get("conformal_q"),
                   alpha=float(data.get("alpha", 0.2)),
                   metrics=data.get("metrics", {}))


# ---------------------------------------------------------------------------
# 3. Repair failure
# ---------------------------------------------------------------------------

class RepairFailureModel:
    """Probability that a repair fails and the defect returns.

    The existing crew board scores a crew on repairs that have *already*
    failed. This is the forward-looking counterpart: given the defect, the
    road, the season and the crew, how likely is this patch to come back?

    The subtle part is censoring. A repair made last week has not failed, but
    it has not survived either — it simply has not had time. Labelling it a
    success teaches the model that recent repairs are good ones, which is
    both false and self-reinforcing, since recent repairs are always the
    largest group. Only repairs whose monitoring window has fully elapsed
    contribute a negative label; failures count as soon as they happen.
    """

    kind = "failure"

    def __init__(self, model: GradientBoostedTrees | None = None,
                 spec: FeatureSpec | None = None,
                 base_rate: float = 0.0,
                 metrics: dict | None = None):
        self.model = model
        self.spec = spec or FAILURE_SPEC
        self.base_rate = float(base_rate)
        self.metrics = metrics or {}

    @property
    def trained(self) -> bool:
        return self.model is not None

    @staticmethod
    def build_rows(recurrence_records: list[dict], tickets_by_id: dict,
                   monitoring_window_days: int = 90,
                   now: datetime | None = None) -> list[dict]:
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=monitoring_window_days)

        rows = []
        for rec in recurrence_records:
            failed = int(rec.get("recurrence_count") or 0) > 0
            fixed_at = parse_ts(rec.get("first_fixed_at"), now)
            # Right-censored: still inside its window and not yet failed.
            if not failed and fixed_at > cutoff:
                continue
            ticket = tickets_by_id.get(rec.get("original_ticket_id"), {})
            rows.append({
                "features": failure_row({
                    "area_ratio": ticket.get("area_ratio"),
                    "severity_level": ticket.get("severity_level"),
                    "priority_score": ticket.get("priority_score"),
                    "first_fixed_at": rec.get("first_fixed_at"),
                    "prior_recurrences": 0,
                    "defect_type": rec.get("defect_type") or ticket.get("defect_type"),
                    "road_class": ticket.get("road_class"),
                    "assigned_crew": rec.get("assigned_crew"),
                }),
                "label": 1.0 if failed else 0.0,
            })
        return rows

    @classmethod
    def train(cls, recurrence_records: list[dict], tickets: list[dict],
              monitoring_window_days: int = 90, random_state: int = 0,
              now: datetime | None = None, **gbt_kwargs) -> "RepairFailureModel":
        tickets_by_id = {t["id"]: t for t in tickets}
        rows = cls.build_rows(recurrence_records, tickets_by_id,
                              monitoring_window_days, now=now)

        labels = np.array([r["label"] for r in rows], dtype=np.float64)
        base_rate = float(labels.mean()) if labels.size else 0.0

        # Both classes must be present; a single-class column teaches nothing
        # and makes AUC undefined.
        if len(rows) < MIN_FAILURE_ROWS or labels.min() == labels.max():
            logger.info("Failure model not trained: insufficient or single-class data",
                        rows=len(rows), required=MIN_FAILURE_ROWS)
            return cls(base_rate=base_rate, metrics={
                "status": "cold_start",
                "uncensored_rows": len(rows),
                "required_rows": MIN_FAILURE_ROWS,
                "observed_failure_rate": round(base_rate, 4),
            })

        # The crew vocabulary is data-dependent, so the spec is built here and
        # travels with the model — a crew added later lands in __other__
        # rather than shifting every column to its right.
        crews = sorted({r["features"].get("crew") for r in rows
                        if r["features"].get("crew")})
        spec = FeatureSpec(numeric=list(FAILURE_SPEC.numeric),
                           categorical={**FAILURE_SPEC.categorical, "crew": crews})

        X = spec.matrix([r["features"] for r in rows])
        fit_idx, cal_idx = _split(len(rows), random_state)

        if labels[fit_idx].min() == labels[fit_idx].max() or \
           labels[cal_idx].min() == labels[cal_idx].max():
            return cls(base_rate=base_rate, metrics={
                "status": "cold_start_single_class_split",
                "uncensored_rows": len(rows),
                "observed_failure_rate": round(base_rate, 4),
            })

        params = dict(loss="logistic", n_estimators=200, learning_rate=0.06,
                      max_depth=3, min_samples_leaf=5, subsample=0.8,
                      random_state=random_state)
        params.update(gbt_kwargs)
        gbt = GradientBoostedTrees(**params).fit(X[fit_idx], labels[fit_idx])

        p = gbt.predict(X[cal_idx])
        auc = _auc(labels[cal_idx], p)
        brier = float(np.mean((p - labels[cal_idx]) ** 2))
        # The baseline is predicting the fleet-wide failure rate for everyone.
        fit_rate = float(labels[fit_idx].mean())
        base_brier = float(np.mean((fit_rate - labels[cal_idx]) ** 2))

        if auc is None or auc <= 0.5 or brier >= base_brier:
            logger.warning("Failure model rejected: no better than the base rate",
                           auc=auc, brier=round(brier, 4), base_brier=round(base_brier, 4))
            return cls(base_rate=base_rate, metrics={
                "status": "rejected_not_better_than_base_rate",
                "uncensored_rows": len(rows),
                "auc": round(auc, 4) if auc is not None else None,
                "brier": round(brier, 4),
                "base_rate_brier": round(base_brier, 4),
                "observed_failure_rate": round(base_rate, 4),
            })

        metrics = {
            "status": "trained",
            "uncensored_rows": len(rows),
            "fit_rows": int(fit_idx.size),
            "calibration_rows": int(cal_idx.size),
            "auc": round(auc, 4),
            "brier": round(brier, 4),
            "base_rate_brier": round(base_brier, 4),
            "observed_failure_rate": round(base_rate, 4),
            "crews_known": len(crews),
            "trees": len(gbt.trees),
        }
        logger.info("Failure model trained", **{k: metrics[k] for k in
                    ("uncensored_rows", "auc", "brier")})
        return cls(model=gbt, spec=spec, base_rate=base_rate, metrics=metrics)

    def predict(self, ticket: dict, crew: str | None = None) -> dict:
        if not self.trained:
            return {
                "failure_probability": round(self.base_rate, 4),
                "source": "base_rate",
                "note": self.metrics.get("status", "cold_start"),
                "risk_label": _failure_label(self.base_rate),
            }
        row = failure_row({
            "area_ratio": ticket.get("area_ratio"),
            "severity_level": ticket.get("severity_level"),
            "priority_score": ticket.get("priority_score"),
            "first_fixed_at": ticket.get("repaired_at") or ticket.get("created_at"),
            "prior_recurrences": ticket.get("recurrence_count") or 0,
            "defect_type": ticket.get("defect_type"),
            "road_class": ticket.get("road_class"),
            "assigned_crew": crew or ticket.get("assigned_to"),
        })
        p = float(self.model.predict(self.spec.vector(row).reshape(1, -1))[0])
        return {
            "failure_probability": round(p, 4),
            "base_rate": round(self.base_rate, 4),
            "lift_vs_base_rate": round(p / self.base_rate, 2) if self.base_rate else None,
            "risk_label": _failure_label(p),
            "source": "model",
        }

    def to_dict(self) -> dict:
        return {"kind": self.kind, "spec": self.spec.to_dict(),
                "base_rate": self.base_rate, "metrics": self.metrics,
                "gbt": self.model.to_dict() if self.model else None}

    @classmethod
    def from_dict(cls, data: dict) -> "RepairFailureModel":
        gbt = GradientBoostedTrees.from_dict(data["gbt"]) if data.get("gbt") else None
        return cls(model=gbt, spec=FeatureSpec.from_dict(data["spec"]),
                   base_rate=float(data.get("base_rate", 0.0)),
                   metrics=data.get("metrics", {}))


def _failure_label(p: float) -> str:
    if p >= 0.5:
        return "high"
    if p >= 0.25:
        return "elevated"
    return "normal"


def _auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based AUC, ties averaged. None when only one class is present."""
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # Average ranks within tied score groups, or a model that outputs one
    # constant would score 1.0 or 0.0 instead of the correct 0.5.
    s = scores[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


# ---------------------------------------------------------------------------
# 4. Budget forecast
# ---------------------------------------------------------------------------

# Statuses that still represent money the city has not yet spent.
OPEN_STATUSES = ("OPEN", "ASSIGNED", "IN_PROGRESS", "REOPENED")


class BudgetForecast:
    """Ward-level 30/60/90-day spend, composed from the other models.

    This head trains nothing of its own. It answers the question a finance
    officer actually asks — *how much do I need to release this quarter, and
    how wrong could that be* — by combining three things the models already
    know: what each open ticket will cost, when its SLA falls due, and how
    likely it is to need doing twice.

    The band comes from a Monte Carlo over per-ticket uncertainty rather than
    from summing the per-ticket intervals, because summing intervals assumes
    every ticket is wrong in the same direction at once. That produces a band
    so wide it is useless; independent errors partly cancel, and the
    simulation captures that.

    Two assumptions are made here that the models themselves do not make, and
    both are stated in the output:

      * **Cost errors are lognormal.** Conformal intervals are
        distribution-free, but adding random variables needs a shape. The
        log-scale sigma is backed out of the conformal half-width, so the
        simulated marginal still matches the calibrated interval.
      * **Cost errors are independent across tickets.** A common shock — a
        bitumen price rise — would correlate them and widen the true band.
        A city with enough history can measure that; this does not guess it.
    """

    kind = "budget"

    def __init__(self, cost_model: CostModel, failure_model: RepairFailureModel,
                 random_state: int = 0):
        self.cost_model = cost_model
        self.failure_model = failure_model
        self.random_state = int(random_state)

    def forecast(self, tickets: list[dict], horizons: tuple = (30, 60, 90),
                 now: datetime | None = None, n_samples: int = 4000,
                 group_by: str = "department") -> dict:
        now = now or datetime.now(timezone.utc)
        horizons = tuple(sorted(int(h) for h in horizons))

        open_tickets = [t for t in tickets if t.get("status") in OPEN_STATUSES]

        rows = []
        for t in open_tickets:
            cost = self.cost_model.predict(t)
            fail = self.failure_model.predict(t)
            due = parse_ts(t.get("sla_due_at"), now)
            days_out = (due - now).total_seconds() / 86400.0
            # An overdue ticket is money owed now, so it lands in the nearest
            # horizon rather than dropping out of the forecast entirely.
            bucket = next((h for h in horizons if days_out <= h), None)
            rows.append({
                "ticket_id": t.get("id"),
                "group": t.get(group_by) or "Unassigned",
                "point_inr": float(cost["predicted_inr"]),
                "low_inr": cost.get("low_inr"),
                "high_inr": cost.get("high_inr"),
                "failure_probability": float(fail["failure_probability"]),
                "days_to_due": round(days_out, 1),
                "horizon": bucket,
                "cost_source": cost["source"],
            })

        sigma = self._log_sigma()
        rng = np.random.default_rng(self.random_state)

        summary = {}
        for h in horizons:
            in_horizon = [r for r in rows if r["horizon"] is not None and r["horizon"] <= h]
            summary[f"{h}_day"] = self._simulate(in_horizon, sigma, rng, n_samples)

        by_group: dict[str, dict] = {}
        for r in rows:
            if r["horizon"] is None:
                continue
            g = by_group.setdefault(r["group"], {"tickets": 0, "point_inr": 0.0,
                                                 "expected_rework_inr": 0.0})
            g["tickets"] += 1
            g["point_inr"] += r["point_inr"]
            g["expected_rework_inr"] += r["point_inr"] * r["failure_probability"]

        beyond = [r for r in rows if r["horizon"] is None]
        return {
            "generated_at": now.isoformat(),
            "open_tickets": len(open_tickets),
            "tickets_in_horizon": len(rows) - len(beyond),
            "tickets_beyond_horizon": len(beyond),
            "beyond_horizon_inr": int(round(sum(r["point_inr"] for r in beyond))),
            "horizons": summary,
            "by_group": {
                name: {
                    "tickets": v["tickets"],
                    "point_inr": int(round(v["point_inr"])),
                    "expected_rework_inr": int(round(v["expected_rework_inr"])),
                    "total_inr": int(round(v["point_inr"] + v["expected_rework_inr"])),
                }
                for name, v in sorted(by_group.items(),
                                      key=lambda kv: -kv[1]["point_inr"])
            },
            "cost_model": self.cost_model.metrics.get("status", "cold_start"),
            "failure_model": self.failure_model.metrics.get("status", "cold_start"),
            "assumptions": [
                "Repair cost errors modelled as lognormal, calibrated to the "
                "cost model's conformal interval width.",
                "Cost errors assumed independent across tickets; a common "
                "price shock would widen the true band.",
                "A ticket is booked to the horizon containing its SLA due "
                "date; overdue tickets fall into the nearest horizon.",
                "Rework is charged in the same horizon as the original "
                "repair, at cost x P(repair fails).",
            ],
        }

    def _log_sigma(self) -> float:
        """Log-scale sigma implied by the cost model's conformal half-width.

        Zero when the model is untrained, which makes the simulation collapse
        onto the rules point estimate — the honest answer when there is no
        calibrated uncertainty to report.
        """
        q = self.cost_model.conformal_q
        if not q or not math.isfinite(q):
            return 0.0
        from statistics import NormalDist
        z = NormalDist().inv_cdf(1.0 - self.cost_model.alpha / 2.0)
        return float(q / z) if z else 0.0

    def _simulate(self, rows: list[dict], sigma: float, rng, n_samples: int) -> dict:
        point = sum(r["point_inr"] for r in rows)
        rework = sum(r["point_inr"] * r["failure_probability"] for r in rows)

        if not rows:
            return {"tickets": 0, "point_inr": 0, "expected_rework_inr": 0,
                    "total_inr": 0, "p10_inr": 0, "p90_inr": 0,
                    "uncertainty": "no tickets in this horizon"}

        costs = np.array([r["point_inr"] for r in rows], dtype=np.float64)
        probs = np.array([r["failure_probability"] for r in rows], dtype=np.float64)

        # exp(N(0, sigma)) has median 1, so scaling by it leaves each ticket's
        # median at its point estimate rather than inflating it by exp(s^2/2).
        noise = np.exp(rng.normal(0.0, sigma, size=(n_samples, costs.size))) \
            if sigma > 0 else np.ones((n_samples, costs.size))
        sampled = costs * noise
        redo = rng.random((n_samples, costs.size)) < probs
        totals = (sampled * (1.0 + redo)).sum(axis=1)

        return {
            "tickets": len(rows),
            "point_inr": int(round(point)),
            "expected_rework_inr": int(round(rework)),
            "total_inr": int(round(point + rework)),
            "p10_inr": int(round(float(np.percentile(totals, 10)))),
            "p50_inr": int(round(float(np.percentile(totals, 50)))),
            "p90_inr": int(round(float(np.percentile(totals, 90)))),
            "uncertainty": "simulated" if sigma > 0 else "point estimate only "
                           "(cost model untrained, no calibrated interval)",
        }
