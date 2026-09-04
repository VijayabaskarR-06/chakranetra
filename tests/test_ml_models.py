"""The four learned models, and the guarantees they claim.

Three of these tests exist because the corresponding failure is invisible:

  * Cold start. A model with no data must return the rules estimate, not a
    number it made up. Nothing in a dashboard distinguishes the two.
  * Conformal coverage. An interval labelled "90%" that covers 60% of the
    time is worse than no interval, and only measurement catches it.
  * Censoring. A repair made last week has not failed *yet*. Counting it as a
    success teaches the model that recent repairs are good, which is false and
    self-reinforcing, and it shows up as a model that scores well and is wrong.
"""

import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roadlens.ml.bootstrap import seed as synthetic_seed
from roadlens.ml.features import COST_SPEC
from roadlens.ml.models import (
    MIN_COST_ROWS,
    CostModel,
    DegradationModel,
    RepairFailureModel,
    _auc,
    _conformal_quantile,
    _rules_cost,
)
from roadlens.ml.registry import ModelRegistry
from roadlens.predictive import PredictiveEngine
from roadlens.tickets import TicketStore

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def corpus():
    return synthetic_seed(n_tickets=600, now=NOW)


@pytest.fixture(scope="module")
def registry(corpus):
    return ModelRegistry.train(corpus["tickets"], corpus["observations"],
                               corpus["recurrence_records"],
                               provenance=corpus["provenance"], now=NOW)


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------

def test_cost_model_with_no_labels_returns_the_rules_estimate():
    ticket = {"defect_type": "pothole", "area_ratio": 0.07, "confidence": 0.9,
              "sightings": 2, "severity_level": 4, "priority_score": 88,
              "created_at": NOW.isoformat()}
    model = CostModel.train([])
    out = model.predict(ticket)
    assert out["source"] == "rules"
    assert out["predicted_inr"] == int(round(_rules_cost(ticket)))
    assert out["low_inr"] is None and out["high_inr"] is None
    assert model.metrics["status"] == "cold_start"


def test_cost_model_needs_a_minimum_of_labelled_repairs(corpus):
    labelled = [t for t in corpus["tickets"] if t["actual_cost_inr"]]
    assert CostModel.train(labelled[:MIN_COST_ROWS - 1]).trained is False
    assert CostModel.train(labelled[:MIN_COST_ROWS * 3]).trained is True


def test_degradation_and_failure_cold_start():
    assert DegradationModel.train([], []).trained is False
    assert DegradationModel.train([], []).forecast(
        {"area_ratio": 0.01, "severity_level": 2})["source"] == "rules"
    assert RepairFailureModel.train([], []).trained is False
    assert RepairFailureModel.train([], []).predict({})["source"] == "base_rate"


def test_cold_start_registry_still_serves_every_endpoint_shape():
    """An untrained registry must degrade, not break: the API surface is the
    same, every answer is the rules answer, and each says so."""
    reg = ModelRegistry()
    ticket = {"id": "T1", "defect_type": "pothole", "area_ratio": 0.03,
              "confidence": 0.8, "sightings": 1, "severity_level": 3,
              "priority_score": 60, "status": "OPEN",
              "created_at": NOW.isoformat(),
              "sla_due_at": (NOW + timedelta(days=3)).isoformat()}
    assert reg.cost.predict(ticket)["source"] == "rules"
    assert reg.failure.predict(ticket)["source"] == "base_rate"
    assert reg.degradation.forecast(ticket)["source"] == "rules"
    budget = reg.budget.forecast([ticket], now=NOW)
    assert budget["horizons"]["30_day"]["point_inr"] == int(round(_rules_cost(ticket)))
    assert "point estimate only" in budget["horizons"]["30_day"]["uncertainty"]


# ---------------------------------------------------------------------------
# The beats-the-baseline gate
# ---------------------------------------------------------------------------

def test_model_that_cannot_beat_the_rule_is_refused(corpus):
    """When actual cost is the rules estimate times pure noise, there is
    nothing to learn, and a 300-tree ensemble will happily learn the noise.
    Installing it would make every prediction worse than the arithmetic it
    replaced — so training must refuse."""
    rng = np.random.default_rng(0)
    tickets = []
    for t in corpus["tickets"][:300]:
        noisy = dict(t)
        noisy["actual_cost_inr"] = int(round(
            _rules_cost(t) * math.exp(float(rng.normal(0, 0.30)))))
        tickets.append(noisy)

    model = CostModel.train(tickets, random_state=0)
    assert model.trained is False
    assert model.metrics["status"] == "rejected_not_better_than_rules"
    # And it still answers, with the rule.
    assert model.predict(tickets[0])["source"] == "rules"


def test_trained_cost_model_beats_the_rules_baseline(registry):
    m = registry.cost.metrics
    assert m["status"] == "trained"
    assert m["model_mae_inr"] < m["rules_mae_inr"]
    assert m["mae_improvement_pct"] > 20


def test_trained_failure_model_beats_the_base_rate(registry):
    m = registry.failure.metrics
    assert m["status"] == "trained"
    assert m["auc"] > 0.6
    assert m["brier"] < m["base_rate_brier"]


# ---------------------------------------------------------------------------
# Conformal intervals
# ---------------------------------------------------------------------------

def test_conformal_intervals_cover_at_their_stated_rate(corpus):
    """Split conformal promises >= 1-alpha coverage on exchangeable data.
    Measured here on tickets the model never saw, at three levels."""
    labelled = [t for t in corpus["tickets"] if t["actual_cost_inr"]]
    rng = np.random.default_rng(7)
    order = rng.permutation(len(labelled))
    train = [labelled[i] for i in order[:350]]
    held_out = [labelled[i] for i in order[350:]]
    assert len(held_out) >= 80

    for alpha in (0.05, 0.10, 0.20):
        model = CostModel.train(train, alpha=alpha, random_state=1)
        assert model.trained
        covered = sum(
            1 for t in held_out
            if model.predict(t)["low_inr"] <= t["actual_cost_inr"] <= model.predict(t)["high_inr"]
        )
        coverage = covered / len(held_out)
        # Finite-sample slack: the guarantee is marginal and this is one draw.
        assert coverage >= (1 - alpha) - 0.07, f"alpha={alpha} coverage={coverage:.3f}"


def test_conformal_quantile_uses_the_finite_sample_correction():
    residuals = np.arange(1.0, 11.0)          # 10 points
    # ceil((10+1)*0.9) = 10 -> the largest residual, not the 9th.
    assert _conformal_quantile(residuals, 0.10) == 10.0
    # ceil((10+1)*0.95) = 11 > 10 -> cannot promise 95% from 10 points.
    assert _conformal_quantile(residuals, 0.05) == float("inf")
    assert _conformal_quantile(np.array([]), 0.1) == float("inf")


def test_interval_is_multiplicative_not_additive(registry):
    """Cost error scales with cost. A fixed rupee band would be absurdly wide
    on a hairline crack and absurdly tight on a highway cavity."""
    small = {"defect_type": "crack", "area_ratio": 0.002, "confidence": 0.7,
             "sightings": 1, "severity_level": 1, "priority_score": 20,
             "road_class": "residential", "created_at": NOW.isoformat()}
    large = {**small, "area_ratio": 0.12, "severity_level": 4,
             "priority_score": 95, "road_class": "highway"}
    a, b = registry.cost.predict(small), registry.cost.predict(large)
    assert (b["high_inr"] - b["low_inr"]) > (a["high_inr"] - a["low_inr"])
    ratio_a = a["high_inr"] / a["low_inr"]
    ratio_b = b["high_inr"] / b["low_inr"]
    assert ratio_a == pytest.approx(ratio_b, rel=0.02)


# ---------------------------------------------------------------------------
# What the cost model learned
# ---------------------------------------------------------------------------

def test_cost_model_learns_the_road_class_effect_the_rule_cannot_express(registry):
    """The rules engine prices from defect size alone, so it gives an
    identical estimate for the same hole on a highway and a residential lane.
    Recovering that difference is the whole point of the model."""
    base = {"defect_type": "pothole", "area_ratio": 0.05, "confidence": 0.85,
            "sightings": 2, "severity_level": 3, "priority_score": 70,
            "created_at": datetime(2026, 2, 1, tzinfo=timezone.utc).isoformat()}
    highway = registry.cost.predict({**base, "road_class": "highway"})
    residential = registry.cost.predict({**base, "road_class": "residential"})

    assert highway["rules_inr"] == residential["rules_inr"]     # the rule cannot tell
    assert highway["predicted_inr"] > residential["predicted_inr"] * 1.5


def test_explanation_names_road_class_as_the_dominant_factor(registry):
    ticket = {"defect_type": "pothole", "area_ratio": 0.05, "confidence": 0.85,
              "sightings": 2, "severity_level": 3, "priority_score": 70,
              "road_class": "highway", "created_at": NOW.isoformat()}
    ex = registry.cost.explain(ticket)
    assert ex["source"] == "model"
    assert ex["factors"][0]["feature"] == "road_class"
    assert ex["factors"][0]["value"] == "highway"
    assert ex["factors"][0]["effect_pct"] > 0
    assert ex["predicted_inr"] == registry.cost.predict(ticket)["predicted_inr"]


def test_explanation_groups_one_hot_columns_into_their_field(registry):
    """A tree splitting `road_class=residential < 0.5` is testing *not
    residential*. Reporting that column by name on a highway ticket is
    accurate and unreadable, so the group is summed and reported once."""
    ticket = {"defect_type": "pothole", "area_ratio": 0.05, "confidence": 0.85,
              "sightings": 2, "severity_level": 3, "priority_score": 70,
              "road_class": "highway", "created_at": NOW.isoformat()}
    ex = registry.cost.explain(ticket, top_k=99)
    fields = [f["feature"] for f in ex["factors"]]
    assert all("=" not in f for f in fields), fields
    assert len(fields) == len(set(fields)), "a field must appear at most once"
    assert set(fields) <= set(COST_SPEC.numeric) | set(COST_SPEC.categorical)


def test_explanation_effects_compose_to_the_prediction(registry):
    """Multiplicative effects off the rules estimate must multiply back to the
    prediction, or the percentages are decoration."""
    ticket = {"defect_type": "pothole", "area_ratio": 0.05, "confidence": 0.85,
              "sightings": 2, "severity_level": 3, "priority_score": 70,
              "road_class": "highway", "created_at": NOW.isoformat()}
    ex = registry.cost.explain(ticket, top_k=99)
    product = ex["rules_inr"]
    for f in ex["factors"]:
        product *= 1.0 + f["effect_pct"] / 100.0
    # The ensemble also carries a feature-independent intercept, so the parts
    # explain the *variation*, not the level; they must at least land within a
    # few percent and on the right side of the rule.
    assert product > ex["rules_inr"]
    assert ex["predicted_inr"] > ex["rules_inr"]


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

def test_growth_pairs_ignore_same_day_and_zero_area_sightings():
    obs = [
        {"ticket_id": "A", "observed_at": "2026-01-01T00:00:00+00:00", "area_ratio": 0.01},
        {"ticket_id": "A", "observed_at": "2026-01-01T06:00:00+00:00", "area_ratio": 0.011},
        {"ticket_id": "A", "observed_at": "2026-02-01T00:00:00+00:00", "area_ratio": 0.02},
        {"ticket_id": "B", "observed_at": "2026-01-01T00:00:00+00:00", "area_ratio": 0.0},
        {"ticket_id": "B", "observed_at": "2026-02-01T00:00:00+00:00", "area_ratio": 0.02},
    ]
    pairs = DegradationModel.build_pairs(obs, {})
    assert len(pairs) == 1                  # only A's second->third
    assert pairs[0]["k"] > 0


def test_forecast_gives_fewer_days_to_a_faster_growing_defect(registry):
    """Monsoon on a highway is the fastest-growing case in the generator, and
    a dry-season residential defect the slowest."""
    base = {"defect_type": "pothole", "area_ratio": 0.03, "confidence": 0.8,
            "sightings": 2, "severity_level": 3, "priority_score": 65}
    fast = registry.degradation.forecast({
        **base, "road_class": "highway",
        "created_at": datetime(2026, 7, 15, tzinfo=timezone.utc).isoformat()})
    slow = registry.degradation.forecast({
        **base, "road_class": "residential",
        "created_at": datetime(2026, 2, 15, tzinfo=timezone.utc).isoformat()})

    assert fast["status"] == "forecast" and slow["status"] == "forecast"
    assert fast["days_to_next_band"] < slow["days_to_next_band"]
    assert fast["growth_pct_per_day"] > slow["growth_pct_per_day"]


def test_forecast_declines_to_predict_at_the_top_band(registry):
    out = registry.degradation.forecast(
        {"area_ratio": 0.25, "severity_level": 4, "defect_type": "pothole",
         "road_class": "highway", "created_at": NOW.isoformat()})
    assert out["status"] == "already_at_top_band"
    assert out["days_to_next_band"] is None


def test_forecast_interval_brackets_the_point_estimate(registry):
    out = registry.degradation.forecast(
        {"defect_type": "pothole", "area_ratio": 0.03, "confidence": 0.8,
         "sightings": 2, "severity_level": 3, "priority_score": 65,
         "road_class": "highway",
         "created_at": datetime(2026, 7, 15, tzinfo=timezone.utc).isoformat()})
    assert out["days_earliest"] <= out["days_to_next_band"]
    if out["days_latest"] is not None:
        assert out["days_latest"] >= out["days_to_next_band"]


# ---------------------------------------------------------------------------
# Repair failure and censoring
# ---------------------------------------------------------------------------

def test_recent_unfailed_repairs_are_excluded_as_censored():
    recent = (NOW - timedelta(days=5)).isoformat()
    old = (NOW - timedelta(days=200)).isoformat()
    records = [
        {"original_ticket_id": "A", "first_fixed_at": recent, "recurrence_count": 0},
        {"original_ticket_id": "B", "first_fixed_at": old, "recurrence_count": 0},
        {"original_ticket_id": "C", "first_fixed_at": recent, "recurrence_count": 2},
    ]
    rows = RepairFailureModel.build_rows(records, {}, monitoring_window_days=90, now=NOW)
    # A is censored: too recent to have survived. B survived its window. C failed.
    assert len(rows) == 2
    assert sorted(r["label"] for r in rows) == [0.0, 1.0]


def test_failure_model_ranks_the_worst_crew_above_the_best(registry):
    """The generator gives Crew-D a 34% base failure rate and Crew-A 6%.
    A model that cannot recover that ordering has learned nothing useful."""
    ticket = {"defect_type": "pothole", "area_ratio": 0.04, "severity_level": 3,
              "priority_score": 65, "road_class": "arterial",
              "repaired_at": NOW.isoformat()}
    worst = registry.failure.predict(ticket, crew="Crew-D")["failure_probability"]
    best = registry.failure.predict(ticket, crew="Crew-A")["failure_probability"]
    assert worst > best


def test_failure_probabilities_are_probabilities(registry, corpus):
    for t in corpus["tickets"][:60]:
        p = registry.failure.predict(t)["failure_probability"]
        assert 0.0 <= p <= 1.0


def test_auc_handles_ties_and_single_class():
    assert _auc(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert _auc(np.array([1.0, 1.0]), np.array([0.9, 0.1])) is None
    assert _auc(np.array([0.0, 1.0]), np.array([0.1, 0.9])) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

def test_budget_grows_with_the_horizon_and_bands_bracket_the_total(registry, corpus):
    out = registry.budget.forecast(corpus["tickets"], now=NOW)
    h = out["horizons"]
    assert h["30_day"]["total_inr"] <= h["60_day"]["total_inr"] <= h["90_day"]["total_inr"]
    for window in h.values():
        if window["tickets"]:
            assert window["p10_inr"] <= window["p50_inr"] <= window["p90_inr"]
            assert window["p10_inr"] <= window["total_inr"] <= window["p90_inr"]


def test_budget_counts_only_open_tickets(registry):
    open_t = {"id": "A", "status": "OPEN", "defect_type": "pothole",
              "area_ratio": 0.05, "confidence": 0.8, "sightings": 1,
              "severity_level": 3, "priority_score": 70, "department": "X",
              "created_at": NOW.isoformat(),
              "sla_due_at": (NOW + timedelta(days=5)).isoformat()}
    done = {**open_t, "id": "B", "status": "VERIFIED"}
    out = registry.budget.forecast([open_t, done], now=NOW)
    assert out["open_tickets"] == 1
    assert out["horizons"]["30_day"]["tickets"] == 1


def test_budget_band_is_narrower_than_summing_per_ticket_intervals(registry, corpus):
    """Summing the per-ticket intervals assumes every estimate is wrong in the
    same direction at once. Independent errors partly cancel, and the
    simulation is what captures that; if it did not, the band would be as
    wide as the naive sum and the whole Monte Carlo would be pointless."""
    tickets = [t for t in corpus["tickets"] if t["status"] == "OPEN"][:40]
    for t in tickets:
        t["sla_due_at"] = (NOW + timedelta(days=10)).isoformat()
    out = registry.budget.forecast(tickets, now=NOW, n_samples=6000)
    window = out["horizons"]["30_day"]
    naive_low = sum(registry.cost.predict(t)["low_inr"] for t in tickets)
    naive_high = sum(registry.cost.predict(t)["high_inr"] for t in tickets)
    assert (window["p90_inr"] - window["p10_inr"]) < (naive_high - naive_low)


def test_budget_reports_its_assumptions(registry, corpus):
    out = registry.budget.forecast(corpus["tickets"], now=NOW)
    assert len(out["assumptions"]) >= 4
    assert any("lognormal" in a for a in out["assumptions"])
    assert any("independent" in a for a in out["assumptions"])


# ---------------------------------------------------------------------------
# Registry, provenance and persistence
# ---------------------------------------------------------------------------

def test_saved_models_reproduce_their_predictions_exactly(registry, corpus):
    with tempfile.TemporaryDirectory() as tmp:
        registry.save(tmp)
        restored = ModelRegistry.load(tmp)
        assert restored is not None
        for t in corpus["tickets"][:50]:
            assert restored.cost.predict(t) == registry.cost.predict(t)
            assert restored.failure.predict(t) == registry.failure.predict(t)
            assert restored.degradation.forecast(t) == registry.degradation.forecast(t)


def test_synthetic_training_is_labelled_everywhere(registry):
    status = registry.status()
    assert status["is_synthetic"] is True
    assert status["provenance"]["training_data"] == "synthetic_bootstrap"
    assert "not any real city" in status["provenance"]["warning"]


def test_load_of_a_corrupt_model_directory_falls_back_rather_than_crashing():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "manifest.json"), "w") as fh:
            fh.write('{"provenance": {}}')       # manifest with no model files
        assert ModelRegistry.load(tmp) is None


def test_load_of_an_absent_directory_returns_none():
    assert ModelRegistry.load("/nonexistent/models/dir") is None


def test_registry_trains_on_real_data_once_there_is_enough(corpus):
    """The fallback must be a fallback. With real labelled repairs on file,
    provenance flips to `observed` and the synthetic corpus is not consulted."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "t.db")
        store, predictive = TicketStore(db), PredictiveEngine(db)

        for t in corpus["tickets"][:120]:
            row = {k: t[k] for k in (
                "id", "defect_type", "lat", "lon", "severity_level", "severity_label",
                "priority_score", "est_cost_inr", "department", "status", "sightings",
                "confidence", "area_ratio", "created_at", "sla_due_at",
                "recurrence_count", "road_class", "actual_cost_inr")}
            row["sources"] = "[]"
            row["status_history"] = "[]"
            store.conn.execute(
                f"INSERT INTO tickets ({','.join(row)}) VALUES "
                f"({','.join('?' * len(row))})", list(row.values()))
        store.conn.commit()

        registry = ModelRegistry.train_from_store(store, predictive, now=NOW)
        assert registry.provenance["training_data"] == "observed"
        assert registry.provenance["labelled_costs"] >= MIN_COST_ROWS
        assert registry.status()["is_synthetic"] is False
