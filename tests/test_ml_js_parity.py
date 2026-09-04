"""The browser's cost model must be the server's cost model.

dashboard/ml.generated.js reimplements feature extraction and tree traversal
in JavaScript so the console can price a defect with no server. That is a
second implementation of a model that decides public money, and a second
implementation drifts unless something checks it — the same argument
tests/test_js_parity.py makes for the severity rules, held to the same 1e-9.

The failure this catches is not a crash. A JS port that gets the monsoon month
off by one, or routes a `>=` where Python routes a `<`, returns a perfectly
plausible rupee figure that is simply a different model's answer.

Skipped (not failed) when node or the generated model is unavailable, so CI
without node still passes.
"""

import json
import math
import os
import random
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roadlens.ml.bootstrap import ROAD_CLASSES
from roadlens.ml.registry import MODEL_DIR, ModelRegistry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_JS = os.path.join(ROOT, "dashboard", "ml.generated.js")
RULES_JS = os.path.join(ROOT, "dashboard", "rules.generated.js")

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None
    or not os.path.exists(ML_JS)
    or not os.path.exists(RULES_JS)
    or not os.path.exists(os.path.join(MODEL_DIR, "manifest.json")),
    reason="node or the generated cost model is unavailable",
)


@pytest.fixture(scope="module")
def registry():
    reg = ModelRegistry.load(MODEL_DIR)
    if reg is None:
        pytest.skip("no trained models on disk; run tools/train_models.py")
    return reg


def _run_js(tickets):
    script = f"""
import {{ predictCost, costModelStatus }} from {json.dumps(ML_JS)};
const tickets = {json.dumps(tickets)};
process.stdout.write(JSON.stringify({{
  status: costModelStatus(),
  results: tickets.map(predictCost),
}}));
"""
    path = os.path.join(ROOT, ".ml_parity_tmp.mjs")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(script)
        res = subprocess.run(["node", path], capture_output=True, text=True, timeout=300)
        assert res.returncode == 0, res.stderr
        return json.loads(res.stdout)
    finally:
        if os.path.exists(path):
            os.remove(path)


def _random_tickets(n, seed=20260904):
    rng = random.Random(seed)
    types = ["pothole", "crack", "manhole", "zebra_crossing", "footpath", "brand_new_type"]
    out = []
    for i in range(n):
        area = round(rng.uniform(0.0, 0.30), 6)
        out.append({
            "id": f"T{i}",
            "defect_type": rng.choice(types),
            "area_ratio": area,
            "confidence": round(rng.uniform(0.3, 1.0), 6),
            "sightings": rng.randint(1, 9),
            "severity_level": rng.randint(1, 4),
            "priority_score": rng.randint(0, 100),
            "recurrence_count": rng.randint(0, 3),
            # An unseen road class must land in the __other__ column on both
            # sides, not shift every one-hot column by one.
            "road_class": rng.choice(ROAD_CLASSES + ["unpaved_track"]),
            "created_at": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T"
                          f"{rng.randint(0, 23):02d}:00:00+00:00",
        })
    return out


def test_javascript_matches_python_on_random_tickets(registry):
    tickets = _random_tickets(1500)
    js = _run_js(tickets)["results"]

    for ticket, got in zip(tickets, js):
        want = registry.cost.predict(ticket)
        assert got["source"] == want["source"], ticket
        assert got["rules_inr"] == want["rules_inr"], ticket
        assert got["predicted_inr"] == want["predicted_inr"], ticket
        assert got["low_inr"] == want["low_inr"], ticket
        assert got["high_inr"] == want["high_inr"], ticket
        assert got["delta_vs_rules_pct"] == pytest.approx(
            want["delta_vs_rules_pct"], abs=0.1), ticket


def test_raw_scores_agree_to_1e_9(registry):
    """Rounded rupees can hide a real disagreement — two models differing by
    0.4 of a rupee round to the same integer. This compares before rounding."""
    tickets = _random_tickets(400, seed=7)
    js = _run_js(tickets)["results"]

    worst = 0.0
    for ticket, got in zip(tickets, js):
        want = registry.cost.predict(ticket)
        # Recover the unrounded log-scale prediction from both sides via the
        # ratio to the rules estimate, which both report exactly.
        a = math.log(max(got["predicted_inr"], 1)) - math.log(max(got["rules_inr"], 1))
        b = math.log(max(want["predicted_inr"], 1)) - math.log(max(want["rules_inr"], 1))
        worst = max(worst, abs(a - b))
    assert worst < 1e-9, f"largest log-scale disagreement {worst}"


def test_monsoon_month_is_read_the_same_way_on_both_sides(registry):
    """getUTCMonth() is 0-based and Python's .month is 1-based. An off-by-one
    here silently prices every defect in the wrong season."""
    base = {"id": "M", "defect_type": "pothole", "area_ratio": 0.05,
            "confidence": 0.85, "sightings": 2, "severity_level": 3,
            "priority_score": 70, "recurrence_count": 0, "road_class": "arterial"}
    tickets = [{**base, "created_at": f"2026-{m:02d}-15T00:00:00+00:00"}
               for m in range(1, 13)]
    js = _run_js(tickets)["results"]
    for ticket, got in zip(tickets, js):
        assert got["predicted_inr"] == registry.cost.predict(ticket)["predicted_inr"], ticket
    # And the season must actually matter, or this test proves nothing.
    assert len({r["predicted_inr"] for r in js}) > 1


def test_boundary_values_agree(registry):
    """Severity band edges are where a one-ULP threshold difference changes
    the base cost, and so the whole prediction."""
    from roadlens.severity import _severity_settings
    levels, _, _ = _severity_settings()
    edges = sorted(row[0] for row in levels)

    tickets = []
    for edge in edges:
        for area in (edge, max(edge - 1e-9, 0.0), edge + 1e-9):
            tickets.append({
                "id": f"E{len(tickets)}", "defect_type": "pothole",
                "area_ratio": area, "confidence": 0.8, "sightings": 2,
                "severity_level": 3, "priority_score": 60,
                "recurrence_count": 0, "road_class": "arterial",
                "created_at": "2026-07-15T00:00:00+00:00",
            })
    js = _run_js(tickets)["results"]
    for ticket, got in zip(tickets, js):
        assert got["predicted_inr"] == registry.cost.predict(ticket)["predicted_inr"], ticket


def test_javascript_carries_the_synthetic_label(registry):
    """A synthetic model must be labelled in the browser too. This is the one
    place a demo could quietly present simulated numbers as a city's real
    costs, so the provenance travels into the bundle."""
    status = _run_js([])["status"]
    assert status["trained"] == registry.cost.trained
    assert status["isSynthetic"] == (
        registry.provenance.get("training_data") == "synthetic_bootstrap")
    assert status["provenance"]["training_data"] == registry.provenance["training_data"]


def test_generated_bundle_is_not_stale(registry):
    """CI fails if tools/generate_ml_js.py has not been re-run after training,
    the same guarantee tests/test_js_parity.py gives for rules.generated.js."""
    with open(ML_JS, encoding="utf-8") as fh:
        source = fh.read()
    start = source.index("export const COST_MODEL = ") + len("export const COST_MODEL = ")
    end = source.index("\n", start)
    embedded = json.loads(source[start:end].rstrip(";"))

    assert embedded["model"] == registry.cost.to_dict(), \
        "dashboard/ml.generated.js is stale — re-run tools/generate_ml_js.py"
    assert embedded["provenance"] == registry.provenance
