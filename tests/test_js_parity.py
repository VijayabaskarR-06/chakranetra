"""The browser scores defects without a server, so the scoring rules exist in
JavaScript as well as Python. Duplicated logic drifts unless something checks
it, so this runs the generated JS under node and asserts it agrees with
roadlens.severity.assess on thousands of inputs, boundary values included.

Skipped (not failed) when node is unavailable, so CI without node still passes.
"""

import json
import os
import random
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roadlens.dedup import haversine_m
from roadlens.severity import _severity_settings, assess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_JS = os.path.join(ROOT, "dashboard", "rules.generated.js")

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not os.path.exists(RULES_JS),
    reason="node or generated rules unavailable",
)


def _run_js(cases):
    """Score every case with the generated JS and return its answers."""
    script = f"""
import {{ assess, haversineM }} from {json.dumps(RULES_JS)};
const cases = {json.dumps(cases)};
const out = cases.map(c => c.kind === "assess"
  ? assess(c.defect_type, c.area_ratio, c.confidence, c.sightings)
  : {{ d: haversineM(c.lat1, c.lon1, c.lat2, c.lon2) }});
process.stdout.write(JSON.stringify(out));
"""
    path = os.path.join(ROOT, ".js_parity_tmp.mjs")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(script)
        res = subprocess.run(["node", path], capture_output=True, text=True, timeout=120)
        assert res.returncode == 0, res.stderr
        return json.loads(res.stdout)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_assess_matches_python_on_random_inputs():
    rng = random.Random(20260827)
    types = ["pothole", "crack", "manhole", "zebra_crossing", "footpath", "unknown_type"]
    cases = [{
        "kind": "assess",
        "defect_type": rng.choice(types),
        "area_ratio": round(rng.uniform(0.0, 0.30), 6),
        "confidence": round(rng.uniform(0.0, 1.0), 6),
        "sightings": rng.randint(1, 9),
    } for _ in range(2000)]

    js = _run_js(cases)
    for c, got in zip(cases, js):
        want = assess(c["defect_type"], c["area_ratio"], c["confidence"], c["sightings"])
        assert got["severity_level"] == want.severity_level, c
        assert got["severity_label"] == want.severity_label, c
        assert got["priority_score"] == want.priority_score, c
        assert got["est_cost_inr"] == want.est_cost_inr, c
        assert got["sla_hours"] == want.sla_hours, c
        assert got["department"] == want.department, c


def test_assess_matches_python_exactly_on_band_boundaries():
    """Severity flips here, so an off-by-epsilon is a wrong ticket."""
    levels, _, _ = _severity_settings()
    edges = [row[0] for row in levels]
    ratios = []
    for e in edges:
        ratios += [e, e - 1e-9, e + 1e-9, e - 1e-4, e + 1e-4]
    cases = [{"kind": "assess", "defect_type": "pothole", "area_ratio": r,
              "confidence": 0.5, "sightings": 2} for r in ratios]

    for c, got in zip(cases, _run_js(cases)):
        want = assess(c["defect_type"], c["area_ratio"], c["confidence"], c["sightings"])
        assert got["severity_level"] == want.severity_level, c
        assert got["est_cost_inr"] == want.est_cost_inr, c


def test_haversine_matches_python():
    rng = random.Random(7)
    cases = [{
        "kind": "haversine",
        "lat1": round(rng.uniform(12.90, 13.05), 6), "lon1": round(rng.uniform(77.55, 77.75), 6),
        "lat2": round(rng.uniform(12.90, 13.05), 6), "lon2": round(rng.uniform(77.55, 77.75), 6),
    } for _ in range(500)]

    for c, got in zip(cases, _run_js(cases)):
        want = haversine_m(c["lat1"], c["lon1"], c["lat2"], c["lon2"])
        assert abs(got["d"] - want) < 1e-6, (c, got, want)


def test_generated_file_is_current():
    """Regenerating must be a no-op; otherwise the checked-in JS is stale."""
    before = open(RULES_JS, encoding="utf-8").read()
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "generate_rules_js.py")],
                   capture_output=True, check=True, cwd=ROOT)
    assert open(RULES_JS, encoding="utf-8").read() == before, (
        "dashboard/rules.generated.js is stale — run tools/generate_rules_js.py")
