"""The dashboard's Budget panel, executed rather than eyeballed.

The panel is the only place a viewer meets the cost model, and the risk it
carries is specific: a hosted console with no API must still price tickets
*and* must never present a synthetically-trained model as a real one. Both
failures render as a perfectly normal-looking page.

There is no browser in CI, so this extracts the panel's own source out of
dashboard/index.html and runs it under node against a minimal DOM stub. That
is weaker than a real browser, and it is much stronger than trusting that a
1100-line file still works after an edit.

Skipped (not failed) when node or the generated model is unavailable.
"""

import json
import os
import re
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "dashboard", "index.html")
ML_JS = os.path.join(ROOT, "dashboard", "ml.generated.js")

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not os.path.exists(ML_JS),
    reason="node or the generated cost model is unavailable",
)

# The panel's source, delimited by the two section banners around it.
PANEL_RE = re.compile(
    r"/\* ── Budget panel:.*?(?=/\* ── Crew panel)", re.S
)


def _panel_source() -> str:
    with open(INDEX, encoding="utf-8") as fh:
        match = PANEL_RE.search(fh.read())
    assert match, "Budget panel block not found in dashboard/index.html"
    return match.group(0)


def _render(tickets, budget=None):
    """Run drawBudget() over `tickets` and return the HTML it wrote."""
    harness = f"""
import {{ predictCost, costModelStatus }} from {json.dumps(ML_JS)};

// Just enough of the dashboard's environment for the panel to run. The
// textContent setter coerces to string exactly as a real DOM node does, so a
// number assigned here is caught the same way the browser would show it.
const elements = {{}};
function makeNode() {{
  let text = "", html = "";
  return {{
    get textContent() {{ return text; }},
    set textContent(v) {{ text = String(v); }},
    get innerHTML() {{ return html; }},
    set innerHTML(v) {{ html = String(v); }},
  }};
}}
const $ = (id) => (elements[id] ??= makeNode());
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, c =>
  ({{ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }}[c]));
const inr = (n) => (Number(n) || 0).toLocaleString("en-IN");
const state = {{ tickets: {json.dumps(tickets)}, budget: {json.dumps(budget)} }};

{_panel_source()}

drawBudget();
process.stdout.write(JSON.stringify({{
  html: $("panel-budget").innerHTML,
  count: $("c-budget").textContent,
  drawerLine: state.tickets.length ? drawerCostLine(state.tickets[0]) : "",
  status: costModelStatus(),
}}));
"""
    path = os.path.join(ROOT, ".budget_panel_tmp.mjs")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", path], capture_output=True, text=True, timeout=120)
        assert res.returncode == 0, res.stderr
        return json.loads(res.stdout)
    finally:
        if os.path.exists(path):
            os.remove(path)


def _ticket(**kw):
    base = {
        "id": "RL-POT-2026-0001", "defect_type": "pothole", "status": "OPEN",
        "area_ratio": 0.07, "confidence": 0.9, "sightings": 2,
        "severity_level": 4, "severity_label": "Critical", "priority_score": 88,
        "est_cost_inr": 23040, "road_class": "highway",
        "created_at": "2026-07-01T00:00:00+00:00",
        "sla_due_at": "2026-09-20T00:00:00+00:00",
    }
    base.update(kw)
    return base


def test_panel_renders_offline_with_no_api():
    """GitHub Pages serves the console with no server at all. The panel must
    still price every open ticket, from the model in the bundle."""
    out = _render([_ticket()], budget=None)
    assert "Forecast horizons" in out["html"]
    assert "Open tickets, priced by the model" in out["html"]
    assert "RL-POT-2026-0001" in out["html"]
    assert out["count"] == "1"
    # And it must be honest that the offline view is not the full forecast.
    assert "need the API" in out["html"]


def test_panel_prefers_the_api_forecast_when_one_is_present():
    budget = {"horizons": {
        "30_day": {"tickets": 3, "point_inr": 100000, "expected_rework_inr": 25000,
                   "total_inr": 125000, "p10_inr": 110000, "p50_inr": 124000,
                   "p90_inr": 141000},
        "60_day": {"tickets": 5, "point_inr": 180000, "expected_rework_inr": 40000,
                   "total_inr": 220000, "p10_inr": 200000, "p50_inr": 219000,
                   "p90_inr": 244000},
        "90_day": {"tickets": 7, "point_inr": 260000, "expected_rework_inr": 60000,
                   "total_inr": 320000, "p10_inr": 295000, "p50_inr": 318000,
                   "p90_inr": 349000},
    }}
    out = _render([_ticket()], budget=budget)
    assert "expected rework" in out["html"]
    assert "p10-p90" in out["html"]
    assert "1,25,000" in out["html"]        # en-IN grouping, from the API total
    assert "need the API" not in out["html"]


def test_synthetic_training_is_disclosed_on_the_page():
    """The one failure that matters most: simulated repair costs presented as
    a real city's. The warning has to be in the rendered HTML, not just the
    JSON behind it."""
    out = _render([_ticket()])
    if out["status"]["isSynthetic"]:
        assert "Trained on synthetic data" in out["html"]
        assert "demo only" in out["html"]
        assert "no figure here should be quoted" in out["html"]
        assert "synthetic model" in out["drawerLine"]
    else:
        assert "Trained on recorded repairs" in out["html"]
        assert "observed" in out["html"]


def test_only_open_tickets_are_priced():
    tickets = [_ticket(id="A", status="OPEN"),
               _ticket(id="B", status="VERIFIED"),
               _ticket(id="C", status="REOPENED")]
    out = _render(tickets)
    assert out["count"] == "2"
    assert "B</h3>" not in out["html"]


def test_empty_queue_does_not_render_a_broken_panel():
    out = _render([])
    assert out["count"] == "0"
    assert "No open tickets to price" in out["html"]


def test_ticket_ids_are_escaped():
    """Ticket ids reach the panel from the API and are interpolated into HTML."""
    out = _render([_ticket(id="<img src=x onerror=alert(1)>")])
    assert "<img src=x" not in out["html"]
    assert "&lt;img src=x" in out["html"]


def test_drawer_line_reports_the_model_against_the_rule():
    line = _render([_ticket()])["drawerLine"]
    assert "vs rule" in line
    assert "INR" in line
