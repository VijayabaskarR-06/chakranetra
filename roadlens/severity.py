"""
RoadLens AI — Severity Engine
=============================
Detection tells us WHERE a defect is. Severity tells us HOW MUCH IT MATTERS.

A municipality has a limited budget and limited crews. If we send them 500
tickets that all look equally important, we have not helped them — we have
buried them. So every ticket gets:

  1. A severity level  (L1 minor → L4 critical)
  2. A priority score  (0–100, used to sort the work queue)
  3. A repair cost estimate (so budgets can be planned)
  4. An SLA — a deadline by which it should be fixed

How severity is decided (simple version):
  - Bigger defect  -> more dangerous            (area_ratio)
  - Model very confident -> we trust it more    (confidence)
  - Seen many times across trips -> definitely real (sighting count)

Everything here is plain arithmetic — no ML — on purpose. Rules that decide
public spending should be explainable to a city official in one sentence.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tunable constants — a city can change these without touching any code logic
# ---------------------------------------------------------------------------

# area_ratio thresholds: what fraction of the camera frame the defect fills.
# A dashcam at ~1.2 m height looking at the road: 2% of frame ≈ a defect
# roughly 30–40 cm across at typical distance. These cut-offs were chosen so
# the four levels match how road crews already talk: patch / repair /
# resurface / emergency.
SEVERITY_LEVELS = [
    # (min_area_ratio, level, label, sla_hours, base_cost_inr)
    (0.060, 4, "Critical", 24, 18000),   # deep/large cavity — accident risk NOW
    (0.025, 3, "High", 72, 9000),        # damaged patch, two-wheeler hazard
    (0.008, 2, "Medium", 168, 4500),     # forming pothole, will grow
    (0.000, 1, "Low", 336, 1800),        # surface wear, schedule with next batch
]

# Departments that receive tickets, by defect type. New defect types
# (from a bigger model later) only need a new line here.
DEPARTMENT_ROUTING = {
    "pothole": "Road Maintenance Division",
    "crack": "Road Maintenance Division",
    "manhole": "Storm Water & Drainage",
    "zebra_crossing": "Traffic Engineering Cell",
    "footpath": "Footpath & Streetscape Wing",
}
DEFAULT_DEPARTMENT = "Road Maintenance Division"


@dataclass
class Assessment:
    severity_level: int      # 1–4
    severity_label: str      # Low / Medium / High / Critical
    priority_score: int      # 0–100 work-queue sort key
    est_cost_inr: int        # estimated repair cost in rupees
    sla_hours: int           # fix-by window
    department: str          # who the ticket is routed to


def _severity_settings() -> tuple[list, dict, str]:
    """Levels / routing from config.yaml when present, defaults otherwise."""
    try:
        from .config import get_config

        cfg = get_config().severity
        levels = [tuple(row) for row in cfg.levels]
        return levels, cfg.department_routing, cfg.default_department
    except Exception:
        return SEVERITY_LEVELS, DEPARTMENT_ROUTING, DEFAULT_DEPARTMENT


def assess(defect_type: str, area_ratio: float, confidence: float, sightings: int = 1) -> Assessment:
    """Turn raw detection numbers into a decision the city can act on."""

    levels, routing, default_dept = _severity_settings()
    # Guard against a nonsensical input silently landing in the wrong band:
    # a negative area_ratio used to fall through the loop and inherit
    # whatever band was last examined.
    area_ratio = max(0.0, float(area_ratio))
    confidence = min(max(float(confidence), 0.0), 1.0)
    sightings = max(1, int(sightings))

    # 1) Pick the severity band from the defect's size. Bands are ordered
    #    largest-threshold-first, so the first match is the right one.
    min_area, level, label, sla_hours, base_cost = levels[-1]
    for row in levels:
        if area_ratio >= row[0]:
            min_area, level, label, sla_hours, base_cost = row
            break

    # 2) Priority score (0–100):
    #    size matters most, confidence and repeat sightings add trust.
    size_component = min(area_ratio / 0.08, 1.0) * 60          # up to 60 pts
    confidence_component = confidence * 25                      # up to 25 pts
    sighting_component = min(sightings, 5) / 5 * 15             # up to 15 pts
    priority = round(size_component + confidence_component + sighting_component)

    # 3) Cost estimate: base cost for the band, scaled a little by size
    #    within the band. (Real deployments calibrate ₹/m² per road class;
    #    this constant-based estimate is stated openly as an estimate.)
    cost = int(base_cost * (1 + min(area_ratio * 4, 1.0)))

    return Assessment(
        severity_level=level,
        severity_label=label,
        priority_score=min(priority, 100),
        est_cost_inr=cost,
        sla_hours=sla_hours,
        department=routing.get(defect_type, default_dept),
    )
