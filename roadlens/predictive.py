"""
RoadLens AI — Predictive Analytics & Recurrence Prevention Engine
=================================================================
THE problem this solves:

  Cities fix potholes reactively — after they appear and cause damage.
  By then, water has already seeped into the sub-base, the damage has
  spread, and the repair cost is 5-10x what preventive maintenance would
  have cost.

RoadLens Predictive turns detection data into foresight:

  1. RECURRENCE TRACKING — detects when a "fixed" defect reappears at the
     same location, flags poor repair quality, and escalates automatically.

  2. RISK HEATMAPS — identifies road segments with historically high defect
     density so cities can schedule preventive resurfacing BEFORE new holes
     form.

  3. REPAIR QUALITY SCORE — rates each repair job based on whether the
     defect recurs within a monitoring window (30/60/90 days). Crews with
     consistently low scores get retrained.

  4. PREDICTIVE ALERTS — when a road segment crosses the recurrence
     threshold, the system auto-generates a preventive maintenance ticket
     BEFORE citizens complain.

This is the difference between a city that patches holes and a city that
prevents them.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from .dedup import haversine_m

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------
RECURRENCE_RADIUS_METERS = 15.0
MONITORING_WINDOW_DAYS = 90
HIGH_RECURRENCE_THRESHOLD = 2
CRITICAL_RECURRENCE_THRESHOLD = 4
HEATMAP_GRID_SIZE_KM = 0.5
RISK_SEGMENT_RADIUS_METERS = 500.0

# Rough metres-per-degree at the equator. Latitude degrees are near-constant;
# longitude degrees shrink with cos(latitude), which is why the grid below
# scales the longitude step by the cell's own latitude.
_METERS_PER_DEG_LAT = 110_574.0
_METERS_PER_DEG_LON_EQUATOR = 111_320.0


def _pconf():
    """Predictive thresholds from config.yaml when available."""
    try:
        from .config import get_config

        return get_config().predictive
    except Exception:
        return None


def _setting(name: str, default):
    cfg = _pconf()
    if cfg is not None and hasattr(cfg, name):
        return getattr(cfg, name)
    return default


def _grid_cell(lat: float, lon: float, grid_km: float) -> tuple[float, float]:
    """Snap a GPS point to the centre of its grid cell.

    The previous implementation did `round(lat / grid_km * 1000) * grid_km / 1000`,
    which divides *degrees* by *kilometres* — dimensionally meaningless. With
    grid_km = 0.5 it produced 0.0005-degree cells (~55 m), a tenth of the
    documented 500 m, so the "risk segments" were really single potholes.
    """
    grid_km = float(grid_km) or 0.5
    step_lat = (grid_km * 1000.0) / _METERS_PER_DEG_LAT
    cell_lat = round(lat / step_lat) * step_lat

    cos_lat = max(math.cos(math.radians(cell_lat)), 1e-6)
    step_lon = (grid_km * 1000.0) / (_METERS_PER_DEG_LON_EQUATOR * cos_lat)
    cell_lon = round(lon / step_lon) * step_lon
    return round(cell_lat, 6), round(cell_lon, 6)


def _parse_ts(value, fallback: datetime) -> datetime:
    """Parse an ISO timestamp into an aware UTC datetime.

    Ticket rows written by TicketStore are timezone-aware, but rows handed in
    by callers/tests may be naive; comparing the two raises TypeError, which
    used to blow up the whole risk-segment computation.
    """
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
class RecurrenceRecord:
    """Tracks how many times a defect reappears after being marked FIXED."""
    original_ticket_id: str
    location_lat: float
    location_lon: float
    defect_type: str
    first_fixed_at: str
    recurrence_count: int = 0
    recurrence_dates: list = field(default_factory=list)
    repair_quality_score: float = 1.0
    assigned_crew: str | None = None


@dataclass
class RiskSegment:
    """A road segment flagged as high-risk based on historical defect density."""
    center_lat: float
    center_lon: float
    radius_meters: float
    total_defects_ever: int
    defects_last_30_days: int
    recurrence_count: int
    risk_score: float
    risk_label: str
    recommended_action: str


RECURRENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS recurrence_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_ticket_id TEXT NOT NULL,
    location_lat REAL NOT NULL,
    location_lon REAL NOT NULL,
    defect_type TEXT NOT NULL,
    first_fixed_at TEXT NOT NULL,
    recurrence_count INTEGER DEFAULT 0,
    recurrence_dates TEXT,
    repair_quality_score REAL DEFAULT 1.0,
    assigned_crew TEXT
);
"""

RISK_SEGMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS risk_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    center_lat REAL NOT NULL,
    center_lon REAL NOT NULL,
    radius_meters REAL NOT NULL,
    total_defects_ever INTEGER DEFAULT 0,
    defects_last_30_days INTEGER DEFAULT 0,
    recurrence_count INTEGER DEFAULT 0,
    risk_score REAL NOT NULL,
    risk_label TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    computed_at TEXT NOT NULL
);
"""


class PredictiveEngine:
    """Analyzes historical ticket data to predict and prevent future defects."""

    def __init__(self, db_path: str = "roadlens.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(RECURRENCE_SCHEMA)
        self.conn.execute(RISK_SEGMENTS_SCHEMA)
        self.conn.commit()

        # Thresholds come from config.yaml / env when present, and fall back
        # to the documented module constants otherwise.
        self.recurrence_radius_m = float(_setting("recurrence_radius_meters", RECURRENCE_RADIUS_METERS))
        self.monitoring_window_days = int(_setting("monitoring_window_days", MONITORING_WINDOW_DAYS))
        self.high_threshold = int(_setting("high_recurrence_threshold", HIGH_RECURRENCE_THRESHOLD))
        self.critical_threshold = int(_setting("critical_recurrence_threshold", CRITICAL_RECURRENCE_THRESHOLD))
        self.grid_km = float(_setting("heatmap_grid_size_km", HEATMAP_GRID_SIZE_KM))
        self.segment_radius_m = float(_setting("risk_segment_radius_meters", RISK_SEGMENT_RADIUS_METERS))

    def check_recurrence(self, ticket_id: str, lat: float, lon: float,
                         defect_type: str, now: datetime | None = None) -> dict | None:
        """When a defect is detected, check if this location was previously
        fixed. If yes, record a recurrence and flag repair quality issues."""
        now = now or datetime.now(timezone.utc)
        cur = self.conn.execute(
            "SELECT * FROM recurrence_records WHERE original_ticket_id = ?",
            (ticket_id,)
        )
        record = cur.fetchone()

        if record is None:
            return None

        dist = haversine_m(record["location_lat"], record["location_lon"], lat, lon)
        if dist > self.recurrence_radius_m:
            return None

        recurrence_dates = json.loads(record["recurrence_dates"] or "[]")
        recurrence_dates.append(now.isoformat())
        new_count = record["recurrence_count"] + 1

        quality_score = max(0.0, 1.0 - (new_count * 0.25))

        self.conn.execute(
            """UPDATE recurrence_records
               SET recurrence_count = ?, recurrence_dates = ?, repair_quality_score = ?
               WHERE original_ticket_id = ?""",
            (new_count, json.dumps(recurrence_dates), quality_score, ticket_id)
        )
        self.conn.commit()

        severity = "low"
        if new_count >= self.critical_threshold:
            severity = "critical"
        elif new_count >= self.high_threshold:
            severity = "high"

        return {
            "original_ticket_id": ticket_id,
            "recurrence_count": new_count,
            "severity": severity,
            "repair_quality_score": quality_score,
            "assigned_crew": record["assigned_crew"],
            "recommended_action": self._recurrence_action(severity, new_count),
        }

    def find_repair_at_location(self, lat: float, lon: float, defect_type: str,
                                now: datetime | None = None) -> str | None:
        """Return the ticket id of a previously-FIXED defect at this spot.

        `check_recurrence` can only match a ticket id you already know. But a
        defect that comes back is detected as a *brand new* sighting with a
        *brand new* ticket id, so nothing in the live scan path could ever
        match — the recurrence feature was unreachable outside the demo.
        This looks the location up instead, which is how a real re-scan works.
        """
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.monitoring_window_days)

        rows = self.conn.execute(
            "SELECT * FROM recurrence_records WHERE defect_type = ?", (defect_type,)
        ).fetchall()

        best_id, best_dist = None, None
        for r in rows:
            # Only defects still inside their monitoring window count; an old
            # hole reappearing after two years is new wear, not a bad repair.
            if _parse_ts(r["first_fixed_at"], cutoff) < cutoff:
                continue
            dist = haversine_m(r["location_lat"], r["location_lon"], lat, lon)
            if dist <= self.recurrence_radius_m and (best_dist is None or dist < best_dist):
                best_id, best_dist = r["original_ticket_id"], dist
        return best_id

    def check_recurrence_at_location(self, lat: float, lon: float, defect_type: str,
                                     now: datetime | None = None) -> dict | None:
        """Location-based recurrence check, for the live re-scan path."""
        ticket_id = self.find_repair_at_location(lat, lon, defect_type, now=now)
        if ticket_id is None:
            return None
        return self.check_recurrence(ticket_id, lat, lon, defect_type, now=now)

    def register_fixed_ticket(self, ticket_id: str, lat: float, lon: float,
                              defect_type: str, assigned_crew: str | None = None,
                              now: datetime | None = None) -> None:
        """When a ticket is marked FIXED, register it for recurrence monitoring."""
        now = now or datetime.now(timezone.utc)
        self.conn.execute(
            """INSERT OR REPLACE INTO recurrence_records
               (original_ticket_id, location_lat, location_lon, defect_type,
                first_fixed_at, recurrence_count, recurrence_dates, assigned_crew)
               VALUES (?, ?, ?, ?, ?, 0, '[]', ?)""",
            (ticket_id, lat, lon, defect_type, now.isoformat(), assigned_crew)
        )
        self.conn.commit()

    def compute_risk_segments(self, tickets: list[dict],
                              now: datetime | None = None) -> list[RiskSegment]:
        """Analyze all tickets to identify high-risk road segments."""
        now = now or datetime.now(timezone.utc)
        thirty_days_ago = now - timedelta(days=30)

        grid: dict[tuple, list] = defaultdict(list)
        for t in tickets:
            if t.get("lat") is None or t.get("lon") is None:
                continue
            grid[_grid_cell(t["lat"], t["lon"], self.grid_km)].append(t)

        segments = []
        for (center_lat, center_lon), segment_tickets in grid.items():
            total = len(segment_tickets)
            recent = sum(
                1 for t in segment_tickets
                if _parse_ts(t.get("created_at"), now) >= thirty_days_ago
            )
            recurrences = sum(
                1 for t in segment_tickets
                if (t.get("recurrence_count") or 0) > 0
            )

            risk_score = self._compute_risk_score(total, recent, recurrences)
            risk_label, action = self._classify_risk(risk_score, recent, recurrences)

            segments.append(RiskSegment(
                center_lat=center_lat,
                center_lon=center_lon,
                radius_meters=self.segment_radius_m,
                total_defects_ever=total,
                defects_last_30_days=recent,
                recurrence_count=recurrences,
                risk_score=round(risk_score, 2),
                risk_label=risk_label,
                recommended_action=action,
            ))

        segments.sort(key=lambda s: s.risk_score, reverse=True)

        self.conn.execute("DELETE FROM risk_segments")
        for s in segments:
            self.conn.execute(
                """INSERT INTO risk_segments
                   (center_lat, center_lon, radius_meters, total_defects_ever,
                    defects_last_30_days, recurrence_count, risk_score,
                    risk_label, recommended_action, computed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (s.center_lat, s.center_lon, s.radius_meters, s.total_defects_ever,
                 s.defects_last_30_days, s.recurrence_count, s.risk_score,
                 s.risk_label, s.recommended_action, now.isoformat())
            )
        self.conn.commit()

        return segments

    def get_predictive_alerts(self, tickets: list[dict]) -> list[dict]:
        """Generate actionable alerts for preventive maintenance."""
        alerts = []
        risk_segments = self.compute_risk_segments(tickets)

        for seg in risk_segments:
            if seg.risk_score >= 0.7:
                alerts.append({
                    "type": "preventive_maintenance",
                    "severity": "high" if seg.risk_score >= 0.85 else "medium",
                    "location": {"lat": seg.center_lat, "lon": seg.center_lon},
                    "risk_score": seg.risk_score,
                    "defect_count_30d": seg.defects_last_30_days,
                    "recurrence_count": seg.recurrence_count,
                    "recommendation": seg.recommended_action,
                    "estimated_preventive_cost": seg.total_defects_ever * 1200,
                })

        recurrence_records = self.conn.execute(
            "SELECT * FROM recurrence_records WHERE recurrence_count >= ?",
            (self.high_threshold,)
        ).fetchall()

        for rec in recurrence_records:
            alerts.append({
                "type": "repair_quality_issue",
                "severity": "critical" if rec["recurrence_count"] >= self.critical_threshold else "high",
                "original_ticket_id": rec["original_ticket_id"],
                "location": {"lat": rec["location_lat"], "lon": rec["location_lon"]},
                "recurrence_count": rec["recurrence_count"],
                "repair_quality_score": rec["repair_quality_score"],
                "assigned_crew": rec["assigned_crew"],
                "recommendation": "Escalate to quality assurance team; consider contractor review",
            })

        rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        alerts.sort(key=lambda a: rank.get(a.get("severity"), 9))
        return alerts

    def get_crew_performance(self) -> list[dict]:
        """Return repair quality scores grouped by assigned crew."""
        records = self.conn.execute(
            "SELECT assigned_crew, COUNT(*) as total_monitored, "
            "AVG(recurrence_count) as avg_recurrences, "
            "AVG(repair_quality_score) as avg_quality_score "
            "FROM recurrence_records WHERE assigned_crew IS NOT NULL "
            "GROUP BY assigned_crew"
        ).fetchall()

        return [
            {
                "crew": r["assigned_crew"],
                "tickets_monitored": r["total_monitored"],
                "avg_recurrences": round(r["avg_recurrences"] or 0.0, 2),
                "avg_quality_score": round(r["avg_quality_score"] or 0.0, 2),
                "performance_label": self._crew_label(r["avg_quality_score"] or 0.0),
            }
            for r in records
        ]

    def get_heatmap_data(self, tickets: list[dict]) -> list[dict]:
        """Return grid data for the predictive heatmap visualization."""
        grid: dict[tuple, dict] = defaultdict(lambda: {"count": 0, "max_severity": 0, "recurrences": 0})

        for t in tickets:
            if t.get("lat") is None or t.get("lon") is None:
                continue
            key = _grid_cell(t["lat"], t["lon"], self.grid_km)
            grid[key]["count"] += 1
            grid[key]["max_severity"] = max(grid[key]["max_severity"], t.get("severity_level") or 0)
            grid[key]["recurrences"] += (t.get("recurrence_count") or 0)

        return [
            {
                "lat": lat,
                "lon": lon,
                "defect_count": data["count"],
                "max_severity": data["max_severity"],
                "recurrence_count": data["recurrences"],
                "intensity": min(data["count"] / 10.0, 1.0),
            }
            for (lat, lon), data in grid.items()
        ]

    def _compute_risk_score(self, total: int, recent: int, recurrences: int) -> float:
        """Composite risk score 0-1."""
        if total == 0:
            return 0.0
        frequency = min(recent / 5.0, 1.0) * 0.4
        density = min(total / 20.0, 1.0) * 0.3
        recurrence_factor = min(recurrences / 3.0, 1.0) * 0.3
        return frequency + density + recurrence_factor

    def _classify_risk(self, score: float, recent: int, recurrences: int) -> tuple[str, str]:
        if score >= 0.85 or recurrences >= 3:
            return "critical", "Immediate resurfacing required; investigate sub-base damage"
        if score >= 0.6 or recent >= 5:
            return "high", "Schedule preventive overlay within 30 days"
        if score >= 0.35 or recent >= 2:
            return "medium", "Monitor closely; schedule seal coating in next quarter"
        return "low", "Routine maintenance schedule sufficient"

    def _recurrence_action(self, severity: str, count: int) -> str:
        if severity == "critical":
            return "Escalate to quality assurance; consider contractor penalty and full-depth repair"
        if severity == "high":
            return "Schedule full-depth repair; review original patch quality"
        return "Monitor closely; schedule inspection within 7 days"

    def _crew_label(self, avg_score: float) -> str:
        if avg_score >= 0.9:
            return "excellent"
        if avg_score >= 0.7:
            return "good"
        if avg_score >= 0.5:
            return "needs_improvement"
        return "requires_retraining"
