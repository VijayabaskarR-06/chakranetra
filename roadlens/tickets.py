"""
RoadLens AI — Ticket Engine
===========================
This is where computer vision becomes civic action. The problem statement
asks for exactly this loop:

  road issue -> ticket -> municipal department -> site engineer -> fixed

Each unique defect (one DefectCluster from dedup.py) becomes exactly ONE
ticket with a lifecycle:

  OPEN -> ASSIGNED -> IN_PROGRESS -> FIXED -> VERIFIED
                                     |-> REOPENED (if a later scan
                                          still sees the defect)

The VERIFIED step is RoadLens's accountability loop: when a later vehicle
pass no longer detects the defect at that location, the fix is confirmed by
the same AI that reported it. No self-reporting, no paperwork.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from .dedup import DefectCluster, _merge_radius, haversine_m
from .severity import assess
from .logger import get_logger

logger = get_logger("tickets")

STATUSES = ["OPEN", "ASSIGNED", "IN_PROGRESS", "FIXED", "VERIFIED", "REOPENED"]


def _as_utc(value, fallback: datetime) -> datetime:
    """Parse a stored ISO timestamp as an aware UTC datetime.

    Rows created with a naive `now` produce naive strings, and comparing
    those against an aware datetime raises TypeError mid-stats().
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

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    defect_type TEXT NOT NULL,
    lat REAL, lon REAL,
    severity_level INTEGER,
    severity_label TEXT,
    priority_score INTEGER,
    est_cost_inr INTEGER,
    department TEXT,
    status TEXT DEFAULT 'OPEN',
    sightings INTEGER DEFAULT 1,
    confidence REAL,
    area_ratio REAL,
    sources TEXT,
    created_at TEXT,
    sla_due_at TEXT,
    assigned_to TEXT,
    status_history TEXT,
    recurrence_count INTEGER DEFAULT 0,
    repair_quality_score REAL DEFAULT 1.0,
    last_recurrence_at TEXT,
    road_class TEXT DEFAULT 'arterial',
    actual_cost_inr INTEGER,
    repaired_at TEXT
);
"""

# Every sighting of a defect, not just the first. `tickets.area_ratio` keeps
# only the largest area ever seen, which is the right number for severity but
# destroys the one thing a degradation model needs: how fast the hole grew
# between two dates. This table is that history.
OBSERVATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS defect_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    area_ratio REAL NOT NULL,
    confidence REAL,
    severity_level INTEGER,
    source TEXT,
    UNIQUE (ticket_id, observed_at, source)
);
"""

# Columns added after v2.0 shipped. SQLite has no "ADD COLUMN IF NOT EXISTS",
# and an existing roadlens.db predates all of them, so they are applied by
# comparing against PRAGMA table_info rather than by catching the error --
# catching it would also swallow a genuinely malformed ALTER.
_MIGRATIONS = {
    "road_class": "ALTER TABLE tickets ADD COLUMN road_class TEXT DEFAULT 'arterial'",
    "actual_cost_inr": "ALTER TABLE tickets ADD COLUMN actual_cost_inr INTEGER",
    "repaired_at": "ALTER TABLE tickets ADD COLUMN repaired_at TEXT",
}


class TicketStore:
    """SQLite-backed ticket registry. SQLite = zero setup, one file,
    perfect for a pilot; swap for Postgres in production without changing
    the interface."""

    def __init__(self, db_path: str = "roadlens.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(SCHEMA)
        self.conn.execute(OBSERVATIONS_SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Bring an older roadlens.db up to the current column set."""
        existing = {r["name"] for r in self.conn.execute("PRAGMA table_info(tickets)")}
        for column, ddl in _MIGRATIONS.items():
            if column not in existing:
                self.conn.execute(ddl)
                logger.info("Schema migrated", column=column)

    def create_from_cluster(self, cluster: DefectCluster, now: datetime | None = None,
                            road_class: str = "arterial") -> dict:
        """One unique physical defect -> one ticket."""
        now = now or datetime.now(timezone.utc)

        a = assess(
            defect_type=cluster.defect_type,
            area_ratio=cluster.max_area_ratio,
            confidence=cluster.max_confidence,
            sightings=cluster.sightings,
        )

        ticket_id = self._next_id(cluster.defect_type, now)
        sla_due = now + timedelta(hours=a.sla_hours)

        row = {
            "id": ticket_id,
            "defect_type": cluster.defect_type,
            "lat": round(cluster.lat, 6),
            "lon": round(cluster.lon, 6),
            "severity_level": a.severity_level,
            "severity_label": a.severity_label,
            "priority_score": a.priority_score,
            "est_cost_inr": a.est_cost_inr,
            "department": a.department,
            "status": "OPEN",
            "sightings": cluster.sightings,
            "confidence": cluster.max_confidence,
            "area_ratio": cluster.max_area_ratio,
            "sources": json.dumps(cluster.sources),
            "created_at": now.isoformat(),
            "sla_due_at": sla_due.isoformat(),
            "assigned_to": None,
            "status_history": json.dumps(
                [{"status": "OPEN", "at": now.isoformat(), "note": "Auto-created by Chakranetra"}]
            ),
            "recurrence_count": 0,
            "repair_quality_score": 1.0,
            "last_recurrence_at": None,
            "road_class": road_class,
            "actual_cost_inr": None,
            "repaired_at": None,
        }
        self.conn.execute(
            f"INSERT OR REPLACE INTO tickets ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
            list(row.values()),
        )
        self.conn.commit()
        self.record_observation(
            ticket_id, cluster.max_area_ratio, cluster.max_confidence,
            a.severity_level, now=now,
            source=(cluster.sources[0] if cluster.sources else None),
        )
        logger.info("Ticket created", ticket_id=ticket_id, severity=a.severity_label, defect_type=cluster.defect_type)
        return row

    def update_status(self, ticket_id: str, status: str, note: str = "", assigned_to: str | None = None) -> dict:
        if status not in STATUSES:
            raise ValueError(f"Unknown status {status}. Use one of {STATUSES}")
        row = self.get(ticket_id)
        if row is None:
            raise KeyError(f"No ticket {ticket_id}")

        history = json.loads(row["status_history"])
        history.append({"status": status, "at": datetime.now(timezone.utc).isoformat(), "note": note})

        # First transition into FIXED starts the repair-monitoring clock; a
        # later REOPENED -> FIXED must not reset it, or the failure model
        # would measure the age of the newest patch against the oldest defect.
        repaired_at = row.get("repaired_at") or (
            datetime.now(timezone.utc).isoformat() if status == "FIXED" else None
        )
        self.conn.execute(
            "UPDATE tickets SET status=?, status_history=?, "
            "assigned_to=COALESCE(?, assigned_to), repaired_at=? WHERE id=?",
            (status, json.dumps(history), assigned_to, repaired_at, ticket_id),
        )
        self.conn.commit()
        logger.info("Ticket status updated", ticket_id=ticket_id, status=status, assigned_to=assigned_to)
        return self.get(ticket_id)

    def record_recurrence(self, ticket_id: str, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        row = self.get(ticket_id)
        if row is None:
            raise KeyError(f"No ticket {ticket_id}")

        new_count = (row.get("recurrence_count") or 0) + 1
        quality_score = max(0.0, 1.0 - (new_count * 0.25))

        self.conn.execute(
            "UPDATE tickets SET recurrence_count=?, repair_quality_score=?, last_recurrence_at=? WHERE id=?",
            (new_count, quality_score, now.isoformat(), ticket_id),
        )
        self.conn.commit()
        logger.warning("Defect recurrence recorded", ticket_id=ticket_id, recurrence_count=new_count, quality_score=quality_score)
        return self.get(ticket_id)

    OPEN_STATUSES = ("OPEN", "ASSIGNED", "IN_PROGRESS", "REOPENED")

    def find_open_at_location(self, lat: float, lon: float, defect_type: str,
                              radius_m: float | None = None) -> str | None:
        """The still-open ticket for this same physical defect, if there is one.

        `cluster_detections` deduplicates *within* one scan. Across scans it
        could not: a second pass over the same road tomorrow produced a second
        ticket for the same hole, and the queue filled with duplicates of the
        defect a crew had already been sent to.

        It also made the degradation model unreachable on real data. Growth is
        measured between two sightings of *one* ticket, and every re-sighting
        was landing on a brand-new id, so no ticket ever accumulated a second
        observation. The model could only ever have trained on simulated data.
        """
        radius = _merge_radius() if radius_m is None else float(radius_m)
        rows = self.conn.execute(
            "SELECT id, lat, lon FROM tickets WHERE defect_type=? AND status IN "
            f"({','.join('?' * len(self.OPEN_STATUSES))})",
            (defect_type, *self.OPEN_STATUSES),
        ).fetchall()

        best_id, best_dist = None, None
        for r in rows:
            if r["lat"] is None or r["lon"] is None:
                continue
            dist = haversine_m(r["lat"], r["lon"], lat, lon)
            if dist <= radius and (best_dist is None or dist < best_dist):
                best_id, best_dist = r["id"], dist
        return best_id

    def record_growth(self, ticket_id: str, cluster: DefectCluster,
                      now: datetime | None = None) -> dict:
        """Fold a fresh sighting into an existing open ticket.

        The sighting is always appended to the growth history. The ticket's
        own numbers are only revised upward: `area_ratio` is documented as the
        worst size ever seen, and letting a distant or partly-occluded frame
        shrink it would quietly downgrade a defect's severity — and its SLA —
        on the strength of a bad camera angle.
        """
        now = now or datetime.now(timezone.utc)
        row = self.get(ticket_id)
        if row is None:
            raise KeyError(f"No ticket {ticket_id}")

        area = max(float(row["area_ratio"] or 0.0), float(cluster.max_area_ratio))
        confidence = max(float(row["confidence"] or 0.0), float(cluster.max_confidence))
        sightings = int(row["sightings"] or 0) + int(cluster.sightings)
        a = assess(cluster.defect_type, area, confidence, sightings)

        self.record_observation(
            ticket_id, cluster.max_area_ratio, cluster.max_confidence,
            assess(cluster.defect_type, cluster.max_area_ratio,
                   cluster.max_confidence, cluster.sightings).severity_level,
            now=now, source=(cluster.sources[0] if cluster.sources else None),
        )

        sources = json.loads(row["sources"] or "[]")
        for src in cluster.sources:
            if src not in sources:
                sources.append(src)

        self.conn.execute(
            "UPDATE tickets SET area_ratio=?, confidence=?, sightings=?, "
            "severity_level=?, severity_label=?, priority_score=?, "
            "est_cost_inr=?, department=?, sources=? WHERE id=?",
            (area, confidence, sightings, a.severity_level, a.severity_label,
             a.priority_score, a.est_cost_inr, a.department,
             json.dumps(sources), ticket_id),
        )
        self.conn.commit()

        if a.severity_level > (row["severity_level"] or 0):
            history = json.loads(row["status_history"] or "[]")
            history.append({
                "status": row["status"], "at": now.isoformat(),
                "note": f"Re-sighted and grown: severity raised to "
                        f"{a.severity_label} (L{a.severity_level})",
            })
            self.conn.execute("UPDATE tickets SET status_history=? WHERE id=?",
                              (json.dumps(history), ticket_id))
            self.conn.commit()
            logger.warning("Defect grew between scans", ticket_id=ticket_id,
                           severity=a.severity_label, area_ratio=area)
        else:
            logger.info("Existing defect re-sighted", ticket_id=ticket_id,
                        sightings=sightings)
        return self.get(ticket_id)

    def record_observation(self, ticket_id: str, area_ratio: float,
                           confidence: float | None = None,
                           severity_level: int | None = None,
                           now: datetime | None = None,
                           source: str | None = None) -> None:
        """Append one sighting to the defect's growth history.

        Silently ignores a repeat of the same (ticket, timestamp, source),
        so re-running a scan over the same footage does not manufacture
        growth data out of duplicate rows.
        """
        now = now or datetime.now(timezone.utc)
        self.conn.execute(
            """INSERT OR IGNORE INTO defect_observations
               (ticket_id, observed_at, area_ratio, confidence, severity_level, source)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ticket_id, now.isoformat(), float(area_ratio),
             confidence, severity_level, source),
        )
        self.conn.commit()

    def observations(self, ticket_id: str | None = None) -> list[dict]:
        """Sighting history, oldest first — the degradation model's input."""
        if ticket_id is None:
            rows = self.conn.execute(
                "SELECT * FROM defect_observations ORDER BY ticket_id, observed_at"
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM defect_observations WHERE ticket_id=? ORDER BY observed_at",
                (ticket_id,),
            )
        return [dict(r) for r in rows]

    def record_actual_cost(self, ticket_id: str, actual_cost_inr: int,
                           now: datetime | None = None) -> dict:
        """Record what the repair actually cost.

        This is the label the cost model learns from, and the only reason it
        is ever more than a restatement of the rules formula. Until a city
        starts entering these, `roadlens.ml` correctly reports that it is
        falling back to the rules engine.
        """
        if actual_cost_inr is None or int(actual_cost_inr) <= 0:
            raise ValueError("actual_cost_inr must be a positive number of rupees")
        row = self.get(ticket_id)
        if row is None:
            raise KeyError(f"No ticket {ticket_id}")
        now = now or datetime.now(timezone.utc)
        self.conn.execute(
            "UPDATE tickets SET actual_cost_inr=?, repaired_at=COALESCE(repaired_at, ?) WHERE id=?",
            (int(actual_cost_inr), now.isoformat(), ticket_id),
        )
        self.conn.commit()
        logger.info("Actual repair cost recorded", ticket_id=ticket_id,
                    actual_cost_inr=int(actual_cost_inr),
                    est_cost_inr=row.get("est_cost_inr"))
        return self.get(ticket_id)

    def get(self, ticket_id: str) -> dict | None:
        cur = self.conn.execute("SELECT * FROM tickets WHERE id=?", (ticket_id,))
        r = cur.fetchone()
        return dict(r) if r else None

    def list(self, status: str | None = None, department: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM tickets", []
        clauses = []
        if status:
            clauses.append("status=?"); args.append(status)
        if department:
            clauses.append("department=?"); args.append(department)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY priority_score DESC"
        return [dict(r) for r in self.conn.execute(q, args)]

    def stats(self) -> dict:
        """Numbers the commissioner's office actually asks for."""
        rows = self.list()
        now = datetime.now(timezone.utc)
        open_like = [r for r in rows if r["status"] in ("OPEN", "ASSIGNED", "IN_PROGRESS", "REOPENED")]
        overdue = [r for r in open_like if _as_utc(r["sla_due_at"], now) < now]
        total_recurrences = sum(r.get("recurrence_count", 0) or 0 for r in rows)
        poor_repairs = sum(1 for r in rows if (r.get("repair_quality_score") or 1.0) < 0.5)
        return {
            "total_tickets": len(rows),
            "open": len(open_like),
            "fixed_or_verified": len([r for r in rows if r["status"] in ("FIXED", "VERIFIED")]),
            "overdue_sla": len(overdue),
            "critical_open": len([r for r in open_like if r["severity_level"] == 4]),
            "est_backlog_cost_inr": sum(r["est_cost_inr"] for r in open_like),
            "total_recurrences": total_recurrences,
            "poor_repair_count": poor_repairs,
        }

    def _next_id(self, defect_type: str, now: datetime) -> str:
        """Human-friendly IDs like RL-POT-2026-0007 — crews say these on
        the phone, so they must be short and speakable.

        Numbered off the highest existing suffix, not COUNT(*): counting rows
        reissues a live id as soon as any earlier ticket is deleted, and the
        INSERT OR REPLACE below would then silently overwrite it.
        """
        prefix = f"RL-{defect_type[:3].upper()}-{now.year}"
        cur = self.conn.execute(
            "SELECT id FROM tickets WHERE id LIKE ?", (prefix + "-%",)
        )
        highest = 0
        for (existing,) in cur.fetchall():
            suffix = existing.rsplit("-", 1)[-1]
            if suffix.isdigit():
                highest = max(highest, int(suffix))
        return f"{prefix}-{highest + 1:04d}"
