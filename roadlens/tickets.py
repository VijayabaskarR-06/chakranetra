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

from .dedup import DefectCluster
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
    last_recurrence_at TEXT
);
"""


class TicketStore:
    """SQLite-backed ticket registry. SQLite = zero setup, one file,
    perfect for a pilot; swap for Postgres in production without changing
    the interface."""

    def __init__(self, db_path: str = "roadlens.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def create_from_cluster(self, cluster: DefectCluster, now: datetime | None = None) -> dict:
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
                [{"status": "OPEN", "at": now.isoformat(), "note": "Auto-created by RoadLens AI"}]
            ),
            "recurrence_count": 0,
            "repair_quality_score": 1.0,
            "last_recurrence_at": None,
        }
        self.conn.execute(
            f"INSERT OR REPLACE INTO tickets ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
            list(row.values()),
        )
        self.conn.commit()
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

        self.conn.execute(
            "UPDATE tickets SET status=?, status_history=?, assigned_to=COALESCE(?, assigned_to) WHERE id=?",
            (status, json.dumps(history), assigned_to, ticket_id),
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
