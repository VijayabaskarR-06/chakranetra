"""Tests for the ticket engine."""

import os
import tempfile
import pytest
from datetime import datetime, timezone, timedelta

from roadlens.tickets import TicketStore
from roadlens.dedup import DefectCluster


def make_cluster(
    defect_type="pothole",
    lat=12.9716,
    lon=77.5946,
    max_area_ratio=0.02,
    max_confidence=0.8,
    sightings=1,
    sources=None,
) -> DefectCluster:
    return DefectCluster(
        defect_type=defect_type,
        lat=lat,
        lon=lon,
        max_area_ratio=max_area_ratio,
        max_confidence=max_confidence,
        sightings=sightings,
        sources=sources or ["test.jpg"],
    )


@pytest.fixture
def store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    s = TicketStore(db_path)
    yield s
    s.conn.close()
    if os.path.exists(db_path):
        os.remove(db_path)


class TestTicketCreation:
    def test_create_ticket(self, store):
        cluster = make_cluster()
        ticket = store.create_from_cluster(cluster)
        assert ticket["id"].startswith("RL-POT-")
        assert ticket["status"] == "OPEN"
        assert ticket["severity_level"] == 2

    def test_ticket_id_format(self, store):
        cluster = make_cluster()
        ticket = store.create_from_cluster(cluster)
        parts = ticket["id"].split("-")
        assert parts[0] == "RL"
        assert parts[1] == "POT"
        assert len(parts[2]) == 4

    def test_sequential_ids(self, store):
        t1 = store.create_from_cluster(make_cluster())
        t2 = store.create_from_cluster(make_cluster())
        assert t1["id"] != t2["id"]

    def test_ticket_has_all_fields(self, store):
        cluster = make_cluster()
        ticket = store.create_from_cluster(cluster)
        required_fields = [
            "id", "defect_type", "lat", "lon", "severity_level",
            "severity_label", "priority_score", "est_cost_inr",
            "department", "status", "sightings", "confidence",
            "area_ratio", "sources", "created_at", "sla_due_at",
            "status_history", "recurrence_count", "repair_quality_score",
        ]
        for field in required_fields:
            assert field in ticket


class TestTicketStatus:
    def test_update_status(self, store):
        ticket = store.create_from_cluster(make_cluster())
        updated = store.update_status(ticket["id"], "ASSIGNED", note="Test", assigned_to="Engineer A")
        assert updated["status"] == "ASSIGNED"
        assert updated["assigned_to"] == "Engineer A"

    def test_status_history_grows(self, store):
        ticket = store.create_from_cluster(make_cluster())
        store.update_status(ticket["id"], "ASSIGNED")
        store.update_status(ticket["id"], "IN_PROGRESS")
        t = store.get(ticket["id"])
        import json
        history = json.loads(t["status_history"])
        assert len(history) == 3

    def test_invalid_status_raises(self, store):
        ticket = store.create_from_cluster(make_cluster())
        with pytest.raises(ValueError):
            store.update_status(ticket["id"], "INVALID_STATUS")

    def test_unknown_ticket_raises(self, store):
        with pytest.raises(KeyError):
            store.update_status("RL-POT-2026-9999", "ASSIGNED")


class TestTicketQuery:
    def test_get_ticket(self, store):
        ticket = store.create_from_cluster(make_cluster())
        found = store.get(ticket["id"])
        assert found is not None
        assert found["id"] == ticket["id"]

    def test_get_nonexistent(self, store):
        assert store.get("RL-POT-2026-9999") is None

    def test_list_all(self, store):
        store.create_from_cluster(make_cluster())
        store.create_from_cluster(make_cluster())
        assert len(store.list()) == 2

    def test_filter_by_status(self, store):
        t1 = store.create_from_cluster(make_cluster())
        t2 = store.create_from_cluster(make_cluster())
        store.update_status(t1["id"], "FIXED")
        open_tickets = store.list(status="OPEN")
        assert len(open_tickets) == 1

    def test_list_sorted_by_priority(self, store):
        store.create_from_cluster(make_cluster(max_area_ratio=0.01))
        store.create_from_cluster(make_cluster(max_area_ratio=0.05))
        tickets = store.list()
        assert tickets[0]["priority_score"] >= tickets[1]["priority_score"]


class TestTicketStats:
    def test_basic_stats(self, store):
        store.create_from_cluster(make_cluster())
        stats = store.stats()
        assert stats["total_tickets"] == 1
        assert stats["open"] == 1

    def test_stats_with_fixed(self, store):
        t = store.create_from_cluster(make_cluster())
        store.update_status(t["id"], "FIXED")
        stats = store.stats()
        assert stats["fixed_or_verified"] == 1
        assert stats["open"] == 0

    def test_stats_overdue(self, store):
        past = datetime.now(timezone.utc) - timedelta(days=30)
        t = store.create_from_cluster(make_cluster(), now=past)
        stats = store.stats()
        assert stats["overdue_sla"] == 1

    def test_recurrence_stats(self, store):
        t = store.create_from_cluster(make_cluster())
        store.record_recurrence(t["id"])
        store.record_recurrence(t["id"])
        stats = store.stats()
        assert stats["total_recurrences"] == 2


class TestRecurrence:
    def test_record_recurrence(self, store):
        t = store.create_from_cluster(make_cluster())
        updated = store.record_recurrence(t["id"])
        assert updated["recurrence_count"] == 1
        assert updated["repair_quality_score"] == 0.75

    def test_multiple_recurrences(self, store):
        t = store.create_from_cluster(make_cluster())
        store.record_recurrence(t["id"])
        store.record_recurrence(t["id"])
        store.record_recurrence(t["id"])
        updated = store.get(t["id"])
        assert updated["recurrence_count"] == 3
        assert updated["repair_quality_score"] == 0.25

    def test_quality_score_minimum(self, store):
        t = store.create_from_cluster(make_cluster())
        for _ in range(10):
            store.record_recurrence(t["id"])
        updated = store.get(t["id"])
        assert updated["repair_quality_score"] >= 0.0
