"""Tests for the predictive analytics engine."""

import os
import tempfile
import pytest
from datetime import datetime, timezone, timedelta

from roadlens.predictive import PredictiveEngine, RECURRENCE_RADIUS_METERS


@pytest.fixture
def engine():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    e = PredictiveEngine(db_path)
    yield e
    e.conn.close()
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def sample_tickets():
    now = datetime.now(timezone.utc)
    return [
        {
            "id": "RL-POT-2026-0001",
            "defect_type": "pothole",
            "lat": 12.9716,
            "lon": 77.5946,
            "severity_level": 3,
            "created_at": (now - timedelta(days=10)).isoformat(),
            "status": "FIXED",
            "recurrence_count": 0,
        },
        {
            "id": "RL-POT-2026-0002",
            "defect_type": "pothole",
            "lat": 12.9718,
            "lon": 77.5948,
            "severity_level": 4,
            "created_at": (now - timedelta(days=5)).isoformat(),
            "status": "OPEN",
            "recurrence_count": 0,
        },
        {
            "id": "RL-POT-2026-0003",
            "defect_type": "crack",
            "lat": 12.9800,
            "lon": 77.6000,
            "severity_level": 2,
            "created_at": (now - timedelta(days=2)).isoformat(),
            "status": "OPEN",
            "recurrence_count": 0,
        },
    ]


class TestRecurrenceTracking:
    def test_register_fixed_ticket(self, engine):
        engine.register_fixed_ticket("RL-POT-2026-0001", 12.97, 77.59, "pothole")
        result = engine.check_recurrence("RL-POT-2026-0001", 12.97, 77.59, "pothole")
        assert result is not None
        assert result["recurrence_count"] == 1

    def test_no_recurrence_for_unregistered(self, engine):
        result = engine.check_recurrence("RL-POT-2026-9999", 12.97, 77.59, "pothole")
        assert result is None

    def test_recurrence_outside_radius(self, engine):
        engine.register_fixed_ticket("RL-POT-2026-0001", 12.97, 77.59, "pothole")
        result = engine.check_recurrence("RL-POT-2026-0001", 13.5, 78.0, "pothole")
        assert result is None

    def test_multiple_recurrences_decrease_quality(self, engine):
        engine.register_fixed_ticket("RL-POT-2026-0001", 12.97, 77.59, "pothole")
        engine.check_recurrence("RL-POT-2026-0001", 12.97, 77.59, "pothole")
        engine.check_recurrence("RL-POT-2026-0001", 12.97, 77.59, "pothole")
        engine.check_recurrence("RL-POT-2026-0001", 12.97, 77.59, "pothole")
        result = engine.check_recurrence("RL-POT-2026-0001", 12.97, 77.59, "pothole")
        assert result["recurrence_count"] == 4
        assert result["severity"] == "critical"
        assert result["repair_quality_score"] <= 0.0


class TestRiskSegments:
    def test_compute_segments(self, engine, sample_tickets):
        segments = engine.compute_risk_segments(sample_tickets)
        assert len(segments) > 0

    def test_segments_sorted_by_risk(self, engine, sample_tickets):
        segments = engine.compute_risk_segments(sample_tickets)
        for i in range(len(segments) - 1):
            assert segments[i].risk_score >= segments[i + 1].risk_score

    def test_risk_score_range(self, engine, sample_tickets):
        segments = engine.compute_risk_segments(sample_tickets)
        for s in segments:
            assert 0.0 <= s.risk_score <= 1.0


class TestPredictiveAlerts:
    def test_alerts_generated(self, engine, sample_tickets):
        alerts = engine.get_predictive_alerts(sample_tickets)
        assert isinstance(alerts, list)

    def test_alerts_have_required_fields(self, engine, sample_tickets):
        alerts = engine.get_predictive_alerts(sample_tickets)
        for alert in alerts:
            assert "type" in alert
            assert "severity" in alert
            assert "recommendation" in alert


class TestCrewPerformance:
    def test_empty_when_no_crews(self, engine):
        performance = engine.get_crew_performance()
        assert performance == []

    def test_crew_scores(self, engine):
        engine.register_fixed_ticket("T1", 12.97, 77.59, "pothole", assigned_crew="Crew A")
        engine.register_fixed_ticket("T2", 12.98, 77.60, "pothole", assigned_crew="Crew A")
        engine.check_recurrence("T1", 12.97, 77.59, "pothole")
        performance = engine.get_crew_performance()
        assert len(performance) == 1
        assert performance[0]["crew"] == "Crew A"


class TestHeatmapData:
    def test_heatmap_generated(self, engine, sample_tickets):
        data = engine.get_heatmap_data(sample_tickets)
        assert isinstance(data, list)

    def test_heatmap_has_required_fields(self, engine, sample_tickets):
        data = engine.get_heatmap_data(sample_tickets)
        for point in data:
            assert "lat" in point
            assert "lon" in point
            assert "defect_count" in point
            assert "intensity" in point
