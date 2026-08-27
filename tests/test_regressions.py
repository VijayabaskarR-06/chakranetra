"""Regression tests for bugs found while running RoadLens end to end.

Each test here maps to one concrete defect that was fixed; they exist so the
same bug cannot come back quietly.
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roadlens.config import AppConfig, get_config, reset_config
from roadlens.dedup import DefectCluster, cluster_detections
from roadlens.detector import Detection
from roadlens.logger import ContextLogger
from roadlens.predictive import PredictiveEngine, _grid_cell, haversine_m
from roadlens.severity import assess
from roadlens.tickets import TicketStore


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


def _cluster(lat=12.97, lon=77.59, area=0.03, conf=0.9, sightings=1):
    return DefectCluster(
        defect_type="pothole", lat=lat, lon=lon,
        max_area_ratio=area, max_confidence=conf, sightings=sightings,
    )


# --------------------------------------------------------------------------
# config: environment must override a YAML file, not be ignored by it
# --------------------------------------------------------------------------
class TestConfigEnvOverride:
    def teardown_method(self):
        reset_config()

    def test_env_wins_over_yaml(self, monkeypatch, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("detector:\n  confidence_threshold: 0.35\ndb_path: from_yaml.db\n")
        monkeypatch.setenv("ROADLENS_CONFIG", str(cfg_file))
        monkeypatch.setenv("ROADLENS_DB_PATH", "/tmp/from_env.db")
        monkeypatch.setenv("ROADLENS_CONFIDENCE", "0.6")
        reset_config()

        cfg = get_config()
        assert cfg.db_path == "/tmp/from_env.db"
        assert cfg.detector.confidence_threshold == 0.6

    def test_yaml_still_applies_without_env(self, tmp_path):
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text("db_path: only_yaml.db\nlog_format: text\n")
        cfg = AppConfig.from_yaml(str(cfg_file))
        assert cfg.db_path == "only_yaml.db"
        assert cfg.log_format == "text"


# --------------------------------------------------------------------------
# logger: the "module" field was empty on every single log line
# --------------------------------------------------------------------------
class TestLoggerModuleField:
    def test_module_field_is_populated(self, capsys):
        # A fresh logger name so we attach a handler to the captured stdout.
        ContextLogger("regression_probe").info("created", ticket_id="RL-POT-2026-0001")
        import json
        entry = json.loads(capsys.readouterr().out.strip())
        assert entry["module"] == "regression_probe"
        assert entry["data"]["ticket_id"] == "RL-POT-2026-0001"


# --------------------------------------------------------------------------
# tickets: ids must not be reissued after a delete
# --------------------------------------------------------------------------
class TestTicketIdAllocation:
    def test_id_not_reused_after_delete(self, db_path):
        store = TicketStore(db_path)
        a = store.create_from_cluster(_cluster(lat=12.97))
        b = store.create_from_cluster(_cluster(lat=12.98))
        store.conn.execute("DELETE FROM tickets WHERE id=?", (a["id"],))
        store.conn.commit()

        c = store.create_from_cluster(_cluster(lat=12.99))
        assert c["id"] != b["id"], "new ticket overwrote a live ticket"
        assert store.get(b["id"]) is not None
        store.conn.close()

    def test_stats_handles_naive_timestamps(self, db_path):
        store = TicketStore(db_path)
        naive = datetime.now()  # no tzinfo — used to raise TypeError in stats()
        store.create_from_cluster(_cluster(), now=naive)
        stats = store.stats()
        assert stats["total_tickets"] == 1
        store.conn.close()


# --------------------------------------------------------------------------
# severity: a bad area_ratio must not inherit whatever band ran last
# --------------------------------------------------------------------------
class TestSeverityBandSelection:
    def test_negative_area_falls_to_lowest_band(self):
        a = assess("pothole", area_ratio=-1.0, confidence=0.9)
        assert a.severity_level == 1
        assert a.severity_label == "Low"

    def test_bands_are_monotonic(self):
        levels = [assess("pothole", r, 0.9).severity_level
                  for r in (0.0, 0.01, 0.03, 0.09)]
        assert levels == [1, 2, 3, 4]


# --------------------------------------------------------------------------
# dedup: evidence_frames must record every merged sighting
# --------------------------------------------------------------------------
class TestDedupEvidenceFrames:
    def test_absorb_records_frames(self):
        dets = [
            Detection("pothole", 0.9, [0, 0, 10, 10], 0.02, frame_index=0, source="a", lat=12.97, lon=77.59),
            Detection("pothole", 0.8, [0, 0, 10, 10], 0.03, frame_index=5, source="a", lat=12.97, lon=77.59),
        ]
        clusters = cluster_detections(dets)
        assert len(clusters) == 1
        assert clusters[0].sightings == 2
        assert sorted(clusters[0].evidence_frames) == [0, 5]


# --------------------------------------------------------------------------
# predictive: the heatmap grid must actually be the configured size
# --------------------------------------------------------------------------
class TestGridGeometry:
    def test_cell_step_matches_configured_km(self):
        """The old code produced ~55 m cells for a 0.5 km setting."""
        centres = sorted({_grid_cell(12.96 + i * 0.0002, 77.59, 0.5)[0]
                          for i in range(120)})
        steps = [haversine_m(a, 77.59, b, 77.59)
                 for a, b in zip(centres, centres[1:])]
        assert steps, "grid produced a single cell"
        for step in steps:
            assert 450 < step < 550, f"cell step {step:.0f} m is not ~500 m"

    def test_smaller_grid_gives_smaller_cells(self):
        big = {_grid_cell(12.96 + i * 0.0002, 77.59, 0.5) for i in range(120)}
        small = {_grid_cell(12.96 + i * 0.0002, 77.59, 0.1) for i in range(120)}
        assert len(small) > len(big)

    def test_points_far_apart_land_in_different_cells(self):
        assert _grid_cell(12.9700, 77.5900, 0.5) != _grid_cell(12.9900, 77.6100, 0.5)


# --------------------------------------------------------------------------
# predictive: recurrence must be findable by LOCATION, not just ticket id
# --------------------------------------------------------------------------
class TestLocationRecurrence:
    def test_finds_repair_at_same_spot(self, db_path):
        e = PredictiveEngine(db_path)
        e.register_fixed_ticket("RL-POT-2026-0001", 12.97, 77.59, "pothole")
        hit = e.check_recurrence_at_location(12.970005, 77.590005, "pothole")
        assert hit is not None
        assert hit["original_ticket_id"] == "RL-POT-2026-0001"
        assert hit["recurrence_count"] == 1
        e.conn.close()

    def test_ignores_far_away_defects(self, db_path):
        e = PredictiveEngine(db_path)
        e.register_fixed_ticket("RL-POT-2026-0001", 12.97, 77.59, "pothole")
        assert e.check_recurrence_at_location(12.99, 77.62, "pothole") is None
        e.conn.close()

    def test_ignores_other_defect_types(self, db_path):
        e = PredictiveEngine(db_path)
        e.register_fixed_ticket("RL-POT-2026-0001", 12.97, 77.59, "pothole")
        assert e.check_recurrence_at_location(12.97, 77.59, "crack") is None
        e.conn.close()

    def test_one_repair_cannot_fail_twice_in_one_scan(self, db_path):
        """Two separate potholes in one frame both used to match the same
        repaired spot, recording two failures of a single patch and halving
        the crew's quality score for one bad repair."""
        e = PredictiveEngine(db_path)
        e.register_fixed_ticket("RL-POT-2026-0001", 12.97, 77.59, "pothole",
                                assigned_crew="Crew A")
        claimed = set()
        hits = []
        # two distinct defects, both within the recurrence radius
        for dlat in (0.00001, 0.00002):
            hit = e.check_recurrence_at_location(
                12.97 + dlat, 77.59, "pothole", exclude_ticket_ids=claimed)
            if hit:
                claimed.add(hit["original_ticket_id"])
                hits.append(hit)
        assert len(hits) == 1, "one repair was blamed twice by one scan"
        assert hits[0]["recurrence_count"] == 1
        crew = e.get_crew_performance()[0]
        assert crew["avg_quality_score"] == 0.75
        e.conn.close()

    def test_ignores_repairs_outside_monitoring_window(self, db_path):
        e = PredictiveEngine(db_path)
        long_ago = datetime.now(timezone.utc) - timedelta(days=e.monitoring_window_days + 30)
        e.register_fixed_ticket("RL-POT-2026-0001", 12.97, 77.59, "pothole", now=long_ago)
        assert e.check_recurrence_at_location(12.97, 77.59, "pothole") is None
        e.conn.close()


# --------------------------------------------------------------------------
# predictive: risk segments must survive naive/missing timestamps
# --------------------------------------------------------------------------
class TestRiskSegmentRobustness:
    def test_naive_created_at(self, db_path):
        e = PredictiveEngine(db_path)
        tickets = [{"lat": 12.97, "lon": 77.59, "severity_level": 3,
                    "created_at": datetime.now().isoformat(), "recurrence_count": None}]
        segments = e.compute_risk_segments(tickets)
        assert len(segments) == 1
        e.conn.close()

    def test_missing_created_at(self, db_path):
        e = PredictiveEngine(db_path)
        segments = e.compute_risk_segments([{"lat": 12.97, "lon": 77.59}])
        assert len(segments) == 1
        e.conn.close()

    def test_alerts_sorted_critical_first(self, db_path):
        e = PredictiveEngine(db_path)
        e.register_fixed_ticket("A", 12.97, 77.59, "pothole", assigned_crew="Crew A")
        e.register_fixed_ticket("B", 12.99, 77.61, "pothole", assigned_crew="Crew B")
        for _ in range(4):
            e.check_recurrence("A", 12.97, 77.59, "pothole")
        for _ in range(2):
            e.check_recurrence("B", 12.99, 77.61, "pothole")
        alerts = e.get_predictive_alerts([])
        severities = [a["severity"] for a in alerts]
        assert severities[0] == "critical"
        assert severities == sorted(severities, key=lambda s: {"critical": 0, "high": 1, "medium": 2, "low": 3}[s])
        e.conn.close()
