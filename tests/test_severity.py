"""Tests for the severity engine."""

import pytest
from roadlens.severity import assess, SEVERITY_LEVELS, DEPARTMENT_ROUTING


class TestSeverityLevels:
    def test_critical_threshold(self):
        a = assess("pothole", area_ratio=0.07, confidence=0.9, sightings=3)
        assert a.severity_level == 4
        assert a.severity_label == "Critical"
        assert a.sla_hours == 24

    def test_high_threshold(self):
        a = assess("pothole", area_ratio=0.03, confidence=0.8, sightings=2)
        assert a.severity_level == 3
        assert a.severity_label == "High"
        assert a.sla_hours == 72

    def test_medium_threshold(self):
        a = assess("pothole", area_ratio=0.01, confidence=0.7, sightings=1)
        assert a.severity_level == 2
        assert a.severity_label == "Medium"
        assert a.sla_hours == 168

    def test_low_threshold(self):
        a = assess("pothole", area_ratio=0.001, confidence=0.5, sightings=1)
        assert a.severity_level == 1
        assert a.severity_label == "Low"
        assert a.sla_hours == 336

    def test_priority_score_range(self):
        a = assess("pothole", area_ratio=0.05, confidence=0.9, sightings=5)
        assert 0 <= a.priority_score <= 100

    def test_priority_increases_with_sightings(self):
        a1 = assess("pothole", area_ratio=0.02, confidence=0.8, sightings=1)
        a2 = assess("pothole", area_ratio=0.02, confidence=0.8, sightings=5)
        assert a2.priority_score > a1.priority_score

    def test_cost_scales_with_size(self):
        a1 = assess("pothole", area_ratio=0.01, confidence=0.8, sightings=1)
        a2 = assess("pothole", area_ratio=0.05, confidence=0.8, sightings=1)
        assert a2.est_cost_inr > a1.est_cost_inr


class TestDepartmentRouting:
    def test_pothole_routing(self):
        a = assess("pothole", 0.02, 0.8)
        assert a.department == "Road Maintenance Division"

    def test_manhole_routing(self):
        a = assess("manhole", 0.02, 0.8)
        assert a.department == "Storm Water & Drainage"

    def test_zebra_crossing_routing(self):
        a = assess("zebra_crossing", 0.02, 0.8)
        assert a.department == "Traffic Engineering Cell"

    def test_unknown_defect_default(self):
        a = assess("unknown_type", 0.02, 0.8)
        assert a.department == "Road Maintenance Division"
