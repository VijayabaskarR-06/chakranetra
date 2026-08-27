"""Tests for the deduplication engine."""

import pytest
from roadlens.dedup import (
    DefectCluster,
    cluster_detections,
    haversine_m,
    MERGE_RADIUS_METERS,
)
from roadlens.detector import Detection


def make_detection(
    defect_type="pothole",
    confidence=0.8,
    area_ratio=0.02,
    lat=None,
    lon=None,
    frame_index=0,
    source="test.jpg",
) -> Detection:
    return Detection(
        defect_type=defect_type,
        confidence=confidence,
        box=[0, 0, 100, 100],
        area_ratio=area_ratio,
        frame_index=frame_index,
        source=source,
        lat=lat,
        lon=lon,
    )


class TestHaversine:
    def test_same_point(self):
        assert haversine_m(12.9716, 77.5946, 12.9716, 77.5946) == 0.0

    def test_known_distance(self):
        d = haversine_m(12.9716, 77.5946, 12.9816, 77.5946)
        assert 1000 < d < 1200

    def test_symmetric(self):
        d1 = haversine_m(12.0, 77.0, 13.0, 78.0)
        d2 = haversine_m(13.0, 78.0, 12.0, 77.0)
        assert abs(d1 - d2) < 0.001


class TestDefectCluster:
    def test_new_cluster(self):
        det = make_detection(lat=12.97, lon=77.59, confidence=0.8, area_ratio=0.02)
        c = DefectCluster(
            defect_type=det.defect_type,
            lat=det.lat,
            lon=det.lon,
            max_area_ratio=det.area_ratio,
            max_confidence=det.confidence,
            sightings=1,
            sources=[det.source],
            member_keys={(det.source, det.frame_index)},
        )
        assert c.sightings == 1
        assert c.max_confidence == 0.8

    def test_absorb_increases_sightings(self):
        det1 = make_detection(lat=12.97, lon=77.59, confidence=0.7, area_ratio=0.02, frame_index=0, source="a.jpg")
        det2 = make_detection(lat=12.9701, lon=77.5901, confidence=0.9, area_ratio=0.03, frame_index=1, source="b.jpg")
        c = DefectCluster(
            defect_type="pothole", lat=12.97, lon=77.59,
            max_area_ratio=0.02, max_confidence=0.7,
            member_keys={("a.jpg", 0)},
        )
        c.absorb(det2)
        assert c.sightings == 2
        assert c.max_confidence == 0.9
        assert c.max_area_ratio == 0.03


class TestClusterDetections:
    def test_single_detection(self):
        det = make_detection(lat=12.97, lon=77.59)
        clusters = cluster_detections([det])
        assert len(clusters) == 1

    def test_same_location_merges(self):
        det1 = make_detection(lat=12.97, lon=77.59, frame_index=0, source="a.jpg")
        det2 = make_detection(lat=12.97001, lon=77.59001, frame_index=1, source="a.jpg")
        clusters = cluster_detections([det1, det2])
        assert len(clusters) == 1
        assert clusters[0].sightings == 2

    def test_far_apart_stays_separate(self):
        det1 = make_detection(lat=12.97, lon=77.59, frame_index=0, source="a.jpg")
        det2 = make_detection(lat=13.0, lon=77.62, frame_index=1, source="a.jpg")
        clusters = cluster_detections([det1, det2])
        assert len(clusters) == 2

    def test_different_defect_types_stay_separate(self):
        det1 = make_detection(defect_type="pothole", lat=12.97, lon=77.59, frame_index=0, source="a.jpg")
        det2 = make_detection(defect_type="crack", lat=12.97, lon=77.59, frame_index=1, source="a.jpg")
        clusters = cluster_detections([det1, det2])
        assert len(clusters) == 2

    def test_same_frame_different_defects(self):
        det1 = make_detection(lat=12.97, lon=77.59, frame_index=0, source="a.jpg")
        det2 = make_detection(lat=12.97001, lon=77.59001, frame_index=0, source="a.jpg")
        clusters = cluster_detections([det1, det2])
        assert len(clusters) == 2

    def test_no_gps_each_unique(self):
        det1 = make_detection(lat=None, lon=None, frame_index=0, source="a.jpg")
        det2 = make_detection(lat=None, lon=None, frame_index=1, source="b.jpg")
        clusters = cluster_detections([det1, det2])
        assert len(clusters) == 2

    def test_multiple_sightings_increase_confidence(self):
        dets = [
            make_detection(lat=12.97, lon=77.59, confidence=0.5 + i * 0.1, frame_index=i, source=f"frame_{i}.jpg")
            for i in range(5)
        ]
        clusters = cluster_detections(dets)
        assert len(clusters) == 1
        assert clusters[0].sightings == 5
        assert clusters[0].max_confidence == 0.9
