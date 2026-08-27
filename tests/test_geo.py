"""Tests for GPS attachment (GPX parsing and interpolation)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roadlens.detector import Detection
from roadlens.geo import GpxTrack, attach_gps_from_track, attach_gps_manual

GPX_11 = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk><trkseg>
    <trkpt lat="12.9500" lon="77.7000"><time>2026-08-25T09:12:00Z</time></trkpt>
    <trkpt lat="12.9510" lon="77.7010"><time>2026-08-25T09:12:10Z</time></trkpt>
    <trkpt lat="12.9520" lon="77.7020"><time>2026-08-25T09:12:20Z</time></trkpt>
  </trkseg></trk>
</gpx>
"""

GPX_10 = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.0" xmlns="http://www.topografix.com/GPX/1/0">
  <trk><trkseg>
    <trkpt lat="12.9500" lon="77.7000"/>
    <trkpt lat="12.9510" lon="77.7010"/>
  </trkseg></trk>
</gpx>
"""


def _write(tmp_path, text, name="t.gpx"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


class TestGpxLoading:
    def test_loads_gpx_11(self, tmp_path):
        track = GpxTrack.load(_write(tmp_path, GPX_11))
        assert len(track.points) == 3
        assert track.times == [0.0, 10.0, 20.0]

    def test_loads_gpx_10_namespace(self, tmp_path):
        """GPX 1.0 used to parse into an empty track."""
        track = GpxTrack.load(_write(tmp_path, GPX_10))
        assert len(track.points) == 2

    def test_untimed_points_fall_back_to_one_per_second(self, tmp_path):
        track = GpxTrack.load(_write(tmp_path, GPX_10))
        assert track.times == [0.0, 1.0]

    def test_empty_track_raises_clear_error(self, tmp_path):
        empty = '<?xml version="1.0"?><gpx version="1.1"><trk><trkseg/></trk></gpx>'
        with pytest.raises(ValueError, match="No track points"):
            GpxTrack.load(_write(tmp_path, empty))

    def test_real_demo_route_loads(self):
        gpx = os.path.join(os.path.dirname(__file__), "..", "data", "demo_route.gpx")
        track = GpxTrack.load(gpx)
        assert len(track.points) == 60


class TestInterpolation:
    @pytest.fixture
    def track(self, tmp_path):
        return GpxTrack.load(_write(tmp_path, GPX_11))

    def test_exact_point(self, track):
        assert track.position_at(0.0) == pytest.approx((12.9500, 77.7000))

    def test_midpoint(self, track):
        lat, lon = track.position_at(5.0)
        assert lat == pytest.approx(12.9505, abs=1e-6)
        assert lon == pytest.approx(77.7005, abs=1e-6)

    def test_clamps_before_start(self, track):
        assert track.position_at(-100) == pytest.approx((12.9500, 77.7000))

    def test_clamps_after_end(self, track):
        assert track.position_at(1e6) == pytest.approx((12.9520, 77.7020))

    def test_empty_track_raises(self):
        with pytest.raises(ValueError):
            GpxTrack([]).position_at(1.0)


class TestAttach:
    def _det(self, frame_index=0):
        return Detection("pothole", 0.9, [0, 0, 10, 10], 0.02, frame_index=frame_index)

    def test_attach_from_track(self, tmp_path):
        track = GpxTrack.load(_write(tmp_path, GPX_11))
        dets = [self._det(0), self._det(300)]  # 0 s and 10 s at 30 fps
        attach_gps_from_track(dets, track, fps=30.0)
        assert dets[0].lat == pytest.approx(12.9500)
        assert dets[1].lat == pytest.approx(12.9510)

    def test_attach_manual(self):
        dets = [self._det(), self._det()]
        attach_gps_manual(dets, 12.9, 77.6)
        assert all(d.lat == 12.9 and d.lon == 77.6 for d in dets)
