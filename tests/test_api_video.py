"""API tests for POST /api/scan/video.

The detector needs YOLO, which is heavy, so — same as tests/test_api.py for
scan/image — most of this exercises validation that fails before the
detector is ever touched. The happy path is different from the image tests:
it's worth proving that a real video + a real GPX track actually flow through
`_file_and_track` correctly, so those tests monkeypatch `get_detector` with a
small stand-in that returns canned detections instead of running YOLO, and
drive it with a genuine tiny .mp4 written by cv2.VideoWriter.
"""

import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient
cv2 = pytest.importorskip("cv2")

from roadlens.config import reset_config
from roadlens.detector import Detection

GPX_SAMPLE = """<?xml version="1.0"?>
<gpx version="1.1"><trk><trkseg>
<trkpt lat="12.9700" lon="77.5900"><time>2026-01-01T00:00:00Z</time></trkpt>
<trkpt lat="12.9705" lon="77.5905"><time>2026-01-01T00:00:10Z</time></trkpt>
<trkpt lat="12.9710" lon="77.5910"><time>2026-01-01T00:00:20Z</time></trkpt>
</trkseg></trk></gpx>"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADLENS_DB_PATH", str(tmp_path / "video_api_test.db"))
    reset_config()

    import server.app as app_module
    app_module = importlib.reload(app_module)

    with TestClient(app_module.app) as c:
        c.app_module = app_module
        yield c

    app_module.store and app_module.store.conn.close()
    reset_config()


def _make_video(path, n_frames=20, fps=30.0, size=(64, 64)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), (i * 7) % 255, dtype=np.uint8))
    writer.release()
    return path


class FakeVideoDetector:
    """Stands in for RoadDefectDetector so tests don't need YOLO.

    `detections_by_call` lets each test script exactly what `detect_video`
    returns, in call order, to drive the create-vs-grow and recurrence logic
    deterministically.
    """

    def __init__(self, detections_sequence):
        self._sequence = list(detections_sequence)
        self.calls = []

    def detect_video(self, video_path, sample_every_n_frames=15, save_annotated_dir=None):
        self.calls.append({"video_path": video_path, "sample_every_n_frames": sample_every_n_frames,
                           "save_annotated_dir": save_annotated_dir})
        if save_annotated_dir:
            os.makedirs(save_annotated_dir, exist_ok=True)
        return self._sequence.pop(0) if self._sequence else []


def _det(frame_index=0, area=0.03, conf=0.9, defect_type="pothole"):
    return Detection(defect_type=defect_type, confidence=conf, box=[0, 0, 10, 10],
                     area_ratio=area, frame_index=frame_index)


def _install_fake_detector(app_module, monkeypatch, sequence):
    fake = FakeVideoDetector(sequence)
    monkeypatch.setattr(app_module, "get_detector", lambda: fake)
    return fake


class TestVideoValidation:
    def test_rejects_when_no_location_given(self, client, tmp_path):
        video = _make_video(str(tmp_path / "a.mp4"))
        with open(video, "rb") as fh:
            r = client.post("/api/scan/video", files={"file": ("a.mp4", fh, "video/mp4")})
        assert r.status_code == 400
        assert "GPS track" in r.json()["detail"]

    def test_rejects_bad_latitude_with_manual_location(self, client, tmp_path):
        video = _make_video(str(tmp_path / "a.mp4"))
        with open(video, "rb") as fh:
            r = client.post("/api/scan/video", files={"file": ("a.mp4", fh, "video/mp4")},
                            data={"lat": 999, "lon": 77.6})
        assert r.status_code == 400
        assert "Latitude" in r.json()["detail"]

    def test_rejects_non_positive_fps(self, client, tmp_path):
        video = _make_video(str(tmp_path / "a.mp4"))
        with open(video, "rb") as fh:
            r = client.post("/api/scan/video", files={"file": ("a.mp4", fh, "video/mp4")},
                            data={"lat": 12.9, "lon": 77.6, "fps": 0})
        assert r.status_code == 400
        assert "fps" in r.json()["detail"]

    def test_rejects_non_video_file(self, client):
        r = client.post("/api/scan/video",
                        files={"file": ("a.txt", b"hello", "text/plain")},
                        data={"lat": 12.9, "lon": 77.6})
        assert r.status_code == 400
        assert "video" in r.json()["detail"].lower()

    def test_accepts_video_extension_with_generic_content_type(self, client, monkeypatch, tmp_path):
        """Some upload clients report application/octet-stream for an mp4;
        the extension is the fallback, and this must get past validation to
        the detector (which is faked so the test stays YOLO-free)."""
        video = _make_video(str(tmp_path / "a.mp4"))
        _install_fake_detector(client.app_module, monkeypatch, [[]])
        with open(video, "rb") as fh:
            r = client.post("/api/scan/video",
                            files={"file": ("a.mp4", fh, "application/octet-stream")},
                            data={"lat": 12.9, "lon": 77.6})
        assert r.status_code == 200

    def test_corrupt_video_is_400_not_500(self, client):
        r = client.post("/api/scan/video",
                        files={"file": ("junk.mp4", os.urandom(2048), "video/mp4")},
                        data={"lat": 12.9, "lon": 77.6})
        assert r.status_code == 400
        assert "readable video" in r.json()["detail"]

    def test_oversized_video_rejected(self, client, monkeypatch):
        monkeypatch.setattr(client.app_module, "MAX_VIDEO_UPLOAD_BYTES", 1024)
        r = client.post("/api/scan/video",
                        files={"file": ("big.mp4", os.urandom(4096), "video/mp4")},
                        data={"lat": 12.9, "lon": 77.6})
        assert r.status_code == 413
        assert "limit" in r.json()["detail"].lower()

    def test_invalid_gpx_is_400(self, client, tmp_path):
        video = _make_video(str(tmp_path / "a.mp4"))
        with open(video, "rb") as vfh:
            r = client.post("/api/scan/video",
                            files={"file": ("a.mp4", vfh, "video/mp4"),
                                  "gpx": ("t.gpx", b"<not><valid", "application/gpx+xml")})
        assert r.status_code == 400
        assert "GPS track" in r.json()["detail"]

    def test_empty_gpx_is_400(self, client, tmp_path):
        video = _make_video(str(tmp_path / "a.mp4"))
        empty_gpx = b'<?xml version="1.0"?><gpx version="1.1"></gpx>'
        with open(video, "rb") as vfh:
            r = client.post("/api/scan/video",
                            files={"file": ("a.mp4", vfh, "video/mp4"),
                                  "gpx": ("t.gpx", empty_gpx, "application/gpx+xml")})
        assert r.status_code == 400


class TestVideoHappyPath:
    def test_gpx_track_positions_every_detection(self, client, monkeypatch, tmp_path):
        video = _make_video(str(tmp_path / "a.mp4"), n_frames=20, fps=30.0)
        _install_fake_detector(client.app_module, monkeypatch, [[_det(frame_index=0)]])

        with open(video, "rb") as vfh:
            r = client.post("/api/scan/video",
                            files={"file": ("a.mp4", vfh, "video/mp4"),
                                  "gpx": ("t.gpx", GPX_SAMPLE.encode(), "application/gpx+xml")},
                            data={"fps": 30.0})
        assert r.status_code == 200
        body = r.json()
        assert body["gps_source"] == "gpx"
        assert body["defects_found"] == 1
        assert len(body["tickets_created"]) == 1

        ticket = client.app_module.get_store().get(body["tickets_created"][0])
        # frame_index 0 -> t=0s -> the GPX track's first point exactly.
        assert ticket["lat"] == pytest.approx(12.9700, abs=1e-4)
        assert ticket["lon"] == pytest.approx(77.5900, abs=1e-4)

    def test_manual_location_pins_every_detection_to_one_point(self, client, monkeypatch, tmp_path):
        video = _make_video(str(tmp_path / "a.mp4"))
        _install_fake_detector(client.app_module, monkeypatch,
                               [[_det(frame_index=0), _det(frame_index=15, area=0.09)]])

        with open(video, "rb") as vfh:
            r = client.post("/api/scan/video", files={"file": ("a.mp4", vfh, "video/mp4")},
                            data={"lat": 12.9, "lon": 77.6})
        assert r.status_code == 200
        body = r.json()
        assert body["gps_source"] == "manual"
        # Both detections are within the dedup merge radius of the *same*
        # manual point, so they merge into one cluster/ticket, not two.
        assert body["defects_found"] == 2
        assert body["unique_defects"] == 1

    def test_second_pass_grows_the_existing_ticket_not_a_duplicate(self, client, monkeypatch, tmp_path):
        """The whole point of routing video scans through _file_and_track:
        driving the same stretch twice must accumulate one ticket's growth
        history, not file the pothole a second time."""
        video = _make_video(str(tmp_path / "a.mp4"))
        _install_fake_detector(
            client.app_module, monkeypatch,
            [[_det(frame_index=0, area=0.02)], [_det(frame_index=0, area=0.03)]],
        )

        def _scan():
            with open(video, "rb") as vfh:
                return client.post("/api/scan/video", files={"file": ("a.mp4", vfh, "video/mp4")},
                                   data={"lat": 12.9, "lon": 77.6})

        first = _scan().json()
        second = _scan().json()

        assert len(first["tickets_created"]) == 1
        assert first["tickets_updated"] == []
        assert second["tickets_created"] == []
        assert second["tickets_updated"] == first["tickets_created"]

        ticket = client.app_module.get_store().get(first["tickets_created"][0])
        assert ticket["sightings"] == 2
        assert ticket["area_ratio"] == pytest.approx(0.03)

    def test_reports_frames_analyzed_and_vehicle_id(self, client, monkeypatch, tmp_path):
        video = _make_video(str(tmp_path / "a.mp4"), n_frames=45, fps=30.0)
        _install_fake_detector(client.app_module, monkeypatch, [[]])

        with open(video, "rb") as vfh:
            r = client.post("/api/scan/video", files={"file": ("a.mp4", vfh, "video/mp4")},
                            data={"lat": 12.9, "lon": 77.6, "vehicle_id": "KA-01-AB-1234"})
        body = r.json()
        assert body["vehicle_id"] == "KA-01-AB-1234"
        assert body["source"] == "a.mp4"
        assert body["frames_analyzed"] is not None and body["frames_analyzed"] > 0

    def test_uses_the_configured_sample_rate(self, client, monkeypatch, tmp_path):
        video = _make_video(str(tmp_path / "a.mp4"))
        fake = _install_fake_detector(client.app_module, monkeypatch, [[]])
        with open(video, "rb") as vfh:
            client.post("/api/scan/video", files={"file": ("a.mp4", vfh, "video/mp4")},
                        data={"lat": 12.9, "lon": 77.6})
        assert fake.calls[0]["sample_every_n_frames"] == \
            client.app_module.CONFIG.detector.video_sample_every_n_frames


class TestRecurrenceOnlyFiresForNewTickets:
    def test_growth_match_does_not_trigger_a_recurrence_check(self, client, monkeypatch, tmp_path):
        """Regression test for a bug this endpoint's refactor fixed: before,
        _every_ cluster ran the recurrence check, including ones that only
        grew an already-open ticket — which has nothing to "reopen". Seed a
        FIXED-and-monitored record right on top of an OPEN ticket and prove a
        second sighting near it grows the open ticket without also filing a
        recurrence against the unrelated fixed one."""
        from roadlens.dedup import DefectCluster

        store = client.app_module.get_store()
        predictive = client.app_module.get_predictive()

        open_ticket = store.create_from_cluster(DefectCluster(
            defect_type="pothole", lat=12.90, lon=77.60,
            max_area_ratio=0.02, max_confidence=0.8, sightings=1,
        ))
        # A separately monitored, previously fixed defect at the same spot.
        predictive.register_fixed_ticket("RL-POT-2020-9999", 12.90, 77.60, "pothole")

        video = _make_video(str(tmp_path / "a.mp4"))
        _install_fake_detector(client.app_module, monkeypatch,
                               [[_det(frame_index=0, area=0.04)]])
        with open(video, "rb") as vfh:
            r = client.post("/api/scan/video", files={"file": ("a.mp4", vfh, "video/mp4")},
                            data={"lat": 12.90, "lon": 77.60})
        body = r.json()

        assert body["tickets_updated"] == [open_ticket["id"]]
        assert body["tickets_created"] == []
        assert body["recurrences"] == []

        # check_recurrence() itself mutates the row it inspects, so the
        # unrelated fixed record's state is read directly rather than by
        # calling it — a bug that only fires the recurrence check on new
        # tickets should leave this record untouched, at count 0.
        row = predictive.conn.execute(
            "SELECT recurrence_count FROM recurrence_records WHERE original_ticket_id=?",
            ("RL-POT-2020-9999",)).fetchone()
        assert row["recurrence_count"] == 0
