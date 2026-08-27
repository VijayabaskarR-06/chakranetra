"""API-level tests for the FastAPI server.

These exercise the routes without loading the YOLO model — the detector is
imported lazily, so every endpoint below is model-free.
"""

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

from roadlens.config import reset_config
from roadlens.dedup import DefectCluster


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A server bound to a throwaway database."""
    monkeypatch.setenv("ROADLENS_DB_PATH", str(tmp_path / "api_test.db"))
    reset_config()

    import server.app as app_module
    app_module = importlib.reload(app_module)

    with TestClient(app_module.app) as c:
        c.app_module = app_module
        yield c

    app_module.store and app_module.store.conn.close()
    reset_config()


def _seed(app_module, lat=12.97, lon=77.59, area=0.03):
    cluster = DefectCluster(
        defect_type="pothole", lat=lat, lon=lon,
        max_area_ratio=area, max_confidence=0.9, sightings=2,
    )
    return app_module.get_store().create_from_cluster(cluster)


class TestStartup:
    def test_uses_configured_db_path(self, client, tmp_path):
        """ROADLENS_DB_PATH used to be ignored — the server always wrote
        roadlens.db next to the source tree."""
        assert client.app_module.DB_PATH == str(tmp_path / "api_test.db")

    def test_output_dir_created_on_import(self, client):
        """StaticFiles refused to mount a missing output/, so a fresh
        checkout could not start the server at all."""
        assert os.path.isdir(client.app_module.OUTPUT_DIR)


class TestTicketRoutes:
    def test_empty_queue(self, client):
        assert client.get("/api/tickets").json() == []

    def test_list_and_get(self, client):
        t = _seed(client.app_module)
        assert len(client.get("/api/tickets").json()) == 1
        assert client.get(f"/api/tickets/{t['id']}").json()["id"] == t["id"]

    def test_unknown_ticket_404(self, client):
        assert client.get("/api/tickets/NOPE").status_code == 404

    def test_invalid_status_filter_400(self, client):
        r = client.get("/api/tickets?status=BOGUS")
        assert r.status_code == 400

    def test_status_transition(self, client):
        t = _seed(client.app_module)
        r = client.post(f"/api/tickets/{t['id']}/status",
                        json={"status": "ASSIGNED", "assigned_to": "Crew A"})
        assert r.status_code == 200
        assert r.json()["status"] == "ASSIGNED"
        assert r.json()["assigned_to"] == "Crew A"

    def test_bad_status_rejected(self, client):
        t = _seed(client.app_module)
        r = client.post(f"/api/tickets/{t['id']}/status", json={"status": "NOPE"})
        assert r.status_code == 400

    def test_status_on_missing_ticket_404(self, client):
        r = client.post("/api/tickets/NOPE/status", json={"status": "FIXED"})
        assert r.status_code == 404

    def test_fixed_registers_for_monitoring(self, client):
        t = _seed(client.app_module)
        client.post(f"/api/tickets/{t['id']}/status",
                    json={"status": "FIXED", "assigned_to": "Crew A"})
        found = client.app_module.get_predictive().find_repair_at_location(
            t["lat"], t["lon"], "pothole")
        assert found == t["id"]

    def test_fixed_at_null_island_still_registers(self, client):
        """lat/lon of exactly 0.0 are valid coordinates but falsy."""
        t = _seed(client.app_module, lat=0.0, lon=0.0)
        client.post(f"/api/tickets/{t['id']}/status", json={"status": "FIXED"})
        assert client.app_module.get_predictive().find_repair_at_location(
            0.0, 0.0, "pothole") == t["id"]


class TestScanValidation:
    def test_rejects_bad_latitude(self, client):
        r = client.post("/api/scan/image",
                        files={"file": ("a.jpg", b"x", "image/jpeg")},
                        data={"lat": 999, "lon": 77.6})
        assert r.status_code == 400
        assert "Latitude" in r.json()["detail"]

    def test_rejects_bad_longitude(self, client):
        r = client.post("/api/scan/image",
                        files={"file": ("a.jpg", b"x", "image/jpeg")},
                        data={"lat": 12.9, "lon": 999})
        assert r.status_code == 400

    def test_rejects_non_image(self, client):
        r = client.post("/api/scan/image",
                        files={"file": ("a.txt", b"hello", "text/plain")},
                        data={"lat": 12.9, "lon": 77.6})
        assert r.status_code == 400

    def test_corrupt_image_is_400_not_500(self, client):
        """An undecodable upload used to reach YOLO and surface as an opaque
        500 ("need at least one array to stack")."""
        r = client.post("/api/scan/image",
                        files={"file": ("junk.jpg", os.urandom(512), "image/jpeg")},
                        data={"lat": 12.9, "lon": 77.6})
        assert r.status_code == 400
        assert "readable image" in r.json()["detail"]

    def test_oversized_upload_rejected(self, client):
        """Unauthenticated endpoint + no size cap = one request fills the disk."""
        big = b"\xff\xd8\xff" + os.urandom(13 * 1024 * 1024)
        r = client.post("/api/scan/image",
                        files={"file": ("big.jpg", big, "image/jpeg")},
                        data={"lat": 12.9, "lon": 77.6})
        assert r.status_code == 413
        assert "limit" in r.json()["detail"].lower()

    def test_cors_does_not_allow_credentials(self, client):
        """'*' plus allow_credentials=True lets any site make credentialed
        requests, because Starlette reflects the caller's origin back."""
        r = client.get("/api/stats", headers={"Origin": "https://evil.example"})
        assert r.headers.get("access-control-allow-credentials") != "true"

    def test_missing_coordinates_422(self, client):
        r = client.post("/api/scan/image",
                        files={"file": ("a.jpg", b"x", "image/jpeg")})
        assert r.status_code == 422


class TestStatsAndPredictive:
    def test_stats_shape(self, client):
        _seed(client.app_module)
        s = client.get("/api/stats").json()
        for key in ("total_tickets", "open", "overdue_sla", "critical_open",
                    "est_backlog_cost_inr", "total_recurrences"):
            assert key in s

    @pytest.mark.parametrize("path", [
        "/api/predictive/alerts",
        "/api/predictive/heatmap",
        "/api/predictive/crews",
        "/api/predictive/segments",
    ])
    def test_predictive_routes(self, client, path):
        _seed(client.app_module)
        r = client.get(path)
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestErrorHandling:
    def test_value_error_returns_400_not_500(self, client):
        """The ValueError handler used to *raise* HTTPException from inside
        the handler, which escapes the chain and produced a 500."""
        app_module = client.app_module

        @app_module.app.get("/__boom")
        def boom():
            raise ValueError("bad input from a caller")

        with TestClient(app_module.app, raise_server_exceptions=False) as c:
            r = c.get("/__boom")
        assert r.status_code == 400
        assert "bad input from a caller" in r.json()["detail"]


class TestDashboardRoutes:
    def test_dashboard_served(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Chakranetra" in r.text
