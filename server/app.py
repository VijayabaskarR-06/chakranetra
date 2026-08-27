"""
RoadLens AI — API Server
========================
A small FastAPI app that ties the pipeline together and serves the
municipal dashboard.

Run it:
    uvicorn server.app:app --reload
Then open:
    http://127.0.0.1:8000            (dashboard)
    http://127.0.0.1:8000/docs       (auto-generated API docs)

Endpoints:
    POST /api/scan/image      upload a road photo (+ lat/lon) -> tickets
    GET  /api/tickets         list tickets (filter by status/department)
    GET  /api/tickets/{id}    one ticket with full history
    POST /api/tickets/{id}/status   move it through the workflow
    GET  /api/stats           the commissioner's numbers
    GET  /api/predictive/alerts     predictive maintenance alerts
    GET  /api/predictive/heatmap    risk heatmap data
    GET  /api/predictive/crews      crew performance scores
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roadlens.dedup import cluster_detections          # noqa: E402
from roadlens.geo import attach_gps_manual             # noqa: E402
from roadlens.tickets import TicketStore, STATUSES     # noqa: E402
from roadlens.predictive import PredictiveEngine       # noqa: E402
from roadlens.logger import get_logger                 # noqa: E402
from roadlens.config import get_config                 # noqa: E402

logger = get_logger("api")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = get_config()

# Honour config.yaml / ROADLENS_DB_PATH instead of always writing next to the
# source tree — the README documents that override, and Docker relies on it.
DB_PATH = CONFIG.db_path or "roadlens.db"
if not os.path.isabs(DB_PATH):
    DB_PATH = os.path.join(ROOT, DB_PATH)

DASHBOARD_DIR = os.path.join(ROOT, "dashboard")
OUTPUT_DIR = os.path.join(ROOT, "output")

# StaticFiles raises at import time if this is missing, so a fresh checkout
# (output/ is gitignored) could not start the server at all until run_demo.py
# had been run once.
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

store: TicketStore | None = None
predictive: PredictiveEngine | None = None
_detector = None


def get_store() -> TicketStore:
    global store
    if store is None:
        store = TicketStore(DB_PATH)
    return store


def get_predictive() -> PredictiveEngine:
    global predictive
    if predictive is None:
        predictive = PredictiveEngine(DB_PATH)
    return predictive


def get_detector():
    global _detector
    if _detector is None:
        from roadlens.detector import RoadDefectDetector
        _detector = RoadDefectDetector(confidence=CONFIG.detector.confidence_threshold)
    return _detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Chakranetra API starting up")
    yield
    logger.info("Chakranetra API shutting down")
    if store:
        store.conn.close()
    if predictive:
        predictive.conn.close()


app = FastAPI(title="Chakranetra", version="2.0.0", lifespan=lifespan)

# allow_origins=["*"] with allow_credentials=True makes Starlette reflect the
# caller's origin back, which effectively lets any site issue credentialed
# requests. This API carries no cookies or auth, so credentials are simply off
# and the wildcard is then safe and honest.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Largest scan upload we will buffer. Without a cap, one request can fill the
# container's disk — the endpoint is unauthenticated by design (citizen
# reporting), so the cap is the only thing standing between it and a bad day.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


class ScanRequest(BaseModel):
    """Validated scan coordinates. (The multipart endpoint below validates
    lat/lon directly; this model is what a JSON client would post.)"""

    lat: float
    lon: float

    @field_validator("lat")
    @classmethod
    def validate_lat(cls, v: float) -> float:
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v

    @field_validator("lon")
    @classmethod
    def validate_lon(cls, v: float) -> float:
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v


class StatusUpdate(BaseModel):
    status: str
    note: str = ""
    assigned_to: str | None = None


# An exception handler must RETURN a response. Raising HTTPException from
# inside one escapes the handler chain, so these used to turn every ValueError
# into an unhandled 500 with a traceback instead of the intended 400.
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.warning("Bad request", error=str(exc))
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def general_error_handler(request, exc):
    logger.error("Unhandled error", error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.post("/api/scan/image")
async def scan_image(
    file: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
):
    """Citizen / field-officer flow: one photo + its location -> ticket(s)."""
    if not -90 <= lat <= 90:
        raise HTTPException(400, "Latitude must be between -90 and 90")
    if not -180 <= lon <= 180:
        raise HTTPException(400, "Longitude must be between -180 and 180")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    tmp_path = None
    try:
        written = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp_path = tmp.name
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"Image exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB limit"
                    )
                tmp.write(chunk)

        # A declared image/* content-type is not proof the bytes decode.
        # Without this check a truncated or corrupt upload reached YOLO and
        # surfaced as an opaque 500 ("need at least one array to stack")
        # for what is really a bad request.
        import cv2

        if cv2.imread(tmp_path) is None:
            raise HTTPException(400, "File is not a readable image")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        annotated = os.path.join(OUTPUT_DIR, f"scan_{os.path.basename(tmp_path)}")

        detections = get_detector().detect_image(tmp_path, save_annotated_to=annotated)
        attach_gps_manual(detections, lat, lon)
        clusters = cluster_detections(detections)
        tickets = [get_store().create_from_cluster(c) for c in clusters]

        # Close the accountability loop: if this spot was repaired recently,
        # a fresh detection here is a failed repair, not just a new pothole.
        recurrences = []
        for cluster, ticket in zip(clusters, tickets):
            hit = get_predictive().check_recurrence_at_location(
                lat=cluster.lat, lon=cluster.lon, defect_type=cluster.defect_type
            )
            if hit:
                get_store().record_recurrence(hit["original_ticket_id"])
                get_store().update_status(
                    hit["original_ticket_id"], "REOPENED",
                    note=f"Defect detected again during scan; new ticket {ticket['id']}",
                )
                recurrences.append(hit)

        logger.info("Scan completed", defects_found=len(detections),
                    unique_defects=len(clusters), tickets_created=len(tickets),
                    recurrences=len(recurrences))
        return {
            "defects_found": len(detections),
            "unique_defects": len(clusters),
            "tickets_created": [t["id"] for t in tickets],
            "recurrences": recurrences,
            "annotated_image": os.path.basename(annotated) if detections else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Scan failed", error=str(e))
        raise HTTPException(500, f"Scan failed: {str(e)}")
    finally:
        # The old code only unlinked on the happy path, so every failed scan
        # leaked a full-size JPEG into the system temp directory.
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.get("/api/tickets")
def list_tickets(status: str | None = None, department: str | None = None):
    if status and status not in STATUSES:
        raise HTTPException(400, f"Invalid status. Must be one of: {STATUSES}")
    return get_store().list(status=status, department=department)


@app.get("/api/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    t = get_store().get(ticket_id)
    if not t:
        raise HTTPException(404, f"No ticket {ticket_id}")
    return t


@app.post("/api/tickets/{ticket_id}/status")
def set_status(ticket_id: str, body: StatusUpdate):
    if body.status not in STATUSES:
        raise HTTPException(400, f"status must be one of {STATUSES}")
    try:
        result = get_store().update_status(ticket_id, body.status, body.note, body.assigned_to)
        if body.status == "FIXED":
            ticket = get_store().get(ticket_id)
            # `and ticket["lat"]` would skip lat/lon of exactly 0.0
            if ticket and ticket.get("lat") is not None and ticket.get("lon") is not None:
                get_predictive().register_fixed_ticket(
                    ticket_id=ticket_id,
                    lat=ticket["lat"],
                    lon=ticket["lon"],
                    defect_type=ticket["defect_type"],
                    assigned_crew=body.assigned_to,
                )
        return result
    except KeyError:
        raise HTTPException(404, f"No ticket {ticket_id}")


@app.get("/api/stats")
def stats():
    return get_store().stats()


@app.get("/api/predictive/alerts")
def predictive_alerts():
    """Get predictive maintenance alerts and repair quality issues."""
    tickets = get_store().list()
    return get_predictive().get_predictive_alerts(tickets)


@app.get("/api/predictive/heatmap")
def predictive_heatmap():
    """Get risk heatmap data for visualization."""
    tickets = get_store().list()
    return get_predictive().get_heatmap_data(tickets)


@app.get("/api/predictive/crews")
def crew_performance():
    """Get repair quality scores by crew."""
    return get_predictive().get_crew_performance()


@app.get("/api/predictive/segments")
def risk_segments():
    """Get computed risk segments."""
    tickets = get_store().list()
    segments = get_predictive().compute_risk_segments(tickets)
    return [
        {
            "center_lat": s.center_lat,
            "center_lon": s.center_lon,
            "radius_meters": s.radius_meters,
            "total_defects_ever": s.total_defects_ever,
            "defects_last_30_days": s.defects_last_30_days,
            "recurrence_count": s.recurrence_count,
            "risk_score": s.risk_score,
            "risk_label": s.risk_label,
            "recommended_action": s.recommended_action,
        }
        for s in segments
    ]


app.mount("/evidence", StaticFiles(directory=OUTPUT_DIR), name="evidence")


@app.get("/")
def dashboard():
    return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))


@app.get("/demo_data.js")
def demo_data():
    path = os.path.join(DASHBOARD_DIR, "demo_data.js")
    if not os.path.exists(path):
        raise HTTPException(404, "Run `python run_demo.py` first")
    return FileResponse(path)
