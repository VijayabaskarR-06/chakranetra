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
    POST /api/scan/video      upload dashcam footage (+ GPS track or lat/lon) -> tickets
    GET  /api/tickets         list tickets (filter by status/department)
    GET  /api/tickets/{id}    one ticket with full history
    POST /api/tickets/{id}/status   move it through the workflow
    GET  /api/stats           the commissioner's numbers
    GET  /api/predictive/alerts     predictive maintenance alerts
    GET  /api/predictive/heatmap    risk heatmap data
    GET  /api/predictive/crews      crew performance scores
    POST /api/tickets/{id}/cost     record what a repair actually cost
    GET  /api/ml/status             which models are trained, and on what data
    GET  /api/ml/cost/{id}          predicted repair cost + conformal interval
    GET  /api/ml/forecast/{id}      days until the next severity band
    GET  /api/ml/failure/{id}       P(this repair comes back)
    GET  /api/ml/budget             30/60/90-day spend forecast
"""

from __future__ import annotations

import math
import os
import tempfile
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roadlens.dedup import cluster_detections                          # noqa: E402
from roadlens.geo import GpxTrack, attach_gps_from_track, attach_gps_manual  # noqa: E402
from roadlens.tickets import TicketStore, STATUSES     # noqa: E402
from roadlens.predictive import PredictiveEngine       # noqa: E402
from roadlens.logger import get_logger                 # noqa: E402
from roadlens.config import get_config                 # noqa: E402
from roadlens.ml.registry import get_registry, reset_registry, ModelRegistry  # noqa: E402

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

# Video is the same story at a different scale: a few minutes of 720p dashcam
# footage runs tens of MB. 150 MB covers a realistic clip without letting one
# upload park a feature-length file on disk.
MAX_VIDEO_UPLOAD_BYTES = 150 * 1024 * 1024
MAX_GPX_UPLOAD_BYTES = 5 * 1024 * 1024      # a GPS track is plain text; this is generous

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


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


class CostReport(BaseModel):
    """What a repair actually cost — the label the cost model learns from."""

    actual_cost_inr: int
    note: str = ""

    @field_validator("actual_cost_inr")
    @classmethod
    def validate_cost(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("actual_cost_inr must be a positive number of rupees")
        # A typo of a few extra zeroes in a public, unauthenticated endpoint
        # would drag the whole trained model with it, and one bad label is
        # much harder to find later than one rejected request.
        if v > 100_000_000:
            raise ValueError("actual_cost_inr above Rs 10 crore looks like a typo")
        return v


# Retraining reads the whole ticket table and fits three ensembles. That is
# fine on a laptop and a denial-of-service vector on a public demo host, so
# the endpoint exists but is off unless the operator turns it on.
TRAINING_ENABLED = os.getenv("ROADLENS_ALLOW_TRAINING", "").lower() in ("1", "true", "yes")


def _ml():
    return get_registry(get_store(), get_predictive())


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


def _save_capped_upload(file: UploadFile, suffix: str, max_bytes: int) -> str:
    """Stream an upload to a temp file, aborting once it exceeds `max_bytes`.

    Shared by every upload endpoint so the cap-and-cleanup behaviour — and the
    413 it raises — is written once. Streaming in chunks rather than reading
    the whole body first means an oversized upload is rejected without ever
    holding the full thing in memory.
    """
    written = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                tmp.close()
                os.unlink(tmp_path)
                raise HTTPException(
                    413, f"{file.filename or 'upload'} exceeds the "
                         f"{max_bytes // (1024 * 1024)} MB limit"
                )
            tmp.write(chunk)
    return tmp_path


def _file_and_track(detections: list) -> dict:
    """Turn raw detections into filed tickets — shared by the image and video
    scan endpoints, so a photo and a dashcam clip land in the queue the same
    way and are tested once.

    Every cluster either grows an already-open ticket for the same physical
    defect, or becomes a new one. Recurrence — "was this spot repaired and is
    it damaged again?" — is only meaningful for the *new* branch: a cluster
    that matched an open ticket is adding a sighting to a defect already in
    the queue, not reappearing at a spot someone signed off as fixed, so
    running the recurrence check on it as well would be checking a question
    that does not apply and could, on a coincidence of geography, misfire.
    """
    clusters = cluster_detections(detections)

    filed = []   # [{"cluster": DefectCluster, "ticket": dict, "is_new": bool}, ...]
    for c in clusters:
        existing = get_store().find_open_at_location(c.lat, c.lon, c.defect_type)
        if existing:
            filed.append({"cluster": c, "ticket": get_store().record_growth(existing, c),
                          "is_new": False})
        else:
            filed.append({"cluster": c, "ticket": get_store().create_from_cluster(c),
                          "is_new": True})

    recurrences = []
    claimed: set[str] = set()
    for f in filed:
        if not f["is_new"]:
            continue
        c = f["cluster"]
        hit = get_predictive().check_recurrence_at_location(
            lat=c.lat, lon=c.lon, defect_type=c.defect_type, exclude_ticket_ids=claimed,
        )
        if hit:
            claimed.add(hit["original_ticket_id"])
            get_store().record_recurrence(hit["original_ticket_id"])
            get_store().update_status(
                hit["original_ticket_id"], "REOPENED",
                note=f"Defect detected again during scan; new ticket {f['ticket']['id']}",
            )
            recurrences.append(hit)

    result = {
        "defects_found": len(detections),
        "unique_defects": len(clusters),
        "tickets_created": [f["ticket"]["id"] for f in filed if f["is_new"]],
        # Re-sightings of defects already in the queue. Each one adds a point
        # to that defect's growth history, which is what the degradation
        # model trains on.
        "tickets_updated": [f["ticket"]["id"] for f in filed if not f["is_new"]],
        "recurrences": recurrences,
    }
    logger.info("Scan filed", defects_found=result["defects_found"],
               unique_defects=result["unique_defects"],
               tickets_created=len(result["tickets_created"]),
               tickets_updated=len(result["tickets_updated"]),
               recurrences=len(recurrences))
    return result


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
        tmp_path = _save_capped_upload(file, suffix=".jpg", max_bytes=MAX_UPLOAD_BYTES)

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

        result = _file_and_track(detections)
        result["annotated_image"] = os.path.basename(annotated) if detections else None
        return result
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


@app.post("/api/scan/video")
async def scan_video(
    file: UploadFile = File(...),
    gpx: Optional[UploadFile] = File(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    fps: float = Form(30.0),
    vehicle_id: Optional[str] = Form(None),
):
    """Dashcam / CCTV flow: a video clip (+ its GPS track) -> ticket(s).

    Exactly the photo flow, run over sampled frames instead of one image: the
    same detector, the same clustering, the same open-ticket matching and
    recurrence check via `_file_and_track`. The only new work here is turning
    "a video plus a GPS track" into the (lat, lon)-tagged detections that flow
    already expects — from a real GPX recording when the vehicle was moving,
    or a single fixed point for a stationary CCTV camera.
    """
    if gpx is None and (lat is None or lon is None):
        raise HTTPException(
            400, "Provide a GPS track (gpx) for a moving vehicle, or lat and "
                 "lon for a stationary camera"
        )
    if lat is not None and not -90 <= lat <= 90:
        raise HTTPException(400, "Latitude must be between -90 and 90")
    if lon is not None and not -180 <= lon <= 180:
        raise HTTPException(400, "Longitude must be between -180 and 180")
    if fps <= 0:
        raise HTTPException(400, "fps must be positive")

    ext = os.path.splitext(file.filename or "")[1].lower()
    looks_like_video = (file.content_type or "").startswith("video/") or ext in ALLOWED_VIDEO_EXTENSIONS
    if not looks_like_video:
        raise HTTPException(400, "File must be a video")

    video_path, gpx_path, annotated_dir = None, None, None
    try:
        video_path = _save_capped_upload(
            file, suffix=ext if ext in ALLOWED_VIDEO_EXTENSIONS else ".mp4",
            max_bytes=MAX_VIDEO_UPLOAD_BYTES,
        )

        import cv2

        cap = cv2.VideoCapture(video_path)
        opened = cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        readable, _ = cap.read() if opened else (False, None)
        cap.release()
        if not opened or not readable:
            raise HTTPException(400, "File is not a readable video")

        track = None
        if gpx is not None:
            gpx_path = _save_capped_upload(gpx, suffix=".gpx", max_bytes=MAX_GPX_UPLOAD_BYTES)
            try:
                track = GpxTrack.load(gpx_path)
            except (ET.ParseError, ValueError) as e:
                raise HTTPException(400, f"Invalid GPS track: {e}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        annotated_dir = os.path.join(OUTPUT_DIR, f"scan_video_{os.path.basename(video_path)}")

        sample_every = CONFIG.detector.video_sample_every_n_frames
        detections = get_detector().detect_video(
            video_path, sample_every_n_frames=sample_every, save_annotated_dir=annotated_dir,
        )

        if track is not None:
            attach_gps_from_track(detections, track, fps=fps)
            gps_source = "gpx"
        else:
            # Every detection pins to the one point given. Fine for a fixed
            # CCTV camera; a moving vehicle with no track means every defect
            # in the clip lands on the same coordinate, which is worth
            # knowing about rather than discovering on the map later.
            attach_gps_manual(detections, lat, lon)
            gps_source = "manual"
            logger.warning("Video scan used a single manual GPS point",
                           source=file.filename, lat=lat, lon=lon)

        result = _file_and_track(detections)
        result["gps_source"] = gps_source
        result["source"] = file.filename
        if vehicle_id:
            result["vehicle_id"] = vehicle_id
        result["frames_analyzed"] = (
            math.ceil(frame_count / sample_every) if frame_count > 0 else None
        )
        annotated_files = sorted(os.listdir(annotated_dir)) if os.path.isdir(annotated_dir) else []
        result["annotated_frames"] = [
            f"{os.path.basename(annotated_dir)}/{name}" for name in annotated_files
        ]
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Video scan failed", error=str(e))
        raise HTTPException(500, f"Video scan failed: {str(e)}")
    finally:
        for p in (video_path, gpx_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
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


# ---------------------------------------------------------------------------
# Learned models (roadlens.ml)
# ---------------------------------------------------------------------------

@app.post("/api/tickets/{ticket_id}/cost")
def report_actual_cost(ticket_id: str, body: CostReport):
    """Record what a repair actually cost.

    This is the endpoint that makes the cost model more than a restatement of
    the rules formula. Until a city posts real invoices here, every cost
    prediction is labelled `source: "rules"` or trained on the synthetic
    bootstrap corpus, and says so.
    """
    try:
        ticket = get_store().record_actual_cost(ticket_id, body.actual_cost_inr)
    except KeyError:
        raise HTTPException(404, f"No ticket {ticket_id}")
    return {
        "ticket": ticket,
        "rules_estimate_inr": ticket.get("est_cost_inr"),
        "actual_cost_inr": ticket.get("actual_cost_inr"),
        "note": "Retrain with `python tools/train_models.py` to fold this into the model.",
    }


@app.get("/api/ml/status")
def ml_status():
    """Which models are trained, on what data, and how well they scored.

    Deliberately the most detailed endpoint in the API. A prediction about
    public money should come with its own audit trail attached, and
    `is_synthetic` is the field the dashboard renders as a warning badge.
    """
    return _ml().status()


@app.get("/api/ml/cost/{ticket_id}")
def ml_cost(ticket_id: str, explain: bool = True):
    """Predicted repair cost with a conformal interval, and why."""
    ticket = get_store().get(ticket_id)
    if not ticket:
        raise HTTPException(404, f"No ticket {ticket_id}")
    registry = _ml()
    out = registry.cost.predict(ticket)
    out["is_synthetic_model"] = registry.status()["is_synthetic"]
    if explain:
        out["explanation"] = registry.cost.explain(ticket)
    return out


@app.get("/api/ml/forecast/{ticket_id}")
def ml_forecast(ticket_id: str):
    """Days until this defect reaches the next severity band."""
    ticket = get_store().get(ticket_id)
    if not ticket:
        raise HTTPException(404, f"No ticket {ticket_id}")
    registry = _ml()
    out = registry.degradation.forecast(ticket)
    out["observations"] = len(get_store().observations(ticket_id))
    out["is_synthetic_model"] = registry.status()["is_synthetic"]
    return out


@app.get("/api/ml/failure/{ticket_id}")
def ml_failure(ticket_id: str, crew: str | None = None):
    """Probability this repair fails and the defect returns."""
    ticket = get_store().get(ticket_id)
    if not ticket:
        raise HTTPException(404, f"No ticket {ticket_id}")
    registry = _ml()
    out = registry.failure.predict(ticket, crew=crew)
    out["is_synthetic_model"] = registry.status()["is_synthetic"]
    return out


@app.get("/api/ml/budget")
def ml_budget(group_by: str = "department"):
    """30/60/90-day spend forecast with a simulated uncertainty band."""
    if group_by not in ("department", "severity_label", "defect_type", "road_class"):
        raise HTTPException(400, "group_by must be one of: department, "
                                 "severity_label, defect_type, road_class")
    registry = _ml()
    out = registry.budget.forecast(get_store().list(), group_by=group_by)
    out["is_synthetic_model"] = registry.status()["is_synthetic"]
    return out


@app.post("/api/ml/train")
def ml_train():
    """Retrain from the live database. Off unless ROADLENS_ALLOW_TRAINING is set."""
    if not TRAINING_ENABLED:
        raise HTTPException(
            403, "Training is disabled on this host. Set ROADLENS_ALLOW_TRAINING=1 "
                 "to enable it, or run `python tools/train_models.py` locally."
        )
    registry = ModelRegistry.train_from_store(get_store(), get_predictive())
    registry.save()
    reset_registry()
    return registry.status()


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
