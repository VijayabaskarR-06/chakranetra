# RoadLens AI — Architecture

## Data flow

```
  dashcam video / CCTV / citizen photo
              │
              ▼
   ┌─────────────────────┐    frame sampling (every Nth frame —
   │  detector.py        │    consecutive frames are near-identical,
   │  YOLOv8-seg model   │    so 2 fps is enough and 15× faster)
   └─────────┬───────────┘
             │  Detection {type, confidence, box, area_ratio, frame}
             ▼
   ┌─────────────────────┐    GPX track shares the video's timeline:
   │  geo.py             │    frame 450 @30fps = t=15s → interpolate
   │  GPS attach         │    vehicle position at t=15s
   └─────────┬───────────┘
             │  Detection + lat/lon
             ▼
   ┌─────────────────────┐    haversine clustering (≤12 m = same hole)
   │  dedup.py           │    + same-frame separation rule
   │  N sightings →      │    sightings make a defect STRONGER,
   │  M unique defects   │    never noisier
   └─────────┬───────────┘
             │  DefectCluster {worst size, best confidence, sightings}
             ▼
   ┌─────────────────────┐    plain, explainable arithmetic:
   │  severity.py        │    severity L1–L4, priority 0–100,
   │  assessment         │    ₹ cost, SLA hours, department route
   └─────────┬───────────┘
             ▼
   ┌─────────────────────┐    OPEN → ASSIGNED → IN_PROGRESS →
   │  tickets.py         │    FIXED → VERIFIED (AI re-scan confirms)
   │  SQLite registry    │              └→ REOPENED if still detected
   └─────────┬───────────┘
             ▼
   ┌─────────────────────────────────────────┐
   │  server/app.py (FastAPI)                │
   │  /api/scan  /api/tickets  /api/stats    │
   └─────────┬───────────────────────────────┘
             ▼
   dashboard/index.html — map + prioritized work queue + SLA tracking
```

## Design decisions worth defending

**The model is a plug-in.** Nothing outside `detector.py` knows YOLO exists — the
rest of the system consumes plain `Detection` objects. Swapping in a multi-class
model (cracks, manholes, footpaths) or an entirely different architecture changes
one file.

**Dedup is geometric, not learned.** Same-defect merging uses GPS distance with an
explicit radius and an explicit same-frame rule. A learned re-identification model
would be more precise but unexplainable; for a system that directs public money,
we start explainable and add ML only where it earns its complexity.

**Severity is arithmetic on purpose.** A commissioner must be able to answer "why
is this ticket Critical?" in one sentence: "It fills 6% of the camera frame, the
model is 75% confident, and two vehicles reported it." Every threshold sits in one
table in `severity.py`.

**Evidence travels with the ticket.** Annotated frames are saved per scan and served
at `/evidence/…` — a crew never argues with a photo of the hole with the AI's own
outline drawn on it.

**SQLite → Postgres is an interface swap.** `TicketStore` is the only class that
touches the database.

## Scaling path

1 vehicle → edge inference on-device (YOLOv8s runs on a phone/Jetson), tickets sync
over 4G. 1,000 vehicles → detections stream to a queue, dedup runs as a windowed
job per road segment, Postgres + PostGIS replaces SQLite, tiles pre-aggregate for
the dashboard. The module boundaries in this repo are drawn exactly where those
production seams are.
