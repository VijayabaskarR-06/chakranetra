# Chakranetra v2.0

**Every vehicle becomes a road inspector. Every defect becomes an accountable municipal ticket. Every repair is tracked for quality. Problems are predicted before they happen.**

### ▶ [Open the live operations console](https://vijayabaskarr-06.github.io/chakranetra/)

The console runs on real output from this repository — 7 tickets produced by scanning the
sample trip in `data/samples/`. Pan the map, filter the work queue, move a ticket through
the civic workflow. For live image scanning, run the API locally (see Quickstart below).

Chakranetra takes ordinary dashcam or CCTV footage, finds road infrastructure defects
(potholes today; cracks, broken footpaths, missing zebra crossings next), and converts
each *unique physical defect* into a routed, costed, SLA-tracked ticket for the
municipal road department — closing the loop the problem statement asks for:

> *issue detected → ticket created → routed to department → assigned to site engineer → fixed → verified → monitored for recurrence*

India's Ministry of Road Transport & Highways attributes thousands of road deaths in
recent years to pothole-related accidents. Cities don't lack the will to fix roads —
they lack a **trustworthy, deduplicated, prioritized list of what is broken, where,
and how urgent it is**. That list is what Chakranetra produces, automatically, from
vehicles that are already driving the roads anyway.

---

## Why this is different from "a model that draws boxes on potholes"

Most computer-vision demos stop at detection. Detection alone is useless to a city.
Chakranetra is built around the problems that actually break real deployments:

### 1. Deduplication — the killer problem
A dashcam at 30 fps sees the same pothole in ~40 consecutive frames. Tomorrow, three
more vehicles drive past it. A naive system files 160 tickets for one hole, and the
municipal officer stops trusting the dashboard on day one. Chakranetra merges every
sighting of the same physical defect (GPS clustering via haversine distance, with a
same-frame separation rule so two adjacent potholes are never wrongly merged) into
**one ticket that gets *stronger* with each sighting** — more sightings means higher
confidence and higher priority, not more noise.

### 2. Prioritization a city can defend
Every ticket carries a severity level (L1–L4), a 0–100 priority score, an estimated
repair cost, an SLA deadline, and a department route. The scoring is deliberately
plain arithmetic, not a black box — rules that direct public money must be explainable
to a commissioner in one sentence. All thresholds live in `config.yaml` for easy tuning.

### 3. Accountability, both directions
Tickets move through `OPEN → ASSIGNED → IN_PROGRESS → FIXED → VERIFIED`. The
`VERIFIED` step is the honest part: when a later vehicle pass no longer detects the
defect at that location, the *same AI that reported the problem confirms the fix*.
No self-reported paperwork. And if a later scan still sees it → `REOPENED`.

### 4. Predictive Recurrence Prevention (NEW in v2.0)
This is the feature that **prevents problems from coming back**:

- **Recurrence Tracking**: When a defect reappears at a previously fixed location, Chakranetra automatically flags it, tracks how many times it has recurred, and calculates a repair quality score for the assigned crew.
- **Repair Quality Scores**: Each crew gets a quality score based on how often their repairs hold up. Crews with consistently low scores are flagged for retraining.
- **Risk Heatmaps**: Chakranetra identifies road segments with historically high defect density so cities can schedule preventive resurfacing BEFORE new holes form.
- **Predictive Alerts**: When a road segment crosses the recurrence threshold, the system raises a preventive-maintenance alert on `/api/predictive/alerts` BEFORE citizens complain, with the recommended action and an estimated preventive cost.

This transforms Chakranetra from a reactive patching system into a **preventive maintenance platform**.

### 5. Production-Ready Infrastructure (NEW in v2.0)
- Structured JSON logging for every operation
- Configuration management via YAML or environment variables
- Input validation and error handling on all API endpoints
- Docker & Docker Compose support
- CI/CD pipeline with GitHub Actions
- Comprehensive test suite with pytest

---

## What's in the box

```
chakranetra/
├── run_demo.py                ← one command, full pipeline, no API keys, CPU-only
├── config.yaml                ← all tunable parameters in one place
├── Dockerfile                 ← production-ready container
├── docker-compose.yml         ← one-command deployment
├── roadlens/
│   ├── detector.py            YOLOv8 segmentation inference (images + video)
│   ├── geo.py                 GPS attach: GPX interpolation / manual (citizen app)
│   ├── dedup.py               sighting → unique-defect clustering
│   ├── severity.py            severity, priority, cost, SLA, dept routing
│   ├── tickets.py             SQLite ticket registry + lifecycle + recurrence
│   ├── predictive.py          recurrence tracking, risk heatmaps, crew scores
│   ├── config.py              configuration management (YAML + env vars)
│   └── logger.py              structured JSON logging
├── server/app.py              FastAPI: scan upload API + ticket API + predictive API + dashboard
├── dashboard/index.html       municipal road-operations console (map + work queue)
├── data/samples/              real road photos (pothole-segmentation dataset)
├── data/demo_route.gpx        demo dashcam GPS track (Bengaluru ORR)
├── tests/                     comprehensive test suite
├── .github/workflows/ci.yml   CI/CD pipeline
└── docs/ARCHITECTURE.md       data flow, module boundaries, design decisions
```

## Quickstart (3 commands)

```bash
pip install -r requirements.txt
python run_demo.py                 # full pipeline on real images, ~1 min on CPU
uvicorn server.app:app             # → open http://127.0.0.1:8000
```

`run_demo.py` downloads the pretrained pothole segmentation model once
(`keremberke/yolov8s-pothole-segmentation`, cached afterward), scans the sample trip,
deduplicates across two simulated vehicle passes, files tickets, demonstrates the
predictive recurrence engine, and exports data so `dashboard/index.html` also works
standalone — just double-click it.

### Docker Quickstart

```bash
docker compose up --build
# → open http://127.0.0.1:8000
```

---

## The API

| Endpoint | What it does |
|---|---|
| `POST /api/scan/image` | photo + lat/lon → detect → dedup → ticket(s) + annotated evidence; also re-opens a recently-repaired ticket if the defect is back |
| `GET /api/tickets` | work queue, sorted by priority (filter by status / department) |
| `GET /api/tickets/{id}` | single ticket with full history |
| `POST /api/tickets/{id}/status` | move a ticket through the civic workflow |
| `GET /api/stats` | commissioner's numbers: open, critical, past-SLA, backlog, recurrences |
| `GET /api/predictive/alerts` | predictive maintenance alerts & repair quality issues |
| `GET /api/predictive/heatmap` | risk heatmap data for visualization |
| `GET /api/predictive/crews` | crew performance scores |
| `GET /api/predictive/segments` | computed risk segments |
| `GET /` | the road-operations dashboard |

Interactive docs are auto-generated at `/docs`.

---

## How severity works (fully transparent)

| Level | Meaning | Frame area | SLA | Base cost |
|---|---|---|---|---|
| L4 Critical | accident risk now | ≥ 6% | 24 h | 18,000 |
| L3 High | two-wheeler hazard | ≥ 2.5% | 72 h | 9,000 |
| L2 Medium | forming, will grow | ≥ 0.8% | 7 d | 4,500 |
| L1 Low | surface wear | any | 14 d | 1,800 |

Priority (0–100) = size (up to 60) + model confidence (up to 25) + repeat sightings
(up to 15). Every constant is documented in `config.yaml` and tunable per city.

---

## Predictive Engine — How It Prevents Problems

### Recurrence Tracking Flow
1. When a ticket is marked `FIXED`, Chakranetra registers it for monitoring (90-day window by default).
2. If a later scan detects a defect within 15 m of the original location, a recurrence is recorded against the *original* ticket, that ticket moves to `REOPENED`, and the new sighting still gets its own ticket. Matching is by location, not by ticket id — a defect that comes back is always detected as a brand-new sighting.
3. Each recurrence reduces the repair quality score by 25%.
4. At 2+ recurrences: "high" severity alert, full-depth repair recommended.
5. At 4+ recurrences: "critical" alert, contractor review triggered.

### Risk Heatmap
Chakranetra divides the city into a grid and computes a risk score for each cell based on:
- Defect frequency in the last 30 days (40% weight)
- Total historical defect density (30% weight)
- Recurrence rate (30% weight)

Segments scoring above 0.85 trigger immediate preventive resurfacing recommendations.

### Crew Performance
Each crew gets an average quality score across all their monitored repairs:
- 0.9–1.0: Excellent
- 0.7–0.9: Good
- 0.5–0.7: Needs improvement
- Below 0.5: Requires retraining

---

## Configuration

All tunable parameters are in `config.yaml`:

```yaml
detector:
  confidence_threshold: 0.35
  video_sample_every_n_frames: 15

dedup:
  merge_radius_meters: 12.0

predictive:
  recurrence_radius_meters: 15.0
  monitoring_window_days: 90
  high_recurrence_threshold: 2
  critical_recurrence_threshold: 4
  heatmap_grid_size_km: 0.5

log_level: INFO
```

Environment variables are layered **on top of** `config.yaml`, so they win:
```bash
export ROADLENS_CONFIDENCE=0.5
export ROADLENS_MERGE_RADIUS=15
export ROADLENS_RECURRENCE_RADIUS=20
export ROADLENS_LOG_LEVEL=DEBUG
export ROADLENS_LOG_FORMAT=text
export ROADLENS_DB_PATH=/data/roadlens.db
```

---

## Running Tests

```bash
pytest tests/ -v --cov=roadlens --cov-report=term-missing
```

---

## Honest limitations (and the plan for each)

- **Cost estimates use a calibration constant**, not measured metres. Production
  calibrates per m² per road class from the camera geometry; the constants ship as
  clearly-labelled defaults.
- **The bundled model detects potholes only.** The detector is a plug-in behind one
  interface — training YOLOv8 on a multi-class dataset extends coverage with zero
  changes elsewhere; department routing for those classes is already wired.
- **GPS dedup radius (12 m) is a heuristic** chosen for consumer GPS error. Dense
  urban deployments can add visual re-identification as a second merge signal.
- **SQLite is a pilot database.** Single-writer is fine for one city ward; the
  `TicketStore` interface swaps to Postgres unchanged at scale.
- **Predictive model is rules-based**, not ML. This is intentional — cities need
  explainable predictions. A future version can layer ML-based deterioration modeling
  on top of the existing rules engine.

---

## Where this goes

Ward-level pilot with municipal garbage trucks and buses (they already cover every
street weekly) → citizen photo reporting through the same `/api/scan/image` endpoint →
integration with existing municipal grievance systems → a public transparency map so
residents see the same queue the city sees → **predictive maintenance scheduling that
fixes roads before they break**.

---

*Built as a complete, working system — every number in this README comes from actually
running the code in this repository. v2.0 adds predictive analytics, recurrence
prevention, structured logging, configuration management, Docker support, CI/CD,
and a comprehensive test suite.*
