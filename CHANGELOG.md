# Changelog

All notable changes to Chakranetra are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] — Predictive Maintenance Platform

### Added

#### Predictive Engine (`roadlens/predictive.py`)
- **Recurrence Tracking**: Automatically detects when a defect reappears at a previously repaired location (within 15 m, configurable) and records it against the original ticket.
- **Repair Quality Scores**: Each repair crew accumulates a quality score based on how often their repairs hold. Every recurrence reduces the crew's score by 25%.
- **Crew Performance Ratings**: Crews are rated Excellent / Good / Needs Improvement / Requires Retraining based on their aggregate quality score across all monitored repairs.
- **Risk Heatmap**: The city is divided into a configurable grid (default 0.5 km²). Each cell receives a risk score from defect frequency (40%), historical density (30%), and recurrence rate (30%).
- **Preventive Alerts**: Road segments exceeding the risk threshold trigger a preventive-maintenance alert with recommended action and estimated preventive cost — raised *before* citizens report a problem.

#### New API Endpoints (`server/app.py`)
- `GET /api/predictive/alerts` — active preventive maintenance alerts
- `GET /api/predictive/heatmap` — risk heatmap data per grid cell
- `GET /api/predictive/crews` — crew performance scores
- `GET /api/predictive/segments` — road segments ranked by risk score

#### Production Infrastructure
- **Structured JSON logging** (`roadlens/logger.py`) — every operation emits a machine-parseable log event with timestamp, module, level, and context.
- **Configuration management** (`roadlens/config.py`) — all parameters are loaded from `config.yaml` with environment variable overrides (`ROADLENS_*`). No more hardcoded constants.
- **Docker support** — `Dockerfile` and `docker-compose.yml` for one-command deployment.
- **CI/CD pipeline** — GitHub Actions workflow runs the full test suite on every push and pull request.
- **Comprehensive test suite** — `tests/` now covers detector parity, JS scoring parity, severity rules, dedup logic, and ticket lifecycle.

#### Browser-Side AI (`dashboard/scan.js`)
- The same 45 MB YOLOv8s-seg model exported to ONNX runs directly in the browser via ONNX Runtime WebAssembly.
- Pixel-identical mask output verified against the Python pipeline — IoU 1.000000, 0 differing pixels.
- Non-square photos are letterboxed correctly; the overlay is composited in letterbox space and cropped back to the photo's original aspect ratio.

#### Parity Tests
- `tools/check_onnx_parity.py` — verifies the ONNX graph reproduces ultralytics output (`area_ratio` delta `0.000000`).
- `tests/test_js_inference_parity.py` — verifies `dashboard/scan.js` reproduces Python post-processing boxes, confidences, `area_ratio` to `1e-9`, and masks pixel-for-pixel.
- `tests/test_js_parity.py` — verifies JS scoring matches `roadlens.severity.assess` on 2,000 random inputs plus every band boundary.

### Changed
- Ticket status `REOPENED` is now triggered automatically by the predictive engine when a post-repair scan detects a defect at the repaired location — no manual intervention required.
- Dashboard (`dashboard/index.html`) updated with a Predictive board tab showing risk alerts, crew leaderboard, and heatmap overlay.
- `config.yaml` expanded with `predictive` section (recurrence radius, monitoring window, alert thresholds, heatmap grid size).
- All severity thresholds and scoring weights moved from inline constants to `config.yaml`.

### Fixed
- Mask thresholding now correctly operates on logits (`> 0`) rather than sigmoid probabilities (`> 0.5`), matching ultralytics' own pipeline. The previous behaviour could shift mask areas by up to 0.0034 of the frame — enough to misclassify severity.
- Same-frame separation rule correctly prevents two adjacent potholes in one frame from being merged into a single defect cluster.

---

## [1.0.0] — Initial Release

### Added

#### Core Detection Pipeline
- **`roadlens/detector.py`** — YOLOv8-seg inference on images and video. Frame sampling (every Nth frame, configurable) avoids processing near-identical consecutive frames.
- **`roadlens/geo.py`** — GPS attachment: GPX track interpolation maps each video frame to a vehicle position by timestamp. Supports manual lat/lon for citizen photo reports.
- **`roadlens/dedup.py`** — Haversine-distance clustering merges multiple sightings of the same physical defect into one `DefectCluster`. Same-frame separation rule keeps two adjacent potholes distinct.
- **`roadlens/severity.py`** — Plain-arithmetic severity scoring: L1–L4 bands by frame-area ratio, priority score (0–100) from size + confidence + sighting count, SLA deadlines, repair cost estimates, and department routing. All thresholds in one table.
- **`roadlens/tickets.py`** — SQLite-backed ticket registry. Full lifecycle: `OPEN → ASSIGNED → IN_PROGRESS → FIXED → VERIFIED`. AI re-scan confirms fixes; detected recurrence moves ticket to `REOPENED`.

#### API (`server/app.py`)
- `POST /api/scan/image` — photo + lat/lon → detect → dedup → ticket(s) + annotated evidence image.
- `GET /api/tickets` — work queue sorted by priority; filterable by status and department.
- `GET /api/tickets/{id}` — single ticket with full sighting history.
- `POST /api/tickets/{id}/status` — civic workflow status transitions.
- `GET /api/stats` — commissioner-level summary numbers.
- `GET /` — serves `dashboard/index.html`.

#### Dashboard (`dashboard/index.html`)
- Interactive map with defect pins coloured by severity.
- Filterable, sortable work queue with SLA countdown timers.
- Ticket detail panel with full lifecycle history and evidence photos.
- Status deep-linking (`#t/RL-POT-2026-0003`).
- Status changes persist across reloads via `localStorage`.

#### Demo & Tooling
- `run_demo.py` — end-to-end pipeline on the sample dataset. Downloads the pretrained model once, scans two simulated vehicle passes, deduplicates, files tickets.
- `data/samples/` — real road photos from a pothole-segmentation dataset.
- `data/demo_route.gpx` — sample GPS track along Bengaluru's Outer Ring Road.

---

[2.0.0]: https://github.com/VijayabaskarR-06/chakranetra/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/VijayabaskarR-06/chakranetra/releases/tag/v1.0.0
