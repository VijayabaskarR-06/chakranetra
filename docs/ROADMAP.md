# Chakranetra — Product Roadmap

This document describes the planned direction for Chakranetra. Items are grouped by release milestone. Community contributions toward any of these are welcome — see [CONTRIBUTING.md](../CONTRIBUTING.md).

> **Note**: Roadmap items are targets, not commitments. Priorities may shift based on pilot feedback from municipal deployments.

---

## ✅ Shipped — v2.0

- Predictive recurrence engine (crew quality scores, risk heatmaps, preventive alerts)
- Browser-side YOLOv8-seg inference via ONNX Runtime WebAssembly (pixel-identical to server)
- Structured JSON logging and YAML/env-var configuration management
- Docker & Docker Compose support
- GitHub Actions CI/CD pipeline
- Comprehensive test suite including JS parity tests

---

## 🚧 In Progress — v2.x

### v2.1 — Multi-Class Detection

- **Goal**: Extend beyond potholes to cracks, broken footpaths, missing zebra crossings, damaged signage, and open manholes.
- **Approach**: Plug a multi-class YOLOv8 model into the existing `detector.py` interface — no changes to dedup, severity, or ticketing logic. Department routing tables in `config.yaml` already have slots for these classes.
- **Status**: Waiting on a labelled multi-class dataset. Contributions of annotated road images are welcome.

### v2.2 — Citizen Reporting App (PWA)

- **Goal**: Allow any citizen with a smartphone to report a road defect via the same `/api/scan/image` endpoint.
- **Features**:
  - Progressive Web App installable from the browser
  - Camera capture → client-side inference (already working in Scan tab) → one-tap report
  - Citizen sees the ticket number and can track its status
  - Report is tagged as `source: citizen` for weighted confidence scoring
- **Status**: Dashboard Scan tab already handles the inference; needs a simplified reporting UI and push notification integration.

### v2.3 — Postgres + PostGIS Migration

- **Goal**: Replace SQLite with Postgres + PostGIS for multi-writer deployments and spatial queries.
- **Approach**: `TicketStore` in `tickets.py` is the only class that touches the database. The interface is already defined; only the implementation changes.
- **Benefits**:
  - Concurrent writes from multiple edge devices
  - Native spatial indexing — dedup and heatmap queries run as DB-level `ST_DWithin` calls instead of Python loops
  - Seamless integration with existing GIS tooling cities already use
- **Status**: Interface ready; implementation pending.

---

## 🔮 Planned — v3.0 (Preventive Maintenance Platform)

### ML-Based Deterioration Forecasting

- Replace the rules-based risk heatmap with a learned deterioration model trained on historical ticket data, weather data, and traffic volume.
- The rules engine remains as an explainable fallback — cities need to justify predictions to commissioners.
- Output: probability of a new pothole forming at a given location within the next 30/60/90 days.

### Integration with Municipal Grievance Systems

- Bi-directional sync with existing civic grievance portals (e.g., Sampark, 311 equivalents).
- A Chakranetra ticket can automatically file a grievance on the city portal; status updates flow back in real time.
- Prevents double-entry and keeps citizens informed through the platform they already use.

### Public Transparency Map

- A read-only public view of the ticket queue: residents see the same defect map the city sees.
- Shows open defects, SLA deadlines, and repair status in their ward.
- Builds public trust — citizens can see that their pothole is in the queue and being tracked.

### Edge Inference on Vehicles

- Package the ONNX model and the dedup + severity pipeline as an edge binary that runs on a Jetson Nano or a mobile phone mounted in a municipal bus or garbage truck.
- Detections are queued locally and synced to the server over 4G when connectivity is available.
- Removes dependency on continuous internet connectivity for data collection.

### Video Stream Support

- Process live RTSP streams from CCTV cameras in addition to dashcam recordings.
- Windowed dedup: defects are clustered within a rolling time window rather than a full batch, allowing near-real-time ticketing.

---

## 💬 How to Influence the Roadmap

- **Vote on existing issues** using 👍 reactions — high-vote items get prioritized.
- **Open a feature request** using the [Feature Request template](../.github/ISSUE_TEMPLATE/feature_request.md).
- **Start a discussion** in [GitHub Discussions](https://github.com/VijayabaskarR-06/chakranetra/discussions) for larger architectural questions.
- **Contribute data**: Labelled road images or GPX tracks from new cities are the single highest-leverage contribution you can make right now.

---

## 🏙️ Pilot Deployment Path

```
Phase 1 (Current): Ward-level pilot
  └─ Municipal garbage trucks & buses (already cover every street weekly)
  └─ Manual GPS logging → dashcam video upload batch pipeline

Phase 2: Citizen augmentation
  └─ PWA citizen reporting app
  └─ Weighted confidence (vehicle > citizen, calibrated per source)

Phase 3: City-wide rollout
  └─ Postgres + PostGIS for scale
  └─ Edge inference on all enrolled vehicles
  └─ Integration with municipal grievance systems

Phase 4: Preventive maintenance
  └─ ML deterioration forecasting
  └─ Proactive resurfacing scheduling
  └─ Public transparency map
```
