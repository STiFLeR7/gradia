# gradia v2.0.0: Learning Timeline

*"From observing training to understanding learning."*

We are excited to announce **Gradia v2.0.0**, a major release that transforms Gradia from a metrics dashboard into a **learning behavior explorer**.

## 🚀 Flagship Feature: Learning Timeline

The Learning Timeline is a real-time, sample-centric view of how your models learn over time. It answers questions that aggregate metrics cannot:

- 🎯 **When did this sample become correctly classified?**
- 🔄 **Which samples keep flipping predictions?**
- 📈 **Is the model memorizing or stabilizing?**
- ⚠️ **Which data points drive learning instability?**

### Timeline UI Blocks

| Block | Purpose |
|-------|---------|
| **Timeline Overview** | High-level view of learning stability across all tracked samples |
| **Sample Inspector** | Deep-dive into individual sample trajectories with confidence curves |
| **Instability Panel** | Top flipping samples, late learners, and persistent errors |
| **Training Context** | Current epoch, status, and tracking metadata |

### Sample Classification

Gradia automatically classifies tracked samples:

- 🟢 **Stable Correct** — Consistently correct predictions
- 🟡 **Late Learner** — Became correct after epoch N
- 🟠 **Unstable** — Predictions flip frequently
- 🔴 **Persistent Error** — Never correctly classified

## ✨ New Features

### Events Module (`gradia.events`)
- **LearningEvent** — Atomic event model for sample-level tracking
- **SampleState** — Tracks prediction history, flip counts, stability classification
- **EpochSummary** — Aggregated statistics per epoch
- **SampleTracker** — Boundary-aware sample selection (default: 100 samples)
- **TimelineLogger** — Structured event persistence to `timeline_events.jsonl`

### Migration System
- **SchemaMigrator** — Automatic v1.x to v2.0 config migration
- **Backward Compatible** — All v1.x runs load without modification
- **Schema Versioning** — Configs now include `schema_version: "2.0"`

### Enhanced Trainer
- **Timeline Integration** — Automatic sample tracking during training
- **`evaluate_full()`** — Extended evaluation with timeline insights
- **`get_sample_timeline()`** — Retrieve individual sample history
- **`get_timeline_insights()`** — Summary of learning behavior

### New API Endpoints
- `GET /timeline` — Learning Timeline page
- `GET /api/timeline/events` — Fetch timeline events
- `GET /api/timeline/summaries` — Epoch summaries
- `GET /api/timeline/sample/{id}` — Individual sample history
- `GET /api/timeline/insights` — Aggregated insights

## 🛠️ Technical Improvements

- **FastAPI Lifespan** — Migrated from deprecated `on_event` to lifespan context manager
- **TemplateResponse Update** — Updated to new Starlette parameter order
- **Zero Warnings** — All deprecation warnings resolved
- **Comprehensive Tests** — 29 tests covering core, events, migration, API, and integration

## 🔄 Migration from v1.x

Migration is automatic! When you run `gradia run .` on a v1.x project:

```bash
gradia run .
# Output: Config migrated: Added timeline config, Set schema_version to 2.0
```

## 📦 Installation

```bash
pip install gradia --upgrade
```

## 💻 Quick Start

```bash
# Auto-detect datasets and start with Learning Timeline
gradia run .

# Access the dashboard
# http://localhost:8000          (Metrics)
# http://localhost:8000/timeline (Learning Timeline)
```

## 📊 Configuration

```yaml
# New v2.0 config options
timeline:
  enabled: true
  max_samples: 100
  sampling_strategy: "boundary"
```

## 🐛 Bug Fixes

- Fixed race conditions in UI updates
- Fixed `time` module shadowing in trainer engine
- Fixed inconsistent metric timestamps
- Improved Windows/Linux path handling

## 🤝 Contributors

- @STiFLeR7 (v2.0.0 Release)

---

## Previous Releases

### v1.0.0: Next-Gen Local MLOps

- Zero-Config Dashboard with auto-detection
- Real-Time Telemetry (Loss, Accuracy, CPU, RAM)
- Smart Model Suggestions
- Validation Suite with Confusion Matrix
- PDF/JSON Report Export

---
*Happy Training!* 🚀
