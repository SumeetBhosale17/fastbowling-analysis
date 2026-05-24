# Fast Bowling Analysis

Computer-vision pipeline for fast-bowling biomechanics from cricket footage. Long-term goal: a web platform where users upload a delivery and get back per-frame pose, segmented bowling phases, and biomechanical metrics.

This iteration (**v0.2**) is the foundation only: a clean, runnable pipeline that turns a video into a compact landmarks array on disk. No event detection, no metrics, no UI yet.

---

## What v0.2 does

For each video in `data/raw_videos/`:

1. Builds a `VideoContext` (immutable: id, path, fps, frame count, duration, stride, start timestamp).
2. Writes `metadata.json` to the per-video output directory.
3. **Streams** frames from the video file — no on-disk frame cache — and runs MediaPipe `PoseLandmarker` in `VIDEO` mode with monotonic timestamps.
4. Writes a compact NumPy landmarks array, a sidecar JSON for per-frame metadata, and a QA report.

Per-video outputs land in `data/processed/<video_id>/`:

```
data/processed/<video_id>/
├── metadata.json          # video-level: fps, total_frames, duration, ...
├── landmarks.npy          # float32 array, shape [T, 33, 4]  (x, y, z, visibility)
├── landmarks_meta.json    # per-frame timestamps_ms, has_pose mask, source_frame_indices
└── qa_report.json         # coverage stats + qa_passed flag
```

`video_id` is `sha1(str(video_path))[:12]` — moving the source file changes the id.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Place the MediaPipe pose model at `models/pose_landmarker_full.task` (already in repo). Drop any `.mp4` / `.avi` / `.mov` files into `data/raw_videos/`.

## Run

From the repo root:

```bash
python main.py
```

That's it. `main.py` is a thin wrapper around `src.pipeline.main`.

---

## Configuration

Everything is in `configs/config.yaml`. The code reads no hardcoded paths or thresholds. Notable knobs:

- `video.frame_stride` — default `1` (process every frame). Increase only if pose inference is too slow; raising this risks missing the release frame.
- `pose.model.confidences.*` — detection / presence / tracking thresholds passed straight to MediaPipe.
- `pose.qa.max_no_pose_ratio` / `max_pose_gap` — the pipeline raises `RuntimeError` if either is exceeded.

---

## Project layout

```
fast-bowling-analysis/
├── main.py                       # entrypoint (thin wrapper)
├── configs/
│   └── config.yaml               # single source of truth for parameters
├── src/
│   ├── pipeline.py               # orchestration
│   ├── core/
│   │   ├── video_context.py      # VideoContext dataclass (frozen)
│   │   └── context_factory.py    # build_video_context()
│   ├── video/
│   │   ├── reader.py             # streaming frame iterator
│   │   └── metadata.py           # write_metadata()
│   ├── pose/
│   │   └── estimator.py          # MediaPipe pose, writes landmarks.npy
│   └── utils/
│       └── logging.py
├── data/
│   ├── raw_videos/               # input videos (gitignored)
│   ├── processed/<video_id>/     # outputs (gitignored)
│   └── schemas/                  # JSON schemas
├── docs/
│   ├── coordinate_system.md
│   └── data_schema.md
├── models/
│   └── pose_landmarker_full.task
├── scripts/
│   └── run_pipeline.sh           # `python main.py`
└── README.md
```

---

## Roadmap

| Phase | Goal | Status |
|------|------|--------|
| Phase 0 | Video ingest + metadata | done |
| Phase 1 | Pose estimation + landmarks.npy | done (v0.2) |
| Phase 2 | Motion signals + filters (joint angles, COM, velocities) | planned |
| Phase 3 | Event detection (BFC, FFC, release, jump initiation) | planned |
| Phase 4 | Per-delivery metrics (trunk angle, hip-shoulder separation, …) | planned |
| Phase 5 | Visualization (annotated video + signal charts) | planned |
| Phase 6 | FastAPI service + worker queue | planned |
| Phase 7 | Frontend (upload + results) | planned |

---

## Known limitations

- One labeled video so far. No ground truth for events or metrics yet — event detection (Phase 3) is blocked on this.
- `video_id` is path-based; renaming or moving a video creates a new id.
- MediaPipe pose has known difficulty with motion blur, self-occlusion at release, and partial-body frames. Treat low `visibility` landmarks as missing.
- Annotated-video output and visualization snapshots are intentionally deferred to Phase 5.
