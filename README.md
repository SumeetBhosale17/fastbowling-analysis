# Fast Bowling Analysis

A computer vision-based project to analyze fast bowling biomechanics from cricket fast bowling videos.
This project is structured in multiple phases, starting from raw video ingestion to pose-based and temporal biomechanical analysis.

---

## Phase 0 - Video Ingestion & Frame Extraction

### Objective
Phase 0 prepares raw bowling videos for downstream analysis by:
- Discovering videos in batch
- Extracting container-level metadata
- Sampling frames at a configurable rate
- Persisting metadata for reproduciblility

This phase **does not perform any pose estimation or biomechanics**.
Its sole purpose is to create a clean, reproducible data foundation.

---

## What Phase 0 does

For each input video:
1. Reads video metadata (FPS, frame count, duration)
2. Saves metadata to structured JSON file
3. Extracts every *N*th frame (configurable)
4. Stores frames in a deterministic directory structure

---

```text
fast-bowling-analysis/
│
├── main.py                 # Pipeline entry point
├── config.yaml             # All configurable parameters
├── extract_frames.py       # Frame extraction logic
├── metadata.py             # Metadata extraction & persistence
├── utils.py                # Shared utilities (logging, etc.)
│
├── scripts/
│   └── run_pipeline.sh     # Reproducible execution script
│
├── videos/                 # Input videos (ignored by git)
├── frames/                 # Extracted frames (ignored by git)
│   └── <video_name>/
│       └── frame_<index>.jpg
│
├── metadata/
│   └── metadata.json       # Generated metadata (ignored by git)
│
└── README.md
```

--- 

## Configuration (`config.yaml`)

All pipeline behavior is controlled via `config.yaml`. \
The code does not contain hardcoded paths or parameters.

Key configuration sections:
- `data.video_dir` - directory containing input videos
- `data.frame_output_dir` - where the extracted frames are stored
- `frame_extraction.nth` - sampling rate
- `metadata.output_path` - metadata storage location
- `formats` - supported video formats

This design enables reproducibility and easy experimentation without modifying source code.

---

## How to Run Phase 0

From the project root:

```bash
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh
```

This pipeline will:
- Discover all supported videos in `videos/`
- Extract metadata and frames
- Save outputs to `frames/` and `metadata/`

---

## Outputs

### Frames

Extracted frames are stored as:

```text
frames/<video_name>/frame_<frame_index>.jpg
```

### Metadata

Metadata is stored in JSON format:

```json
{
    "video_name": "video.mp4",
    "fps": 29,
    "total_frames": 684,
    "duration_seconds": 23.586206896551722,
    "frame_extraction": {
        "nth_frame": 10,
        "frames_extracted": 68
    }
}
```

This metadata is required for:
- Temporal alignment
- Velocity Calculations
- Reproducibility across experiments

---

### Git Tracking Policy

The following directories are intentionally excluded from version control:
- `videos/`
- `frames/`
- `metadata/*.json`

Only source code and configuration are tracked. \
This keeps the repository lightweight and reproducible.

---