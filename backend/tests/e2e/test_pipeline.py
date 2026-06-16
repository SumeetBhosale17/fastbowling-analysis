"""End-to-end smoke test for the pose pipeline.

Runs `process_video` against a synthetic clip and asserts contracts on the
four output files. Does not assert pose values - synthetic frames don't
contain human, so `qa_passed` may be False. We're verifying that the
pipeline runs end-to-end and writes the expected file shapes, not that
pose detection finds anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pacelab.core.settings import Settings, load_settings
from pacelab.pipeline import process_video


def _settings_for_test(output_dir: Path) -> Settings:
    """Load real config and redirect only `data.output_dir` to a tmp path."""
    settings = load_settings()
    return settings.model_copy(
        update={
            "data": settings.data.model_copy(update={"output_dir": output_dir}),
        }
    )


def test_pipeline_smoke(synthetic_video: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    settings = _settings_for_test(output_dir=output_dir)
    process_video(synthetic_video, settings)

    # One output dir per video, named by video_id.
    produced = list(output_dir.iterdir())
    assert len(produced) == 1
    out = produced[0]

    # landmarks.npy: float32 array of shape (T, 33, 4) for some T >= 1.
    landmarks = np.load(out / "landmarks.npy")
    assert landmarks.dtype == np.float32
    assert landmarks.ndim == 3
    assert landmarks.shape[1:] == (33, 4)
    n_frames = landmarks.shape[0]
    assert n_frames >= 1

    # landmarks_meta.json: scheme + per-frame metadata aligned with landmarks.
    meta = json.loads((out / "landmarks_meta.json").read_text())
    assert meta["schema_version"] == "pose_landmarks_v2"
    assert meta["coordinate_system"] == "mediapipe_normalized_v1"
    assert meta["num_frames"] == n_frames
    assert len(meta["has_pose"]) == n_frames
    assert len(meta["timestamps_ms"]) == n_frames
    assert len(meta["source_frame_indices"]) == n_frames

    # qa_report.json required keys present (values depend on pose detection).
    qa = json.loads((out / "qa_report.json").read_text())
    assert qa["total_frames_processed"] == n_frames
    assert "qa_passed" in qa
    assert isinstance(qa["qa_passed"], bool)
    assert "no_pose_ratio" in qa
    assert "max_consecutive_no_pose" in qa

    # metadata.json: video-level info.
    vmeta = json.loads((out / "metadata.json").read_text())
    assert vmeta["video_id"]
    assert vmeta["fps"] > 0
    assert vmeta["total_frames"] >= 1
