"""Motion stage: condition raw pose landmarks into analysis-ready signals.

Masks low-confidence landmarks, smooths the sagittal (x, y) trajectories,
differentiates them, and derives a COM proxy. Consumes the pose stage's
landmarks.npy from disk rather than an in-memory array so the stage can be
re-run against existing pose output without paying for MediaPipe again."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from pacelab.core.settings import Settings
from pacelab.core.video_context import VideoContext
from pacelab.motion.kinematics import com
from pacelab.motion.signals import (
    extract_xy,
    mask_low_visibility,
    smooth_positions,
    velocities,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "motion_signals_v1"
COORDINATE_SYSTEM = "mediapipe_normalized_v1"


def run_motion(ctx: VideoContext, settings: Settings) -> Path:
    """Derive smoothed positions, velocities, and COM from persisted landmarks.

    Outputs (under data/processed/<video_id>):
        - motion.npz        positions [T, 33, 2], velocity [T, 33, 2],
                            valid [T, 33] bool, com [T, 2]. All float64 except
                            valid. NaN wherever a frame is untrustworthy.
        - motion_meta.json  schema + sampling rate + settings used.
    """
    out_dir = settings.data.output_dir / ctx.video_id
    landmarks = np.load(out_dir / "landmarks.npy")

    motion = settings.motion
    smoothing = motion.smoothing
    # Sampling rate, not source rate: consecutive rows of landmarks.npy are
    # frame_stride source frames apart, so velocity must divide by that step.
    sample_fps = ctx.fps / ctx.frame_stride

    positions_raw = extract_xy(
        mask_low_visibility(landmarks, motion.visibility_threshold)
    )

    positions, valid = smooth_positions(
        positions_raw,
        window=smoothing.window,
        polyorder=smoothing.polyorder,
        max_gap=motion.max_interpolation_gap,
    )

    velocity, _ = velocities(
        positions_raw,
        window=smoothing.window,
        polyorder=smoothing.polyorder,
        max_gap=motion.max_interpolation_gap,
        fps=sample_fps,
    )
    com_xy = com(positions)

    np.savez_compressed(
        out_dir / "motion.npz",
        positions=positions,
        velocity=velocity,
        valid=valid,
        com=com_xy,
    )

    valid_ratio = float(valid.mean())
    meta = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_system": COORDINATE_SYSTEM,
        "video_id": ctx.video_id,
        "source_fps": ctx.fps,
        "frame_stride": ctx.frame_stride,
        "sample_fps": sample_fps,
        "num_frames": int(positions.shape[0]),
        "landmark_count": int(positions.shape[1]),
        "velocity_units": "normalized_units_per_second",
        "com_definition": "hip_midpoint",
        "params": {
            "visibility_threshold": motion.visibility_threshold,
            "max_interpolation_gap": motion.max_interpolation_gap,
            "smoothing_window": smoothing.window,
            "smoothing_polyorder": smoothing.polyorder,
        },
        "valid_ratio": valid_ratio,
    }
    with open(out_dir / "motion_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Motion done for %s: %d frames, valid_ratio=%.2f%%, sample_fps=%.2f",
        ctx.video_id,
        positions.shape[0],
        valid_ratio * 100,
        sample_fps,
    )

    return out_dir
