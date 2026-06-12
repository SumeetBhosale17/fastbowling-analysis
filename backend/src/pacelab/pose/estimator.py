from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from pacelab.core.settings import Settings
from pacelab.core.video_context import VideoContext

logger = logging.getLogger(__name__)

LANDMARK_COUNT = 33
LANDMARK_DIMS = ("x", "y", "z", "visibility")
SCHEMA_VERSION = "pose_landmarks_v2"
COORDINATE_SYSTEM = "mediapipe_normalized_v1"


def _timestamp_ms(sampled_idx: int, ctx: VideoContext) -> int:
    return int(
        ctx.start_timestamp_ms + (sampled_idx * ctx.frame_stride * 1000.0 / ctx.fps)
    )


def _build_landmarker(settings: Settings):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    pose_model = settings.pose.model
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model.path)),
        running_mode=RunningMode.VIDEO,
        num_poses=pose_model.num_poses,
        min_pose_detection_confidence=pose_model.confidences.detection,
        min_pose_presence_confidence=pose_model.confidences.presence,
        min_tracking_confidence=pose_model.confidences.tracking,
    )
    return PoseLandmarker.create_from_options(options)


def run_pose(
    ctx: VideoContext,
    frames: Iterable[tuple[int, np.ndarray]],
    settings: Settings,
) -> Path:
    """Run MediaPipe pose estimation on a stream of (source_frame_index, bgr) pairs.

    Outputs (under data/processed/<video_id>/):
      - landmarks.npy        float32 array [T, 33, 4] with (x, y, z, visibility).
                             Missing-pose frames are filled with NaN.
      - landmarks_meta.json  schema + per-frame timestamps_ms, has_pose mask,
                             source_frame_indices.
      - qa_report.json       coverage statistics + pass/fail.
    """
    out_dir = settings.data.output_dir / ctx.video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    landmarks_buf: list[np.ndarray] = []
    has_pose_mask: list[bool] = []
    timestamps_ms: list[int] = []
    source_frame_indices: list[int] = []
    no_pose_indices: list[int] = []

    max_gap = settings.pose.qa.max_pose_gap
    current_gap = 0
    max_consecutive_no_pose = 0

    with _build_landmarker(settings) as landmarker:
        for sampled_idx, (source_idx, image) in enumerate(frames):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = _timestamp_ms(sampled_idx, ctx)

            result = landmarker.detect_for_video(mp_img, ts)
            has_pose = bool(result.pose_landmarks)

            if has_pose:
                lm = result.pose_landmarks[0]
                arr = np.array(
                    [[p.x, p.y, p.z, getattr(p, "visibility", 0.0)] for p in lm],
                    dtype=np.float32,
                )
                current_gap = 0
            else:
                arr = np.full(
                    (LANDMARK_COUNT, len(LANDMARK_DIMS)),
                    np.nan,
                    dtype=np.float32,
                )
                no_pose_indices.append(sampled_idx)
                current_gap += 1
                max_consecutive_no_pose = max(max_consecutive_no_pose, current_gap)

            landmarks_buf.append(arr)
            has_pose_mask.append(has_pose)
            timestamps_ms.append(ts)
            source_frame_indices.append(source_idx)

    if not landmarks_buf:
        raise RuntimeError("No frames received from video reader")

    landmarks = np.stack(landmarks_buf, axis=0)
    np.save(out_dir / "landmarks.npy", landmarks)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_system": COORDINATE_SYSTEM,
        "video_id": ctx.video_id,
        "fps": ctx.fps,
        "frame_stride": ctx.frame_stride,
        "landmark_count": LANDMARK_COUNT,
        "landmark_dims": list(LANDMARK_DIMS),
        "num_frames": int(landmarks.shape[0]),
        "timestamps_ms": timestamps_ms,
        "has_pose": has_pose_mask,
        "source_frame_indices": source_frame_indices,
    }
    with open(out_dir / "landmarks_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    total = len(landmarks_buf)
    no_pose_ratio = len(no_pose_indices) / total
    qa = {
        "video_id": ctx.video_id,
        "fps": ctx.fps,
        "frame_stride": ctx.frame_stride,
        "total_frames_processed": total,
        "frames_with_pose": total - len(no_pose_indices),
        "no_pose_frames": len(no_pose_indices),
        "no_pose_indices": no_pose_indices,
        "max_consecutive_no_pose": max_consecutive_no_pose,
        "no_pose_ratio": no_pose_ratio,
        "qa_passed": (
            no_pose_ratio <= settings.pose.qa.max_no_pose_ratio
            and max_consecutive_no_pose <= max_gap
        ),
    }
    with open(out_dir / "qa_report.json", "w", encoding="utf-8") as f:
        json.dump(qa, f, indent=2)

    logger.info(
        "Pose done for %s: %d frames, no_pose_ratio=%.2f%%, max_gap=%d",
        ctx.video_id,
        total,
        no_pose_ratio * 100,
        max_consecutive_no_pose,
    )

    if not qa["qa_passed"]:
        logger.warning(
            "QA failed for %s: no_pose_ratio=%.2f%% (max %.0f%%), "
            "max_consecutive_no_pose=%d (max %d). Outputs written for inspection.",
            ctx.video_id,
            no_pose_ratio * 100,
            settings.pose.qa.max_no_pose_ratio * 100,
            max_consecutive_no_pose,
            max_gap,
        )

    return out_dir
