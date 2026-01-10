from __future__ import annotations

import logging
from core.video_context import VideoContext
from glob import glob
from pathlib import Path
import json
from typing import List

import cv2
import mediapipe as mp

POSE_CONNECTIONS = [
    # Arms
    (11, 13), (13, 15),     # Left arm
    (12, 14), (14, 16),     # Right arm

    # Upper body
    (11, 12),               # Shoulders
    (23, 24),               # Hips
    (11, 23), (12, 24),     # Torso

    # Legs
    (23, 25), (25, 27),     # Left leg
    (24, 26), (26, 28),     # Right leg

    # Feet
    (27, 29), (27, 31),     # Left foot
    (28, 30), (28, 32),     # Right foot
    (29, 31), (30, 32),
]

logger = logging.getLogger(__name__)
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

def load_frames(video_ctx: VideoContext, cfg: dict) -> List[str]:
    """
    Load and sort frame paths.

    Args:
        video_ctx: VideoContext.
        cfg: config dict

    Returns:
        Sorted list of frame file paths.
    """
    frames_dir = (
        Path(cfg["frames"]["output_dir"]) /
        video_ctx.video_id
    )
    pattern = cfg["frames"]["frame_glob"]
    paths = glob(str(Path(frames_dir) / pattern))
    paths.sort()
    if not paths:
        raise RuntimeError("No frames found. Check frames_dir and pattern.")
    return paths


def compute_timestamp_ms(frame_index: int, video_ctx: VideoContext) -> int:
    """
    Compute monotonic timestamp for VIDEO mode.

    Args:
        frame_index: Frame index (0-based).
        video_ctx: VideoContext dataclass of video.
    
    Returns:
        Timestamp in milliseconds.
    """
    return int(
        video_ctx.start_timestamp_ms +
        (frame_index * video_ctx.frame_stride * 1000.0 / video_ctx.fps)
    )


def mp_image_from_bgr(bgr_image) -> mp.Image:
    """
    Convert OpenCV BGR image to MediaPipe Image.
    
    Args:
        bgr_image: OpenCV image (BGR)
    
    Returns:
        MediaPipe Image in SRGB format.
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def draw_pose_landmarks(
        image,
        pose_landmarks,
        visibility_threshold: float,
        point_radius: int,
        line_thickness: int
) -> None:
    """
    Draw pose landmarks and skeleton using OpenCV

    Args:
        image: BGR image to draw on.
        pose_landmarks: List of landmarks from PoseLandmarker.
        visibility_threshold: Minimum visibility to draw
        point_radius: Radius of landmark points
        line_thickness: Thickness of skeleton lines.
    """
    h, w = image.shape[:2]

    # Draw Points
    for lm in pose_landmarks:
        if getattr(lm, "visibility", 1.0) < visibility_threshold:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), point_radius, (0, 255, 0), -1)
    
    # Draw skeleton
    for a, b in POSE_CONNECTIONS:
        lm1, lm2 = pose_landmarks[a], pose_landmarks[b]

        if (
            getattr(lm1, "visibility", 1.0) < visibility_threshold
            or getattr(lm2, "visibility", 1.0) < visibility_threshold
        ):
            continue
        x1, y1 = int(lm1.x * w), int(lm1.y * h)
        x2, y2 = int(lm2.x * w), int(lm2.y * h)
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), line_thickness)


def run_pose_pipeline(video_ctx: VideoContext, cfg: dict) -> None:
    """
    Run pose estimation pipeline.

    Args:
        config_path: Path to YAML configuration.
    """

    logger.info("Processing Video id: %s", video_ctx.video_id)

    frames = load_frames(
        video_ctx, 
        cfg
    )

    output_dir = (
        Path(cfg["pose"]["output"]["root_dir"]) /
        video_ctx.video_id
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_dir = output_dir / cfg["pose"]["output"]["annotated_dir"]
    if cfg["pose"]["output"]["write_annotated_frames"]:
        annotated_dir.mkdir(exist_ok=True)
    
    json_file = None
    if cfg["pose"]["output"]["write_jsonl"]:
        json_file = open(output_dir / cfg["pose"]["output"]["jsonl_name"], "w", encoding="utf-8")
    
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=cfg["pose"]["model"]["path"]
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=int(cfg["pose"]["model"]["num_poses"]),
        min_pose_detection_confidence=float(cfg["pose"]["model"]["confidences"]["detection"]),
        min_pose_presence_confidence=float(cfg["pose"]["model"]["confidences"]["presence"]),
        min_tracking_confidence=float(cfg["pose"]["model"]["confidences"]["tracking"])
    )

    qa = {
        "video_id": video_ctx.video_id,
        "fps": video_ctx.fps,
        "frame_stride": video_ctx.frame_stride,
        "total_frames_processed": len(frames),
        "frames_with_pose": 0,
        "no_pose_frames": 0,
        "frames_without_pose": [],
        "max_consecutive_no_pose": 0
    }

    no_pose_count = 0
    pose_gap = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        for i, frame_path in enumerate(frames):
            image = cv2.imread(frame_path)
            if image is None:
                logger.warning("Unreadable frames: %s", frame_path)
                continue

            mp_img = mp_image_from_bgr(image)
            ts = compute_timestamp_ms(i, video_ctx)

            result = landmarker.detect_for_video(mp_img, ts)
            has_pose = bool(result.pose_landmarks)

            if not has_pose:
                qa["no_pose_frames"] += 1
                qa["frames_without_pose"].append(i)
                no_pose_count += 1
                pose_gap += 1
                qa["max_consecutive_no_pose"] = max(
                    qa["max_consecutive_no_pose"], pose_gap
                )
            
                if pose_gap >= cfg["pose"]["qa"]["max_pose_gap"]:
                    raise RuntimeError("QA failed: pose gap exceeded")
            
            else:
                qa["frames_with_pose"] += 1
                pose_gap = 0

            if json_file:
                json_file.write(json.dumps({
                    "schema_version": "pose_landmarks_v1",
                    "coordinate_system": "mediapipe_normalized_v1",

                    "video_id": video_ctx.video_id,
                    "fps": video_ctx.fps,
                    "frame_stride": video_ctx.frame_stride,

                    "frame_index": i,
                    "timestamp_ms": ts,

                    "has_pose": has_pose,
                    "num_poses": len(result.pose_landmarks or []),

                    "pose_landmarks": [
                        [
                            {
                                "x": lm.x,
                                "y": lm.y,
                                "z": lm.z,
                                "visibility": getattr(lm, "visibility", None)
                            }
                            for lm in pose 
                        ]
                        for pose in (result.pose_landmarks or [])
                    ],
                }) + "\n")

            if cfg["pose"]["output"]["write_annotated_frames"] and has_pose:
                draw_pose_landmarks(
                    image,
                    result.pose_landmarks[0],
                    cfg["pose"]["overlay"]["visibility_threshold"],
                    cfg["pose"]["overlay"]["point_radius"],
                    cfg["pose"]["overlay"]["line_thickness"]
                )
                frame_name = Path(frame_path).name
                cv2.imwrite(str(annotated_dir / frame_name), image)
    
    if json_file:
        json_file.close()
    
    qa["no_pose_ratio"] = qa["no_pose_frames"] / qa["total_frames_processed"]
    qa["qa_passed"] = (
        qa["no_pose_ratio"] <= cfg["pose"]["qa"]["max_no_pose_ratio"] and
        qa["max_consecutive_no_pose"] <= cfg["pose"]["qa"]["max_pose_gap"]
    )
    with open(output_dir / "qa_report.json", "w") as f:
        json.dump(qa, f, indent=4)

    ratio = no_pose_count / len(frames)
    logger.info("Completed. No-pose ratio: %.2f%%", ratio*100)

    if ratio > cfg["pose"]["qa"]["max_no_pose_ratio"]:
        raise RuntimeError("QA failed: too many frames without pose.")