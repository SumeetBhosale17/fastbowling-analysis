from __future__ import annotations

import json
import os
import logging
from utils import setup_logging
from glob import glob
from pathlib import Path
from typing import Dict, Any, List

import cv2
import mediapipe as mp
import yaml

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

def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML configutation file.

    Args:
        path: Path to YAML config.
    
    Returns:
        Parsed configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

def load_frames(frames_dir: str, pattern: str) -> List[str]:
    """
    Load and sort frame paths.

    Args:
        frames_dir: Directory containing extracted frames.
        pattern: Filename glob pattern.

    Returns:
        Sorted list of frame file paths.
    """
    paths = glob(str(Path(frames_dir) / pattern))
    paths.sort()
    if not paths:
        raise RuntimeError("No frames found. Check frames_dir and pattern.")
    return paths


def compute_timestamp_ms(index: int, fps: float, start_ms: int) -> int:
    """
    Compute monotonic timestamp for VIDEO mode.

    Args: 
        index: Frame index (0-based)
        fps: Frames per second.
        start_ms: Initial timestamp offset.

    Returns:
        Timestamp in milliseconds.
    """
    return int(start_ms + (index * 1000.0 / fps))


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


def run_pose_pipeline(config_path: str) -> None:
    """
    Run pose estimation pipeline.

    Args:
        config_path: Path to YAML configuration.
    """
    setup_logging()
    log = logging.getLogger("Pose")

    if not Path(config_path).exists():
        log.error(f"Config path {config_path} does not exist")
        raise FileNotFoundError(f"Config path {config_path} does not exist")

    cfg = load_config(config_path)

    videos = [
        f for f in os.listdir(cfg["io"]["frames_dir"])
        if os.path.isdir(os.path.join(cfg["io"]["frames_dir"], f))
    ]

    for video in videos:
        log.info(f"Processing {video}.")

        frames = load_frames(Path(cfg["io"]["frames_dir"])/ video, cfg["io"]["frame_glob"])

        output_dir = Path(cfg["io"]["output_dir"]) / os.path.basename(video)

        if output_dir.exists():
            log.info(f"Output directory {os.path.basename(video)} already exists")
            continue

        output_dir.mkdir(parents=True, exist_ok=False)

        annotated_dir = output_dir / cfg["io"]["annotated_dir"]
        if cfg["io"]["write_annotated_frames"]:
            annotated_dir.mkdir(exist_ok=False)

        json_file = None
        if cfg["io"]["write_json"]:
            json_file = open(output_dir /  cfg["io"]["json_path"], "w", encoding="utf-8")
        
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        if os.path.exists(config["metadata"]["output_path"]):
            with open(config["metadata"]["output_path"], 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        fps = next(
            (float(m["fps"]) for m in data if m["video_name"] == os.path.basename(video)),
            30.0
        )


        start_ms = int(cfg["video"]["start_timestamp_ms"])

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=cfg["model"]["task_model_path"]
            ),
            running_mode=RunningMode.VIDEO,
            num_poses = int(cfg["model"]["num_poses"]),
            min_pose_detection_confidence=float(cfg["model"]["min_pose_detection_confidence"]),
            min_pose_presence_confidence=float(cfg["model"]["min_pose_presence_confidence"]),
            min_tracking_confidence=float(cfg["model"]["min_tracking_confidence"]),
        )

        no_pose_count = 0

        with PoseLandmarker.create_from_options(options) as landmarker:
            for i, frame_path in enumerate(frames):
                image = cv2.imread(frame_path)
                if image is None:
                    log.warning("Unreadable frames: %s", frame_path)
                    continue

                mp_img = mp_image_from_bgr(image)
                ts = compute_timestamp_ms(i, fps, start_ms)

                result = landmarker.detect_for_video(mp_img, ts)
                has_pose = bool(result.pose_landmarks)

                if not has_pose:
                    no_pose_count += 1
                
                if json_file:
                    json_file.write(json.dumps({
                        "frame_index": i,
                        "timestamp_ms": ts,
                        "has_pose": has_pose,
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
                
                if cfg["io"]["write_annotated_frames"] and has_pose:
                    draw_pose_landmarks(
                        image,
                        result.pose_landmarks[0],
                        cfg["overlay"]["visibility_threshold"],
                        cfg["overlay"]["point_radius"],
                        cfg["overlay"]["line_thickness"]
                    )
                    cv2.imwrite(str(annotated_dir / Path(frame_path).name), image)

        if json_file:
            json_file.close()

        ratio = no_pose_count / len(frames)
        log.info("Completed. No-pose ratio: %.2f%%", ratio * 100)

        if ratio > cfg["qa"]["max_no_pose_ratio"]:
            raise RuntimeError("QA failed: too many frames without pose.")

def main() -> None:
    run_pose_pipeline('configs/pose.yaml')

# =============================
# CLI
# =============================

# def main() -> None:
#     parser = argparse.ArgumentParser(description="Enterprise Pose Estimation")
#     parser.add_argument("--config", required=True)
#     args = parser.parse_args()
#     run_pose_pipeline(args.config)


if __name__ == '__main__':
    main()