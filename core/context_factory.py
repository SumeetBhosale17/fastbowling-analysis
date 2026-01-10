import hashlib
import cv2
from pathlib import Path
from core.video_context import VideoContext

def generate_video_id(video_path: Path) -> str:
    return hashlib.sha1(str(video_path).encode()).hexdigest()[:12]

def build_video_context(video_path: Path, cfg: dict) -> VideoContext:
    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS for {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    cap.release()

    return VideoContext(
        video_id=generate_video_id(video_path=video_path),
        video_path=video_path,
        fps=fps,
        total_frames=total_frames,
        duration_sec=duration,
        frame_stride=cfg["frames"]["nth_frame"],
        start_timestamp_ms=cfg["video"]["start_timestamp_ms"]
    )