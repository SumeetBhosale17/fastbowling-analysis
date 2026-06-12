from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoContext:
    video_id: str
    video_path: Path

    fps: float
    total_frames: int
    duration_sec: float

    frame_stride: int
    start_timestamp_ms: int
