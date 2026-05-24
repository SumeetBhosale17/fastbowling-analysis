from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np


def iter_video_frames(
    video_path: Path,
    frame_stride: int = 1,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (source_frame_index, bgr_frame) for every `frame_stride`-th frame.

    `source_frame_index` is the index in the ORIGINAL video, not the sampled
    stream. Frames are decoded on demand and never written to disk.
    """
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_stride == 0:
                yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()
