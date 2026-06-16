"""pytest fixtures shared across the test suite."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Generate a 30-frame, 480x270 synthetic video in tmp_path.

    Frames are solid colors that vary across the clips, so the file is real
    H.264-encoded mp4 that the pipeline can decode end-to-end. No human in
    frame, so pose detection will mostly fail - the smoke test asserts
    contracts (file existence, array shapes, JSON keys), not pose values.
    """

    video_path = tmp_path / "synthetic.mp4"
    width, height, fps, n_frames = 480, 270, 30, 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    try:
        for i in range(n_frames):
            shade = (i * 8) % 256
            frame = np.full((height, width, 3), shade, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return video_path
