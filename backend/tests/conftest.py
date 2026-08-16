"""pytest fixtures shared across the test suite."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

_LABEL_DIR = Path(__file__).parent / "fixtures" / "events"
_VIDEO_DIR = Path("data/raw_videos")
_LABEL_FILES = sorted(_LABEL_DIR.glob("*.json"))


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


@dataclass(frozen=True)
class LabelledClip:
    video_path: Path
    labels: dict


def _sha1_of_file(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(
    params=_LABEL_FILES or [None],
    ids=lambda p: p.stem if p is not None else "no-labels",
)
def labelled_clip(request: pytest.FixtureRequest) -> LabelledClip:
    """A hand-labelled clip, or a skip when the footage isn't present.

    Clips are gitignored - they are not ours to redistribute - so any test
    using this fixture is local-only and CI skips it. That is a real coverage
    gap, not a design win; `unit/test_event_labels.py` covers what CI can.
    """
    if request.param is None:
        pytest.skip(f"no label fixtures in {_LABEL_DIR}")

    labels = json.loads(Path(request.param).read_text(encoding="utf-8"))
    video_path = _VIDEO_DIR / labels["video_filename"]
    if not video_path.exists():
        pytest.skip(f"{video_path} not available (labelled clips are local-only)")

    # Fail, not skip: silently-changed footage would invalidate the ground
    # truth while the suite still reported green.
    if _sha1_of_file(video_path) != labels["video_sha1"]:
        pytest.fail(
            f"{video_path} content does not match {Path(request.param).name}. "
            "The clip was re-encoded or replaced; re-label it."
        )

    return LabelledClip(video_path=video_path, labels=labels)
