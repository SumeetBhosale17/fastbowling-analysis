from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import yaml

from src.core.context_factory import build_video_context
from src.pose.estimator import run_pose
from src.utils.logging import setup_logging
from src.video.metadata import write_metadata
from src.video.reader import iter_video_frames

logger = logging.getLogger(__name__)


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_videos(cfg: dict) -> List[Path]:
    video_dir = Path(cfg["data"]["video_dir"])
    formats = cfg["data"]["formats"]
    paths: List[Path] = []
    for fmt in formats:
        paths.extend(video_dir.glob(f"*.{fmt}"))
    return sorted(paths)


def process_video(video_path: Path, cfg: dict) -> Path:
    logger.info("Processing %s", video_path.name)
    ctx = build_video_context(video_path, cfg)
    write_metadata(ctx, cfg)
    frames = iter_video_frames(ctx.video_path, ctx.frame_stride)
    return run_pose(ctx, frames, cfg)


def main() -> None:
    setup_logging()
    cfg = load_config()
    videos = discover_videos(cfg)
    logger.info("Found %d video(s)", len(videos))
    for video_path in videos:
        process_video(video_path, cfg)


if __name__ == "__main__":
    main()
