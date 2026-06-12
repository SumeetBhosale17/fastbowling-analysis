from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from pacelab.core.context_factory import build_video_context
from pacelab.core.settings import Settings, load_settings
from pacelab.pose.estimator import run_pose
from pacelab.utils.logging import setup_logging
from pacelab.video.metadata import write_metadata
from pacelab.video.reader import iter_video_frames

logger = logging.getLogger(__name__)


def discover_videos(settings: Settings) -> List[Path]:
    paths: List[Path] = []
    for fmt in settings.data.formats:
        paths.extend(settings.data.video_dir.glob(f"*.{fmt}"))
    return sorted(paths)


def process_video(video_path: Path, settings: Settings) -> Path:
    logger.info("Processing %s", video_path.name)
    ctx = build_video_context(video_path, settings)
    write_metadata(ctx, settings)
    frames = iter_video_frames(ctx.video_path, ctx.frame_stride)
    return run_pose(ctx, frames, settings)


def main() -> None:
    setup_logging()
    settings = load_settings()
    videos = discover_videos(settings)
    logger.info("Found %d video(s)", len(videos))
    for video_path in videos:
        process_video(video_path, settings)


if __name__ == "__main__":
    main()
