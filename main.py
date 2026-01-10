from pathlib import Path
import yaml
import logging

from core.context_factory import build_video_context
from run_pose import run_pose_pipeline
from utils import setup_logging
from metadata import append_metadata
from extract_frames import extract_frames

logger = logging.getLogger(__name__)

def main():
    setup_logging()

    # Load config
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    video_dir = Path(cfg["data"]["video_dir"])
    formats = cfg["data"]["formats"]

    video_paths = []
    for fmt in formats:
        video_paths.extend(video_dir.glob(f"*.{fmt}"))
    
    logger.info("Found %d videos", len(video_paths))

    for video_path in video_paths:
        logging.info("Processing %s", video_path.name)

        video_ctx = build_video_context(video_path=video_path, cfg=cfg)

        append_metadata(video_ctx, cfg)
        extract_frames(video_ctx, cfg)
        run_pose_pipeline(video_ctx, cfg)


if __name__ == '__main__':
    main()