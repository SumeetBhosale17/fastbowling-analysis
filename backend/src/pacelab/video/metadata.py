import json
import logging
from pathlib import Path

from pacelab.core.settings import Settings
from pacelab.core.video_context import VideoContext

logger = logging.getLogger(__name__)


def build_video_metadata(ctx: VideoContext) -> dict:
    return {
        "video_id": ctx.video_id,
        "video_path": str(ctx.video_path),
        "fps": ctx.fps,
        "total_frames": ctx.total_frames,
        "duration_seconds": ctx.duration_sec,
        "frame_extraction": {
            "frame_stride": ctx.frame_stride,
            "frames_processed": ctx.total_frames // ctx.frame_stride,
        },
    }


def write_metadata(ctx: VideoContext, settings: Settings) -> Path:
    metadata = build_video_metadata(ctx)
    output_dir = settings.data.output_dir / ctx.video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metadata.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info("Wrote metadata: %s", output_path)
    return output_path
