import json
import logging
import os

from core.video_context import VideoContext

logger = logging.getLogger(__name__)

def build_video_metadata(ctx: VideoContext) -> dict:
    return {
        "video_id": ctx.video_id,
        "video_path": str(ctx.video_path),
        "fps": ctx.fps,
        "total_frames": ctx.total_frames,
        "duration_seconds": ctx.duration_sec,
        "frame_extration": {
            "nth_frame": ctx.frame_stride,
            "frames_extracted": ctx.total_frames // ctx.frame_stride
        }
    }

def append_metadata(ctx: VideoContext, cfg: dict) -> None:
    metadata = build_video_metadata(ctx)
    output_path = cfg["metadata"]["output_path"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    
    if not any(m["video_id"] == metadata["video_id"] for m in data):
        data.append(metadata)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    logger.info(f"Metadata saved for video id: {ctx.video_id}, path: {str(ctx.video_path)}")