from pathlib import Path
import cv2
import logging

from core.video_context import VideoContext

logger = logging.getLogger(__name__)

def extract_frames(video_ctx: VideoContext, cfg: dict) -> None:
    cap = cv2.VideoCapture(str(video_ctx.video_path))

    output_dir = (
        Path(cfg["frames"]["output_dir"]) /
        video_ctx.video_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stride = video_ctx.frame_stride
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            out_path = output_dir / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1
        
        frame_idx += 1
    
    cap.release()
    logger.info(f"Frames extracted for video id: {video_ctx.video_id}")