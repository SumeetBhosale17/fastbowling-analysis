import json
import os
import cv2
import logging

def save_metadata(cap, video_name, metadata_path, nth):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    metadata = {
        "video_name": os.path.basename(video_name),
        "fps": fps,
        "total_frames": frame_count,
        "duration_seconds": duration,
        "frame_extraction": {
            "nth_frame": nth,
            "frames_extracted": frame_count // nth
        }
    }

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    if not any(m["video_name"] == metadata["video_name"] for m in data):
        data.append(metadata)

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(data, f, indent=4)

    logging.info(f"Metadata saved for {metadata['video_name']}")
