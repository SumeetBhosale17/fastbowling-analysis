import os
import glob
import yaml
import cv2
import logging

from utils import setup_logging
from metadata import save_metadata
from extract_frames import extract_frames

def main():
    setup_logging()

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    video_dir = config["data"]["video_dir"]
    frame_dir = config["data"]["frame_output_dir"]
    nth = config["frame_extraction"]["nth_frame"]
    metadata_path = config["metadata"]["output_path"]
    formats = config.get("formats", ["mp4", "avi", "mov"])

    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # Collect videos
    video_files = []
    for fmt in formats:
        video_files.extend(glob.glob(os.path.join(video_dir, f"*.{fmt}")))

    logging.info(f"Found {len(video_files)} videos")

    for video in video_files:
        logging.info(f"Processing: {video}")
        cap = cv2.VideoCapture(video)
        save_metadata(cap, video, metadata_path, nth)
        extract_frames(cap, nth, video, frame_dir)
        cap.release()

if __name__ == "__main__":
    main()
