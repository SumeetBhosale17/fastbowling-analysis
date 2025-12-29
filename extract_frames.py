import cv2
import os
import logging

def extract_frames(cap, nth, video_name, frame_dir):
    frame_idx = 0
    output_dir = os.path.join(frame_dir, os.path.splitext(os.path.basename(video_name))[0])
    os.makedirs(output_dir, exist_ok=True)

    last_saved = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % nth == 0:
            filename = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
            success = cv2.imwrite(filename, frame)
            if success:
                last_saved = frame_idx
            else:
                logging.error(f"Failed to save frame {frame_idx} for video {video_name} at {filename}")
            if frame_idx % (nth * 50) == 0:
                logging.info(f"Extracted {frame_idx} frames...")
        frame_idx += 1
    
    if last_saved is None:
        logging.warning(f"No frames extraced for {os.path.basename(video_name)}")
    else:
        logging.info(f"Last frame extracted for {os.path.basename(video_name)}: frame_{last_saved}")
        logging.info(f"Frames extracted from {os.path.basename(video_name)}")
