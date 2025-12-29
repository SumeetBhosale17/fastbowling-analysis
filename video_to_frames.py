import cv2
import os

# Loading the Video

video_path = 'videos/jofra.mp4'
os.makedirs('frames', exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError('Error: Cannot open video file')

# Querying the metadata

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration = frame_count / fps if fps > 0 else 0

print(f"FPS: {fps}")
print(f"Total Frames: {frame_count}")
print(f"Duration (s): {duration}")

# Extracting Frames

nth = 10
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % nth == 0:
        filename = f"frames/frame__{frame_idx}.jpg"
        cv2.imwrite(filename, frame)
        # cv2.imshow('Frame', frame)

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()