# Per-video output schemas

Outputs land in `data/processed/<video_id>/`.

## `landmarks.npy`

NumPy array, saved with `np.save`.

- **dtype**: `float32`
- **shape**: `[T, 33, 4]`
  - `T` = number of frames processed (one per sampled frame)
  - `33` = MediaPipe pose landmarks
  - `4` = `(x, y, z, visibility)`
- Frames where no pose was detected are filled with `NaN`.
- Coordinate system: `mediapipe_normalized_v1` (see `coordinate_system.md`).

## `landmarks_meta.json`

Sidecar for `landmarks.npy`. One object per video.

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | Currently `pose_landmarks_v2`. |
| `coordinate_system` | string | `mediapipe_normalized_v1`. |
| `video_id` | string | 12-char SHA-1 of the video path. |
| `fps` | float | Source video FPS. |
| `frame_stride` | int | Stride used when sampling the source video. |
| `landmark_count` | int | Always 33 for MediaPipe pose. |
| `landmark_dims` | string[] | `["x", "y", "z", "visibility"]`. |
| `num_frames` | int | `T`, matches `landmarks.npy` axis 0. |
| `timestamps_ms` | int[] | Monotonic ms timestamps for `RunningMode.VIDEO`. Length `T`. |
| `has_pose` | bool[] | `True` iff pose was detected for that frame. Length `T`. |
| `source_frame_indices` | int[] | Index of each sampled frame in the original video. Length `T`. |

## `motion.npz`

NumPy archive, saved with `np.savez_compressed`. Written by the motion stage
from `landmarks.npy`. Sagittal plane only — the `z` channel is dropped, since
v1 is side-on.

| Array | dtype | shape | Notes |
|---|---|---|---|
| `positions` | `float64` | `[T, 33, 2]` | Savitzky-Golay smoothed `(x, y)`. |
| `velocity` | `float64` | `[T, 33, 2]` | SG first derivative, normalized units/second. |
| `valid` | `bool` | `[T, 33]` | Per-frame, per-landmark trustworthiness. Describes both `positions` and `velocity`. |
| `com` | `float64` | `[T, 2]` | Hip-midpoint centre-of-mass proxy. |

`positions`, `velocity`, and `com` are `NaN` wherever `valid` is `False`.
A frame is invalid when the landmark's `visibility` was below
`motion.visibility_threshold`, when pose was missing and the gap exceeded
`motion.max_interpolation_gap`, or when the missing run was leading/trailing
(no interior data to interpolate between). Interior gaps at or under
`max_interpolation_gap` are linearly bridged and kept valid.

Velocity is in MediaPipe-normalized units per second — scale- and pan-dependent,
so it is intended for event *timing*, not absolute speed.

## `motion_meta.json`

Sidecar for `motion.npz`.

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | Currently `motion_signals_v1`. |
| `coordinate_system` | string | `mediapipe_normalized_v1`. |
| `video_id` | string | |
| `source_fps` | float | Source video FPS. |
| `frame_stride` | int | |
| `sample_fps` | float | `source_fps / frame_stride`. The rate velocity is differentiated against. |
| `num_frames` | int | `T`, matches `motion.npz` axis 0. |
| `landmark_count` | int | 33. |
| `velocity_units` | string | `normalized_units_per_second`. |
| `com_definition` | string | `hip_midpoint`. |
| `params` | object | The `motion:` settings used: `visibility_threshold`, `max_interpolation_gap`, `smoothing_window`, `smoothing_polyorder`. |
| `valid_ratio` | float | Fraction of `[T, 33]` entries that are valid. |

## `metadata.json`

Video-level metadata.

| Field | Type |
|---|---|
| `video_id` | string |
| `video_path` | string |
| `fps` | float |
| `total_frames` | int |
| `duration_seconds` | float |
| `frame_extraction.frame_stride` | int |
| `frame_extraction.frames_processed` | int |

## `qa_report.json`

| Field | Type | Notes |
|---|---|---|
| `video_id` | string | |
| `fps` | float | |
| `frame_stride` | int | |
| `total_frames_processed` | int | |
| `frames_with_pose` | int | |
| `no_pose_frames` | int | |
| `no_pose_indices` | int[] | Sampled-stream indices, not source-video indices. |
| `max_consecutive_no_pose` | int | |
| `no_pose_ratio` | float | `no_pose_frames / total_frames_processed`. |
| `qa_passed` | bool | False if `no_pose_ratio > max_no_pose_ratio` or `max_consecutive_no_pose > max_pose_gap`. |
