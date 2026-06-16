"""Signal conditioning for pose landmark trajectories.

Savitzky-Golay smoothing and its first derivative (velocity), computed over the
sagittal-plane (x, y) channels of a [T, 33, 4] landmarks array. SG is chosen
because it preserves peaks/edges (the release window is ~1 frame at 60 fps) and
yields a smooth derivative from the same polynomial fit, so velocity needs no
separate noisy finite-difference step.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

# MediaPipe landmarks are (x, y, z, visibility); v1 is side-on, so the camera
# plane is the sagittal plane and the 2D (x, y) channels are what we analyze.
_XY = slice(0, 2)


def extract_xy(landmarks: np.ndarray) -> np.ndarray:
    """Slice the (x, y) channels from [T, 33, 4] landmarks into [T, 33, 2]."""
    if landmarks.ndim != 3 or landmarks.shape[2] < 2:
        raise ValueError(
            f"expected [T, 33, >=2] landmarks, got shape {landmarks.shape}"
        )
    return landmarks[:, :, _XY].astype(np.float64)


def _short_interior_gap_mask(present: np.ndarray, max_gap: int) -> np.ndarray:
    """Mask over time, True where a frame is trustworthy: either pose was
    present, or it sits in an interior NaN run no longer than max_gap (so a
    short occlusion can be safely bridged by interpolation)."""
    n = present.size
    keep = present.copy()
    i = 0
    while i < n:
        if not present[i]:
            j = i
            while j < n and not present[j]:
                j += 1
            interior = i > 0 and j < n
            if interior and (j - i) <= max_gap:
                keep[i:j] = True
            i = j
        else:
            i += 1
    return keep


def _fill_series(series: np.ndarray) -> np.ndarray:
    """Linear-interpolate interior NaN; clamp leading/trailing to nearest valid.

    Produces a NaN-free series so the SG filter does not propagate NaN across
    its window. Frames that should not be trusted are masked out by the caller.
    """
    out = series.copy()
    valid = ~np.isnan(series)
    if not valid.any():
        return out
    idx = np.arange(series.size)
    out[~valid] = np.interp(idx[~valid], idx[valid], series[valid])
    return out


def _apply_sg(
    positions: np.ndarray,
    *,
    window: int,
    polyorder: int,
    max_gap: int,
    deriv: int,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-landmark, per-coordinate Savitzky-Golay over the time axis.

    Returns (out [T, L, C] float64, valid [T, L] bool). out is NaN where the
    frame is not trustworthy (long gap, or leading/trailing missing pose).
    """
    t_len, n_landmarks, n_coords = positions.shape
    if t_len < window:
        raise ValueError(f"need at least window={window} frames, got T={t_len}")

    out = np.full((t_len, n_landmarks, n_coords), np.nan, dtype=np.float64)
    valid = np.zeros((t_len, n_landmarks), dtype=bool)

    for lm in range(n_landmarks):
        # Pose presence is per-landmark (all coords share it), keyed off x.
        present = ~np.isnan(positions[:, lm, 0])
        keep = _short_interior_gap_mask(present, max_gap)
        valid[:, lm] = keep
        for c in range(n_coords):
            filled = _fill_series(positions[:, lm, c])
            if np.isnan(filled).all():
                continue  # landmark never detected
            res = np.asarray(
                savgol_filter(filled, window, polyorder, deriv=deriv, delta=delta)
            )
            out[:, lm, c] = np.where(keep, res, np.nan)

    return out, valid


def smooth_positions(
    positions: np.ndarray,
    *,
    window: int,
    polyorder: int,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Savitzky-Golay smoothing of [T, 33, 2] positions over time.

    Returns (smoothed [T, 33, 2] float64, valid [T, 33] bool).
    """
    return _apply_sg(
        positions,
        window=window,
        polyorder=polyorder,
        max_gap=max_gap,
        deriv=0,
        delta=1.0,
    )


def velocities(
    positions: np.ndarray,
    *,
    window: int,
    polyorder: int,
    max_gap: int,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First time-derivative of [T, 33, 2] positions via the SG filter.

    Returns (velocity [T, 33, 2] float64, valid [T, 33] bool). Units are
    normalized-coordinate units per second (positions are MediaPipe-normalized,
    so this is scale/pan dependent and intended for event *timing*, not absolute
    speed).
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return _apply_sg(
        positions,
        window=window,
        polyorder=polyorder,
        max_gap=max_gap,
        deriv=1,
        delta=1.0 / fps,
    )
