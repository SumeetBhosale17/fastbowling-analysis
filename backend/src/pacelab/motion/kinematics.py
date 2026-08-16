"""Planar geometry primitives over pose landmark trajectories.

Generic, joint-agnostic: functions take a [T, 33, 2] positions array (sagittal
x, y) and landmark indices, and return per-frame quantities. Bowling-specific
named metrics (which landmarks are the knee, etc.) live in the metrics layer.

Angles are returned in degrees. A frame is NaN wherever a required landmark is
missing, so callers can mask on the result.
"""

from __future__ import annotations

import numpy as np

from pacelab.core.landmarks import PoseLandmark


def angle_at(
    positions: np.ndarray,
    a: int,
    b: int,
    c: int,
) -> np.ndarray:
    """Interior angle in degrees at landmark b, between segments b->a and b->c.

    positions: [T, 33, 2]. Returns [T] degrees in [0, 180]. A frame is NaN if any
    of the three landmarks is missing, or if a segment has zero length.
    """
    ba = positions[:, a, :] - positions[:, b, :]
    bc = positions[:, c, :] - positions[:, b, :]
    dot = np.sum(ba * bc, axis=1)
    norm = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.divide(dot, norm, out=np.full_like(dot, np.nan), where=norm > 0)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def torso_length(positions: np.ndarray) -> np.ndarray:
    """Shoulder-midpoint to hip-midpoint distance per frame, as a body-scale
    reference.

    positions: [T, 33, 2]. Returns [T]. Normalized coordinates shrink as the
    bowler moves away from the camera, so thresholds expressed in torso
    lengths survive a clip shot from further back; thresholds in raw
    normalized units do not.
    """
    shoulders = (
        positions[:, PoseLandmark.LEFT_SHOULDER, :]
        + positions[:, PoseLandmark.RIGHT_SHOULDER, :]
    ) / 2.0
    hips = (
        positions[:, PoseLandmark.LEFT_HIP, :] + positions[:, PoseLandmark.RIGHT_HIP, :]
    ) / 2.0
    return np.linalg.norm(shoulders - hips, axis=1)


def com(positions: np.ndarray) -> np.ndarray:
    """Approximate centre of mass per frame as the hip midpoint.

    positions: [T, 33, 2]. Returns [T, 2]. NaN at frames where either hip is
    missing. This is a v1 approximation; a segment-weighted model is deferred
    until a metric needs the extra precision.
    """
    left = positions[:, PoseLandmark.LEFT_HIP, :]
    right = positions[:, PoseLandmark.RIGHT_HIP, :]
    return (left + right) / 2.0
