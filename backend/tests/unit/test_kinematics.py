"""Unit tests for motion.kinematics: angle_at and com."""

from __future__ import annotations

import numpy as np
from pacelab.motion.kinematics import LEFT_HIP, RIGHT_HIP, angle_at, com


def _positions(points: dict[int, tuple[float, float]], t_len: int = 1) -> np.ndarray:
    """Build a [T, 33, 2] array, NaN everywhere except the given landmark points."""
    arr = np.full((t_len, 33, 2), np.nan, dtype=np.float64)
    for idx, (x, y) in points.items():
        arr[:, idx, :] = (x, y)
    return arr


def test_right_angle() -> None:
    pos = _positions({0: (1.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 1.0)})
    np.testing.assert_allclose(angle_at(pos, 0, 1, 2), 90.0)


def test_straight_angle() -> None:
    pos = _positions({0: (1.0, 0.0), 1: (0.0, 0.0), 2: (-1.0, 0.0)})
    np.testing.assert_allclose(angle_at(pos, 0, 1, 2), 180.0)


def test_zero_angle() -> None:
    pos = _positions({0: (1.0, 0.0), 1: (0.0, 0.0), 2: (2.0, 0.0)})
    np.testing.assert_allclose(angle_at(pos, 0, 1, 2), 0.0, atol=1e-7)


def test_missing_landmark_is_nan() -> None:
    pos = _positions({0: (1.0, 0.0), 1: (0.0, 0.0)})  # landmark 2 left NaN
    assert np.isnan(angle_at(pos, 0, 1, 2)).all()


def test_degenerate_segment_is_nan() -> None:
    pos = _positions({0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 1.0)})  # b == a
    assert np.isnan(angle_at(pos, 0, 1, 2)).all()


def test_com_is_hip_midpoint() -> None:
    pos = _positions({LEFT_HIP: (0.0, 0.0), RIGHT_HIP: (1.0, 0.4)})
    result = com(pos)
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result[0], (0.5, 0.2))


def test_com_nan_when_hip_missing() -> None:
    pos = _positions({LEFT_HIP: (0.0, 0.0)})  # RIGHT_HIP left NaN
    assert np.isnan(com(pos)).all()
