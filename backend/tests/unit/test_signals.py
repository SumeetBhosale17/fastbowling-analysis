"""Unit tests for motion.signals: SG smoothing, velocity, and gap handling."""

from __future__ import annotations

import numpy as np
import pytest
from pacelab.motion.signals import extract_xy, smooth_positions, velocities


def _ramp(t_len: int, slope: float = 0.01, intercept: float = 0.1) -> np.ndarray:
    """Single-landmark [T, 1, 2] with linear ramps in x and y (slope, 2*slope)."""
    t = np.arange(t_len, dtype=np.float64)
    xy = np.stack([slope * t + intercept, 2 * slope * t + intercept], axis=1)
    return xy[:, None, :]


def test_extract_xy_slices_two_channels() -> None:
    lm = np.zeros((5, 33, 4), dtype=np.float32)
    lm[..., 0], lm[..., 1] = 0.5, 0.6
    xy = extract_xy(lm)
    assert xy.shape == (5, 33, 2)
    assert np.allclose(xy[..., 0], 0.5)
    assert np.allclose(xy[..., 1], 0.6)


def test_smooth_reproduces_linear() -> None:
    pos = _ramp(21)
    smoothed, valid = smooth_positions(pos, window=7, polyorder=3, max_gap=3)
    assert smoothed.shape == pos.shape
    assert valid.shape == (21, 1)
    assert valid.all()
    np.testing.assert_allclose(smoothed, pos, atol=1e-9)


def test_velocity_of_linear_is_constant() -> None:
    fps = 30.0
    pos = _ramp(21, slope=0.01)
    vel, valid = velocities(pos, window=7, polyorder=3, max_gap=3, fps=fps)
    assert valid.all()
    np.testing.assert_allclose(vel[:, 0, 0], 0.01 * fps, atol=1e-6)
    np.testing.assert_allclose(vel[:, 0, 1], 0.02 * fps, atol=1e-6)


def test_short_gap_is_bridged() -> None:
    pos = _ramp(21)
    pos[10, 0, :] = np.nan  # 1-frame interior gap, <= max_gap
    smoothed, valid = smooth_positions(pos, window=7, polyorder=3, max_gap=3)
    assert valid[10, 0]
    assert np.isfinite(smoothed[10, 0]).all()


def test_long_gap_stays_nan() -> None:
    pos = _ramp(21)
    pos[8:16, 0, :] = np.nan  # 8-frame interior gap, > max_gap
    smoothed, valid = smooth_positions(pos, window=7, polyorder=3, max_gap=3)
    assert not valid[8:16, 0].any()
    assert np.isnan(smoothed[8:16, 0]).all()


def test_leading_nan_stays_nan() -> None:
    pos = _ramp(21)
    pos[0:4, 0, :] = np.nan  # leading run is not interior
    _, valid = smooth_positions(pos, window=7, polyorder=3, max_gap=3)
    assert not valid[0:4, 0].any()


def test_too_short_raises() -> None:
    pos = _ramp(5)
    with pytest.raises(ValueError):
        smooth_positions(pos, window=7, polyorder=3, max_gap=3)


def test_nonpositive_fps_raises() -> None:
    pos = _ramp(21)
    with pytest.raises(ValueError):
        velocities(pos, window=7, polyorder=3, max_gap=3, fps=0.0)
