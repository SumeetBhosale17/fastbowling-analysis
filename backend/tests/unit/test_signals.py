"""Unit tests for motion.signals: SG smoothing, velocity, and gap handling."""

from __future__ import annotations

import numpy as np
import pytest
from pacelab.motion.signals import (
    extract_xy,
    mask_low_visibility,
    smooth_positions,
    velocities,
)


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


def test_mask_low_visibility_nans_below_threshold() -> None:
    lm = np.zeros((2, 33, 4), dtype=np.float32)
    lm[..., :3] = 0.5
    lm[..., 3] = 0.9  # all visible
    lm[:, 5, 3] = 0.3  # landmark 5 below threshold
    out = mask_low_visibility(lm, threshold=0.5)
    assert np.isnan(out[:, 5, :3]).all()  # position NaN'd
    assert (out[:, 5, 3] == 0.3).all()  # visibility preserved
    assert np.isfinite(out[:, 0, :3]).all()  # a visible landmark untouched


def test_mask_threshold_is_exclusive() -> None:
    lm = np.zeros((1, 33, 4), dtype=np.float32)
    lm[..., :3] = 0.5
    lm[..., 3] = 0.5  # exactly at threshold -> kept (not < threshold)
    out = mask_low_visibility(lm, threshold=0.5)
    assert np.isfinite(out[:, 0, :3]).all()


def test_mask_preserves_no_pose_nan() -> None:
    lm = np.full((1, 33, 4), np.nan, dtype=np.float32)  # no pose anywhere
    out = mask_low_visibility(lm, threshold=0.5)
    assert np.isnan(out).all()


def test_mask_wrong_shape_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        mask_low_visibility(np.zeros((2, 33, 2), dtype=np.float32), threshold=0.5)
