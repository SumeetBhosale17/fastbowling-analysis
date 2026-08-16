"""Unit tests for event assignment and detectors.

Synthetic signals only - these run in CI, where the labelled footage is not
available. Accuracy against real bowling is covered by
`integration/test_event_accuracy.py`, which skips without the clip.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from pacelab.core.landmarks import PoseLandmark as LM
from pacelab.core.settings import (
    ArmDecelerationSettings,
    BowlingArmSettings,
    FollowThroughSettings,
    FootContactSettings,
    ReleaseSettings,
)
from pacelab.events.assignment import (
    angle_above_horizontal,
    angular_sweep_deg,
    identify_bowling_arm,
)
from pacelab.events.detectors import (
    find_arm_deceleration,
    find_follow_through,
    find_foot_contacts,
    find_release,
    ground_level,
)
from pacelab.motion.kinematics import torso_length

_ARM = BowlingArmSettings(
    min_angular_sweep_deg=270.0,
    sweep_half_window_frames=30,
    min_radius_torso_frac=0.6,
)
_RELEASE = ReleaseSettings(
    degrees_past_vertical=11.0, max_frames_before_arm_speed_peak=30
)
_CONTACT = FootContactSettings(
    max_vertical_speed=0.15,
    min_stationary_frames=3,
    min_separation_frames=5,
    ground_window_frames=15,
    max_height_above_ground_torso_frac=0.15,
    contact_depth_tolerance=0.01,
    min_stride_separation_torso_frac=0.8,
)
_DECEL = ArmDecelerationSettings(max_frames_after_release=60)
_FOLLOW = FollowThroughSettings(
    max_frames_after_arm_deceleration=150,
    max_com_speed_fraction=0.15,
    min_sustained_frames=5,
)


def _body(t_len: int) -> np.ndarray:
    """A static upright body: shoulders at y=0.4, hips at 0.6, ankles at 0.9.
    Torso length is 0.2, so thresholds in torso fractions are easy to reason
    about."""
    pos = np.full((t_len, 33, 2), np.nan)
    for lm, (x, y) in {
        LM.LEFT_SHOULDER: (0.48, 0.4),
        LM.RIGHT_SHOULDER: (0.52, 0.4),
        LM.LEFT_HIP: (0.48, 0.6),
        LM.RIGHT_HIP: (0.52, 0.6),
        LM.LEFT_ANKLE: (0.48, 0.9),
        LM.RIGHT_ANKLE: (0.52, 0.9),
        LM.LEFT_WRIST: (0.48, 0.5),
        LM.RIGHT_WRIST: (0.52, 0.5),
    }.items():
        pos[:, lm, 0], pos[:, lm, 1] = x, y
    return pos


def _rotate_arm(
    pos: np.ndarray, wrist: int, shoulder: int, degrees: np.ndarray, radius: float
) -> None:
    """Place the wrist on a circle of `radius` about its shoulder."""
    theta = np.radians(degrees)
    pos[:, wrist, 0] = pos[:, shoulder, 0] + radius * np.cos(theta)
    pos[:, wrist, 1] = pos[:, shoulder, 1] - radius * np.sin(theta)


def test_torso_length_is_shoulder_to_hip() -> None:
    np.testing.assert_allclose(torso_length(_body(4)), 0.2, atol=1e-9)


def test_angle_above_horizontal_is_90_when_overhead() -> None:
    pos = _body(3)
    pos[:, LM.RIGHT_WRIST, 0] = pos[:, LM.RIGHT_SHOULDER, 0]
    pos[:, LM.RIGHT_WRIST, 1] = pos[:, LM.RIGHT_SHOULDER, 1] - 0.2
    angle = angle_above_horizontal(pos, LM.RIGHT_WRIST, LM.RIGHT_SHOULDER)
    np.testing.assert_allclose(angle, 90.0, atol=1e-9)


def test_angular_sweep_survives_full_circumduction() -> None:
    """A full turn must not fold back to zero via atan2 wraparound."""
    assert angular_sweep_deg(np.linspace(0, 400, 80)) == pytest.approx(400, abs=1)


def _delivery_sweep(t_len: int, total_deg: float, peak_at: int) -> np.ndarray:
    """Logistic angle ramp: slow, fast through the delivery, slow again. A
    constant-rate rotation has no speed peak, so the anchor would be arbitrary
    and the sweep window would not straddle the delivery."""
    t = np.arange(t_len)
    return total_deg / (1.0 + np.exp(-(t - peak_at) / 8.0))


def test_identify_bowling_arm_prefers_the_circumducting_side() -> None:
    pos = _body(120)
    _rotate_arm(
        pos, LM.RIGHT_WRIST, LM.RIGHT_SHOULDER, _delivery_sweep(120, 400, 60), 0.2
    )
    _rotate_arm(pos, LM.LEFT_WRIST, LM.LEFT_SHOULDER, _delivery_sweep(120, 60, 60), 0.2)
    arm = identify_bowling_arm(pos, _ARM)
    assert arm.side == "right"
    assert arm.sweep_deg["right"] > arm.sweep_deg["left"]
    assert arm.anchor_frame is not None and abs(arm.anchor_frame - 60) <= 3


def test_identify_bowling_arm_rejects_when_neither_circumducts() -> None:
    """Front-on footage foreshortens the sweep; refusing beats guessing."""
    pos = _body(120)
    _rotate_arm(pos, LM.RIGHT_WRIST, LM.RIGHT_SHOULDER, np.linspace(0, 60, 120), 0.2)
    _rotate_arm(pos, LM.LEFT_WRIST, LM.LEFT_SHOULDER, np.linspace(0, 50, 120), 0.2)
    arm = identify_bowling_arm(pos, _ARM)
    assert arm.side is None
    assert arm.reason is not None and "circumduct" in arm.reason


def test_identify_bowling_arm_ignores_folded_arm_spikes() -> None:
    """A wrist passing near its shoulder swings the angle wildly. Without the
    radius guard that spike outranks the real delivery."""
    pos = _body(120)
    _rotate_arm(
        pos, LM.RIGHT_WRIST, LM.RIGHT_SHOULDER, _delivery_sweep(120, 400, 60), 0.2
    )
    _rotate_arm(pos, LM.LEFT_WRIST, LM.LEFT_SHOULDER, _delivery_sweep(120, 60, 60), 0.2)
    # Left wrist collapses onto its shoulder for three frames.
    pos[50:53, LM.LEFT_WRIST, :] = pos[50:53, LM.LEFT_SHOULDER, :] + 1e-4
    assert identify_bowling_arm(pos, _ARM).side == "right"


def test_find_release_lands_past_vertical() -> None:
    pos = _body(80)
    # 5 degrees per frame: vertical at frame 40, 11 degrees past by frame 43.
    _rotate_arm(
        pos,
        LM.RIGHT_WRIST,
        LM.RIGHT_SHOULDER,
        np.arange(80) * 5.0 - 110.0,
        0.2,
    )
    found = find_release(pos, "right", 60, _RELEASE, _ARM.min_radius_torso_frac)
    assert found.frame == 43


def test_find_release_reports_reason_when_never_below_threshold() -> None:
    pos = _body(80)
    _rotate_arm(pos, LM.RIGHT_WRIST, LM.RIGHT_SHOULDER, np.full(80, 150.0), 0.2)
    found = find_release(pos, "right", 60, _RELEASE, _ARM.min_radius_torso_frac)
    assert found.frame is None
    assert found.reason is not None


def test_find_arm_deceleration_finds_lowest_wrist_point() -> None:
    pos = _body(60)
    y = 0.4 + 0.3 * np.sin(np.linspace(0, np.pi, 60))
    pos[:, LM.RIGHT_WRIST, 1] = y
    pos[:, LM.RIGHT_WRIST, 0] = 0.52
    found = find_arm_deceleration(pos, "right", release_frame=5, settings=_DECEL)
    assert found.frame == int(np.argmax(y))


def test_find_arm_deceleration_reports_untracked_wrist() -> None:
    pos = _body(60)
    pos[:, LM.RIGHT_WRIST, :] = np.nan
    found = find_arm_deceleration(pos, "right", release_frame=5, settings=_DECEL)
    assert found.frame is None
    assert found.reason is not None


def test_find_follow_through_detects_settling() -> None:
    com = np.zeros((80, 2))
    com[:40, 0] = np.arange(40) * 0.01  # moving, then still
    com[40:, 0] = com[39, 0]
    found = find_follow_through(com, arm_deceleration_frame=10, settings=_FOLLOW)
    assert found.frame is not None and found.frame >= 40


def test_find_follow_through_reports_reason_when_never_settles() -> None:
    com = np.stack([np.arange(80) * 0.01, np.zeros(80)], axis=1)
    found = find_follow_through(com, arm_deceleration_frame=10, settings=_FOLLOW)
    assert found.frame is None
    assert found.reason is not None


def _striding_body(t_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bowler travelling in -x: back foot plants at 30, front foot at 60 a
    full torso length ahead, both staying down through release at 70."""
    pos = _body(t_len)
    pos[:, LM.LEFT_SHOULDER, 0] = 0.8 - np.arange(t_len) * 0.004
    pos[:, LM.RIGHT_SHOULDER, 0] = pos[:, LM.LEFT_SHOULDER, 0] + 0.04
    pos[:, LM.LEFT_HIP, 0] = pos[:, LM.LEFT_SHOULDER, 0]
    pos[:, LM.RIGHT_HIP, 0] = pos[:, LM.RIGHT_SHOULDER, 0]

    hip_x = (pos[:, LM.LEFT_HIP, 0] + pos[:, LM.RIGHT_HIP, 0]) / 2
    back_y = np.where(np.arange(t_len) < 30, 0.75, 0.9)
    front_y = np.where(np.arange(t_len) < 60, 0.75, 0.9)
    pos[:, LM.RIGHT_ANKLE, 1] = back_y
    pos[:, LM.LEFT_ANKLE, 1] = front_y
    pos[:, LM.RIGHT_ANKLE, 0] = hip_x + 0.10  # behind (travel is -x)
    pos[:, LM.LEFT_ANKLE, 0] = np.where(
        np.arange(t_len) < 60, hip_x - 0.05, hip_x - 0.22
    )

    velocity = np.gradient(pos, axis=0)
    com = (pos[:, LM.LEFT_HIP, :] + pos[:, LM.RIGHT_HIP, :]) / 2
    return pos, velocity, com


def test_find_foot_contacts_orders_and_assigns_legs() -> None:
    pos, velocity, com = _striding_body(90)
    contacts = find_foot_contacts(pos, velocity, com, 70, _CONTACT)
    assert contacts.front_leg == "left"
    assert contacts.back_leg == "right"
    assert contacts.bfc.frame is not None and contacts.ffc.frame is not None
    assert contacts.bfc.frame < contacts.ffc.frame < 70


def test_find_foot_contacts_reports_reason_without_a_stride() -> None:
    """Feet never separate, so no plant reaches delivery-stride extension."""
    pos = _body(90)
    pos[:, LM.LEFT_ANKLE, 1] = 0.9
    pos[:, LM.RIGHT_ANKLE, 1] = 0.9
    velocity = np.zeros_like(pos)
    com = np.stack([0.5 - np.arange(90) * 0.004, np.full(90, 0.6)], axis=1)
    contacts = find_foot_contacts(pos, velocity, com, 70, _CONTACT)
    assert contacts.ffc.frame is None
    assert contacts.front_leg is None
    # Plants were found, they were just too narrow - a different failure from
    # finding none at all, and the reason has to say which.
    assert contacts.ffc.reason is not None
    assert "torso lengths of stride extension" in contacts.ffc.reason


def _untracked_ankles(t_len: int) -> np.ndarray:
    pos = _body(t_len)
    pos[:, LM.LEFT_ANKLE, :] = np.nan
    pos[:, LM.RIGHT_ANKLE, :] = np.nan
    return pos


def test_ground_level_is_nan_without_tracked_ankles() -> None:
    """Neither ankle tracked is a normal state on poor footage, so it must not
    warn on every frame."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ground = ground_level(_untracked_ankles(20), 5)
    assert np.isnan(ground).all()


def test_find_foot_contacts_distinguishes_no_plants_from_short_stride() -> None:
    pos = _untracked_ankles(90)
    com = np.stack([0.5 - np.arange(90) * 0.004, np.full(90, 0.6)], axis=1)
    contacts = find_foot_contacts(pos, np.zeros_like(pos), com, 70, _CONTACT)
    assert contacts.bfc.frame is None
    assert contacts.bfc.reason is not None
    assert "no ankle plant detected" in contacts.bfc.reason


def test_find_foot_contacts_reports_untracked_com() -> None:
    pos = _body(90)
    com = np.full((90, 2), np.nan)
    contacts = find_foot_contacts(pos, np.zeros_like(pos), com, 70, _CONTACT)
    assert contacts.bfc.frame is None
    assert contacts.bfc.reason is not None
    assert "centre of mass" in contacts.bfc.reason
