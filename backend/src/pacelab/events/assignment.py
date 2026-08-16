"""Limb assignment: which arm bowled, and when the delivery happens.

Nothing in motion.npz says which side is which - it depends on the bowler's
handedness and on which way they face the camera. Both are inferred from the
motion itself, so v1 asks the user for nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pacelab.core.landmarks import PoseLandmark
from pacelab.core.settings import BowlingArmSettings
from pacelab.motion.kinematics import torso_length

ARMS: dict[str, tuple[int, int]] = {
    "left": (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_SHOULDER),
    "right": (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_SHOULDER),
}


@dataclass(frozen=True)
class ArmAssignment:
    """`side` and `anchor_frame` are None when the bowling arm is ambiguous;
    `sweep_deg` is kept either way as the evidence behind the decision."""

    side: str | None
    anchor_frame: int | None
    sweep_deg: dict[str, float]
    reason: str | None


def angle_above_horizontal(
    positions: np.ndarray, wrist: int, shoulder: int
) -> np.ndarray:
    """Angle of the wrist about its shoulder, in degrees.

    Image y grows downward, so it is negated to make up positive: 90 degrees
    is the arm straight overhead, more than 90 is past vertical.
    """
    rel = positions[:, wrist, :] - positions[:, shoulder, :]
    return np.degrees(np.arctan2(-rel[:, 1], rel[:, 0]))


def extended_arm_angle(
    positions: np.ndarray, wrist: int, shoulder: int, min_radius_torso_frac: float
) -> np.ndarray:
    """Arm angle with folded-arm frames blanked.

    When the wrist sits close to its shoulder the angle is both numerically
    unstable and not part of the delivery arc - a bowler's arm is extended
    through the swing. Every consumer of the angle needs the same guard, so it
    lives here rather than being reapplied per detector.
    """
    angle = angle_above_horizontal(positions, wrist, shoulder)
    radius = np.linalg.norm(
        positions[:, wrist, :] - positions[:, shoulder, :], axis=1
    ) / torso_length(positions)
    return np.where(radius > min_radius_torso_frac, angle, np.nan)


def unwrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    """Unwrap while leaving NaN frames NaN. Unwrapping across a gap assumes
    continuity through it, which is why long gaps are masked upstream."""
    out = np.full_like(angle_deg, np.nan)
    finite = np.isfinite(angle_deg)
    out[finite] = np.degrees(np.unwrap(np.radians(angle_deg[finite])))
    return out


def angular_speed(angle_deg: np.ndarray) -> np.ndarray:
    """Central-difference speed in degrees per frame.

    Per frame rather than per second because this only ever ranks frames
    against each other, and the sample rate is a playback rate on slow-motion
    footage anyway.
    """
    unwrapped = unwrap_deg(angle_deg)
    speed = np.full_like(unwrapped, np.nan)
    speed[1:-1] = np.abs(unwrapped[2:] - unwrapped[:-2]) / 2.0
    return speed


def angular_sweep_deg(angle_deg: np.ndarray) -> float:
    """Total angle traversed, unwrapped so a full circumduction does not fold
    back on itself."""
    finite = angle_deg[np.isfinite(angle_deg)]
    if finite.size < 2:
        return 0.0
    unwrapped = np.unwrap(np.radians(finite))
    return float(np.degrees(unwrapped.max() - unwrapped.min()))


def identify_bowling_arm(
    positions: np.ndarray, settings: BowlingArmSettings
) -> ArmAssignment:
    """Pick the bowling arm and the delivery anchor from arm rotation.

    The bowling arm reaches far higher angular speed than the front arm, and
    circumducts through more than a full turn where the front arm manages
    about half. Vertical range and apex height do NOT separate them - on the
    reference clip both pick the front arm.
    """
    angles: dict[str, np.ndarray] = {}
    peaks: dict[str, tuple[float, int]] = {}

    for side, (wrist, shoulder) in ARMS.items():
        # Blanking the angle rather than the resulting speed matters: a
        # corrupt sample also poisons the central difference on both its
        # neighbours, which masking the output alone would leave in place.
        angle = extended_arm_angle(
            positions, wrist, shoulder, settings.min_radius_torso_frac
        )
        speed = angular_speed(angle)
        angles[side] = angle
        if np.isnan(speed).all():
            continue
        peaks[side] = (float(np.nanmax(speed)), int(np.nanargmax(speed)))

    if not peaks:
        return ArmAssignment(None, None, {}, "no frames with an extended arm")

    side = max(peaks, key=lambda s: peaks[s][0])
    anchor = peaks[side][1]

    half = settings.sweep_half_window_frames
    window = slice(max(0, anchor - half), anchor + half)
    sweeps = {s: angular_sweep_deg(angles[s][window]) for s in ARMS}

    if sweeps[side] < settings.min_angular_sweep_deg:
        return ArmAssignment(
            None,
            anchor,
            sweeps,
            f"fastest arm ({side}) swept only {sweeps[side]:.0f} deg, under "
            f"{settings.min_angular_sweep_deg:.0f} - no arm circumducts",
        )
    return ArmAssignment(side, anchor, sweeps, None)
