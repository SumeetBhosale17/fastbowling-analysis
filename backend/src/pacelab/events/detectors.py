"""Event detectors over conditioned motion signals.

Each returns a frame index or None with a reason. The order is a cascade -
the bowling arm anchors release, and release anchors the foot contacts - so a
missing prerequisite yields None rather than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pacelab.core.landmarks import PoseLandmark
from pacelab.core.settings import (
    ArmDecelerationSettings,
    FollowThroughSettings,
    FootContactSettings,
    ReleaseSettings,
)
from pacelab.events.assignment import ARMS, angle_above_horizontal, unwrap_deg
from pacelab.motion.kinematics import torso_length

ANKLES: dict[str, int] = {
    "left": PoseLandmark.LEFT_ANKLE,
    "right": PoseLandmark.RIGHT_ANKLE,
}


@dataclass(frozen=True)
class Detection:
    frame: int | None
    reason: str | None = None


@dataclass(frozen=True)
class FootContacts:
    bfc: Detection
    ffc: Detection
    front_leg: str | None
    back_leg: str | None


def _other(side: str) -> str:
    return "left" if side == "right" else "right"


def find_release(
    positions: np.ndarray,
    bowling_arm: str,
    anchor_frame: int,
    settings: ReleaseSettings,
) -> Detection:
    """First frame on the approach to peak arm speed at which the arm has
    passed vertical by `degrees_past_vertical`.

    Scanning backward from the anchor rather than forward from a start frame
    means an earlier crossing during the run-up cannot be mistaken for the
    release.
    """
    wrist, shoulder = ARMS[bowling_arm]
    threshold = 90.0 + settings.degrees_past_vertical
    floor = max(0, anchor_frame - settings.max_frames_before_arm_speed_peak)

    # Unwrap over the search window only. atan2 wraps at 180 degrees, and the
    # arm is often already past that at peak speed - compared raw, the anchor
    # reads as a negative angle and the scan stops on its first step. Local
    # unwrapping keeps the arc continuous without accumulating the turns the
    # arm makes earlier in the run-up.
    angle = unwrap_deg(
        angle_above_horizontal(positions, wrist, shoulder)[floor : anchor_frame + 1]
    )

    i = anchor_frame - floor
    while i > 0 and (not np.isfinite(angle[i]) or angle[i] >= threshold):
        i -= 1
    if i == 0:
        return Detection(
            None,
            f"arm stayed above {threshold:.0f} deg for the whole "
            f"{settings.max_frames_before_arm_speed_peak}-frame window before "
            "peak arm speed",
        )
    return Detection(floor + i + 1)


def ground_level(positions: np.ndarray, window: int) -> np.ndarray:
    """Rolling maximum of both ankle heights - a local estimate of the ground.

    Local rather than global because perspective moves the ground line as the
    bowler travels across the frame.
    """
    ankles = np.vstack([positions[:, lm, 1] for lm in ANKLES.values()])
    # -inf rather than np.nanmax: a window with neither ankle tracked is a
    # normal state on poor footage, and nanmax reports it by warning on every
    # frame. Carrying -inf through and restoring NaN at the end says the same
    # thing without the noise.
    filled = np.where(np.isfinite(ankles), ankles, -np.inf).max(axis=0)
    rolled = np.array(
        [filled[max(0, k - window) : k + window + 1].max() for k in range(filled.size)]
    )
    return np.where(np.isneginf(rolled), np.nan, rolled)


def plant_windows(
    positions: np.ndarray,
    velocity: np.ndarray,
    ankle: int,
    ground: np.ndarray,
    torso: np.ndarray,
    settings: FootContactSettings,
) -> list[tuple[int, int]]:
    """Frame ranges where an ankle is both slow and near the ground.

    Both conditions are needed: an airborne foot at the apex of its swing also
    has zero vertical velocity, and speed alone reports it as a contact.
    """
    vy = np.abs(velocity[:, ankle, 1])
    y = positions[:, ankle, 1]
    planted = (
        np.isfinite(vy)
        & (vy < settings.max_vertical_speed)
        & np.isfinite(y)
        & (y >= ground - settings.max_height_above_ground_torso_frac * torso)
    )

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for k in range(planted.size):
        if planted[k] and start is None:
            start = k
        elif not planted[k] and start is not None:
            if k - start >= settings.min_stationary_frames:
                runs.append((start, k))
            start = None
    if start is not None and planted.size - start >= settings.min_stationary_frames:
        runs.append((start, planted.size))

    # One plant fragments whenever the ankle landmark twitches past the speed
    # threshold; unmerged, a single contact reads as several.
    merged: list[tuple[int, int]] = []
    for start_f, end_f in runs:
        if merged and start_f - merged[-1][1] <= settings.min_separation_frames:
            merged[-1] = (merged[-1][0], end_f)
        else:
            merged.append((start_f, end_f))
    return merged


def _contact_frame(
    positions: np.ndarray, ankle: int, window: tuple[int, int], depth_tol: float
) -> int:
    """First frame in the window where the ankle has reached its floor.

    The window opens while the foot is still descending - its velocity is
    already under threshold - so the window start is a few frames early.
    """
    start, end = window
    y = positions[start:end, ankle, 1]
    floor = np.nanmax(y)
    hit = np.where(np.isfinite(y) & (y >= floor - depth_tol))[0]
    return start + int(hit[0])


def find_foot_contacts(
    positions: np.ndarray,
    velocity: np.ndarray,
    com: np.ndarray,
    release_frame: int,
    settings: FootContactSettings,
) -> FootContacts:
    """FFC is the last plant before release at full stride extension; BFC is
    the last plant of the other ankle before it.

    Extension is what separates the delivery stride from the back leg swinging
    through afterwards, which clears the front leg by roughly half as much.
    Which ankle leads is measured along the direction of travel, so the result
    does not depend on which way the bowler faces.
    """
    torso = torso_length(positions)
    ground = ground_level(positions, settings.ground_window_frames)

    drift = np.diff(com[:release_frame, 0])
    if not np.isfinite(drift).any():
        untracked = Detection(None, "centre of mass not tracked before release")
        return FootContacts(untracked, untracked, None, None)
    travel = np.sign(np.nanmean(drift))

    candidates: list[tuple[int, str, float]] = []
    for side, ankle in ANKLES.items():
        other = ANKLES[_other(side)]
        for window in plant_windows(
            positions, velocity, ankle, ground, torso, settings
        ):
            if window[0] >= release_frame:
                continue
            frame = _contact_frame(
                positions, ankle, window, settings.contact_depth_tolerance
            )
            lead = (
                (positions[frame, ankle, 0] - positions[frame, other, 0])
                * travel
                / torso[frame]
            )
            candidates.append((frame, side, float(lead)))
    candidates.sort()

    if not candidates:
        # Distinct from finding plants that are simply too short a stride:
        # this means no ankle was ever both slow and near the ground, which on
        # real footage points at pose coverage rather than at the threshold.
        no_plant = Detection(
            None,
            "no ankle plant detected before release - no frames with an ankle "
            "both stationary and near the ground",
        )
        return FootContacts(no_plant, no_plant, None, None)

    strides = [
        c for c in candidates if c[2] >= settings.min_stride_separation_torso_frac
    ]
    if not strides:
        widest = max(c[2] for c in candidates)
        no_stride = Detection(
            None,
            f"{len(candidates)} plant(s) found before release but the widest "
            f"reached only {widest:.2f} torso lengths of stride extension, "
            f"under {settings.min_stride_separation_torso_frac}",
        )
        return FootContacts(no_stride, no_stride, None, None)

    ffc_frame, front_leg, _ = strides[-1]
    back_leg = _other(front_leg)
    prior = [c for c in candidates if c[1] == back_leg and c[0] < ffc_frame]
    if not prior:
        return FootContacts(
            Detection(None, f"no {back_leg} ankle plant before front-foot contact"),
            Detection(ffc_frame),
            front_leg,
            back_leg,
        )
    return FootContacts(
        Detection(prior[-1][0]), Detection(ffc_frame), front_leg, back_leg
    )


def find_arm_deceleration(
    positions: np.ndarray,
    bowling_arm: str,
    release_frame: int,
    settings: ArmDecelerationSettings,
) -> Detection:
    """Lowest point of the bowling wrist after release - the end of the
    downswing.

    Position rather than speed: near a turning point the extremum is flat, but
    it is still one well-defined frame, whereas a speed threshold fires
    anywhere across that flat.
    """
    wrist = ARMS[bowling_arm][0]
    end = min(positions.shape[0], release_frame + settings.max_frames_after_release)
    y = positions[release_frame:end, wrist, 1]
    if np.isnan(y).all():
        return Detection(None, "bowling wrist not tracked after release")
    return Detection(release_frame + int(np.nanargmax(y)))


def find_follow_through(
    com: np.ndarray,
    arm_deceleration_frame: int,
    settings: FollowThroughSettings,
) -> Detection:
    """First frame after arm deceleration where whole-body speed stays low for
    `min_sustained_frames`.

    UNVALIDATED: the reference clip ends mid-action, so these thresholds are
    starting points rather than measurements.
    """
    speed = np.linalg.norm(np.gradient(com, axis=0), axis=1)
    end = min(
        speed.size,
        arm_deceleration_frame + settings.max_frames_after_arm_deceleration,
    )
    segment = speed[arm_deceleration_frame:end]
    if np.isnan(segment).all():
        return Detection(None, "centre of mass not tracked after arm deceleration")

    threshold = settings.max_com_speed_fraction * np.nanmax(segment)
    settled = np.isfinite(segment) & (segment <= threshold)
    run = 0
    for k, quiet in enumerate(settled):
        run = run + 1 if quiet else 0
        if run >= settings.min_sustained_frames:
            return Detection(
                arm_deceleration_frame + k - settings.min_sustained_frames + 1
            )
    return Detection(
        None,
        "centre-of-mass speed never settled within "
        f"{settings.max_frames_after_arm_deceleration} frames of arm deceleration",
    )
