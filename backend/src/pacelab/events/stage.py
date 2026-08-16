"""Event stage: locate the delivery events in conditioned motion signals.

Reads motion.npz written by the motion stage and writes events.json. Every
event is reported as an object even when undetected, carrying the reason it
was not found - a metric that depends on a missing event should be able to
say why it is unavailable without re-running detection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from pacelab.core.settings import Settings
from pacelab.core.video_context import VideoContext
from pacelab.events.assignment import identify_bowling_arm
from pacelab.events.detectors import (
    Detection,
    find_arm_deceleration,
    find_follow_through,
    find_foot_contacts,
    find_release,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "events_v1"
EVENT_ORDER = (
    "bfc",
    "ffc",
    "release",
    "arm_deceleration_complete",
    "follow_through",
)


def detect_events(
    positions: np.ndarray,
    velocity: np.ndarray,
    com: np.ndarray,
    settings: Settings,
) -> tuple[dict[str, Detection], dict[str, object]]:
    """Run the detection cascade. Returns per-event detections plus the limb
    assignment that produced them."""
    cfg = settings.events
    arm = identify_bowling_arm(positions, cfg.bowling_arm)
    assignment: dict[str, object] = {
        "bowling_arm": arm.side,
        "front_leg": None,
        "back_leg": None,
        "delivery_anchor_frame": arm.anchor_frame,
        "arm_sweep_deg": arm.sweep_deg,
    }

    if arm.side is None or arm.anchor_frame is None:
        return dict.fromkeys(EVENT_ORDER, Detection(None, arm.reason)), assignment

    found: dict[str, Detection] = {}
    release = find_release(positions, arm.side, arm.anchor_frame, cfg.release)
    found["release"] = release

    if release.frame is None:
        blocked = Detection(None, "release not detected")
        for name in EVENT_ORDER:
            found.setdefault(name, blocked)
        return found, assignment

    contacts = find_foot_contacts(
        positions, velocity, com, release.frame, cfg.foot_contact
    )
    found["bfc"], found["ffc"] = contacts.bfc, contacts.ffc
    assignment["front_leg"] = contacts.front_leg
    assignment["back_leg"] = contacts.back_leg

    decel = find_arm_deceleration(
        positions, arm.side, release.frame, cfg.arm_deceleration
    )
    found["arm_deceleration_complete"] = decel
    found["follow_through"] = (
        find_follow_through(com, decel.frame, cfg.follow_through)
        if decel.frame is not None
        else Detection(None, "arm deceleration not detected")
    )
    return found, assignment


def run_events(ctx: VideoContext, settings: Settings) -> Path:
    """Detect delivery events and write events.json.

    Outputs (under data/processed/<video_id>/):
      - events.json  limb assignment + one entry per event, each with frame,
                     timestamp_ms and reason. frame is null when undetected.
    """
    out_dir = settings.data.output_dir / ctx.video_id
    with np.load(out_dir / "motion.npz") as motion:
        positions = motion["positions"]
        velocity = motion["velocity"]
        com = motion["com"]
    meta = json.loads((out_dir / "landmarks_meta.json").read_text(encoding="utf-8"))
    timestamps = meta["timestamps_ms"]

    found, assignment = detect_events(positions, velocity, com, settings)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "video_id": ctx.video_id,
        "sample_fps": ctx.fps / ctx.frame_stride,
        "num_frames": int(positions.shape[0]),
        "assignment": assignment,
        "events": {
            name: {
                "frame": found[name].frame,
                "timestamp_ms": (
                    timestamps[found[name].frame]
                    if found[name].frame is not None
                    else None
                ),
                "reason": found[name].reason,
            }
            for name in EVENT_ORDER
        },
        "params": settings.events.model_dump(),
    }
    with open(out_dir / "events.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    detected = [n for n in EVENT_ORDER if found[n].frame is not None]
    logger.info(
        "Events for %s: %s arm, %d/%d detected%s",
        ctx.video_id,
        assignment["bowling_arm"],
        len(detected),
        len(EVENT_ORDER),
        (
            " (" + ", ".join(f"{n}={found[n].frame}" for n in detected) + ")"
            if detected
            else ""
        ),
    )
    for name in EVENT_ORDER:
        if found[name].frame is None:
            logger.warning("Event %s not detected: %s", name, found[name].reason)

    return out_dir
