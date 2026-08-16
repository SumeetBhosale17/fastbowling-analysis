"""Detected events versus hand labels, on real footage.

Local-only: the clips are gitignored, so CI skips every test here. This is a
regression guard, not a validation - one bowler, one camera position, one
labeller. It catches "a refactor moved the answer"; it says nothing about
whether the detectors generalise.

Tolerances live in the label file, per event, because the labeller is the one
who knows how crisply each event could be seen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pacelab.core.settings import Settings, load_settings
from pacelab.pipeline import process_video

if TYPE_CHECKING:
    # conftest is loaded by pytest, not importable as a module. Annotations
    # are lazy under `from __future__ import annotations`, so this never runs.
    from tests.conftest import LabelledClip


def _settings_for_test(output_dir: Path) -> Settings:
    settings = load_settings()
    return settings.model_copy(
        update={"data": settings.data.model_copy(update={"output_dir": output_dir})}
    )


def test_detected_events_match_labels(
    labelled_clip: LabelledClip, tmp_path: Path
) -> None:
    out_dir = process_video(labelled_clip.video_path, _settings_for_test(tmp_path))
    detected = json.loads((out_dir / "events.json").read_text(encoding="utf-8"))
    meta = json.loads((out_dir / "landmarks_meta.json").read_text(encoding="utf-8"))
    source_frames = meta["source_frame_indices"]

    failures: list[str] = []
    for name, label in labelled_clip.labels["events"].items():
        found = detected["events"][name]
        if label is None:
            # The label says the clip does not contain this event, so the
            # detector must not claim one.
            if found["frame"] is not None:
                failures.append(
                    f"{name}: detected frame {found['frame']} but the clip is "
                    "labelled as not containing this event"
                )
            continue

        if found["frame"] is None:
            failures.append(f"{name}: not detected ({found['reason']})")
            continue

        # Labels are in source-frame space so they survive a frame_stride
        # change; detections index the sampled stream.
        expected = source_frames.index(label["source_frame"])
        offset = abs(found["frame"] - expected)
        if offset > label["tolerance_frames"]:
            failures.append(
                f"{name}: detected {found['frame']}, labelled {expected}, "
                f"off by {offset} > tolerance {label['tolerance_frames']}"
            )

    assert not failures, "\n".join(failures)


def test_limb_assignment_is_self_consistent(
    labelled_clip: LabelledClip, tmp_path: Path
) -> None:
    out_dir = process_video(labelled_clip.video_path, _settings_for_test(tmp_path))
    assignment = json.loads((out_dir / "events.json").read_text(encoding="utf-8"))[
        "assignment"
    ]

    assert assignment["bowling_arm"] in {"left", "right"}
    assert assignment["front_leg"] in {"left", "right"}
    assert assignment["back_leg"] in {"left", "right"}
    assert assignment["front_leg"] != assignment["back_leg"]
    # The bowling arm has to out-sweep the front arm for the pick to mean
    # anything; equal sweeps would mean the threshold got lucky.
    sweeps = assignment["arm_sweep_deg"]
    other = "left" if assignment["bowling_arm"] == "right" else "right"
    assert sweeps[assignment["bowling_arm"]] > sweeps[other]
