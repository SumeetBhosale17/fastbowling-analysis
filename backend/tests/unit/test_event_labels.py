"""Integrity checks for the hand-labelled event fixtures.

The clips themselves are gitignored, so detection accuracy against them is a
local-only test. The labels *are* committed, so CI can at least verify they
stay well-formed and internally consistent - a mistyped frame index or a
broken ordering should not wait until someone runs the suite locally.

pytest reports an empty parametrize set as a skip, so these are inert until
the first clip is labelled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_LABEL_DIR = Path(__file__).parents[1] / "fixtures" / "events"
_LABEL_FILES = sorted(_LABEL_DIR.glob("*.json"))
_EVENT_ORDER = ("bfc", "ffc", "release", "follow_through")
_CONFIDENCES = {"high", "medium", "low"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", _LABEL_FILES, ids=lambda p: p.stem)
def test_label_file_is_wellformed(path: Path) -> None:
    labels = _load(path)
    assert labels["schema_version"] == "event_labels_v1"
    assert labels["video_filename"]
    assert len(labels["video_sha1"]) == 40  # sha1 hexdigest
    assert labels["source_fps"] > 0
    assert labels["total_frames"] > 0
    assert set(labels["events"]) == set(_EVENT_ORDER)


@pytest.mark.parametrize("path", _LABEL_FILES, ids=lambda p: p.stem)
def test_labelled_events_are_ordered_and_in_range(path: Path) -> None:
    labels = _load(path)
    total = labels["total_frames"]

    marked: list[tuple[str, int]] = []
    for name in _EVENT_ORDER:
        event = labels["events"][name]
        if event is None:
            continue  # not observable in this clip
        assert 0 <= event["source_frame"] < total, name
        assert event["confidence"] in _CONFIDENCES, name
        assert event["tolerance_frames"] >= 1, name
        marked.append((name, event["source_frame"]))

    assert marked, "label file marks no events at all"
    for (a_name, a), (b_name, b) in zip(marked, marked[1:], strict=False):
        assert a < b, f"{a_name}={a} must precede {b_name}={b}"


@pytest.mark.parametrize("path", _LABEL_FILES, ids=lambda p: p.stem)
def test_label_filename_matches_video(path: Path) -> None:
    """The fixture stem keys the clip; a mismatch means a file was copied and
    edited rather than re-labelled."""
    assert Path(_load(path)["video_filename"]).stem == path.stem
