"""Frame-accurate event labelling tool for building ground-truth fixtures.

Opens a clip in a scrubbing window, steps frame by frame, and records the
source frame index of each delivery event. Writes a label JSON that the
event-detection tests assert against.

Run this BEFORE writing detectors. Labels made after seeing detector output
are biased toward whatever the detector happened to find.

Usage:
    uv run python scripts/label_events.py data/raw_videos/jofra.mp4

Re-running against an existing label file reloads its marks, so revising one
event does not mean re-marking the others.

Keys:
    a / d      step back / forward one frame
    A / D      step back / forward ten frames
    [ / ]      jump to previous / next marked frame
    1 2 3 4 5  mark BFC / FFC / release / arm-deceleration-complete /
               follow-through at the current frame
    c          cycle confidence of the mark on this frame (high/medium/low)
    x          clear the mark on this frame
    w          write labels and exit
    q          quit without saving

Leaving an event unmarked writes it as null - the honest record for an event
the footage does not contain, which the tests skip rather than assert.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np

SCHEMA_VERSION = "event_labels_v2"
EVENT_ORDER = (
    "bfc",
    "ffc",
    "release",
    "arm_deceleration_complete",
    "follow_through",
)
_KEY_TO_EVENT = {
    ord("1"): "bfc",
    ord("2"): "ffc",
    ord("3"): "release",
    ord("4"): "arm_deceleration_complete",
    ord("5"): "follow_through",
}

# Tolerance the test allows for each event, in frames. Not uniform: a foot
# plant is a crisp physical instant, whereas "the arm has stopped" and "the
# action has ended" are judgement calls even for a human labeller. Encoding
# that here keeps the test honest instead of pretending all five are equally
# observable. Turning-point events get the widest band because position
# barely changes across the frames either side of an extremum.
_DEFAULT_TOLERANCE = {
    "bfc": 2,
    "ffc": 2,
    "release": 1,
    "arm_deceleration_complete": 3,
    "follow_through": 5,
}
_CONFIDENCES = ("high", "medium", "low")


def sha1_of_file(path: Path) -> str:
    """Content hash of the clip.

    Deliberately not `video_id` (sha1 of the *path*) - labels must stay bound
    to the footage, not to where it happens to sit on disk.
    """
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def _render(
    frame: np.ndarray | None,
    idx: int,
    total: int,
    fps: float,
    marks: dict[str, dict],
    max_height: int,
) -> np.ndarray:
    if frame is None:
        raise RuntimeError("no frame to render")

    out = frame.copy()
    if out.shape[0] > max_height:
        scale = max_height / out.shape[0]
        out = cv2.resize(out, None, fx=scale, fy=scale)

    lines = [f"frame {idx}/{total - 1}    t={idx / fps:.3f}s"]
    for name in EVENT_ORDER:
        mark = marks.get(name)
        if mark is None:
            lines.append(f"{name:26s}   --")
        else:
            here = "  <<" if mark["source_frame"] == idx else ""
            lines.append(
                f"{name:26s}    {mark['source_frame']:5d}  {mark['confidence']}{here}"
            )

    for i, line in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(
            out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            out,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _load_existing(path: Path) -> dict[str, dict]:
    """Reload marks from a previous session. Unmarked (null) events and any
    key no longer in EVENT_ORDER are dropped."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: dict(event)
        for name, event in payload.get("events", {}).items()
        if event is not None and name in EVENT_ORDER
    }


def _marked_at(marks: dict[str, dict], idx: int) -> list[str]:
    return [n for n in EVENT_ORDER if n in marks and marks[n]["source_frame"] == idx]


def _nearest_mark(marks: dict[str, dict], idx: int, *, forward: bool) -> int | None:
    frames = sorted({m["source_frame"] for m in marks.values()})
    candidates = [f for f in frames if (f > idx if forward else f < idx)]
    if not candidates:
        return None
    return candidates[0] if forward else candidates[-1]


def _validate_order(marks: dict[str, dict]) -> str | None:
    """Delivery events are strictly ordered in time. Violations are labelling
    slips, not exotic deliveries, so refuse to save rather than persist them."""
    marked = [(n, marks[n]["source_frame"]) for n in EVENT_ORDER if n in marks]
    for (a_name, a), (b_name, b) in zip(marked, marked[1:], strict=False):
        if b <= a:
            return f"{a_name}={a} must come strictly before {b_name}={b}"
    return None


def label(video_path: Path, out_path: Path, max_height: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    marks = _load_existing(out_path)
    if marks:
        print(f"resuming from {out_path}: {len(marks)} event(s) already marked")
    idx, frame = 0, None
    dirty = True

    try:
        while True:
            if dirty:
                frame = _read_frame(cap, idx)
                if frame is None:
                    raise RuntimeError(f"failed to read frame {idx}")
                dirty = False

            cv2.imshow(
                "label_events", _render(frame, idx, total, fps, marks, max_height)
            )
            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                print("quit without saving")
                return 1
            elif key == ord("w"):
                break
            elif key in (ord("a"), ord("d"), ord("A"), ord("D")):
                step = {ord("a"): -1, ord("d"): 1, ord("A"): -10, ord("D"): 10}[key]
                new = min(max(idx + step, 0), total - 1)
                idx, dirty = new, new != idx
            elif key in (ord("["), ord("]")):
                target = _nearest_mark(marks, idx, forward=key == ord("]"))
                if target is not None:
                    idx, dirty = target, True
            elif key in _KEY_TO_EVENT:
                # Re-marking an event keeps whatever confidence and tolerance
                # it already carried; only its frame moves.
                name = _KEY_TO_EVENT[key]
                previous = marks.get(name, {})
                marks[name] = {
                    "source_frame": idx,
                    "confidence": previous.get("confidence", "high"),
                    "tolerance_frames": previous.get(
                        "tolerance_frames", _DEFAULT_TOLERANCE[name]
                    ),
                }
            elif key == ord("c"):
                # Operates on the mark under the cursor rather than on edit
                # history, so a reloaded mark is editable without re-marking.
                for name in _marked_at(marks, idx):
                    mark = marks[name]
                    nxt = _CONFIDENCES.index(mark["confidence"]) + 1
                    mark["confidence"] = _CONFIDENCES[nxt % len(_CONFIDENCES)]
            elif key == ord("x"):
                for name in _marked_at(marks, idx):
                    del marks[name]
    finally:
        cap.release()
        cv2.destroyAllWindows()

    problem = _validate_order(marks)
    if problem is not None:
        print(f"refusing to save: {problem}")
        return 1
    if not marks:
        print("refusing to save: nothing marked")
        return 1

    events = {name: marks.get(name) for name in EVENT_ORDER}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "video_filename": video_path.name,
        "video_sha1": sha1_of_file(video_path),
        "source_fps": fps,
        "total_frames": total,
        "labelled_on": date.today().isoformat(),
        "notes": "",
        "events": events,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max_height", type=int, default=900)
    args = parser.parse_args()

    out = args.out or (
        Path("backend/tests/fixtures/events") / f"{args.video.stem}.json"
    )
    return label(args.video, out, args.max_height)


if __name__ == "__main__":
    raise SystemExit(main())
