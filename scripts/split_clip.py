"""Split a source clip into two frame-accurate parts.

Written for `jofra.mp4`, which contains a side-on delivery followed by a
front-on one: v1 analyses side-on only (CLAUDE.md section 3), so the two
camera angles have to become two clips before either is usable as a fixture.

Frame-accurate cutting requires re-encoding - a stream copy (`-c copy`) can
only cut on keyframes, which would silently move the boundary by up to a GOP
and invalidate every labelled frame index.

Encodes with MediaFoundation's `h264_mf` rather than libx264, which this
ffmpeg build does not ship. It takes a target bitrate instead of a CRF, so
the quality target is expressed as roughly four times the source bitrate
(~4.1 Mbit/s for jofra.mp4) - generous enough that the second generation
costs nothing that pose estimation can see.

Parts are written to temporary files and moved into place only after their
frame counts verify, so a failed encode cannot leave a half-written clip
where the input used to be. That also makes it safe for part one to reuse
the input's own filename.

Usage:
    uv run python scripts/split_clip.py data/raw_videos/jofra.mp4 \
        --through 370 \
        --out1 data/raw_videos/jofra.mp4 \
        --out2 data/raw_videos/jofra1.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_CODEC = "h264_mf"
VIDEO_BITRATE = "16M"


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{result.stderr[-2000:]}")


def frame_count(path: Path) -> int:
    """Decoded frame count. Counts frames rather than trusting the container's
    `nb_frames`, which is absent or wrong often enough to matter here."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def _encode(src: Path, dst: Path, trim: str) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            # setpts rebases timestamps to zero; without it the second part keeps
            # the source's PTS and reports a duration it does not have.
            "-vf",
            f"{trim},setpts=PTS-STARTPTS",
            "-c:v",
            VIDEO_CODEC,
            "-b:v",
            VIDEO_BITRATE,
            "-pix_fmt",
            "yuv420p",
            "-an",  # the pipeline never reads audio
            str(dst),
        ]
    )


def split(source: Path, through: int, out1: Path, out2: Path) -> int:
    total = frame_count(source)
    if not 0 <= through < total - 1:
        raise ValueError(
            f"--through must be in [0, {total - 2}] for a {total}-frame clip, "
            f"got {through}"
        )

    expected1 = through + 1
    expected2 = total - expected1
    print(f"{source.name}: {total} frames -> {expected1} + {expected2}")

    tmp1 = out1.with_suffix(".part1.tmp.mp4")
    tmp2 = out2.with_suffix(".part2.tmp.mp4")
    try:
        # trim end_frame is exclusive, start_frame inclusive.
        _encode(source, tmp1, f"trim=end_frame={expected1}")
        _encode(source, tmp2, f"trim=start_frame={expected1}")

        for tmp, expected in ((tmp1, expected1), (tmp2, expected2)):
            actual = frame_count(tmp)
            if actual != expected:
                raise RuntimeError(
                    f"{tmp.name}: expected {expected} frames, got {actual}"
                )

        out1.parent.mkdir(parents=True, exist_ok=True)
        out2.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp1), str(out1))
        shutil.move(str(tmp2), str(out2))
    finally:
        for tmp in (tmp1, tmp2):
            tmp.unlink(missing_ok=True)

    print(f"wrote {out1} ({expected1} frames, source frames 0-{through})")
    print(f"wrote {out2} ({expected2} frames, source frames {expected1}-{total - 1})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--through",
        type=int,
        required=True,
        help="last source frame index included in part one (inclusive)",
    )
    parser.add_argument("--out1", type=Path, required=True)
    parser.add_argument("--out2", type=Path, required=True)
    args = parser.parse_args()

    if not args.source.exists():
        print(f"no such file: {args.source}", file=sys.stderr)
        return 1
    return split(args.source, args.through, args.out1, args.out2)


if __name__ == "__main__":
    raise SystemExit(main())
