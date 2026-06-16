from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _StrictBase(BaseModel):
    """Base config model. `extra='forbid'` rejects unknown YAML keys."""

    model_config = ConfigDict(extra="forbid")


class AppSettings(_StrictBase):
    log_level: str
    experiment_name: str


class DataSettings(_StrictBase):
    video_dir: Path
    output_dir: Path
    formats: list[str]


class VideoSettings(_StrictBase):
    frame_stride: int = Field(ge=1)
    start_timestamp_ms: int = Field(ge=0)
    max_input_height: int = Field(ge=1)
    min_fps_warn: float = Field(gt=0)
    min_fps_reject: float = Field(gt=0)
    min_bowler_frame_fraction_warn: float = Field(gt=0, lt=1)
    min_bowler_frame_fraction_reject: float = Field(gt=0, lt=1)
    min_aspect_ratio: float = Field(gt=0)
    max_background_pan_px_per_frame: float = Field(ge=0)


class ConfidenceSettings(_StrictBase):
    detection: float = Field(ge=0, lt=1)
    presence: float = Field(ge=0, lt=1)
    tracking: float = Field(ge=0, lt=1)


class ModelSettings(_StrictBase):
    path: Path
    num_poses: int = Field(ge=1)
    confidences: ConfidenceSettings


class QASettings(_StrictBase):
    max_no_pose_ratio: float = Field(ge=0, le=1)
    max_pose_gap: int = Field(ge=0)


class PoseSettings(_StrictBase):
    model: ModelSettings
    qa: QASettings


class Settings(_StrictBase):
    app: AppSettings
    data: DataSettings
    video: VideoSettings
    pose: PoseSettings


def load_settings(path: str | Path = "configs/config.yaml") -> Settings:
    with open(path, encoding="utf-8") as f:
        data: Any = yaml.safe_load(f)
    return Settings.model_validate(data)
