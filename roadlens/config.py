"""
RoadLens AI — Configuration Management
=======================================
Centralizes all tunable parameters in one place. Replaces hardcoded
constants with a loadable configuration that can be overridden via
environment variables or a YAML file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class DetectorConfig:
    confidence_threshold: float = 0.35
    model_repo: str = "keremberke/yolov8s-pothole-segmentation"
    model_file: str = "best.pt"
    video_sample_every_n_frames: int = 15


@dataclass
class DedupConfig:
    merge_radius_meters: float = 12.0


@dataclass
class SeverityConfig:
    levels: list[tuple] = field(default_factory=lambda: [
        (0.060, 4, "Critical", 24, 18000),
        (0.025, 3, "High", 72, 9000),
        (0.008, 2, "Medium", 168, 4500),
        (0.000, 1, "Low", 336, 1800),
    ])
    department_routing: dict = field(default_factory=lambda: {
        "pothole": "Road Maintenance Division",
        "crack": "Road Maintenance Division",
        "manhole": "Storm Water & Drainage",
        "zebra_crossing": "Traffic Engineering Cell",
        "footpath": "Footpath & Streetscape Wing",
    })
    default_department: str = "Road Maintenance Division"


@dataclass
class PredictiveConfig:
    recurrence_radius_meters: float = 15.0
    monitoring_window_days: int = 90
    high_recurrence_threshold: int = 2
    critical_recurrence_threshold: int = 4
    heatmap_grid_size_km: float = 0.5
    risk_segment_radius_meters: float = 500.0


@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    predictive: PredictiveConfig = field(default_factory=PredictiveConfig)
    db_path: str = "roadlens.db"
    log_level: str = "INFO"
    log_format: str = "json"

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build a config from defaults, then apply environment overrides."""
        return cls().apply_env()

    def apply_env(self) -> "AppConfig":
        """Layer ROADLENS_* environment variables on top of this config.

        Environment always wins over the YAML file — that is what makes
        `docker run -e ROADLENS_DB_PATH=...` work against a baked-in
        config.yaml.
        """
        if os.getenv("ROADLENS_CONFIDENCE"):
            self.detector.confidence_threshold = float(os.getenv("ROADLENS_CONFIDENCE"))
        if os.getenv("ROADLENS_MERGE_RADIUS"):
            self.dedup.merge_radius_meters = float(os.getenv("ROADLENS_MERGE_RADIUS"))
        if os.getenv("ROADLENS_RECURRENCE_RADIUS"):
            self.predictive.recurrence_radius_meters = float(os.getenv("ROADLENS_RECURRENCE_RADIUS"))
        if os.getenv("ROADLENS_DB_PATH"):
            self.db_path = os.getenv("ROADLENS_DB_PATH")
        if os.getenv("ROADLENS_LOG_LEVEL"):
            self.log_level = os.getenv("ROADLENS_LOG_LEVEL")
        if os.getenv("ROADLENS_LOG_FORMAT"):
            self.log_format = os.getenv("ROADLENS_LOG_FORMAT")
        return self

    @classmethod
    def _from_dict(cls, data: dict) -> "AppConfig":
        config = cls()
        if "detector" in data:
            for k, v in data["detector"].items():
                if hasattr(config.detector, k):
                    setattr(config.detector, k, v)
        if "dedup" in data:
            for k, v in data["dedup"].items():
                if hasattr(config.dedup, k):
                    setattr(config.dedup, k, v)
        if "predictive" in data:
            for k, v in data["predictive"].items():
                if hasattr(config.predictive, k):
                    setattr(config.predictive, k, v)
        if "severity" in data:
            for k, v in data["severity"].items():
                if hasattr(config.severity, k):
                    setattr(config.severity, k, v)
        if "db_path" in data:
            config.db_path = data["db_path"]
        if "log_level" in data:
            config.log_level = data["log_level"]
        if "log_format" in data:
            config.log_format = data["log_format"]
        return config

    def to_dict(self) -> dict:
        return asdict(self)


_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        config_path = os.getenv("ROADLENS_CONFIG", "config.yaml")
        if Path(config_path).exists():
            # YAML is the base; environment variables still override it.
            _config = AppConfig.from_yaml(config_path).apply_env()
        else:
            _config = AppConfig.from_env()
    return _config


def reset_config() -> None:
    global _config
    _config = None
