"""Tests for configuration management."""

import os
import tempfile
import pytest

from roadlens.config import AppConfig, get_config, reset_config


class TestAppConfig:
    def test_defaults(self):
        config = AppConfig()
        assert config.detector.confidence_threshold == 0.35
        assert config.dedup.merge_radius_meters == 12.0
        assert config.log_level == "INFO"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("ROADLENS_CONFIDENCE", "0.5")
        monkeypatch.setenv("ROADLENS_LOG_LEVEL", "DEBUG")
        config = AppConfig.from_env()
        assert config.detector.confidence_threshold == 0.5
        assert config.log_level == "DEBUG"

    def test_to_dict(self):
        config = AppConfig()
        d = config.to_dict()
        assert "detector" in d
        assert "dedup" in d
        assert d["detector"]["confidence_threshold"] == 0.35


class TestYamlConfig:
    def test_load_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("detector:\n  confidence_threshold: 0.5\nlog_level: DEBUG\n")
            f.flush()
            config = AppConfig.from_yaml(f.name)
        assert config.detector.confidence_threshold == 0.5
        assert config.log_level == "DEBUG"

    def test_load_nonexistent_returns_defaults(self):
        config = AppConfig.from_yaml("/nonexistent/path.yaml")
        assert config.detector.confidence_threshold == 0.35


class TestConfigSingleton:
    def teardown_method(self):
        reset_config()

    def test_get_config_returns_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2
