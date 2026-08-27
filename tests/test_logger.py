"""Tests for structured logging."""

import json
import logging
import pytest

from roadlens.logger import ContextLogger, get_logger


class TestContextLogger:
    def test_log_info(self, caplog):
        caplog.set_level(logging.INFO)
        logger = ContextLogger("test_info", level=logging.INFO)
        logger.info("test message")
        assert "test message" in caplog.text

    def test_json_format(self, caplog):
        caplog.set_level(logging.INFO)
        logger = ContextLogger("test_json", level=logging.INFO)
        logger.info("test message", key="value")
        assert "test message" in caplog.text

    def test_log_levels(self, caplog):
        caplog.set_level(logging.WARNING)
        logger = ContextLogger("test_levels", level=logging.WARNING)
        logger.debug("should not appear")
        logger.info("should not appear")
        logger.warning("should appear")
        assert "should not appear" not in caplog.text
        assert "should appear" in caplog.text

    def test_extra_data_in_stdout(self, capsys):
        logger = ContextLogger("test_extra", level=logging.INFO)
        logger.info("ticket created", ticket_id="RL-POT-2026-0001", severity="High")
        captured = capsys.readouterr()
        output = captured.out.strip()
        assert output.startswith("{")
        data = json.loads(output)
        assert data["message"] == "ticket created"
        assert data["data"]["ticket_id"] == "RL-POT-2026-0001"


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_module")
        assert isinstance(logger, ContextLogger)
