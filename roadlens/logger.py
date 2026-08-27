"""
RoadLens AI — Structured Logging
=================================
Provides consistent, JSON-structured logging across all modules.
Every log line includes: timestamp, level, module, message, and optional
context fields (ticket_id, defect_type, etc.) for easy filtering.
"""

from __future__ import annotations

import logging
import sys
import json
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines for machine-readable output."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class ContextLogger:
    """Logger that supports attaching contextual data to each log call."""

    def __init__(self, name: str, level: int = logging.INFO, json_format: bool = True):
        self.name = name
        self.logger = logging.getLogger(f"roadlens.{name}")
        self.logger.setLevel(level)

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONFormatter() if json_format else logging.Formatter(
                "%(asctime)s [%(levelname)s] %(module)s: %(message)s"
            ))
            self.logger.addHandler(handler)

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        # makeRecord() derives record.module from the pathname it is given.
        # Passing "" here left every log line with an empty "module" field,
        # which defeats the whole point of filtering logs by module.
        record = self.logger.makeRecord(
            self.logger.name, level, f"{self.name}.py", 0, message, (), None
        )
        if kwargs:
            record.extra_data = kwargs
        self.logger.handle(record)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, message, **kwargs)


def get_logger(name: str) -> ContextLogger:
    """Get a logger scoped to a module name, honouring config.yaml /
    ROADLENS_LOG_LEVEL / ROADLENS_LOG_FORMAT."""
    level = logging.INFO
    json_format = True
    try:
        from .config import get_config

        cfg = get_config()
        level = getattr(logging, str(cfg.log_level).upper(), logging.INFO)
        json_format = str(cfg.log_format).lower() == "json"
    except Exception:  # config is optional — never let logging setup fail
        pass
    return ContextLogger(name, level=level, json_format=json_format)
