"""infra/logging_setup.py

Structured JSON logger. Secrets never logged.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Optional


class JsonFormatter(logging.Formatter):
    SENSITIVE = frozenset({"key", "private_key", "secret", "passphrase", "token", "password"})

    def format(self, record: logging.LogRecord) -> str:
        doc = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        # Scrub any extra fields that might carry secrets
        for k, v in record.__dict__.items():
            if k.startswith("_") or k in (
                "msg", "args", "levelname", "name", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            ):
                continue
            if any(s in k.lower() for s in self.SENSITIVE):
                doc[k] = "***"
            else:
                doc[k] = v
        return json.dumps(doc, default=str)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_console: bool = False,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    if json_console:
        ch.setFormatter(JsonFormatter())
    else:
        ch.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
    root.addHandler(ch)

    # Optional file handler (JSON), rotated to bound disk footprint
    if log_file:
        fh = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "websockets", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
