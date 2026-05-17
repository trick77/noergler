"""Splunk-friendly JSON logging via structlog.

Bridges stdlib loggers (uvicorn, sqlalchemy, httpx, ...) through the same
structlog pipeline so every container log line is one JSON object with the
same schema. Splunk-reserved field names are kept off the wire to avoid
silent overwrite by the forwarder.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

# Field names Splunk treats as built-in input metadata. Using them as JSON
# keys causes Splunk to silently overwrite our values with the forwarder's
# (`source`, `host`, `index`, `sourcetype`, `time`/`_time`/`_raw`), or to
# double-extract (`event`). Keep them off the wire.
_SPLUNK_RESERVED = frozenset(
    {"source", "sourcetype", "host", "index", "time", "_time", "_raw", "event"}
)

_SERVICE_NAME = "noergler"


def _make_service_metadata_processor(env: str):
    def _add(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        del _logger, _name
        event_dict.setdefault("service", _SERVICE_NAME)
        event_dict.setdefault("env", env)
        return event_dict

    return _add


def _rename_level(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    del _logger, _name
    if "level" in event_dict and "log_level" not in event_dict:
        event_dict["log_level"] = event_dict.pop("level")
    return event_dict


def _strip_reserved(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    del _logger, _name
    # `msg` is set by EventRenamer, `log_level` by _rename_level — both
    # already safe. This pass catches accidental kwargs (e.g. `source="jenkins"`)
    # and namespaces them under `splunk_<name>` so the value survives without
    # colliding with Splunk's built-in field of the same name.
    for key in list(event_dict.keys()):
        if key in _SPLUNK_RESERVED:
            event_dict[f"splunk_{key}"] = event_dict.pop(key)
    return event_dict


def configure_logging(level: str = "INFO", env: str = "dev") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _make_service_metadata_processor(env),
        structlog.processors.EventRenamer("msg"),
        _rename_level,
        _strip_reserved,
    ]

    # Bridge: route stdlib logs through the same JSON pipeline so Splunk
    # sees one schema only.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level)

    # uvicorn's access log becomes redundant once our middleware emits
    # http_request; keep only warnings/errors from it. Same for chatty
    # client libraries.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
