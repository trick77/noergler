import json
import logging
from io import StringIO

import pytest
import structlog

from app.logging_config import (
    _make_service_metadata_processor,
    _rename_level,
    _strip_reserved,
    configure_logging,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    # Snapshot/restore root config so other tests aren't affected by
    # configure_logging's global side effects.
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    root.handlers = saved_handlers
    root.setLevel(saved_level)
    structlog.reset_defaults()


def test_strip_reserved_namespaces_splunk_keys():
    event_dict = {
        "msg": "x",
        "source": "jenkins",
        "host": "h1",
        "event": "evt",
        "kept": "yes",
    }
    out = _strip_reserved(None, "", event_dict)
    assert out["splunk_source"] == "jenkins"
    assert out["splunk_host"] == "h1"
    assert out["splunk_event"] == "evt"
    assert out["kept"] == "yes"
    assert "source" not in out
    assert "host" not in out
    assert "event" not in out


def test_rename_level_preserves_existing_log_level():
    out = _rename_level(None, "", {"level": "info", "log_level": "warning"})
    # If both are present, the explicit log_level wins and `level` is left
    # untouched (caller's intent — don't clobber).
    assert out["log_level"] == "warning"
    assert out["level"] == "info"


def test_rename_level_promotes_when_only_level_present():
    out = _rename_level(None, "", {"level": "info"})
    assert out["log_level"] == "info"
    assert "level" not in out


def test_service_metadata_processor_stamps_env_and_service():
    proc = _make_service_metadata_processor("intg")
    out = proc(None, "", {})
    assert out["service"] == "noergler"
    assert out["env"] == "intg"


def test_configure_logging_idempotent_single_handler():
    configure_logging("INFO", env="dev")
    configure_logging("INFO", env="dev")
    root = logging.getLogger()
    assert len(root.handlers) == 1


def test_configure_logging_emits_json_with_expected_schema(monkeypatch):
    configure_logging("INFO", env="intg")
    root = logging.getLogger()
    buffer = StringIO()
    # Redirect the single handler's stream to capture output.
    handler = root.handlers[0]
    monkeypatch.setattr(handler, "stream", buffer)

    log = structlog.stdlib.get_logger("test")
    log.info("riptide_reachable", team="alpha", url="https://r.example")

    line = buffer.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["msg"] == "riptide_reachable"
    assert payload["log_level"] == "info"
    assert payload["service"] == "noergler"
    assert payload["env"] == "intg"
    assert payload["team"] == "alpha"
    assert payload["url"] == "https://r.example"


def test_configure_logging_stdlib_bridge_emits_json(monkeypatch):
    configure_logging("INFO", env="dev")
    root = logging.getLogger()
    buffer = StringIO()
    handler = root.handlers[0]
    monkeypatch.setattr(handler, "stream", buffer)

    # A bog-standard stdlib logger (uvicorn-style) must come out as JSON.
    logging.getLogger("uvicorn.error").warning("uvi %s", "boom")
    payload = json.loads(buffer.getvalue().strip().splitlines()[-1])
    assert payload["msg"] == "uvi boom"
    assert payload["log_level"] == "warning"
    assert payload["service"] == "noergler"
