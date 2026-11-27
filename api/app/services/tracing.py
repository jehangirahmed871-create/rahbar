"""Structured query traces persisted as JSON Lines."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("rahbar.trace")


def write_trace(trace: dict[str, Any], trace_dir: str) -> None:
    """Write a trace to stdout and append the same JSON record to the trace volume."""
    encoded = json.dumps(trace, ensure_ascii=False, separators=(",", ":"), default=str)
    log.info("query_trace", extra={"trace": trace, "trace_id": trace["trace_id"]})
    try:
        directory = Path(trace_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "queries.jsonl"
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, (encoded + "\n").encode("utf-8"))
        finally:
            os.close(descriptor)
    except OSError:
        log.exception("failed to persist query trace", extra={"trace_id": trace["trace_id"]})
