#!/usr/bin/env python3
"""Rebuild the retrieved context and prompt from a persisted trace ID."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_trace(trace_file: Path, trace_id: str) -> dict[str, Any]:
    if not trace_file.is_file():
        raise FileNotFoundError(f"trace file not found: {trace_file}")
    for line_number, line in enumerate(trace_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("trace_id") == trace_id:
            return record
    raise KeyError(f"trace_id {trace_id!r} not found in {trace_file}")


def format_trace(trace: dict[str, Any]) -> str:
    lines = [
        f"Trace ID: {trace['trace_id']}",
        f"Query: {trace.get('user_query', '')}",
        f"Cache: {trace.get('cache', 'unknown')}",
        "Context used for generation:",
    ]
    for passage in trace.get("generation_context", []):
        lines.extend([
            (
                f"- [{passage.get('chunk_id')}] {passage.get('source')} "
                f"({passage.get('document_version')}, page {passage.get('page')}): "
                f"{passage.get('section')}"
            ),
            str(passage.get("text", "")),
        ])
    prompt = trace.get("prompt")
    lines.extend(["", "Prompt:", prompt if prompt is not None else "(no prompt used; cached or stub response)"])
    lines.extend(["", "Prompt hash:", str(trace.get("prompt_hash"))])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_id")
    parser.add_argument("--traces", type=Path, default=Path("traces") / "queries.jsonl")
    args = parser.parse_args()
    try:
        trace = load_trace(args.traces, args.trace_id)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        print(f"trace lookup failed: {exc}", file=sys.stderr)
        return 1
    print(format_trace(trace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
