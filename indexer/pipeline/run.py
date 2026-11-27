"""The indexer Job's entry point. It runs once and exits (section 4.2).

    python -m pipeline.run --dummy    # week 1: three fake chunks, no model needed
    python -m pipeline.run            # the real pipeline: extract, clean, chunk, embed, store

Exit code 0 means the index was written; anything else fails the Job, and
Kubernetes retries it up to the Job's backoffLimit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .config import IndexerSettings, get_settings
from .models import Chunk, Document
from .verify import verify_documents, write_report


def log(msg: str, **fields: object) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "logger": "rahbar.indexer", "msg": msg, **fields}), flush=True)


def _front_matter(path: Path) -> dict[str, str]:
    """Read `key: value` lines between leading '---' markers (used by eval/fixtures)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def discover(corpus_dir: str) -> list[tuple[Path, str, bool]]:
    """List (path, doc_version, superseded) for every document to index.

    corpus/fetch.py writes _index.json next to the files it downloads; use it when
    present. Otherwise read Markdown front matter (the CI fixtures work this way).
    """
    root = Path(corpus_dir)
    index = root / "_index.json"
    indexed = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
    by_file = {str(entry.get("file")): entry for entry in indexed}
    found: list[tuple[Path, str, bool]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(
            part.lower() in {"quarantine", "archive", "verification"}
            for part in path.relative_to(root).parts[:-1]
        ):
            continue
        if path.suffix.lower() not in {".pdf", ".md", ".html", ".txt"}:
            continue
        if path.name.lower().startswith("readme"):
            continue
        relative = str(path.relative_to(root))
        entry = by_file.get(relative) or by_file.get(path.name)
        meta = _front_matter(path) if path.suffix.lower() in {".md", ".txt"} else {}
        version = str((entry or {}).get("doc_version", meta.get("doc_version", "unknown")))
        superseded = (entry or {}).get("superseded", meta.get("superseded", "false"))
        found.append((path, version, superseded is True or str(superseded).lower() == "true"))
    return found


def _metadata_for(path: Path, corpus_root: Path) -> dict[str, str | bool]:
    index = corpus_root / "_index.json"
    if index.exists():
        entries = json.loads(index.read_text(encoding="utf-8"))
        for entry in entries:
            if entry.get("file") in {path.name, str(path.relative_to(corpus_root))}:
                return {
                    "title": str(entry.get("title") or entry.get("name") or path.stem),
                    "document_id": str(entry.get("document_id") or path.stem),
                    "text_truncated": bool(entry.get("text_truncated", False)),
                }
    meta = _front_matter(path) if path.suffix.lower() in {".md", ".txt"} else {}
    title = meta.get("title", "")
    document_id = meta.get("document_id") or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    truncated = meta.get("text_truncated", meta.get("truncated", "false")).lower() == "true"
    return {"title": title, "document_id": document_id, "text_truncated": truncated}


def _wait_for_chroma(settings: IndexerSettings, timeout: int = 60) -> None:
    # In local mode, no server to wait for
    if os.environ.get("CHROMA_LOCAL_DIR", ""):
        log("using local chroma (no server needed)")
        return

    from .store import get_client

    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            client = get_client(settings)
            client.heartbeat()
            log("chroma ready", host=settings.chroma_host, port=settings.chroma_port)
            return
        except Exception as exc:  # noqa: BLE001 - Chroma clients expose several transport errors
            last_err = exc
            time.sleep(2)
    log("warning: chroma did not respond to heartbeat within timeout", error=str(last_err))


def run_pipeline(settings: IndexerSettings) -> int:
    from .chunk import chunk
    from .clean import clean
    from .embed import embed
    from .extract import extract
    from .store import store

    _wait_for_chroma(settings)
    root = Path(settings.corpus_dir)
    sources = discover(settings.corpus_dir)
    if not sources:
        raise SystemExit(f"no documents found in {settings.corpus_dir}")

    extracted: list[Document] = []
    for path, doc_version, superseded in sources:
        metadata = _metadata_for(path, root)
        try:
            document = extract(
                str(path),
                doc_version,
                superseded,
                title=str(metadata["title"]),
                document_id=str(metadata["document_id"]),
                text_truncated=bool(metadata["text_truncated"]),
            )
        except Exception as exc:  # noqa: BLE001 - isolate document extraction failures for quarantine
            document = Document(
                source=path.name,
                doc_version=doc_version,
                superseded=superseded,
                title=str(metadata["title"]),
                document_id=str(metadata["document_id"]),
                extraction_error=f"{type(exc).__name__}: {exc}",
                source_path=str(path.resolve()),
            )
            try:
                document.content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                document.content_hash = ""
        extracted.append(document)

    records = verify_documents(extracted, settings)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for record in records:
        if record.passed and record.document.superseded:
            source_path = Path(record.document.source_path)
            archive_path = (
                Path(settings.archive_dir or root / "archive")
                / record.document.document_id
                / record.document.doc_version
                / source_path.name
            )
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, archive_path)
        elif not record.passed:
            source_path = Path(record.document.source_path)
            if source_path.is_file():
                quarantine_path = (
                    Path(settings.quarantine_dir or root / "quarantine")
                    / run_id
                    / f"{record.sha256[:12]}-{source_path.name}"
                )
                quarantine_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(quarantine_path))
                record.quarantine_path = str(quarantine_path)

    report_path = write_report(
        records,
        settings.verification_dir or str(root / "verification"),
        settings,
    )
    valid_records = [record for record in records if record.passed]
    all_chunks: list[Chunk] = []
    for record in valid_records:
        document = clean(record.document)
        pieces = chunk(document, settings.chunk_size, settings.chunk_overlap)
        log("chunked", source=document.source, chunks=len(pieces),
            document_id=document.document_id, version=document.doc_version)
        all_chunks.extend(pieces)

    desired_versions = {
        f"{record.document.document_id}@{record.document.doc_version}"
        for record in valid_records
    }
    from .store import plan_incremental

    chunks_to_embed = plan_incremental(all_chunks, settings)
    vectors = embed([item.text for item in chunks_to_embed], settings)
    written = store(
        chunks_to_embed,
        vectors,
        settings,
        desired_document_version_ids=desired_versions,
    )
    log(
        "verification and incremental indexing complete",
        verification_report=str(report_path),
        documents=len(records),
        passed=len(valid_records),
        quarantined=len(records) - len(valid_records),
        chunks_embedded=len(chunks_to_embed),
        chunks_written=written,
    )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RAHBAR index, then exit.")
    parser.add_argument("--dummy", action="store_true", help="write three fake chunks")
    parser.add_argument("--restore", nargs=2, metavar=("DOCUMENT_ID", "VERSION"),
                        help="restore an archived document version for retrieval")
    args = parser.parse_args()
    settings = get_settings()
    log("indexer starting", dummy=args.dummy, embed_model=settings.embed_model,
        collection=settings.collection, chroma=f"{settings.chroma_host}:{settings.chroma_port}",
        pod=os.environ.get("HOSTNAME", "-"))
    started = time.perf_counter()
    try:
        if args.restore:
            from .store import restore_version

            count = restore_version(args.restore[0], args.restore[1], settings)
        elif args.dummy:
            from .dummy import write_dummy_index
            from .store import get_client

            count = write_dummy_index(settings, get_client(settings))
        else:
            count = run_pipeline(settings)
    except NotImplementedError as exc:
        log("indexer stage not implemented yet", error=str(exc))
        return 2
    log("indexer finished", chunks=count, seconds=round(time.perf_counter() - started, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
