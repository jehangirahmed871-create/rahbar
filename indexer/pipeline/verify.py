"""Verify extracted documents before cleaning, chunking, or indexing them."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import IndexerSettings
from .embed import embed
from .models import Document

CHECK_NAMES = ("quality", "completeness", "recency", "uniqueness")


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"status": "pass" if self.passed else "fail", "reason": self.reason}


@dataclass
class VerificationRecord:
    document: Document
    path: Path
    sha256: str
    checks: dict[str, CheckResult]
    quarantine_path: str | None = None

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks.values())

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.document.source,
            "title": self.document.title,
            "document_id": self.document.document_id,
            "doc_version": self.document.doc_version,
            "page_count": len(self.document.pages),
            "sha256": self.sha256,
            "status": "pass" if self.passed else "fail",
            "checks": {name: self.checks[name].as_dict() for name in CHECK_NAMES},
            "quarantine_path": self.quarantine_path,
        }


def _document_text(document: Document) -> str:
    return "\n".join(page.text for page in document.pages)


def _version_key(version: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in re.findall(r"\d+", version))
    return values or (0,)


def _quality(document: Document) -> CheckResult:
    text = _document_text(document)
    if document.extraction_error:
        return CheckResult(False, f"extraction failed: {document.extraction_error}")
    if not text.strip():
        return CheckResult(False, "extracted text is empty")
    replacement_ratio = text.count("\ufffd") / max(len(text), 1)
    control_ratio = sum(ord(char) < 32 and char not in "\n\r\t" for char in text) / max(len(text), 1)
    letter_ratio = sum(char.isalpha() for char in text) / max(len(text), 1)
    if replacement_ratio > 0.01:
        return CheckResult(False, f"garbled extraction: {replacement_ratio:.1%} replacement characters")
    if control_ratio > 0.01:
        return CheckResult(False, f"garbled extraction: {control_ratio:.1%} control characters")
    if len(text.strip()) >= 40 and letter_ratio < 0.35:
        return CheckResult(False, f"garbled extraction: only {letter_ratio:.1%} alphabetic characters")
    return CheckResult(True, "extracted text is non-empty and passes corruption heuristics")


def _completeness(document: Document) -> CheckResult:
    missing: list[str] = []
    if not document.source.strip():
        missing.append("source")
    if not document.title.strip():
        missing.append("title")
    if not document.doc_version.strip() or document.doc_version.lower() == "unknown":
        missing.append("date/version")
    if not document.pages:
        missing.append("page count")
    if document.text_truncated:
        missing.append("text not truncated")
    if missing:
        return CheckResult(False, "missing or invalid: " + ", ".join(missing))
    return CheckResult(True, f"required metadata present; page count={len(document.pages)}; text not truncated")


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def verify_documents(
    documents: list[Document],
    settings: IndexerSettings,
    embedder=embed,
) -> list[VerificationRecord]:
    """Evaluate every required check; embedding similarity is used for near duplicates."""
    records: list[VerificationRecord] = []
    for document in documents:
        path = Path(document.source_path or document.source)
        document_text = _document_text(document)
        if not document.content_hash:
            document.content_hash = hashlib.sha256(document_text.encode("utf-8")).hexdigest()
        records.append(VerificationRecord(
            document=document,
            path=path,
            sha256=document.content_hash,
            checks={
                "quality": _quality(document),
                "completeness": _completeness(document),
                "recency": CheckResult(True, "recency evaluated against this ingestion batch"),
                "uniqueness": CheckResult(True, "uniqueness evaluated against this ingestion batch"),
            },
        ))

    by_document_id: dict[str, list[VerificationRecord]] = {}
    for record in records:
        by_document_id.setdefault(record.document.document_id, []).append(record)
    for versions in by_document_id.values():
        newest = max(versions, key=lambda item: _version_key(item.document.doc_version))
        newest_key = _version_key(newest.document.doc_version)
        for record in versions:
            is_newest = _version_key(record.document.doc_version) == newest_key
            if is_newest and record.document.superseded:
                record.checks["recency"] = CheckResult(
                    False, "latest document version is marked superseded"
                )
            elif not is_newest and not record.document.superseded:
                record.checks["recency"] = CheckResult(
                    False, f"newer version {newest.document.doc_version} exists but this version is active"
                )
            elif is_newest:
                record.checks["recency"] = CheckResult(
                    True, f"newest version {record.document.doc_version} detected"
                )
            else:
                record.checks["recency"] = CheckResult(
                    True, f"older version archived as superseded; newest is {newest.document.doc_version}"
                )

    valid_texts = [
        _document_text(record.document)[:100_000]
        for record in records
        if record.checks["quality"].passed and _document_text(record.document).strip()
    ]
    valid_records = [
        record for record in records
        if record.checks["quality"].passed and _document_text(record.document).strip()
    ]
    vectors = embedder(valid_texts, settings) if valid_texts else []
    vector_by_record = {id(record): vector for record, vector in zip(valid_records, vectors)}
    by_hash: dict[str, VerificationRecord] = {}
    for record in records:
        if not record.sha256:
            continue
        duplicate = by_hash.get(record.sha256)
        if duplicate:
            record.checks["uniqueness"] = CheckResult(
                False, f"exact duplicate of {duplicate.document.source} (SHA-256 match)"
            )
        else:
            by_hash[record.sha256] = record

    for left_index, left in enumerate(valid_records):
        if not left.checks["uniqueness"].passed:
            continue
        left_vector = vector_by_record.get(id(left))
        if left_vector is None:
            continue
        best_match: tuple[VerificationRecord, float] | None = None
        for right in valid_records[:left_index]:
            if not right.checks["uniqueness"].passed:
                continue
            if (left.document.document_id == right.document.document_id
                    and left.document.doc_version != right.document.doc_version):
                continue
            right_vector = vector_by_record.get(id(right))
            if right_vector is None:
                continue
            similarity = _cosine(left_vector, right_vector)
            if similarity >= settings.near_duplicate_threshold and (
                best_match is None or similarity > best_match[1]
            ):
                best_match = (right, similarity)
        if best_match:
            prior, similarity = best_match
            left.checks["uniqueness"] = CheckResult(
                False,
                f"near-duplicate of {prior.document.source}: embedding cosine similarity "
                f"{similarity:.4f} >= {settings.near_duplicate_threshold:.4f}",
            )

    return records


def write_report(
    records: list[VerificationRecord],
    output_dir: str,
    settings: IndexerSettings,
) -> Path:
    import json

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
    report_path = directory / f"verification-{timestamp}.json"
    payload = {
        "run_id": timestamp,
        "created_at": now.isoformat(),
        "embedding_model": settings.embed_model,
        "near_duplicate_cosine_threshold": settings.near_duplicate_threshold,
        "embedding_method": (
            "configured embedding provider; deterministic lexical feature-hashing fallback "
            "is used when the provider is unavailable and is logged"
        ),
        "summary": {
            "documents": len(records),
            "passed": sum(record.passed for record in records),
            "failed": sum(not record.passed for record in records),
        },
        "documents": [record.as_dict() for record in records],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path
