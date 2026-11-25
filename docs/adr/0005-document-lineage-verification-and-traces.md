# ADR 0005: Document lineage, verification, and query traces

## Status
Accepted

## Context
Assignment 03 requires per-document verification, incremental updates, version
rollback, and query-level traceability. The original chunk metadata identified
only the source filename and location, so it could not safely distinguish a
changed document from a new version or prove which document produced a chunk.

## Decision
- Add document ID, version ID, source hash, title, page count, and chunking
  settings to chunk metadata.
- Verify quality, completeness, recency, and uniqueness before chunking.
- Quarantine failed sources and preserve verified superseded source files in a
  versioned archive.
- Embed only changed chunk versions, delete stale/deleted versions, and allow a
  stored version's `superseded` filter state to be restored.
- Persist one JSONL trace per query, expose its ID in `AskResponse`, and record
  the exact prompt and safe context when a provider uses a prompt.

## Consequences
- Existing Chroma rows without document-version metadata are treated as legacy
  and removed during the next managed ingestion; current verified inputs are
  then written with the expanded schema.
- A local deterministic lexical feature-hashing fallback supports offline
  verification only. Its scores are not represented as measurements from the
  configured semantic embedding model.
- Synthetic golden questions carry an explicit human-review flag. The team
  must review at least eight before submission.
- The trace volume contains user queries and prompts and therefore must remain
  within the cluster's controlled storage and access boundary.
