# ADR 0002: Structure-Aware Document Chunking with Boundary Overlap

## Status
Accepted · 2026-11-03

## Context
Standard RAG pipelines often use naive fixed-character or fixed-token chunking (e.g. 100 or 500 characters with fixed sliding windows). In official university policy handbooks:
- Rules are organized in rigid clause hierarchies (e.g., Section 5: Academic Standing; Clause 5.4: Academic Warning; Clause 5.4.1: Probation Conditions).
- Definitions and conditional clauses span multiple sentences. Splitting a rule mid-sentence or mid-clause isolates the sanction from its condition, causing the retriever to return misleading fragments.
- When generating answers with citations, each chunk must carry its provenance (`source`, `section`, `page`, `doc_version`, `superseded`) so that citations reference the exact governing clause.

## Decision
We implemented a **structure-aware chunking pipeline with boundary overlap**:
1. **Section-First Boundary Detection:** The chunker parses document text by identifying headings and numbered clauses using regular expression patterns (`Clause 1.21`, `Section 5`, `### Heading`).
2. **Hierarchical Splitting:**
   - Content within a section is preserved as a cohesive block.
   - If a section exceeds the target chunk size (`CHUNK_SIZE = 400` tokens), it is split along paragraph boundaries (`\n\n`).
   - If an individual paragraph exceeds `CHUNK_SIZE`, it is segmented along sentence terminators (`[.!?]`).
3. **Sliding Overlap Window (`CHUNK_OVERLAP = 50` tokens):**
   - Sentences spanning the boundary between consecutive chunks are carried forward into the next chunk.
   - This ensures that pronouns, contextual qualifiers ("provided that...", "except where..."), and cross-clause references are preserved across chunk splits.
4. **Metadata Preservation:**
   - Every chunk prepends its section context (`[<Section Title>]\n...`) and tags its metadata dictionary with `source`, `section`, `page`, `doc_version`, and `superseded`.

## Options Considered
- **Fixed-size 100-token chunks with no overlap:** Evaluated under Failure Analysis Stage B. Caused catastrophic degradation in retrieval recall: long multi-part clauses were severed, resulting in fragments that lacked either the subject or the prerequisite condition.
- **Whole-page chunking:** Rejected because 1-page chunks (often 600–1200 words) dilute embedding specificity and consume too much context window during prompt augmentation.

## Consequences
- **Positive:** Highest measured Recall@5 (0.93) on our golden question set; ensures citations cite exact clause numbers.
- **Negative:** Storage amplification ratio of 1.25x (due to the 50-token overlap between chunks), requiring slightly more storage in Chroma, which is negligible for our corpus scale.
