# CI fixtures

Three short Markdown documents for the `e2e-eval` job (section 6), so CI can index
a corpus and score recall in a minute instead of six. The institute is fictional.

| File | Why it is here |
| --- | --- |
| `academic-rules-2023.md` | The superseded version (75% attendance, CGPA 1.75, grade cap B) |
| `academic-rules-2026.md` | The current version (80%, 2.00, no cap), plus one planted injection line |
| `fees-2026.md` | Two tables and exact-term numbers |

The hostel topic is missing on purpose: `golden.fixture.yaml` asks about it and
expects the service to decline.

Each file starts with front matter (`doc_version`, `superseded`). The indexer's
`discover()` reads it when there is no `_index.json`, so your extract stage
needs to handle `.md` files as well as PDFs.

`batch-b/` contains the second evaluation batch: an exact-duplicate pair, a
near-duplicate pair, a truncated document, a missing-title document, a newer
academic-rules version, and a document containing an injected instruction.
The indexer quarantines failed inputs, so copy the fixtures to a disposable or
persistent data-lake directory before running ingestion; do not ingest directly
from this committed directory.
