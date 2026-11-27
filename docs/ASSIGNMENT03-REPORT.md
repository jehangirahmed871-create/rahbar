# CS4032 (Software Construction and Development) — Assignment 03: RAGOps Report

**Course:** CS4032 — Software Construction and Development (Fall 2026)  
**Group Members:**
- **Muhammad Abdul Wadood** — Roll No: `24i-3055`
- **Jehangir Ahmed** — Roll No: `24i-3143`

**Repository:** `https://github.com/jehangirahmed871-create/rahbar` (branch: `assignment-03/day1`)

---

## 1. Mapping to RAGOps Lifecycle (Paper Figure 3)

| Figure 3 Stage | What Our Pipeline Does | File(s) That Do It | What Is Missing / Limitation |
|---|---|---|---|
| **Ingest** | Extracts text & tables from PDF/HTML/MD with per-page `pdfplumber` cache flush, OCR fallback (`pytesseract`), and HTML `<script>`/`<style>` stripping. | `indexer/pipeline/extract.py`, `indexer/pipeline/clean.py`, `indexer/pipeline/chunk.py` | No streaming/async document queue; batch-only PVC directory scan. |
| **Verify** | Runs 4 pre-chunking checks (quality, completeness, recency, uniqueness via SHA-256 & cosine ≥ 0.95); quarantines failures and writes JSON reports. | `indexer/pipeline/verify.py` (in `indexer/pipeline/run.py`) | Does not scan ingested text for prompt-injection payloads at index time (handled at query time instead). |
| **Update Data Lake** | Seeds `corpus-pvc` via `cp -rn`, moves failed docs to `/indexer-data/quarantine`, and archives superseded versions to `/indexer-data/archive`. | `k8s/base/reindex-cronjob.yaml`, `k8s/base/batch-b-seed-job.yaml`, `indexer/pipeline/run.py` | Manual `kubectl cp` or Job needed to inspect/prune archived files; no object-store lifecycle policy. |
| **Update Retrieval Sources** | Sole-writer indexer upserts only new/changed chunks by SHA-256, deletes orphan chunks by `document_id`, marks `superseded: true`, and supports `--restore`. | `indexer/pipeline/store.py`, `indexer/pipeline/run.py`, `k8s/base/indexer-restore-job.yaml` | Uses deterministic `offline-feature-hash-v1` (768-d lexical hashing) rather than dense semantic embeddings. |
| **Offline Testing** | Runs 3-level suite twice (`seed=42`): module (15 paraphrase vs 15 unrelated pairs), component (Recall@3/5, MRR, nDCG@5 + typo noise), end-to-end (faithfulness proxy, abstention, citations, p50/p95). | `eval/run_assignment03.py`, `eval/golden.jsonl`, `eval/embedding_pairs.jsonl`, `k8s/eval-job.yaml` | Faithfulness uses an exact-substring proxy (`expected_answer in answer`) rather than a full LLM-as-judge + human-labeled calibration. |
| **Live Testing** | Emits structured JSONL query traces to stdout & `traces-pvc`, enforces injection guardrails, exposes Prometheus `/metrics`, and summarizes trace windows. | `api/app/main.py`, `api/app/retrieval/chroma.py`, `eval/summarise_traces.py`, `eval/show_trace.py` | No automated live alerting (e.g., Alertmanager) on abstention spikes; summary is Job-triggered. |
| **Check Coverage** | Compares 40 live queries (`eval/live_queries.jsonl`) against `eval/golden.jsonl`, flags uncovered topics, and exits non-zero below threshold (`0.31`). | `eval/coverage.py`, `eval/live_queries.jsonl`, `eval/coverage-thresholds.json` | Coverage uses lexical feature-hash cosine similarity rather than dense semantic clustering. |

---

## 2. Five RAGOps Qualities (Paper Table 2)

| Quality | What We Measure | Threshold | Evidence (Observed Cluster / Script Output) |
|---|---|---|---|
| **Adaptability** | Release-gate comparison of candidate (`TOP_K=8`, `rahbar_candidate`) vs baseline (`TOP_K=5`) & ConfigMap threshold reload without image rebuild. | `max_recall5_drop ≤ 0.02`, `min_faithfulness ≥ 0.05`, `max_latency_p95 ≤ 30.0s` | `eval/results/part-c-gate.json` (`PASS`), `eval/results/part-c-bad-gate.json` (`FAIL` at `TOP_K=1`), `eval/results/threshold-change-fail.json` / `threshold-change-pass.json`. |
| **Monitorability** | Windowed trace summary (`top1_similarity`, `abstention_rate`, `cache_hit_rate`, stage p95 latencies, `injection_attempts_flagged`) & `/metrics`. | `p95 total ≤ 30s`; zero unhandled 5xx; injection attempts logged | `eval/results/trace-summary.json` (`trace_count: 314`, `top1_similarity p50: 0.5556`, `p95: 0.8571`, `abstention_rate: 0.9236`, `injection_attempts_flagged: 26`). |
| **Observability** | Structured JSONL trace per query (`stdout` + `/traces/queries.jsonl`) and Prometheus counters/histograms on `/metrics`. | Every `/api/ask` writes 1 JSONL trace with stage latencies & guardrail actions | `traces/queries.jsonl` (315 records), `kubectl port-forward svc/api 8000:8000` `/metrics` output. |
| **Traceability** | Reconstruction of exact retrieved chunks, prompt hash, model, collection, and filters from `trace_id`; data-version archiving. | `show_trace.py <trace_id>` reproduces full context & prompt | `eval/show_trace.py a6fb0b044c224a43b56fdd223529f42e` (post-undo trace) & `4086e916ec85413d8f238a63fce09938` (guardrail trace). |
| **Reliability** | Sole-writer Chroma invariant (`concurrencyPolicy: Forbid`), `maxUnavailable: 0` rolling updates, `kubectl rollout undo`, and PVC persistence. | Zero downtime on rollout/undo; `0` chunks embedded on unchanged corpus | `indexer-unchanged` (`chunks_embedded: 0`), `kubectl rollout undo deployment/api`, `indexer-restore-old-version` (`chunks: 9`). |

---

## 3. Experimental Results & Release Decision (Parts A–F)

### Part A: Verification, Incremental Update & Batch B
- **Base Corpus (`indexer-base`):** 9 documents processed → **8 passed**, **1 quarantined** (`undergraduate-academic-rules-aug-2023.pdf`, marked `superseded`). Total indexed chunks: **1,155** (630 existing + 525 newly embedded from `fee-structure-page.html` and `grading-page.html` after fixing HTML `<style>`/`<script>` and whitespace extraction).
- **Unchanged Re-run (`indexer-unchanged`):** 9 documents, 8 passed, 1 quarantined → **`chunks_embedded: 0`, `chunks_written: 0`**.
- **Batch B (`indexer-post-batch-b`, report `verification-20261127T085344.826540Z.json`):** 20 documents → **15 passed**, **5 quarantined**, **14 new chunks embedded**:
  - `exact-duplicate-b.md` → **Fail (uniqueness):** SHA-256 match with `exact-duplicate-a.md`.
  - `near-duplicate-b.md` → **Fail (uniqueness):** cosine similarity `0.9513 ≥ 0.9500` with `near-duplicate-a.md`.
  - `missing-title.md` → **Fail (completeness):** missing or invalid `title`.
  - `truncated-document.md` → **Fail (completeness):** `text_truncated` flag set.
  - `undergraduate-academic-rules-aug-2023.pdf` → **Fail (recency):** marked `superseded`.
- **Archive & Restore (`indexer-restore-old-version`):** Verified archived `academic-rules-2026.md` on `indexer-data-pvc` and restored **9 chunks** (`version: 2026`) into `rahbar`.

### Part B & C: Offline Evaluation (3 Levels), Stochasticity & Release Gate
- **Golden Set (`eval/golden.jsonl`):** 39 questions (27 answerable, 12 unanswerable, 12 hard; 10 human, 29 synthetic with 8 synthetic reviewed by hand).
- **Module Level (`offline-feature-hash-v1`):** Mean cosine similarity: paraphrase `0.1039` vs unrelated `0.0338` (**gap = `0.0701`**); embedding time `0.2056 ms/text`; determinism max absolute difference **`0.0`**.

| Run (Cluster Job) | Recall@3 | Recall@5 | MRR | nDCG@5 | Faithfulness Proxy | Abstention | Citation Validity | Latency p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Pre-Batch-B Baseline** (`part-b-before-batch-b.json`, Run 1) | 0.0370 | 0.1111 | 0.0352 | 0.0536 | 0.0370 | 1.0000 | 1.0000 | 0.857s / 2.730s |
| **Post-Batch-B Baseline** (`part-e3-after-batch-b.json`, Run 1) | 0.1481 | 0.2222 | 0.1216 | 0.1462 | 0.0741 | 1.0000 | 1.0000 | 0.601s / 1.591s |
| **Bad Candidate (`TOP_K=1`)** (`part-c-bad-candidate.json`, Run 1) | 0.1111 | 0.1111 | 0.1111 | 0.1111 | 0.1481 | 1.0000 | 1.0000 | 0.525s / 1.374s |
| **Good Candidate (`TOP_K=8`)** (`part-c-candidate.json`, Run 1) | **0.1852** | **0.2593** | **0.1648** | **0.1881** | **0.0741** | **1.0000** | **1.0000** | **0.746s / 1.490s** |

- **Noise Test (`TOP_K=8`):** Typo perturbations shifted Recall@3 `+0.0370`, Recall@5 `-0.0370`, MRR `+0.0265`, nDCG@5 `+0.0301`.
- **Stochasticity (`seed=42`, Run 2 − Run 1 on `TOP_K=8`):** All retrieval, faithfulness, abstention, and citation metrics had **0.0000** delta; only network/Groq API latency varied (`p50: -0.1285s`, `p95: -0.1502s`).
- **Release Decision (Part C):** **RELEASE `TOP_K=8` (`rahbar_candidate`)** and **REJECT `TOP_K=1`**. `TOP_K=8` improved Recall@5 from `0.2222` to `0.2593` (+3.7 pp) and MRR from `0.1216` to `0.1648`, kept abstention & citation validity at `1.0000`, and stayed well below the `30s` p95 latency ceiling (`1.490s`). `TOP_K=1` was blocked by the gate because Recall@5 dropped by `0.1111 > 0.02`. Controlled rollout (`maxUnavailable: 0`), simulated bad release (`TOP_K=1`), and `kubectl rollout undo deployment/api` succeeded; post-undo trace `a6fb0b044c224a43b56fdd223529f42e` verified live serving.

### Part D: Observability, Guardrails & Root Cause (D5)
- **Windowed Metrics (`trace-summary.json`):** 314 traces; `top1_similarity` mean `0.5463`, p50 `0.5556`, p95 `0.8571`; `abstention_rate` `0.9236`; `cache_hit_rate` `0.0223`; stage p95 latencies: `cache_lookup 0.096s`, `query_embedding 0.001s`, `retrieval 0.820s`, `generation 2.500s`, `total 2.653s`; **`injection_attempts_flagged: 26`**.
- **Guardrail Trace (D4):** Trace `4086e916ec85413d8f238a63fce09938` recorded two guardrail actions: `query` flagged as untrusted & excluded from semantic cache lookup, and `retrieved_context` stripped `injected-support-notice@2026#student-support-notice#p1#c0` before prompt construction.
- **Root Cause from Trace Only (D5 — 5 lines, trace `9690d7906e3546a4810dd388e13887d4`):**
  1. Query `"What is the minimum percentage of attendance required to be eligible for final examinations?"` abstained (`answered_from_context: false`).
  2. Trace `collection` was `rahbar` using `offline-feature-hash-v1` with metadata filter `{"superseded": false}`.
  3. `retrieved_chunks` in the trace was empty (`[]`), meaning zero chunks survived retrieval/thresholding.
  4. Because no context chunks were passed to Groq, the API safely returned its fallback/abstention response rather than hallucinating.
  5. Cause: lexical feature-hash mismatch / empty retrieval set on the `rahbar` collection under `superseded: false` (no chunk scored above cutoff).

### Part E: Query Coverage & Data Drift
- **Coverage (`eval/coverage.py`, threshold `0.31`):** Before adding 6 uncovered-topic golden questions: **16/28 in-domain covered (`57.14%`)**, mean similarity `0.3043`. After adding 6 golden questions (`coverage-after.json`): **18/28 in-domain covered (`64.29%`)**, mean similarity `0.3384`, 0 out-of-domain false positives.
- **Data Drift (E3):** Ingesting Batch B caused Recall@5 to jump from `0.1111` to `0.2222` (+0.1111) and MRR from `0.0352` to `0.1216` because Batch B supplied the missing 2026/2027 academic rules, library, deferral, and fee-payment documents referenced by Batch B golden questions.

### Part F: Measured Kubernetes Resource Usage
| Component | Steady / Peak Measured CPU | Steady / Peak Measured Memory | Configured Request (CPU / Mem) | Configured Limit (CPU / Mem) |
|---|---:|---:|---:|---:|
| `api` Deployment | 20m | 335 MiB | 100m / 384Mi | 500m / 768Mi |
| `chroma-0` StatefulSet | 15m | 562 MiB | 200m / 512Mi | 1 / 1Gi |
| `redis` Deployment | 16m | 7 MiB | 25m / 32Mi | 250m / 128Mi |
| `indexer` Job (Peak) | 982m | 489 MiB | 250m / 512Mi | 1 / 1536Mi |
| Minikube Node Total | 521m (26%) | 2127 MiB (54%) | — | — |

---

## 4. Short Answers

- **(a) Verification checks on Batch B:** `uniqueness` and `completeness` each caught 2 failures (`exact-duplicate-b.md`, `near-duplicate-b.md`, `missing-title.md`, `truncated-document.md`), and `recency` caught 1 (`undergraduate-academic-rules-aug-2023.pdf`). We did **not** catch `injected-instruction.md` at ingestion time; adding a regex/classifier instruction-injection scan during `verify` would catch it before indexing.
- **(b) Module vs Component/End-to-End discrepancy:** At the module level, determinism was `0.0` and paraphrase vs unrelated gap was positive (`0.0701`), yet component Recall@5 was only `0.2593` and faithfulness proxy was `0.0741` because 768-d feature hashing lacks semantic synonym matching across long regulatory PDFs.
- **(c) Archive vs Delete:** We **archived** superseded versions to `/indexer-data/archive/<doc_id>/<version>/` and tagged chunks `superseded: true` in Chroma (while deleting only orphan/changed chunks of updated docs). Deleting old versions outright would break `--restore` rollback and auditability of historical policy answers.
- **(d) Least trusted metric:** `faithfulness_expected_answer_proxy` (`0.0741`), because it checks exact normalized substring containment (`expected_answer in answer`), penalizing valid paraphrased or unit-formatted answers.
- **(e) Hardest paper idea to apply:** Continuous automated online evaluation + dense neural reranking on a 3 GiB local Minikube node; we adapted by keeping Groq for generation, using lightweight deterministic feature hashing (`offline-feature-hash-v1`), and scaling `api`/`ui` to 0 during heavy `indexer` Jobs.
- **(f) Kubernetes object responsibilities:** `CronJob/reindex` (`concurrencyPolicy: Forbid`) + `Job/indexer-*` enforce the sole-writer invariant; `PersistentVolumeClaim` (`corpus-pvc`, `indexer-data-pvc`, `eval-results-pvc`, `traces-pvc`, `chroma-data`) persists state across pod restarts; `ConfigMap/eval-thresholds` decouples release-gate thresholds from images; `Deployment/api` (`maxUnavailable: 0`) handles zero-downtime rollouts and `kubectl rollout undo`.

---

## 5. Team Contributions & AI Usage Statement

| Member | Roll No | Responsibilities |
|---|---|---|
| **Muhammad Abdul Wadood** | `24i-3055` | API tracing & injection guardrails, Kubernetes manifests (`k8s/base`, `overlays/dev`, Jobs, ConfigMaps), rollout/undo, and CI/cluster execution. |
| **Jehangir Ahmed** | `24i-3143` | Indexer verification stage, incremental Chroma store & archive/restore, HTML/PDF memory fixes, golden/live query datasets, and evaluation/coverage scripts. |

**AI Usage Statement:** AI coding assistants (Antigravity / Gemini) were used to help draft boilerplate Kubernetes manifests, refactor HTML/PDF extraction memory handling, and format this report. All pipeline runs, cluster deployments, and metric tables were executed on our own Minikube cluster and verified directly from committed repository outputs.

---

## Appendix: Terminal Screenshots (Placeholders — Does Not Count Toward 8-Page Limit)

> **Instructions:** Replace each placeholder box below with an **uncropped** screenshot showing both the terminal command and its full output from your Minikube cluster. Label each screenshot clearly (`A1`, `A2`, `B`, `C1`, `C2`, `D1`, `D2`, `E1`, `E2`, `F1`–`F5`).

### Screenshot A1 — Batch B Verification Report (`indexer-post-batch-b`)
- **Command:** `kubectl logs job/indexer-post-batch-b -n rahbar` (and verification JSON summary showing pass/fail per check with reasons)
- `[ INSERT SCREENSHOT A1 HERE ]`

### Screenshot A2 — Second Ingestion Run on Unchanged Corpus (`chunks_embedded: 0`)
- **Command:** `kubectl logs job/indexer-unchanged -n rahbar`
- `[ INSERT SCREENSHOT A2 HERE ]`

### Screenshot B — Three-Level Evaluation Summary (`eval-post-batch-b`)
- **Command:** `kubectl logs job/eval-post-batch-b -n rahbar`
- `[ INSERT SCREENSHOT B HERE ]`

### Screenshot C1 — Release Gate Failing (`TOP_K=1` Bad Candidate)
- **Command:** `kubectl logs job/eval-candidate -n rahbar` (from `TOP_K=1` run showing `Release gate: FAIL`)
- `[ INSERT SCREENSHOT C1 HERE ]`

### Screenshot C2 — Release Gate Passing (`TOP_K=8` Good Candidate)
- **Command:** `kubectl logs job/eval-candidate -n rahbar` (from `TOP_K=8` run showing `Release gate: PASS`)
- `[ INSERT SCREENSHOT C2 HERE ]`

### Screenshot D1 — One Full Trace Record
- **Command:** `Get-Content traces\queries.jsonl -Tail 1`
- `[ INSERT SCREENSHOT D1 HERE ]`

### Screenshot D2 — Trace Tool Rebuilding Context & Prompt by Trace ID
- **Command:** `python eval\show_trace.py a6fb0b044c224a43b56fdd223529f42e --traces traces\queries.jsonl`
- `[ INSERT SCREENSHOT D2 HERE ]`

### Screenshot E1 & E2 — Coverage Before & After Adding 6 New Golden Questions
- **Command (Before):** `Get-Content eval\results\coverage-before.json | Select-Object -First 20` (or `python eval\coverage.py` summary)
- **Command (After):** `python eval\coverage.py --out eval\results\coverage-after.json`
- `[ INSERT SCREENSHOT E1 & E2 HERE ]`

### Screenshot F1 — Cluster Resources Overview
- **Command:** `kubectl get all,pvc,configmap,cronjob,job -n rahbar`
- `[ INSERT SCREENSHOT F1 HERE ]`

### Screenshot F2 — Logs of Completed Ingestion Job & Completed Evaluation Job
- **Command:** `kubectl logs job/indexer-base -n rahbar` and `kubectl logs job/eval-baseline -n rahbar`
- `[ INSERT SCREENSHOT F2 HERE ]`

### Screenshot F3 — Gate Result Changing After ConfigMap Edit (Without Rebuilding Image)
- **Command:** `kubectl apply -f k8s\eval-thresholds-fail.yaml` → `kubectl logs job/release-gate -n rahbar` → `kubectl apply -f k8s\base\eval-thresholds-configmap.yaml` → `kubectl logs job/release-gate -n rahbar`
- `[ INSERT SCREENSHOT F3 HERE ]`

### Screenshot F4 — `kubectl rollout status` & `kubectl rollout undo` + Post-Undo Trace
- **Command:** `kubectl rollout status deployment/api -n rahbar`, `kubectl rollout undo deployment/api -n rahbar`, and `python eval\show_trace.py <post-undo-trace-id> --traces traces\queries.jsonl`
- `[ INSERT SCREENSHOT F4 HERE ]`

