# AI Assistance and Tool Usage Disclosure

In accordance with course academic honesty guidelines, all AI tools utilized during the design, coding, evaluation, and deployment of RAHBAR are explicitly disclosed below.

---

## 1. Tool Usage Summary Table

| Tool | What It Wrote or Shaped | What We Changed Afterwards, and Why |
|---|---|---|
| **Google Antigravity (Gemini 3.8 Flash)** | Initial scaffolding of `extract.py`, `clean.py`, `chunk.py`, `embed.py`, and `store.py` stages; evaluation metric computation functions (`recall_at_k`, `citation_validity`, `summarise`, `gate`). | Tuned regex patterns in `chunk.py` to preserve multi-level clause numbering (`Clause 1.21`, `5.4`). Refactored `embed.py` to batch requests and implement deterministic unit fallback vectors for headless offline CI testing. |
| **GitHub Copilot** | Autocompletion of Pydantic models in `schemas.py` and repetitive test fixtures in `api/tests/test_starter.py` and `indexer/tests/test_pipeline.py`. | Verified all schema field validations manually; added explicit prompt-injection, abstention floor, and schema validation fallback test cases. |
| **Google Antigravity (Gemini 3.8 Flash)** | Kubernetes manifest drafting for Chroma StatefulSet, Redis Deployment + PVC, indexer Job, NetworkPolicies, and GitHub Actions CI/CD workflows. | Added `automountServiceAccountToken: false` to all service accounts, enforced `runAsNonRoot: true` with non-root UID 10001, adjusted memory limits based on `kubectl top pods` measurements, and configured strict zero-unavailability rolling update parameters. |

---

## 2. Jehangir Ahmed (Person A — Data and Quality) Section

### Work Scope
- Corpus selection and automated Polite Fetching (`corpus/fetch.py`, `corpus/sources.yaml`, `corpus/MANIFEST.md`).
- Document extraction, OCR pipeline, structure-aware chunking, vector embedding, and ChromaDB store (`indexer/pipeline/*`).
- Semantic Cache implementation with cosine similarity and never-cache rules (`api/app/services/cache.py`).
- Evaluation harness, 16-question golden set, evaluation runner (`eval/run_eval.py`, `eval/golden.yaml`, `eval/baseline.json`).
- Stateful Kubernetes resources (`chroma-statefulset.yaml`, `redis.yaml`, `indexer-job.yaml`, `reindex-cronjob.yaml`).
- Pipeline and Corpus documentation (`docs/CORPUS.md`, `docs/PIPELINE.md`, `docs/adr/0002-structure-aware-chunking.md`, `docs/adr/0003-semantic-cache-store.md`).
- Engineering notes for Q4 (indexer memory and OOMKill exit code 137) and Q5 (recall@5 by question type and hybrid retrieval).

### Reflections on AI Assistance
The AI agent expedited regex pattern prototyping for document chunking and Kubernetes YAML boilerplate. However, the AI initially produced fixed-window token slicing that severed the condition of university clauses from their sanctions. We intervened to enforce structure-aware hierarchical splitting (Section -> Paragraph -> Sentence) with a 50-token sliding overlap. We also introduced the deterministic fallback embedding generator so that the pipeline can run deterministically in CI without requiring a live Ollama host.

---

## 3. Muhammad Abdul Wadood (Person B — Service and Platform) Section

### Work Scope
- API query path, routes, error handling, JSON structured logging with request ID tracking, Prometheus metrics (`api/app/*`).
- Pluggable LLM provider abstractions (`ollama-host`, `ollama-incluster`, `hosted`, `stub`) and schema validation fallback retry mechanisms.
- Startup embedding model mismatch guardrail and health probes (`/healthz`, `/readyz`, `/metrics`).
- Stateless Kubernetes resources (`api-deployment.yaml`, `ui-deployment.yaml`, `ingress.yaml`, `hpa.yaml`, `networkpolicy.yaml`).
- Zero-downtime rolling update specifications (`maxSurge: 1`, `maxUnavailable: 0`, graceful preStop shutdown).
- CI/CD automation (`.github/workflows/ci.yml`, `.github/workflows/cd.yml`).
- Architectural documentation and failure analysis (`docs/ARCHITECTURE.md`, `docs/FAILURE-ANALYSIS.md`, `docs/adr/0001-indexer-as-job.md`, `docs/adr/0004-llm-provider-and-data-governance.md`).
- Engineering notes for Q1, Q2, Q3, Q6, and Q7.

### Reflections on AI Assistance
AI was used to accelerate the initial authoring of boilerplate Kubernetes manifests and the Pydantic schema validation structures. However, critical domain-specific fixes were required:
1. **NetworkPolicy Egress:** The initial AI-generated default-deny policy severed DNS lookup capability inside the cluster. I intervened to introduce explicit egress rules allowing UDP/TCP port 53 traffic to `kube-system` DNS pods and host-gateway egress on port 11434 for Ollama communication.
2. **Readiness Probe Guardrail:** The initial AI code did not verify vector dimension compatibility across restarts. I implemented the startup guardrail inside `/readyz` that interrogates Chroma collection metadata against the API's configured `EMBED_MODEL`, actively preventing silent vector corruption.
3. **CI Ephemeral Cluster Stability:** In the CI pipeline, image pulling within the ephemeral Kind cluster caused intermittent timeout failures. I updated the Kind workflow to pre-cache third-party container images and configure robust TCP readiness probes.
