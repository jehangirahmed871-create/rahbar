# ADR 0004: LLM Provider Selection and Data Governance

## Status
Accepted · 2026-11-03

## Context
RAHBAR processes academic records, university policies, and student inquiries. We need an LLM provider architecture that:
1. Prevents institutional policy data and student query telemetry from leaking to third-party public models without governance.
2. Supports local development and offline CI without incurring API billing costs or requiring internet network access.
3. Allows seamless switching between local execution (`ollama-host`, `ollama-incluster`) and hosted inference (`hosted`) via standard configuration.
4. Keeps API credentials out of Git history and manifests.

## Decision
We implemented a pluggable provider interface (`LLMProvider`) driven by the `LLM_PROVIDER` environment variable:
1. **Ollama as Primary Provider (`ollama-host`):**
   - In production and local minikube clusters, the system connects to Ollama running locally at `host.minikube.internal:11434` or in-cluster.
   - Using `llama3.2:3b` and `nomic-embed-text` keeps all prompt data, retrieved passages, and generated answers on-premises, satisfying data sovereignty requirements.
2. **Offline Stub Provider (`stub`):**
   - Used in unit tests, regression suites, and CI pipelines.
   - Requires zero network calls and deterministic behavior, ensuring CI runs fast without external flakiness.
3. **Secret Isolation for Hosted Fallback:**
   - When `LLM_PROVIDER=hosted` is enabled, the API key is injected strictly via a Kubernetes Secret (`rahbar-secret`) mapped to `LLM_API_KEY`.
   - Manifests in Git commit only `secret.placeholder.yaml` with dummy values.
4. **Constrained JSON Schema Generation:**
   - Regardless of provider, generation enforces strict Pydantic JSON schema output (`format="json"` on Ollama; response schema on hosted) to eliminate unparsed conversational prose.

## Options Considered
- **Public API Only (OpenAI/Groq):** Rejected due to external network dependency in air-gapped or offline CI environments and data privacy concerns.
- **Hardcoded Provider Implementations:** Rejected in favor of the factory pattern `get_provider(settings)` so switching models is a one-line ConfigMap change.

## Consequences
- **Positive:** Full data privacy on local cluster; clean offline testing in CI; no secrets committed to version control.
- **Negative:** Running `llama3.2:3b` on developer laptops requires ~4GB RAM allocated to Ollama.
