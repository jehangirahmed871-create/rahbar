# ADR 0001: Indexing as an Independent Kubernetes Batch Job

## Status
Accepted · 2026-11-03

## Author
Muhammad Abdul Wadood (Person B — Service & Platform)

## Context
When deploying a RAG application on Kubernetes, it is tempting to run document extraction and vector indexing inside the API pod's startup lifecycle (e.g. within FastAPI's `lifespan` or an `initContainer`). However, in a multi-replica production deployment, this creates critical failure modes:
1. **Concurrent Writer Collisions:** If the API Deployment scales to 3 replicas, three pods start simultaneously and race to write to the same single Chroma PersistentVolumeClaim (PVC). Chroma's underlying SQLite/HNSW storage does not support multi-process concurrent write access across independent containers; concurrent writes corrupt the index and lock the database.
2. **Re-indexing on Every Restart:** Every rolling restart, OOMKill recovery, or node eviction would trigger full document extraction, OCR, and embedding generation across the 279-page corpus, incurring immense compute costs and startup delays.
3. **Readiness Probe Failure:** If indexing takes 3–5 minutes during startup, Kubernetes liveness probes either timeout and kill the pod prematurely (triggering a CrashLoopBackOff restart storm), or the pod marks itself ready before the index finishes, returning empty retrievals and false abstentions to users.
4. **Configuration Decoupling:** Re-indexing documents when a policy updates should never require redeploying or rebuilding the API container.

## Decision
We decouple the write path and the read path entirely:
- **Write Path:** Indexing is executed exclusively as a standalone Kubernetes `Job` (`indexer-job.yaml`) and recurring weekly `CronJob` (`reindex-cronjob.yaml`). The Job mounts the Chroma PVC with `ReadWriteOnce`, builds the index, records the model metadata, invalidates the semantic cache, and terminates cleanly (`restartPolicy: OnFailure`, `backoffLimit: 3`).
- **Read Path:** The API Deployment (`api`) mounts the Chroma PVC in `ReadOnlyMany` mode (or queries the Chroma StatefulSet service via HTTP). The API only issues read queries (`retrieve`) and never writes to Chroma.
- **Readiness Integration:** The API's `/readyz` probe checks that the Chroma collection exists and is non-empty, and verifies that the collection's `embed_model` matches the API's configured `EMBED_MODEL`. If the index is missing, the API reports 503 and is kept out of the Service endpoint pool until the indexer Job finishes.

## Options Considered
- **Indexing in API startup:** Rejected for the 4 critical flaws detailed above.
- **InitContainers on API Pods:** Rejected because initContainers run on every replica pod, repeating the multi-writer race condition on the shared volume.

## Consequences
- **Positive:** Zero risk of concurrent write corruption; zero-downtime rolling updates; re-indexing is triggered on-demand via a single command:
  ```bash
  kubectl create job --from=cronjob/reindex manual-reindex-1 -n rahbar
  ```
- **Negative:** Requires orchestrating the indexer Job before the API pods become ready on a fresh cluster install. Documented clearly in the README apply order.
