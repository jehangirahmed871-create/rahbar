# Assignment 03: Kubernetes runbook

This runbook uses the existing Minikube cluster with Docker Desktop as its
container engine. Docker Desktop's separate built-in Kubernetes cluster is not
currently available in this environment. The development overlay is sized for
a single-node desktop cluster; run ingestion and evaluation Jobs sequentially.

The API uses Groq for answer generation; its API key is read from the
`rahbar-secret` Kubernetes Secret. Embedding vectors use deterministic local
feature hashing, and query retrieval uses lexical matching. No Ollama service
or separate embedding API is needed. This keeps resource use low but is less
semantically capable than a trained embedding model.

## 1. Check the cluster and build images

In PowerShell, confirm the intended context and node before applying anything:

```powershell
minikube status
kubectl config current-context
kubectl get nodes
```

For a new Minikube cluster, Docker Desktop must be running:

```powershell
minikube start --driver=docker --cni=calico --memory=3072 --cpus=2
minikube addons enable metrics-server
```

Fetch the source corpus before building the indexer image. The corpus is
git-ignored and is copied into the indexer image as its initial seed.

```powershell
python corpus\fetch.py
docker build -t rahbar-api:dev api
docker build -f indexer\Dockerfile -t rahbar-indexer:dev .
docker build -t rahbar-ui:dev ui
docker build -f eval\Dockerfile -t rahbar-eval:dev .
minikube image load rahbar-api:dev
minikube image load rahbar-indexer:dev
minikube image load rahbar-ui:dev
minikube image load rahbar-eval:dev
```

The `rahbar-secret` Secret must exist in the `rahbar` namespace and contain the
LLM provider key. Supply it from a secure local environment; never put the
value in a manifest, command transcript, or repository file.

## 2. Deploy persistent storage and the live service

```powershell
kubectl apply -k k8s\overlays\dev
kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/corpus-pvc -n rahbar --timeout=120s
kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/indexer-data-pvc -n rahbar --timeout=120s
kubectl rollout status statefulset/chroma -n rahbar --timeout=180s
kubectl rollout status deployment/redis -n rahbar --timeout=120s
kubectl get pvc -n rahbar
```

`corpus-pvc` is the persistent data lake. `indexer-data-pvc` holds verification
reports, quarantined files, and archived versions. The trace and evaluation
PVCs are separate. The CronJob has `concurrencyPolicy: Forbid`; do not launch
manual indexer Jobs concurrently with another writer. If an inherited cluster
already has a pending or active indexer Job, inspect its pod and logs and wait
for it to finish before starting another ingestion.

On a Docker Desktop node limited to 3 GiB, pause the HPA and live API while an
indexer Job runs. The API can be restored before evaluation and demos:

```powershell
kubectl delete hpa api-hpa -n rahbar --ignore-not-found
kubectl scale deployment/api --replicas=0 -n rahbar
```

## 3. Seed and verify the base corpus

The first indexer Job seeds the persistent data lake from the image, verifies
documents, archives superseded versions, quarantines failures, and indexes only
passing documents. The Job uses the same template as the scheduled CronJob:

```powershell
kubectl create job indexer-base --from=cronjob/reindex -n rahbar
kubectl wait --for=condition=complete job/indexer-base -n rahbar --timeout=7200s
kubectl logs job/indexer-base -n rahbar
```

Run it again without changing the corpus to verify incremental indexing:

```powershell
kubectl create job indexer-unchanged --from=cronjob/reindex -n rahbar
kubectl wait --for=condition=complete job/indexer-unchanged -n rahbar --timeout=7200s
kubectl logs job/indexer-unchanged -n rahbar
```

The final structured indexer log must report `chunks_embedded: 0`. Save the
verification report path printed in the log. To copy the verification folder
from the first run:

```powershell
kubectl apply -k k8s\overlays\dev
kubectl rollout status deployment/api -n rahbar --timeout=180s
$indexerPod = kubectl get pods -n rahbar -l job-name=indexer-base -o jsonpath='{.items[0].metadata.name}'
New-Item -ItemType Directory -Force eval\results | Out-Null
kubectl cp -n rahbar "${indexerPod}:/indexer-data/verification" "eval\results\verification-base"
```

The indexer integration test also exercises unchanged ingestion (zero embedded
chunks), a one-document update, orphan deletion by document ID, archive and
restore behavior:

```powershell
python -m pytest indexer\tests\test_pipeline.py -k "incremental_store_skips_unchanged_chunks_and_removes_deleted_documents or pipeline_quarantines_batch_b_and_is_incremental" -q
```

Run the pre-Batch-B evaluation twice as a Kubernetes Job:

```powershell
kubectl apply -f k8s\eval-baseline-job.yaml
kubectl wait --for=condition=complete job/eval-baseline -n rahbar --timeout=7200s
kubectl logs job/eval-baseline -n rahbar
$evalPod = kubectl get pods -n rahbar -l job-name=eval-baseline -o jsonpath='{.items[0].metadata.name}'
kubectl cp -n rahbar "${evalPod}:/results/part-b-before-batch-b.json" "eval\results\part-b-before-batch-b.json"
```

## 4. Ingest Batch B and record data drift

Batch B is copied into the same persistent corpus without replacing the base
corpus. The seed Job is idempotent and only populates its directory once.

```powershell
kubectl apply -f k8s\base\batch-b-seed-job.yaml
kubectl wait --for=condition=complete job/batch-b-seed -n rahbar --timeout=300s
kubectl delete hpa api-hpa -n rahbar --ignore-not-found
kubectl scale deployment/api --replicas=0 -n rahbar
kubectl apply -f k8s\base\indexer-post-batch-b-job.yaml
kubectl wait --for=condition=complete job/indexer-post-batch-b -n rahbar --timeout=7200s
kubectl logs job/indexer-post-batch-b -n rahbar
$indexerPod = kubectl get pods -n rahbar -l job-name=indexer-post-batch-b -o jsonpath='{.items[0].metadata.name}'
kubectl cp -n rahbar "${indexerPod}:/indexer-data/verification" "eval\results\verification-batch-b"
kubectl cp -n rahbar "${indexerPod}:/indexer-data/archive" "eval\results\archive-batch-b"
```

Review the per-document verification checks in the Job's JSON report. Then
re-run the Part B suite against the updated baseline:

```powershell
kubectl apply -f k8s\base\api-evaluation.yaml
kubectl scale deployment/api-baseline --replicas=1 -n rahbar
kubectl rollout status deployment/api-baseline -n rahbar --timeout=180s
kubectl apply -f k8s\eval-post-batch-b-job.yaml
kubectl wait --for=condition=complete job/eval-post-batch-b -n rahbar --timeout=7200s
kubectl logs job/eval-post-batch-b -n rahbar
$evalPod = kubectl get pods -n rahbar -l job-name=eval-post-batch-b -o jsonpath='{.items[0].metadata.name}'
kubectl cp -n rahbar "${evalPod}:/results/part-e3-after-batch-b.json" "eval\results\part-e3-after-batch-b.json"
kubectl scale deployment/api-baseline --replicas=0 -n rahbar
kubectl apply -f k8s\base\indexer-candidate-job.yaml
kubectl wait --for=condition=complete job/indexer-candidate -n rahbar --timeout=7200s
kubectl logs job/indexer-candidate -n rahbar
kubectl scale deployment/api-candidate --replicas=1 -n rahbar
kubectl rollout status deployment/api-candidate -n rahbar --timeout=180s
```

The 2026 academic-rules version is archived and marked superseded while the
2027 version remains active. The restore Job checks that the archived file
survived on the PVC, then restores that version in Chroma. Re-index afterward
to return the current collection to the 2027 version:

```powershell
kubectl apply -f k8s\base\indexer-restore-job.yaml
kubectl wait --for=condition=complete job/indexer-restore-old-version -n rahbar --timeout=300s
kubectl logs job/indexer-restore-old-version -n rahbar
kubectl create job indexer-restore-current --from=cronjob/reindex -n rahbar
kubectl wait --for=condition=complete job/indexer-restore-current -n rahbar --timeout=1800s
kubectl logs job/indexer-restore-current -n rahbar
```

The baseline and candidate APIs use separate Services and Chroma collections.
Their Deployments start at zero replicas so they do not consume desktop memory
until selected. Run one API Deployment at a time; the candidate changes only
`TOP_K` (5 to 8), with the same corpus and embedding model.

## 5. Gate and controlled rollout

The candidate Job runs the same twice-repeated suite and exits non-zero if any
release check fails. For a deliberately weak candidate, set the candidate's
`TOP_K` to 1, rerun the Job, and retain its failed report before trying `TOP_K`
8. A completed Job must be deleted before the same Job manifest can be applied
again; result files are on the PVC and are not deleted with the Job.

```powershell
kubectl set env deployment/api-candidate TOP_K=1 -n rahbar
kubectl rollout status deployment/api-candidate -n rahbar --timeout=180s
kubectl apply -f k8s\eval-job.yaml
kubectl wait --for=condition=failed job/eval-candidate -n rahbar --timeout=7200s
kubectl logs job/eval-candidate -n rahbar
$evalPod = kubectl get pods -n rahbar -l job-name=eval-candidate -o jsonpath='{.items[0].metadata.name}'
kubectl cp -n rahbar "${evalPod}:/results/part-c-candidate.json" "eval\results\part-c-bad-candidate.json"
kubectl cp -n rahbar "${evalPod}:/results/part-c-gate.json" "eval\results\part-c-bad-gate.json"
kubectl delete job eval-candidate -n rahbar
kubectl set env deployment/api-candidate TOP_K=8 -n rahbar
kubectl rollout status deployment/api-candidate -n rahbar --timeout=180s
kubectl apply -f k8s\eval-job.yaml
kubectl wait --for=condition=complete job/eval-candidate -n rahbar --timeout=7200s
kubectl logs job/eval-candidate -n rahbar
$evalPod = kubectl get pods -n rahbar -l job-name=eval-candidate -o jsonpath='{.items[0].metadata.name}'
kubectl cp -n rahbar "${evalPod}:/results/part-c-candidate.json" "eval\results\part-c-candidate.json"
kubectl cp -n rahbar "${evalPod}:/results/part-c-gate.json" "eval\results\part-c-gate.json"
```

Show that the threshold ConfigMap changes the decision without rebuilding the
image. The deliberately strict manifest changes only the latency threshold.

```powershell
kubectl apply -f k8s\eval-thresholds-fail.yaml
kubectl apply -f k8s\release-gate-job.yaml
kubectl wait --for=condition=failed job/release-gate -n rahbar --timeout=300s
kubectl logs job/release-gate -n rahbar
$gatePod = kubectl get pods -n rahbar -l job-name=release-gate -o jsonpath='{.items[0].metadata.name}'
kubectl cp -n rahbar "${gatePod}:/results/threshold-change-gate.json" "eval\results\threshold-change-fail.json"
kubectl delete job release-gate -n rahbar
kubectl apply -f k8s\base\eval-thresholds-configmap.yaml
kubectl apply -f k8s\release-gate-job.yaml
kubectl wait --for=condition=complete job/release-gate -n rahbar --timeout=300s
kubectl logs job/release-gate -n rahbar
$gatePod = kubectl get pods -n rahbar -l job-name=release-gate -o jsonpath='{.items[0].metadata.name}'
kubectl cp -n rahbar "${gatePod}:/results/threshold-change-gate.json" "eval\results\threshold-change-pass.json"
```

Only after the candidate gate passes, release it to the live Deployment. The
live Deployment uses `maxUnavailable: 0`. Then demonstrate a bad release and
undo it; the undo restores the previously gated candidate configuration.

```powershell
kubectl scale deployment/api-candidate --replicas=0 -n rahbar
kubectl set env deployment/api CHROMA_COLLECTION=rahbar_candidate TOP_K=8 -n rahbar
kubectl scale deployment/api --replicas=1 -n rahbar
kubectl rollout status deployment/api -n rahbar --timeout=180s
kubectl set env deployment/api TOP_K=1 -n rahbar
kubectl rollout status deployment/api -n rahbar --timeout=180s
kubectl rollout undo deployment/api -n rahbar
kubectl rollout status deployment/api -n rahbar --timeout=180s
kubectl apply -k k8s\overlays\dev
```

## 6. Traces, metrics, and resource evidence

The live API writes each structured trace to stdout and to `traces-pvc` at
`/traces/queries.jsonl`. Query `/metrics` through a port-forward and issue a
question after the rollback to produce the evidence trace:

```powershell
kubectl port-forward svc/api -n rahbar 8000:8000
```

In a second PowerShell window:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/metrics
$answer = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/ask -ContentType 'application/json' -Body '{"question":"What attendance percentage is required for the final examination?"}'
$answer.trace_id
```

Copy the trace file and rebuild its context and prompt by trace ID:

```powershell
$apiPod = kubectl get pods -n rahbar -l app.kubernetes.io/name=api -o jsonpath='{.items[0].metadata.name}'
New-Item -ItemType Directory -Force traces | Out-Null
kubectl cp -n rahbar "${apiPod}:/traces/queries.jsonl" "traces\queries.jsonl"
python eval\show_trace.py $answer.trace_id --traces traces\queries.jsonl
```

Summarize trace metrics inside the cluster:

```powershell
kubectl apply -f k8s\trace-summary-job.yaml
kubectl wait --for=condition=complete job/trace-summary -n rahbar --timeout=300s
kubectl logs job/trace-summary -n rahbar
```

Capture resource measurements before and during indexing/evaluation; use the
measured steady-state and peak values when discussing requests and limits:

```powershell
kubectl top nodes
kubectl top pods -n rahbar
kubectl get all,pvc,configmap,cronjob,job -n rahbar
```

The 40-query coverage run uses the repository's calibrated threshold and exits
non-zero below the in-domain coverage floor:

```powershell
python eval\coverage.py --out eval\results\coverage-after.json
```

For screenshots and the report, copy every result JSON from `eval-results-pvc`
into `eval\results` and commit only reports produced by this repository's own
cluster runs. Never include secret values in screenshots or output.
