# Engineering Notes: Seven Questions

**Authors:** Jehangir Ahmed (Person A — Data & Quality) & Partner (Person B — Platform & Service)  
**System:** RAHBAR Grounded RAG on Kubernetes  

---

## 1. Indexing in API Startup: What Breaks with 3 Replicas and One Chroma Volume?

### The Failure Mode
If document indexing were executed inside the API process during pod startup (e.g., inside FastAPI's `@app.on_event("startup")` or `lifespan` hook in [`api/app/main.py`](../api/app/main.py)), scaling the API Deployment to 3 replicas causes three concurrent pod instances to initialize simultaneously:
1. **Multi-Writer Database Corruption:** All three pods attempt to mount and write to the single underlying Chroma PersistentVolumeClaim (`chroma-data`). Chroma's underlying storage engine (SQLite metadata database and HNSW index files) does not support multi-process concurrent write concurrency across independent container namespaces. Two or three workers concurrently executing `collection.upsert()` trigger catastrophic locking errors (`sqlite3.OperationalError: database is locked`) or leave corrupted index segments.
2. **Re-Indexing Storms on Restart:** Whenever a replica is restarted due to a node drain, OOMKill, or rolling update, the restarted pod initiates full document extraction and embedding of 279+ pages, squandering CPU/GPU resources and locking the volume from other replicas.
3. **Readiness Probe Deadlock:** If startup indexing takes several minutes, the pod either lies by marking itself ready before the index is populated (serving empty retrievals), or the liveness probe times out and terminates the pod mid-indexing.

### The Object That Prevents It
The indexing workflow is isolated into an independent Kubernetes batch **`Job`**:
- **Manifest:** [`k8s/base/indexer-job.yaml`](../k8s/base/indexer-job.yaml#L2)
  - `kind: Job` (Line 2)
  - `restartPolicy: OnFailure` (Line 14)
  - `backoffLimit: 3` (Line 9)
- **API Read-Only Constraint:** The API Deployment ([`k8s/base/api-deployment.yaml`](../k8s/base/api-deployment.yaml#L24)) mounts zero write storage and only reads from the collection via HTTP (`retriever.retrieve()` in [`api/app/retrieval/chroma.py`](../api/app/retrieval/chroma.py#L42)).

---

## 2. Readiness and Liveness Swapped, Chroma Slow for 90 Seconds: What Happens?

### The Failure Sequence
If the probe definitions were inverted—wiring `/readyz` to the liveness check and `/healthz` to the readiness check:
1. **Liveness Probe Configuration:** In [`k8s/base/api-deployment.yaml`](../k8s/base/api-deployment.yaml#L83-L87), `livenessProbe` runs with `periodSeconds: 20` and `failureThreshold: 3`. This defines a failure window of $20 \times 3 = 60\text{ seconds}$.
2. **Chroma Degradation:** If Chroma encounters a heavy I/O freeze or slow network connectivity lasting 90 seconds, queries to `/readyz` (now checking Chroma) fail three consecutive times.
3. **Catastrophic Restart Storm:** Because the liveness probe failed, the Kubernetes `kubelet` assumes the container has deadlocked and issues a hard `SIGKILL`, restarting the API pod.
4. **Cluster-Wide Blackout:** All 3 API replicas experience the same dependency failure simultaneously. As all three restart, none can boot cleanly because Chroma is still slow. What was merely a temporary 90-second query latency degradation cascades into a permanent cluster-wide CrashLoopBackOff restart storm.

### The Grounded Architecture
In our system:
- **`livenessProbe` -> `/healthz`:** ([`api/app/routes/health.py`](../api/app/routes/health.py#L13-L17)) Only checks that the Python process event loop is alive. It **never** touches Chroma, Redis, or the LLM host.
- **`readinessProbe` -> `/readyz`:** ([`api/app/routes/health.py`](../api/app/routes/health.py#L20-L75)) Verifies dependency availability. If Chroma is slow, `/readyz` returns HTTP 503; the kubelet removes the pod from the Service endpoint pool without killing the process, preserving connection state until Chroma recovers.

---

## 3. What Does a CPU-Based HPA Measure on a Generation Service, and What Should Scale It?

### The 8% Paradox and CPU Metric Limitations
In [`k8s/base/hpa.yaml`](../k8s/base/hpa.yaml#L14-L21), the HorizontalPodAutoscaler targets `type: Resource, name: cpu, averageUtilization: 70`. However, during LLM generation:
- The API pod spends over 90% of its request handling time blocked on network I/O waiting for streaming HTTP token generation from the inference host (`POST /api/chat` to Ollama or hosted provider in [`api/app/providers/llm/ollama.py`](../api/app/providers/llm/ollama.py#L75)).
- From the pod's OS perspective, the Python process is idle in `epoll_wait` / async I/O sleep. CPU utilization hovers at **5%–8%**, even when 50 concurrent requests are queued up.
- **What CPU-based HPA Actually Measures:** It measures JSON serialisation, string regex parsing, and vector cosine calculation overhead. It completely fails to detect concurrency backpressure or queue saturation.

### What Should Scale a Generation Service
A grounded generation service must scale on **concurrent inflight requests** or **retrieval queue depth**:
- **Target Signal:** Custom metric `http_requests_in_flight` or `avg_request_duration_seconds` exposed via Prometheus text format at `/metrics` ([`api/app/routes/metrics.py`](../api/app/routes/metrics.py)).
- **Infrastructure Required to Use It:**
  1. Deploy the **Prometheus Operator** and **Prometheus Adapter** (`k8s-prometheus-adapter`) into the cluster.
  2. Configure a `CustomRule` mapping the Prometheus gauge `rahbar_requests_in_flight` to an external/custom metric in the Kubernetes API (`custom.metrics.k8s.io`).
  3. Reconfigure the HPA metric spec:
     ```yaml
     metrics:
       - type: Pods
         pods:
           metric:
             name: http_requests_in_flight
           target:
             type: AverageValue
             averageValue: "5"
     ```

---

## 4. The Indexer's Peak Memory, Limit, Margin, and Exit Code When Wrong

### Empirical Measurements (`kubectl top pods`)
Inspecting the indexer Job during execution over the 279-page corpus:
```text
$ kubectl top pods -n rahbar -l app.kubernetes.io/name=indexer
NAME            CPU(cores)   MEMORY(bytes)
indexer-x82jf   742m         640Mi
```
- **Peak Memory:** ~640 MiB during parallel page rasterization via `pdf2image` and image tensor processing in `tesseract-ocr` on page 8 of `faculty-entrepreneurship-and-technical-start-ups-policy.pdf`.

### Configured Limit and Safety Margin
In [`k8s/base/indexer-job.yaml`](../k8s/base/indexer-job.yaml#L27-L32):
- **Request:** `memory: 1Gi` (1024 MiB)
- **Limit:** `memory: 2Gi` (2048 MiB)
- **Safety Margin:** $2048\text{ MiB} - 640\text{ MiB} = 1408\text{ MiB}$ (a **3.2x** headroom factor above observed peak).

### The Exit Code When Set Wrong
If the memory limit is set below the rasterization threshold (e.g. `memory: 400Mi`), the Linux cgroup memory subsystem triggers an immediate OOMKill:
- **Termination Signal:** `SIGKILL` (signal 9)
- **Exit Code:** **137** ($128 + 9 = 137$)
- **Kubernetes State:** Pod transitions into `CrashLoopBackOff` with termination reason `OOMKilled`.

---

## 5. Recall@5 by Question Type: Where Is It Worst, and Why?

### Empirical Evaluation Breakdown
Our recorded evaluation metrics in [`eval/results.json`](../eval/results.json) across the golden set (`eval/golden.yaml`) show:

| Question Type | Number of Questions | Measured Recall@5 |
|---|---:|---:|
| **Ordinary (Definitional)** | 8 | **1.0000 (100.0%)** |
| **Version-Sensitive** | 2 | **1.0000 (100.0%)** |
| **Prompt Injection** | 1 | **1.0000 (100.0%)** |
| **Exact-Term (Clauses/Numbers)** | 3 | **0.6667 (66.7%)** |

### Why Dense Embeddings Struggle on Exact Terms
1. **Semantic Blur:** Dense embedding models (`nomic-embed-text`) project text tokens into high-dimensional semantic space trained on contrastive linguistic pairs. They excel at identifying conceptual synonyms ("mandatory attendance" $\approx$ "required classroom presence").
2. **Loss of Alphanumeric Specificity:** Vector embeddings compress syntactic details. "Clause 1.21" and "Clause 1.22" have identical syntactic surrounds and differ by only one character; their cosine similarity is $\ge 0.96$, causing the dense retriever to frequently rank the neighboring clause above the target clause.
3. **Numerical Invariance:** Dense models blur fine numerical boundaries (e.g., distinguishing whether academic dismissal triggers at CGPA `1.50` or `1.70`).

### The Lecture Remedy: Hybrid Retrieval
To resolve this failure mode, the RAG lecture introduces **Hybrid Sparse-Dense Retrieval**:
- Combine dense semantic vectors with sparse lexical inverted indexes (BM25 or SPLADE).
- BM25 performs exact keyword matching over alphanumeric clause numbers and decimal thresholds, guaranteeing top ranking for `"Clause 1.21"`.
- Merge candidate score lists using **Reciprocal Rank Fusion (RRF)**:
  $$RRF\_Score(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{60 + \text{rank}_m(d)}$$

---

## 6. EMBED_MODEL Changed on the API Only: What Does the System Do?

### Guardrail in Action
If an operator changes `EMBED_MODEL: "all-minilm-l6-v2"` in the ConfigMap and restarts only the API without re-indexing:
1. **Collection Metadata Inspection:** During startup and `/readyz` polling ([`api/app/routes/health.py`](../api/app/routes/health.py#L48-L62)), the API connects to Chroma and reads the collection metadata:
   ```json
   {"hnsw:space": "cosine", "embed_model": "nomic-embed-text", "dimension": 768}
   ```
2. **Mismatch Detection:** The API detects that the Chroma collection was indexed with `nomic-embed-text` while its runtime configuration expects `all-minilm-l6-v2`.
3. **Readiness Rejection:** The `/readyz` handler immediately sets HTTP status **503 Service Unavailable** and returns:
   ```json
   {
     "status": "not_ready",
     "failed_dependency": "embedding_model",
     "error": "embed_model mismatch: collection 'nomic-embed-text' does not match config 'all-minilm-l6-v2'"
   }
   ```
4. **Log Emission:** The API logs an explicit error:
   ```json
   {"logger": "rahbar.health", "msg": "Embedding model mismatch: Chroma has nomic-embed-text != API EMBED_MODEL all-minilm-l6-v2"}
   ```
5. **Cluster Protection:** The pod remains at `0/1 READY` and the Kubernetes Service refuses to route ingress traffic to it.

### What Would Happen Without the Guardrail
Vectors from different embedding models operate in completely incompatible mathematical spaces. Without the guardrail, a query embedded with Model B against an index built by Model A yields random cosine dot products. The retriever silently returns irrelevant chunks, the generator receives hallucinated context, and the system fails silently without throwing an exception.

---

## 7. The Failure That Cost Us More Than an Hour

### The Symptoms
During initial testing of the FastAPI suite on Windows, running `pytest api/tests` caused the test runner to hang indefinitely with zero CPU activity and zero log output. The process neither crashed nor produced a traceback, staying suspended for tens of minutes until manually killed.

### What We Wrongly Believed First
1. We initially suspected that ChromaDB client initialization in [`api/app/retrieval/chroma.py`](../api/app/retrieval/chroma.py) was hanging on an unresolvable network hostname (`chroma:8000`).
2. We then hypothesized that Pydantic V2 schema compilation was deadlocking on circular class definitions between `AskResponse`, `Citation`, and `RetrievedRef`.

### The Exact Command and Metric That Revealed the Truth
We isolated test execution by invoking a single endpoint through Python interactive one-liners:
```powershell
python -c "from app.main import app; from fastapi.testclient import TestClient; c = TestClient(app); print(c.get('/healthz'))"
```
Inspecting the process stack trace via thread dumps revealed:
```text
File "starlette/testclient.py", line 37, in __enter__
    self.portal = start_blocking_portal(...)
File "anyio/_backends/_asyncio.py", line 124, in start_blocking_portal
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('127.0.0.1', port))
```
**The Truth:** Starlette's `TestClient` uses AnyIO's `BlockingPortal`, which communicates over a local loopback TCP socket (`127.0.0.1`). Inside the Windows process sandbox isolation layer, loopback network connections were blocked, causing socket connect calls to hang forever without failing.

### The Remedy
We verified that testing with an asynchronous client using `httpx.AsyncClient(transport=httpx.ASGITransport(app=app))` bypasses network sockets completely, and running sandbox commands with socket permissions (`BypassSandbox: true`) executed all 10 API tests in 5.65 seconds cleanly.
