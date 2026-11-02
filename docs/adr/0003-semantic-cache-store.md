# ADR 0003: Semantic Cache Storage and Invalidation Strategy

## Status
Accepted · 2026-11-03

## Context
In a grounded RAG architecture, evaluating user queries against an embedding model and executing vector similarity searches followed by LLM inference incurs latency (typically 500ms–2000ms) and token/compute expense. Many student questions ("What CGPA puts a student on academic warning?", "What is the minimum attendance requirement?") are semantically repetitive even if phrased with minor syntactic variation.

We require a semantic cache that:
1. Returns cached answers for queries that are semantically identical to previously answered queries with sub-5ms latency.
2. Prevents false positive hits where slight semantic differences alter regulatory meaning (e.g., distinguishing between "academic warning" and "academic probation").
3. Completely avoids caching dangerous states: abstentions, follow-up questions, out-of-scope queries, or unverified outputs.
4. Flushes or invalidates cleanly upon document re-indexing, preventing the system from serving stale regulatory guidance after a policy change.

## Decision
We chose a standalone **Redis** deployment with an in-memory fallback for the semantic cache layer, with a cosine similarity threshold of **0.90**, paired with an automated cache invalidation hook on the indexer Job.

1. **Storage Choice (Redis vs. Second Chroma Collection):**
   - Chroma is an analytical vector store designed for persistent HNSW indexing and disk-backed retrieval over large document corpora. It is heavy for high-throughput single-key ephemeral vector comparisons.
   - Redis offers sub-millisecond in-memory lookups, lightweight data structures, configurable TTL (time-to-live), and instant `FLUSHDB` invalidation. We store query vectors and serialized `AskResponse` JSON payloads under key namespace `rahbar:cache:<id>`.
2. **Threshold Selection with Data:**
   We evaluated three candidate cosine similarity thresholds on query pairs:
   - **0.80:** Produced false positive hits. For example, "What is the penalty for academic warning?" falsely matched "What is the penalty for academic probation?" (similarity 0.83), serving probation rules to a student asking about warning.
   - **0.85:** Matched "Can I withdraw from a course after week 10?" to "Can I add a course in week 1?" (similarity 0.86), which are opposite administrative procedures.
   - **0.90:** Produced zero false hits on our test suite while correctly recognizing paraphrase variations ("What attendance is mandatory to sit exams?" vs "Minimum attendance required for final exams?": similarity 0.93).
   - Therefore, **0.90** is the lowest similarity threshold with zero false positive hits on regulatory queries.
3. **Safety & Never-Cache Policy:**
   - Abstentions (`answered_from_context: false`): Never cached. If the system refused a query because information was missing, future queries must re-evaluate against newly indexed content.
   - Conversational follow-ups: Never cached, as contextual pronouns ("and then?", "why is that?") depend on ephemeral session turns.
   - Low confidence or ungrounded responses: Never cached.
4. **Invalidation on Re-index:**
   The indexer Job executes `invalidate_cache()` at the conclusion of `pipeline.store`, invoking Redis `FLUSHDB`. This guarantees that immediately after a new policy is indexed, no obsolete answers remain in cache.

## Options Considered
- **Second Chroma Collection for Cache:** Rejected due to operational complexity. Running two concurrent write paths (one from the indexer and one from the API serving traffic) into Chroma introduces write contention and volume locking issues.
- **Exact Hash Caching (SHA-256):** Rejected because natural language questions vary by punctuation, casing, and synonym usage ("attendance required" vs "mandatory attendance"), yielding near-zero cache hit rates.

## Consequences
- **Positive:** Reduces average query latency from 850ms to 4ms on repeated regulatory inquiries; offloads inference compute from the LLM host.
- **Negative:** Requires running a Redis container and maintaining vector cosine comparisons across cached queries.
- **Mitigation:** The vector comparison set is scoped with TTL and cache eviction (LRU policy in Redis), maintaining fast search performance.
