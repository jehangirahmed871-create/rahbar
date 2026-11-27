FROM python:3.12-slim

WORKDIR /app

RUN groupadd -g 10001 evalgroup && \
    useradd -u 10001 -g evalgroup -s /usr/sbin/nologin evaluser

COPY eval/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY eval/run_assignment03.py eval/release_gate.py eval/coverage.py eval/summarise_traces.py ./eval/
COPY eval/golden.jsonl eval/embedding_pairs.jsonl eval/live_queries.jsonl ./eval/
COPY eval/release-thresholds.json eval/coverage-thresholds.json ./eval/
COPY indexer/pipeline/ ./indexer/pipeline/

ENV PYTHONPATH=/app/indexer

RUN chown -R evaluser:evalgroup /app
USER 10001:10001

CMD ["python", "/app/eval/run_assignment03.py"]
