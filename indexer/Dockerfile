FROM python:3.12-slim

# System packages for PDF processing and OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

COPY indexer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY indexer/pipeline/ ./pipeline/
# The corpus is fetched by corpus/fetch.py and is intentionally not committed.
# It is copied into the immutable indexer image at build time; the API never sees it.
COPY corpus/raw/ /corpus/
COPY eval/fixtures/batch-b/ /batch-b/

RUN chown -R appuser:appgroup /app /corpus

USER appuser

ENTRYPOINT ["python", "-m", "pipeline.run"]
