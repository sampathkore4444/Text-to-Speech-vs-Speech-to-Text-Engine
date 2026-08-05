# syntax=docker/dockerfile:1
# Bank Speech AI - API service image.
# The same image runs the batch worker with a different command (see compose).

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SPEECHAI_CONFIG=/app/configs/config.yaml

# onnxruntime (Piper) needs OpenMP; curl for healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install ".[engines]"

# Runtime assets (models are mounted from the host into /data).
COPY configs ./configs
COPY scripts ./scripts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "speechai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
