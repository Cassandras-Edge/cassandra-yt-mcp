# GPU_BASE is overridden by CI to use the pre-built registry image directly,
# skipping the gpu-base stage entirely. For local builds, falls back to gpu-base.
ARG GPU_BASE=gpu-base

# ── Stage 1: GPU/ML dependencies (cached as separate image in local registry) ──
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS gpu-base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip \
      ffmpeg libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir \
      "nemo_toolkit[asr]>=2.0.0" \
      "pyannote.audio>=3.3.0,<4.0"

# ── Stage 2: App code (fast, changes often) ──
FROM ${GPU_BASE} AS app

ARG YTDLP_CHANNEL=nightly

RUN set -eux; \
    if [ "$YTDLP_CHANNEL" = "nightly" ]; then \
      YTDLP_URL="https://github.com/yt-dlp/yt-dlp-nightly-builds/releases/latest/download/yt-dlp"; \
    elif [ "$YTDLP_CHANNEL" = "stable" ]; then \
      YTDLP_URL="https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp"; \
    else \
      echo "Unsupported YTDLP_CHANNEL: $YTDLP_CHANNEL"; \
      exit 1; \
    fi; \
    curl -fsSL "$YTDLP_URL" -o /usr/local/bin/yt-dlp; \
    chmod +x /usr/local/bin/yt-dlp; \
    yt-dlp --version

RUN groupadd -g 1001 app && useradd -u 1001 -g app -s /bin/bash -m app

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --no-cache-dir "."

RUN mkdir -p /data && chown -R app:app /app /data

USER app

ENV PYTHONUNBUFFERED=1 \
    PORT=3000 \
    WORKER_PORT=3001 \
    HOST=0.0.0.0 \
    DATA_DIR=/data \
    DATABASE_PATH=/data/cassandra_yt_mcp.sqlite3 \
    POLL_INTERVAL_SECONDS=5 \
    ROLE=standalone

EXPOSE 3000 3001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1 \
   || curl -fsS "http://127.0.0.1:${WORKER_PORT}/worker/healthz" >/dev/null 2>&1 \
   || exit 1

CMD ["cassandra-yt-mcp-backend"]
