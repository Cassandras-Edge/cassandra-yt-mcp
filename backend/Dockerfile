FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive
ARG YTDLP_CHANNEL=stable
ARG AUTH_YAML_CONTENT=""

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg curl ca-certificates git nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    if [ "$YTDLP_CHANNEL" = "nightly" ]; then \
      YTDLP_URL="https://github.com/yt-dlp/yt-dlp-nightly-builds/releases/latest/download/yt-dlp"; \
    elif [ "$YTDLP_CHANNEL" = "stable" ]; then \
      YTDLP_URL="https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp"; \
    else \
      echo "Unsupported YTDLP_CHANNEL: $YTDLP_CHANNEL"; exit 1; \
    fi; \
    curl -fsSL "$YTDLP_URL" -o /usr/local/bin/yt-dlp; \
    chmod +x /usr/local/bin/yt-dlp; \
    yt-dlp --version; \
    yt-dlp --remote-components ejs:github

RUN groupadd -g 1001 app && useradd -u 1001 -g app -s /bin/bash -m app

WORKDIR /app

COPY pyproject.toml README.md ./

# Install deps from pyproject.toml with stub package
RUN mkdir -p src/cassandra_yt_mcp && touch src/cassandra_yt_mcp/__init__.py \
 && pip install --no-cache-dir "."

# Copy actual source
COPY --chown=app:app src ./src
RUN pip install --no-cache-dir --no-deps "."

# Bake in ACL policy if provided
RUN if [ -n "$AUTH_YAML_CONTENT" ]; then \
      echo "$AUTH_YAML_CONTENT" > /app/acl.yaml; \
    fi

RUN mkdir -p /data && chown -R app:app /app /data

USER app

ENV PYTHONUNBUFFERED=1 \
    PORT=3003 \
    HOST=0.0.0.0 \
    DATA_DIR=/data \
    DATABASE_PATH=/data/cassandra_yt_mcp.sqlite3 \
    POLL_INTERVAL_SECONDS=5

EXPOSE 3003

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1 || exit 1

CMD ["cassandra-yt-mcp-backend"]
