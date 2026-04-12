FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive
ARG YTDLP_CHANNEL=stable
ARG AUTH_YAML_CONTENT=""

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg curl ca-certificates git nodejs npm \
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
    mkdir -p /etc/yt-dlp && printf "%s\n" "--remote-components ejs:github" "--js-runtimes node" > /etc/yt-dlp.conf

# ── PO Token provider (fixes YouTube 403 on media downloads) ──
# Install yt-dlp plugins for PO token support
RUN mkdir -p /usr/lib/yt-dlp-plugins/yt-dlp-get-pot /usr/lib/yt-dlp-plugins/bgutil-pot \
    && git clone --depth 1 https://github.com/coletdjnz/yt-dlp-get-pot.git /tmp/yt-dlp-get-pot \
    && cp -r /tmp/yt-dlp-get-pot/yt_dlp_plugins /usr/lib/yt-dlp-plugins/yt-dlp-get-pot/ \
    && git clone --depth 1 https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git /tmp/bgutil \
    && cp -r /tmp/bgutil/plugin/yt_dlp_plugins /usr/lib/yt-dlp-plugins/bgutil-pot/ \
    && cd /tmp/bgutil/server && npm ci --omit=dev && npx tsc \
    && mkdir -p /opt/bgutil-server && cp -r /tmp/bgutil/server/build /tmp/bgutil/server/node_modules /tmp/bgutil/server/package.json /opt/bgutil-server/ \
    && rm -rf /tmp/yt-dlp-get-pot /tmp/bgutil \
    && echo '--plugin-dirs /usr/lib/yt-dlp-plugins' >> /etc/yt-dlp.conf

RUN groupadd -g 1001 app && useradd -u 1001 -g app -s /bin/bash -m app

WORKDIR /app

COPY pyproject.toml README.md ./

# Install deps from pyproject.toml with stub package
ARG GITHUB_TOKEN=""
RUN if [ -n "$GITHUB_TOKEN" ]; then printf 'machine github.com\nlogin x-access-token\npassword %s\n' "$GITHUB_TOKEN" > /root/.netrc && chmod 600 /root/.netrc; fi \
 && mkdir -p src/cassandra_yt_mcp && touch src/cassandra_yt_mcp/__init__.py \
 && pip install --no-cache-dir "." \
 && rm -f /root/.netrc

# Copy actual source
COPY --chown=app:app src ./src
RUN pip install --no-cache-dir --no-deps "."

# Bake in ACL policy if provided
RUN if [ -n "$AUTH_YAML_CONTENT" ]; then \
      echo "$AUTH_YAML_CONTENT" > /app/acl.yaml; \
    fi

COPY --chown=app:app entrypoint.sh /app/entrypoint.sh

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

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["cassandra-yt-mcp-backend"]
