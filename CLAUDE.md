# CLAUDE.md — Cassandra YT-MCP

## What This Is

YouTube transcription MCP service. GPU-accelerated ASR (Parakeet TDT 0.6b) + speaker diarization (pyannote), with AssemblyAI fallback. CF Worker gateway with WorkOS OAuth + MCP API key auth.

## Repo Structure

```
cassandra-yt-mcp/
├── worker/                # CF Worker — MCP gateway + OAuth
│   ├── src/
│   ├── wrangler.jsonc.example
│   └── package.json
├── backend/               # Python GPU backend — FastAPI
│   ├── src/cassandra_yt_mcp/
│   │   ├── main.py        # FastAPI app
│   │   ├── config.py      # Settings
│   │   ├── runtime.py     # Model loading
│   │   ├── metrics.py     # Prometheus metrics
│   │   ├── api/           # Route handlers
│   │   ├── db/            # Database layer
│   │   ├── models/        # Data models
│   │   └── services/      # Business logic (ASR, diarization)
│   ├── Dockerfile
│   ├── Dockerfile.coordinator
│   ├── pyproject.toml
│   └── tests/
├── infra/
│   └── modules/           # Terraform: worker-edge + backend-access
└── README.md
```

`wrangler.jsonc` is gitignored — only `.example` is tracked. Real KV IDs stay local.

## Auth Stack (3 layers)

```
Client → CF Worker (WorkOS OAuth OR mcp_ API key)
       → CF Access (service token auth)
       → Backend (Bearer API token)
```

1. **MCP API key** (`Bearer mcp_...`): KV lookup in shared `MCP_KEYS`, must have `service === "yt-mcp"`
2. **WorkOS JWT** (fallback): Standard OAuth for browser clients
3. **Backend API token**: Worker → backend, protected by CF Access service token

## Deploy

Worker auto-deploys on push to main via Woodpecker CI (`.woodpecker.yaml`), triggered only when `worker/` files change. `wrangler.jsonc` is templated from Woodpecker secrets at deploy time.

```bash
# Manual deploy (if needed)
cd worker && npm install && npx wrangler deploy

# Backend image — built by Woodpecker CI, pushed to local registry
# ArgoCD deploys from cassandra-k8s/apps/cassandra-yt-mcp/

# Infra (from cassandra-infra)
cd cassandra-infra/environments/production/yt-mcp
source ../../.env
tofu init -backend-config=production.s3.tfbackend
tofu apply
```

## Worker Secrets (via `wrangler secret put`)

- `WORKOS_CLIENT_ID` — Shared WorkOS app
- `WORKOS_CLIENT_SECRET` — Shared WorkOS app
- `COOKIE_ENCRYPTION_KEY` — Session encryption
- `BACKEND_BASE_URL` — Backend API URL
- `BACKEND_API_TOKEN` — Bearer token for backend
- `CF_ACCESS_CLIENT_ID` — Service token for backend CF Access
- `CF_ACCESS_CLIENT_SECRET` — Service token for backend CF Access
- `VM_PUSH_URL` — VictoriaMetrics push endpoint
- `VM_PUSH_CLIENT_ID` — CF Access service token for metrics
- `VM_PUSH_CLIENT_SECRET` — CF Access service token for metrics

## Worker Bindings

- `MCP_OBJECT` — Durable Object (MUST be this name)
- `OAUTH_KV` — Per-service KV for OAuth state
- `MCP_KEYS` — Shared KV for API key auth

## Backend

- **Image**: `172.20.0.161:30500/cassandra-yt-mcp/backend:latest` (local registry)
- **GPU**: Runs on `role=gpu-node` with `dedicated=gpu-node:NoSchedule` toleration
- **Models**: Parakeet TDT 0.6b (ASR) + pyannote 3.1 (diarization) — loaded at startup
- **Startup**: Model loading takes minutes — needs long `startupProbe` (3 min window)
- **Fallback**: AssemblyAI when GPU is unavailable or for unsupported formats
- **Helm chart**: `cassandra-k8s/apps/cassandra-yt-mcp/`
- **Exposes `/metrics`** for VMAgent scraping (does NOT use push path)

## Observability

- Worker pushes `mcp_requests_total` + `yt_mcp_jobs_total` via `cassandra-observability`
- Backend exposes Prometheus `/metrics` for VMAgent scrape
- Dashboard: `cassandra-observability/dashboards/yt-mcp.json`

## CI

- Woodpecker CI pipeline (`.woodpecker.yaml`)
- Builds Docker image → pushes `:latest` to local registry → pods pick up on creation
