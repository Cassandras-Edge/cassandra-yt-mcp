# Cassandra YT MCP

`cassandra-yt-mcp` is the Cassandra-native rewrite of the old `yt-dlp-mcp`.

It is intentionally split into three ownership layers:

- `backend/` - private HTTP API, coordinator, GPU worker, MCP sidecar (FastMCP), yt-dlp/transcription runtime
- `worker/` - public Cloudflare Worker MCP + OAuth edge using the WorkOS pattern
- `infra/` - service-owned Terraform modules that `cassandra-infra` composes

## Deployment Topology

Two MCP access paths:

```text
# Path 1: CF Worker (existing, WorkOS OAuth)
MCP client → yt-mcp.<domain> (CF Worker) → WorkOS OAuth → CF Tunnel → backend

# Path 2: FastMCP sidecar (new, API key auth)
MCP client → yt-mcp-mcp.<domain> → CF Tunnel → MCP sidecar (port 3003) → direct DB
```

## Repo Layout

```text
cassandra-yt-mcp/
├── backend/        # Backend: coordinator + GPU worker + MCP sidecar + tests
├── worker/         # Public MCP/OAuth Cloudflare Worker
├── infra/          # Service-owned Terraform modules
└── .woodpecker.yaml # CI/CD pipeline (Woodpecker CI)
```

## Responsibilities

- `cassandra-yt-mcp` defines the service contract and Cloudflare module shape.
- `cassandra-infra` instantiates the Cloudflare modules per environment.
- `cassandra-k8s` deploys the backend and tunnel connector into the cluster.

## Image Publishing

Images are built by Woodpecker CI via BuildKit and pushed to the local registry:

- `172.20.0.161:30500/cassandra-yt-mcp/coordinator:latest`
- `172.20.0.161:30500/cassandra-yt-mcp/gpu-worker:latest`

ArgoCD syncs the Helm chart using `:latest` with `pullPolicy: Always`.

## Secrets

All secrets are managed manually — none are stored in git.

### Kubernetes (namespace: `cassandra-yt-mcp`)

```bash
# Backend API token — authenticates Worker -> Backend requests
kubectl create secret generic cassandra-yt-mcp-backend \
  --namespace cassandra-yt-mcp \
  --from-literal=BACKEND_API_TOKEN=<token> \
  --from-literal=ASSEMBLYAI_API_KEY=<key>       # optional fallback transcriber \
  --from-literal=HUGGINGFACE_TOKEN=<token>       # optional model access

# MCP sidecar auth (for FastMCP path)
kubectl create secret generic cassandra-yt-mcp-mcp \
  --namespace cassandra-yt-mcp \
  --from-literal=AUTH_URL=https://auth.<domain> \
  --from-literal=AUTH_SECRET=<auth-secret>
```

### Cloudflare Worker (`wrangler secret put`)

```bash
cd worker/
wrangler secret put WORKOS_CLIENT_ID          # from WorkOS dashboard
wrangler secret put WORKOS_CLIENT_SECRET      # from WorkOS dashboard
wrangler secret put COOKIE_ENCRYPTION_KEY     # openssl rand -hex 32
wrangler secret put BACKEND_BASE_URL          # your backend URL
wrangler secret put BACKEND_API_TOKEN         # must match k8s secret above
wrangler secret put CF_ACCESS_CLIENT_ID       # from tofu output cf_access_client_id
wrangler secret put CF_ACCESS_CLIENT_SECRET   # from tofu output -raw cf_access_client_secret
```

### WorkOS

Add `https://<your-worker-subdomain>.<your-domain>/callback` as an allowed redirect URI.

