# Cassandra YT MCP

`cassandra-yt-mcp` is the Cassandra-native rewrite of the old `yt-dlp-mcp`.

It is intentionally split into three ownership layers:

- `backend/` - private HTTP API for jobs, transcript storage, and yt-dlp/transcription runtime
- `worker/` - public Cloudflare Worker MCP + OAuth edge using the same WorkOS pattern as `fast-mcp-test`
- `infra/` - service-owned Terraform modules that `cassandra-infra` composes into environments

## Deployment Topology

```text
MCP client
  -> yt-mcp.<domain> (Cloudflare Worker)
  -> WorkOS OAuth / M2M token resolution
  -> yt-mcp-api.<domain> (Cloudflare Tunnel + Access)
  -> cassandra-yt-mcp backend in Kubernetes
  -> SQLite + PVC transcript store
```

## Repo Layout

```text
cassandra-yt-mcp/
├── backend/        # Private backend API, worker loop, tests, Docker image
├── worker/         # Public MCP/OAuth Cloudflare Worker
├── infra/          # Service-owned Terraform modules
└── .github/        # CI/CD for the standalone service repo
```

## Responsibilities

- `cassandra-yt-mcp` defines the service contract and Cloudflare module shape.
- `cassandra-infra` instantiates the Cloudflare modules per environment.
- `cassandra-k8s` deploys the backend and tunnel connector into the cluster.

## Image Publishing

The backend image is built by ARC runners and pushed to the local registry:

- `172.20.0.161:30500/cassandra-yt-mcp/backend:latest`

ArgoCD Image Updater detects new tags and auto-syncs deployments.

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

# Cloudflare tunnel connector token
kubectl create secret generic cloudflare-tunnel \
  --namespace cassandra-yt-mcp \
  --from-literal=token=<tunnel-token>
# Get token from: cd cassandra-infra/environments/production/yt-mcp && tofu output -raw tunnel_token
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
