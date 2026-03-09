declare namespace Cloudflare {
  interface Env {
    OAUTH_KV: KVNamespace;
    MCP_KEYS: KVNamespace;
    WORKOS_CLIENT_ID: string;
    WORKOS_CLIENT_SECRET: string;
    COOKIE_ENCRYPTION_KEY: string;
    BACKEND_BASE_URL: string;
    BACKEND_API_TOKEN?: string;
    CF_ACCESS_CLIENT_ID?: string;
    CF_ACCESS_CLIENT_SECRET?: string;
    CASSANDRA_YT_MCP_OBJECT: DurableObjectNamespace<import("./src/index").CassandraYtMCP>;
  }
}

interface Env extends Cloudflare.Env {}
