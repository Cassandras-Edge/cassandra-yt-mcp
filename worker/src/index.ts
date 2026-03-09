import OAuthProvider from "@cloudflare/workers-oauth-provider";
import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { backendGet, backendPost, jsonToolResponse } from "./backend";
import { WorkOSHandler } from "./workos-handler";
import type { Props } from "./utils";

export class CassandraYtMCP extends McpAgent<Env, Record<string, never>, Props> {
  server: any = new McpServer({
    name: "Cassandra YT MCP",
    version: "1.0.0",
  });

  async init() {
    const env = this.env;

    this.server.registerTool(
      "transcribe",
      {
        description: "Queue a video for transcription.",
        annotations: { readOnlyHint: false, idempotentHint: true },
        inputSchema: { url: z.string().describe("The video URL to transcribe (YouTube, etc.)") },
      },
      async ({ url }: { url: string }) =>
        jsonToolResponse((await backendPost(env, "/api/jobs/transcribe", { url })) as Record<string, unknown>),
    );

    this.server.registerTool(
      "job_status",
      {
        description: "Get the status of a transcription job.",
        annotations: { readOnlyHint: true },
        inputSchema: { job_id: z.string().describe("The job ID returned from transcribe()") },
      },
      async ({ job_id }: { job_id: string }) =>
        jsonToolResponse((await backendGet(env, `/api/jobs/${job_id}`)) as Record<string, unknown>),
    );

    this.server.registerTool(
      "search",
      {
        description: "Search transcripts by content.",
        annotations: { readOnlyHint: true },
        inputSchema: {
          query: z.string().describe("Search query string"),
          limit: z.number().int().min(1).max(50).default(10),
        },
      },
      async ({ query, limit }: { query: string; limit: number }) =>
        jsonToolResponse(
          (await backendGet(env, "/api/transcripts/search", { query, limit })) as Record<string, unknown>,
        ),
    );

    this.server.registerTool(
      "list_transcripts",
      {
        description: "List available transcripts.",
        annotations: { readOnlyHint: true },
        inputSchema: {
          platform: z.string().optional().describe("Filter by platform (e.g., \"youtube\")"),
          channel: z.string().optional().describe("Filter by channel name"),
          limit: z.number().int().min(1).max(100).default(20),
        },
      },
      async ({ platform, channel, limit }: { platform?: string; channel?: string; limit: number }) =>
        jsonToolResponse(
          (await backendGet(env, "/api/transcripts", { platform, channel, limit })) as Record<string, unknown>,
        ),
    );

    this.server.registerTool(
      "read_transcript",
      {
        description: "Read a transcript by video ID.",
        annotations: { readOnlyHint: true },
        inputSchema: {
          video_id: z.string().describe("The video ID to read"),
          format: z.enum(["markdown", "text", "json"]).default("markdown").describe("Output format"),
          offset: z.number().int().min(0).default(0).describe("Number of lines/segments to skip"),
          limit: z.number().int().min(1).optional().describe("Max lines/segments to return"),
        },
      },
      async ({
        video_id,
        format,
        offset,
        limit,
      }: {
        video_id: string;
        format: "markdown" | "text" | "json";
        offset: number;
        limit?: number;
      }) =>
        jsonToolResponse(
          (await backendGet(env, `/api/transcripts/${video_id}`, {
            format,
            offset,
            limit,
          })) as Record<string, unknown>,
        ),
    );

    this.server.registerTool(
      "yt_search",
      {
        description: "Search YouTube for videos.",
        annotations: { readOnlyHint: true },
        inputSchema: {
          query: z.string().describe("Search query string"),
          limit: z.number().int().min(1).max(25).default(10),
        },
      },
      async ({ query, limit }: { query: string; limit: number }) =>
        jsonToolResponse(
          (await backendGet(env, "/api/youtube/search", { query, limit })) as Record<string, unknown>,
        ),
    );

    this.server.registerTool(
      "get_metadata",
      {
        description: "Get full metadata for a video.",
        annotations: { readOnlyHint: true },
        inputSchema: { url: z.string().describe("The video URL (YouTube, etc.)") },
      },
      async ({ url }: { url: string }) =>
        jsonToolResponse(
          (await backendGet(env, "/api/youtube/metadata", { url })) as Record<string, unknown>,
        ),
    );

    this.server.registerTool(
      "get_comments",
      {
        description: "Get comments for a video.",
        annotations: { readOnlyHint: true },
        inputSchema: {
          url: z.string().describe("The video URL (YouTube, etc.)"),
          limit: z.number().int().min(1).max(100).default(20),
          sort: z.enum(["top", "new"]).default("top"),
        },
      },
      async ({ url, limit, sort }: { url: string; limit: number; sort: "top" | "new" }) =>
        jsonToolResponse(
          (await backendGet(env, "/api/youtube/comments", { url, limit, sort })) as Record<string, unknown>,
        ),
    );
  }
}

async function resolveExternalToken(input: {
  token: string;
  request: Request;
  env: Env;
}): Promise<{ props: Props; audience?: string | string[] } | null> {
  // Check MCP API key (mcp_ prefix)
  if (input.token.startsWith("mcp_")) {
    const meta = await input.env.MCP_KEYS.get<{
      name?: string;
      service?: string;
      created_by?: string;
    }>(input.token, "json");
    if (meta && meta.service === "yt-mcp") {
      return {
        props: {
          userId: meta.created_by || "api-key",
          email: "api-key@mcp",
          name: meta.name || "API Key",
          accessToken: input.token,
        },
      };
    }
    return null;
  }

  // WorkOS JWT validation
  try {
    const [headerB64, payloadB64, signatureB64] = input.token.split(".");
    const header = JSON.parse(atob(headerB64));
    const jwksResponse = await fetch("https://api.workos.com/sso/jwks");
    if (!jwksResponse.ok) return null;
    const jwks = (await jwksResponse.json()) as { keys: JsonWebKey[] };
    const key = jwks.keys.find(
      (candidate) => (candidate as JsonWebKey & { kid?: string }).kid === header.kid,
    );
    if (!key) return null;

    const cryptoKey = await crypto.subtle.importKey(
      "jwk",
      key,
      { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
      false,
      ["verify"],
    );

    const signatureBytes = Uint8Array.from(
      atob(signatureB64.replace(/-/g, "+").replace(/_/g, "/")),
      (char) => char.charCodeAt(0),
    );
    const dataBytes = new TextEncoder().encode(`${headerB64}.${payloadB64}`);
    const valid = await crypto.subtle.verify(
      "RSASSA-PKCS1-v1_5",
      cryptoKey,
      signatureBytes,
      dataBytes,
    );
    if (!valid) return null;

    const payload = JSON.parse(atob(payloadB64.replace(/-/g, "+").replace(/_/g, "/")));
    if (payload.exp && payload.exp < Date.now() / 1000) return null;

    return {
      props: {
        userId: payload.sub || payload.org_id || "m2m",
        email: payload.sub || "m2m@machine",
        name: payload.azp || "M2M Client",
        accessToken: input.token,
      },
    };
  } catch {
    return null;
  }
}

export default new OAuthProvider({
  apiHandler: CassandraYtMCP.serve("/mcp"),
  apiRoute: "/mcp",
  authorizeEndpoint: "/authorize",
  clientRegistrationEndpoint: "/register",
  defaultHandler: WorkOSHandler as any,
  tokenEndpoint: "/token",
  resolveExternalToken,
});
