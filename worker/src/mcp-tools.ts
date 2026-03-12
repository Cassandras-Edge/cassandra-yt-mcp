import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { backendGet, backendPost, jsonToolResponse } from "./backend";

export function registerMcpTools(server: McpServer, env: Env): void {
  server.registerTool(
    "transcribe",
    {
      description: "Queue a video for transcription. Supports single videos and playlist URLs (all videos in the playlist will be queued).",
      annotations: { readOnlyHint: false, idempotentHint: true },
      inputSchema: { url: z.string().describe("The video URL to transcribe (YouTube, etc.)") },
    },
    async ({ url }) =>
      jsonToolResponse(
        (await backendPost(env, "/api/jobs/transcribe", { url: String(url) })) as Record<string, unknown>,
      ),
  );

  server.registerTool(
    "job_status",
    {
      description:
        "Get the status of a transcription job. Waits for the job to finish (up to ~5 minutes) before returning, using exponential backoff internally. Only returns early if the job is already complete, failed, or the timeout is exceeded.",
      annotations: { readOnlyHint: true },
      inputSchema: { job_id: z.string().describe("The job ID returned from transcribe()") },
    },
    async ({ job_id }) => {
      const id = String(job_id);
      const IN_PROGRESS = new Set(["queued", "downloading", "downloaded", "transcribing"]);
      const MAX_WAIT_MS = 5 * 60 * 1000;
      const BASE_DELAY_MS = 2_000;
      const MAX_DELAY_MS = 30_000;

      const deadline = Date.now() + MAX_WAIT_MS;
      let attempt = 0;

      while (true) {
        const result = (await backendGet(env, `/api/jobs/${id}`)) as Record<string, unknown>;
        const status = result.status as string | undefined;

        if (!status || !IN_PROGRESS.has(status) || Date.now() >= deadline) {
          return jsonToolResponse(result);
        }

        const delay = Math.min(BASE_DELAY_MS * 2 ** attempt, MAX_DELAY_MS);
        await new Promise((r) => setTimeout(r, delay));
        attempt++;
      }
    },
  );

  server.registerTool(
    "search",
    {
      description: "Search transcripts by content.",
      annotations: { readOnlyHint: true },
      inputSchema: {
        query: z.string().describe("Search query string"),
        limit: z.number().int().min(1).max(50).default(10),
      },
    },
    async ({ query, limit }) =>
      jsonToolResponse(
        (await backendGet(env, "/api/transcripts/search", {
          query: String(query),
          limit: limit as number,
        })) as Record<string, unknown>,
      ),
  );

  server.registerTool(
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
    async ({ platform, channel, limit }) =>
      jsonToolResponse(
        (await backendGet(env, "/api/transcripts", {
          channel: channel as string | undefined,
          limit: limit as number,
          platform: platform as string | undefined,
        })) as Record<string, unknown>,
      ),
  );

  server.registerTool(
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
    async ({ video_id, format, offset, limit }) =>
      jsonToolResponse(
        (await backendGet(env, `/api/transcripts/${String(video_id)}`, {
          format: format as string,
          limit: limit as number | undefined,
          offset: offset as number,
        })) as Record<string, unknown>,
      ),
  );

  server.registerTool(
    "yt_search",
    {
      description: "Search YouTube for videos.",
      annotations: { readOnlyHint: true },
      inputSchema: {
        query: z.string().describe("Search query string"),
        limit: z.number().int().min(1).max(25).default(10),
      },
    },
    async ({ query, limit }) =>
      jsonToolResponse(
        (await backendGet(env, "/api/youtube/search", {
          limit: limit as number,
          query: String(query),
        })) as Record<string, unknown>,
      ),
  );

  server.registerTool(
    "get_metadata",
    {
      description: "Get full metadata for a video.",
      annotations: { readOnlyHint: true },
      inputSchema: { url: z.string().describe("The video URL (YouTube, etc.)") },
    },
    async ({ url }) =>
      jsonToolResponse(
        (await backendGet(env, "/api/youtube/metadata", { url: String(url) })) as Record<string, unknown>,
      ),
  );

  server.registerTool(
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
    async ({ url, limit, sort }) =>
      jsonToolResponse(
        (await backendGet(env, "/api/youtube/comments", {
          limit: limit as number,
          sort: sort as string,
          url: String(url),
        })) as Record<string, unknown>,
      ),
  );
}
