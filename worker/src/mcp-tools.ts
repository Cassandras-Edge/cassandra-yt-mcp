import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ResolvedAuth } from "cassandra-mcp-auth";
import { z } from "zod";
import { backendGet, backendPost, jsonToolResponse } from "./backend";

function cookieHeaders(auth: ResolvedAuth): Record<string, string> {
  const cookies = auth.credentials?.youtube_cookies;
  return cookies ? { "X-Cookies-B64": cookies } : {};
}

export function registerMcpTools(server: McpServer, env: Env, auth: ResolvedAuth): void {
  server.registerTool(
    "transcribe",
    {
      description: "Queue a video for transcription. Supports any yt-dlp-compatible URL (YouTube, Twitch VODs/clips, Twitter/X, and 1000+ other sites). Also supports YouTube playlist URLs.",
      annotations: { readOnlyHint: false, idempotentHint: true },
      inputSchema: { url: z.string().describe("The video URL to transcribe (YouTube, Twitch, Twitter/X, etc.)") },
    },
    async ({ url }) => {
      const body: Record<string, unknown> = { url: String(url) };
      // Only attach YouTube cookies for YouTube URLs to avoid leaking cookies to other extractors
      const urlLower = String(url).toLowerCase();
      const isYouTube = ["youtube.com", "youtu.be"].some((h) => urlLower.includes(h));
      if (isYouTube) {
        const cookies = auth.credentials?.youtube_cookies;
        if (cookies) {
          body.cookies_b64 = cookies;
        }
      }
      return jsonToolResponse(
        (await backendPost(env, "/api/jobs/transcribe", body)) as Record<string, unknown>,
      );
    },
  );

  server.registerTool(
    "job_status",
    {
      description:
        "Get the status of a transcription job. Returns immediately with current status. If in progress, includes a retry_after hint (seconds) for when to poll again.",
      annotations: { readOnlyHint: true },
      inputSchema: { job_id: z.string().describe("The job ID returned from transcribe()") },
    },
    async ({ job_id }) =>
      jsonToolResponse(
        (await backendGet(env, `/api/jobs/${String(job_id)}`)) as Record<string, unknown>,
      ),
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
        }, cookieHeaders(auth))) as Record<string, unknown>,
      ),
  );

  server.registerTool(
    "list_channel_videos",
    {
      description:
        "List videos from a YouTube channel. Use tab parameter to browse shorts, videos, or streams.",
      annotations: { readOnlyHint: true },
      inputSchema: {
        url: z.string().describe("The channel URL (e.g. https://www.youtube.com/channel/UC... or https://www.youtube.com/@handle)"),
        tab: z.enum(["shorts", "videos", "streams"]).default("videos").describe("Which tab to list"),
        limit: z.number().int().min(1).max(50).default(20),
      },
    },
    async ({ url, tab, limit }) =>
      jsonToolResponse(
        (await backendGet(env, "/api/youtube/channel", {
          url: String(url),
          tab: tab as string,
          limit: limit as number,
        }, cookieHeaders(auth))) as Record<string, unknown>,
      ),
  );

  server.registerTool(
    "get_metadata",
    {
      description: "Get full metadata for a video (works with any yt-dlp-supported URL).",
      annotations: { readOnlyHint: true },
      inputSchema: { url: z.string().describe("The video URL (YouTube, Twitch, Twitter/X, etc.)") },
    },
    async ({ url }) =>
      jsonToolResponse(
        (await backendGet(env, "/api/youtube/metadata", { url: String(url) }, cookieHeaders(auth))) as Record<string, unknown>,
      ),
  );

  server.registerTool(
    "get_comments",
    {
      description: "Get comments for a video. Comment sorting/limits are optimized for YouTube; other platforms return all available comments.",
      annotations: { readOnlyHint: true },
      inputSchema: {
        url: z.string().describe("The video URL (YouTube, Twitch, etc.)"),
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
        }, cookieHeaders(auth))) as Record<string, unknown>,
      ),
  );

  server.registerTool(
    "watch_later_sync",
    {
      description:
        "Sync your YouTube Watch Later playlist. Finds new videos, queues them for transcription, and tracks which ones have been seen. Requires YouTube cookies to be configured in the portal.",
      annotations: { readOnlyHint: false, idempotentHint: true },
      inputSchema: {},
    },
    async () => {
      const cookies = auth.credentials?.youtube_cookies;
      if (!cookies) {
        return jsonToolResponse({
          error: "no_cookies",
          message:
            "YouTube cookies not configured. Set them in the portal under yt-mcp service credentials.",
        });
      }
      return jsonToolResponse(
        (await backendPost(env, "/api/watch-later/sync", {
          user_id: auth.userId,
          cookies_b64: cookies,
        })) as Record<string, unknown>,
      );
    },
  );

  server.registerTool(
    "watch_later_status",
    {
      description: "Check the status of your Watch Later sync — seen videos, last sync time, etc.",
      annotations: { readOnlyHint: true },
      inputSchema: {},
    },
    async () =>
      jsonToolResponse(
        (await backendGet(env, "/api/watch-later/status", {
          user_id: auth.userId,
        })) as Record<string, unknown>,
      ),
  );
}
