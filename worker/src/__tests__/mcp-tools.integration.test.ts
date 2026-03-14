import { afterEach, describe, expect, it, vi } from "vitest";
import { registerMcpTools } from "../mcp-tools";
import { createMockEnv } from "./test-helpers";

interface ToolServer {
  registerTool(
    name: string,
    definition: Record<string, unknown>,
    handler: (input: Record<string, unknown>) => Promise<unknown>,
  ): void;
}

describe("registerMcpTools", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("registers the expected MCP tools", () => {
    const tools = collectTools();

    expect(Array.from(tools.keys())).toEqual([
      "transcribe",
      "job_status",
      "search",
      "list_transcripts",
      "read_transcript",
      "yt_search",
      "get_metadata",
      "get_comments",
    ]);
  });

  it("maps tools to the correct backend routes, methods, and params", async () => {
    const tools = collectTools();
    const fetchMock = vi.fn(async (url: string) =>
      new Response(JSON.stringify({ route: url }), {
        headers: { "Content-Type": "application/json" },
        status: 200,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(tools.get("transcribe")!.handler({ url: "https://youtu.be/demo" })).resolves.toEqual({
      content: [
        {
          text: JSON.stringify({ route: "https://backend.example.test/api/jobs/transcribe" }),
          type: "text",
        },
      ],
    });
    await tools.get("job_status")!.handler({ job_id: "job-123" });
    await tools.get("search")!.handler({ limit: 7, query: "transcript" });
    await tools.get("list_transcripts")!.handler({ channel: "demo", limit: 3, platform: "youtube" });
    await tools.get("read_transcript")!.handler({
      format: "json",
      limit: 2,
      offset: 1,
      video_id: "vid-123",
    });
    await tools.get("yt_search")!.handler({ limit: 4, query: "cassandra" });
    await tools.get("get_metadata")!.handler({ url: "https://youtu.be/meta" });
    await tools.get("get_comments")!.handler({ limit: 9, sort: "new", url: "https://youtu.be/comments" });

    expect(fetchMock.mock.calls).toHaveLength(8);
    expectRequest(fetchMock.mock.calls[0], "POST", "/api/jobs/transcribe", {});
    expect((fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].body).toBe(
      JSON.stringify({ url: "https://youtu.be/demo" }),
    );
    expectRequest(fetchMock.mock.calls[1], "GET", "/api/jobs/job-123", {});
    expectRequest(fetchMock.mock.calls[2], "GET", "/api/transcripts/search", {
      limit: "7",
      query: "transcript",
    });
    expectRequest(fetchMock.mock.calls[3], "GET", "/api/transcripts", {
      channel: "demo",
      limit: "3",
      platform: "youtube",
    });
    expectRequest(fetchMock.mock.calls[4], "GET", "/api/transcripts/vid-123", {
      format: "json",
      limit: "2",
      offset: "1",
    });
    expectRequest(fetchMock.mock.calls[5], "GET", "/api/youtube/search", {
      limit: "4",
      query: "cassandra",
    });
    expectRequest(fetchMock.mock.calls[6], "GET", "/api/youtube/metadata", {
      url: "https://youtu.be/meta",
    });
    expectRequest(fetchMock.mock.calls[7], "GET", "/api/youtube/comments", {
      limit: "9",
      sort: "new",
      url: "https://youtu.be/comments",
    });
  });
});

function collectTools() {
  const registered = new Map<
    string,
    {
      definition: Record<string, unknown>;
      handler: (input: Record<string, unknown>) => Promise<unknown>;
    }
  >();
  const server: ToolServer = {
    registerTool(name, definition, handler) {
      registered.set(name, { definition, handler });
    },
  };

  registerMcpTools(server as any, createMockEnv(), {
    userId: "test-user",
    email: "test@example.com",
    name: "Test User",
    accessToken: "test-token",
  } as any);

  return registered;
}

function expectRequest(
  call: unknown[] | undefined,
  method: string,
  pathname: string,
  query: Record<string, string>,
) {
  expect(call).toBeDefined();
  const [rawUrl, init] = call as [string, RequestInit];
  const url = new URL(rawUrl);
  const actualQuery: Record<string, string> = {};
  url.searchParams.forEach((value, key) => {
    actualQuery[key] = value;
  });

  expect(url.pathname).toBe(pathname);
  expect(actualQuery).toEqual(query);
  expect(init).toEqual(expect.objectContaining({ method }));
}
