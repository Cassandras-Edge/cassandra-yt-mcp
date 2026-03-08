export async function backendGet(
  env: Env,
  path: string,
  params: Record<string, string | number | undefined> = {},
): Promise<any> {
  const url = new URL(path, env.BACKEND_BASE_URL);
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) {
      url.searchParams.set(key, String(value));
    }
  }
  return fetchBackendJson(env, url, { method: "GET" });
}

export async function backendPost(
  env: Env,
  path: string,
  body: Record<string, unknown>,
): Promise<any> {
  const url = new URL(path, env.BACKEND_BASE_URL);
  return fetchBackendJson(env, url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function fetchBackendJson(env: Env, url: URL, init: RequestInit): Promise<any> {
  const headers = new Headers(init.headers || {});
  if (env.BACKEND_API_TOKEN) {
    headers.set("Authorization", `Bearer ${env.BACKEND_API_TOKEN}`);
  }
  if (env.CF_ACCESS_CLIENT_ID) {
    headers.set("CF-Access-Client-Id", env.CF_ACCESS_CLIENT_ID);
  }
  if (env.CF_ACCESS_CLIENT_SECRET) {
    headers.set("CF-Access-Client-Secret", env.CF_ACCESS_CLIENT_SECRET);
  }
  const response = await fetch(url.toString(), { ...init, headers });
  const text = await response.text();
  const payload = text ? safeJsonParse(text) : {};
  if (!response.ok) {
    throw new Error(
      typeof payload === "object" && payload && "detail" in payload
        ? String(payload.detail)
        : `Backend request failed (${response.status})`,
    );
  }
  return payload;
}

export function jsonToolResponse(payload: Record<string, unknown>) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(payload) }],
  };
}

function safeJsonParse(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return { message: value };
  }
}
