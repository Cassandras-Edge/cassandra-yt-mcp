export function getUpstreamAuthorizeUrl({
  upstream_url,
  client_id,
  redirect_uri,
  state,
}: {
  upstream_url: string;
  client_id: string;
  redirect_uri: string;
  state?: string;
}) {
  const upstream = new URL(upstream_url);
  upstream.searchParams.set("client_id", client_id);
  upstream.searchParams.set("redirect_uri", redirect_uri);
  upstream.searchParams.set("response_type", "code");
  upstream.searchParams.set("provider", "authkit");
  if (state) upstream.searchParams.set("state", state);
  return upstream.href;
}

export async function fetchWorkOSAuthToken({
  client_id,
  client_secret,
  code,
  redirect_uri,
}: {
  code: string | undefined;
  client_secret: string;
  client_id: string;
  redirect_uri?: string;
}): Promise<[WorkOSAuthResult, null] | [null, Response]> {
  if (!code) {
    return [null, new Response("Missing code", { status: 400 })];
  }

  const resp = await fetch("https://api.workos.com/user_management/authenticate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      client_id,
      client_secret,
      code,
      grant_type: "authorization_code",
    }),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    console.error("WorkOS token exchange failed:", errText);
    return [null, new Response("Failed to exchange code for token", { status: 500 })];
  }

  const body = (await resp.json()) as WorkOSTokenResponse;
  if (!body.access_token || !body.user) {
    return [null, new Response("Missing access token or user info from WorkOS", { status: 400 })];
  }

  return [
    {
      accessToken: body.access_token,
      userId: body.user.id,
      email: body.user.email,
      name: [body.user.first_name, body.user.last_name].filter(Boolean).join(" ") || body.user.email,
    },
    null,
  ];
}

interface WorkOSTokenResponse {
  access_token: string;
  user: {
    id: string;
    email: string;
    first_name: string | null;
    last_name: string | null;
  };
}

export interface WorkOSAuthResult {
  accessToken: string;
  userId: string;
  email: string;
  name: string;
}

export type Props = {
  userId: string;
  email: string;
  name: string;
  accessToken: string;
};
