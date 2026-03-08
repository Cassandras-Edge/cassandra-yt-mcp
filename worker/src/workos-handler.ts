import { env } from "cloudflare:workers";
import type { AuthRequest, OAuthHelpers } from "@cloudflare/workers-oauth-provider";
import { Hono } from "hono";
import { fetchWorkOSAuthToken, getUpstreamAuthorizeUrl, type Props } from "./utils";
import {
  bindStateToSession,
  createOAuthState,
  OAuthError,
  validateOAuthState,
} from "./workers-oauth-utils";

const app = new Hono<{ Bindings: Env & { OAUTH_PROVIDER: OAuthHelpers } }>();

app.get("/authorize", async (c) => {
  const oauthReqInfo = await c.env.OAUTH_PROVIDER.parseAuthRequest(c.req.raw);
  const { clientId } = oauthReqInfo;
  if (!clientId) {
    return c.text("Invalid request", 400);
  }

  // Auto-approve all clients — skip consent screen, go straight to WorkOS
  const { stateToken } = await createOAuthState(oauthReqInfo, c.env.OAUTH_KV);
  const { setCookie } = await bindStateToSession(stateToken);
  const headers = new Headers();
  headers.append("Set-Cookie", setCookie);
  return redirectToWorkOS(c.req.raw, stateToken, headers);
});

app.get("/callback", async (c) => {
  let oauthReqInfo: AuthRequest;
  let clearSessionCookie: string;
  try {
    const result = await validateOAuthState(c.req.raw, c.env.OAUTH_KV);
    oauthReqInfo = result.oauthReqInfo;
    clearSessionCookie = result.clearCookie;
  } catch (error) {
    if (error instanceof OAuthError) {
      return error.toResponse();
    }
    return c.text("Internal server error", 500);
  }

  if (!oauthReqInfo.clientId) {
    return c.text("Invalid OAuth request data", 400);
  }

  const [authResult, errResponse] = await fetchWorkOSAuthToken({
    client_id: c.env.WORKOS_CLIENT_ID,
    client_secret: c.env.WORKOS_CLIENT_SECRET,
    code: c.req.query("code"),
    redirect_uri: new URL("/callback", c.req.url).href,
  });
  if (errResponse) return errResponse;

  const { redirectTo } = await c.env.OAUTH_PROVIDER.completeAuthorization({
    request: oauthReqInfo,
    userId: authResult.userId,
    scope: oauthReqInfo.scope,
    metadata: { label: authResult.name },
    props: {
      userId: authResult.userId,
      email: authResult.email,
      name: authResult.name,
      accessToken: authResult.accessToken,
    } as Props,
  });

  const headers = new Headers({ Location: redirectTo });
  if (clearSessionCookie) {
    headers.set("Set-Cookie", clearSessionCookie);
  }
  return new Response(null, { status: 302, headers });
});

function redirectToWorkOS(
  request: Request,
  stateToken: string,
  headers: Headers = new Headers(),
) {
  headers.set(
    "location",
    getUpstreamAuthorizeUrl({
      upstream_url: "https://api.workos.com/user_management/authorize",
      client_id: env.WORKOS_CLIENT_ID,
      redirect_uri: new URL("/callback", request.url).href,
      state: stateToken,
    }),
  );
  return new Response(null, {
    status: 302,
    headers,
  });
}

export { app as WorkOSHandler };
