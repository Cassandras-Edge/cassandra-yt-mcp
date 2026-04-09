"""FastMCP server for Cassandra YT MCP — exposes the same 10 tools as the CF Worker."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.dependencies import CurrentAccessToken
from fastmcp.server.auth import AccessToken
from fastmcp.server.context import Context

from cassandra_mcp_auth import AclMiddleware
from cassandra_yt_mcp.auth import McpKeyAuthProvider, build_auth
from cassandra_yt_mcp.config import Settings
from cassandra_yt_mcp.runtime import AppRuntime

logger = logging.getLogger(__name__)

SERVICE_ID = "yt-mcp"

# Tools that need ACL enforcement (all of them)
ALL_TOOL_NAMES = [
    "transcribe",
    "job_status",
    "search",
    "list_transcripts",
    "read_transcript",
    "yt_search",
    "list_channel_videos",
    "get_comments",
    "watch_later_sync",
    "watch_later_status",
]


def _get_email(token: AccessToken | None) -> str:
    if token is None:
        return ""
    return token.claims.get("email", "")


def _get_credentials(token: AccessToken) -> dict[str, str]:
    """Get per-user credentials from token claims.

    Both mcp_ keys and WorkOS OAuth tokens carry credentials in claims —
    the auth layer (McpKeyAuthProvider / UserInfoEnrichingVerifier) handles
    fetching them from the auth service.
    """
    return token.claims.get("credentials", {})


def _get_youtube_cookies(token: AccessToken) -> str | None:
    return _get_credentials(token).get("youtube_cookies") or None


def _is_youtube_url(url: str) -> bool:
    lower = url.lower()
    return any(h in lower for h in ("youtube.com", "youtu.be"))


def create_mcp_server(settings: Settings) -> FastMCP:
    """Create and configure the FastMCP server with all tools."""

    auth_provider = None
    mcp_key_provider: McpKeyAuthProvider | None = None
    if settings.auth_url and settings.auth_secret:
        if settings.workos_client_id and settings.workos_authkit_domain and settings.base_url:
            auth_provider, mcp_key_provider = build_auth(
                acl_url=settings.auth_url,
                acl_secret=settings.auth_secret,
                service_id=SERVICE_ID,
                base_url=settings.base_url,
                workos_client_id=settings.workos_client_id,
                workos_authkit_domain=settings.workos_authkit_domain,
            )
        else:
            mcp_key_provider = McpKeyAuthProvider(
                acl_url=settings.auth_url,
                acl_secret=settings.auth_secret,
                service_id=SERVICE_ID,
            )
            auth_provider = mcp_key_provider

    # Shared state populated by lifespan, accessible to custom routes via closure.
    _state: dict[str, object] = {}

    @asynccontextmanager
    async def lifespan(mcp: FastMCP):
        runtime = AppRuntime(settings)
        runtime.start()
        _state["runtime"] = runtime
        try:
            yield {"runtime": runtime}
        finally:
            _state.clear()
            runtime.close()
            if mcp_key_provider is not None:
                mcp_key_provider.close()

    acl_mw = AclMiddleware(service_id=SERVICE_ID, acl_path=settings.auth_yaml_path)

    mcp_kwargs: dict = {
        "name": "YouTube",
        "instructions": (
            "# Cassandra YouTube/Media\n\n"
            "Use this when the user wants to transcribe or search video/audio content. "
            "GPU-accelerated speech recognition with speaker diarization. Works with "
            "YouTube, Twitch, Twitter/X videos, and 1000+ other sites via yt-dlp.\n\n"
            "Think of this server when the user asks about:\n"
            "- Transcribing a video or audio URL\n"
            "- What was said in a video (searching transcript content)\n"
            "- Browsing YouTube — searching videos, listing channels, reading comments\n"
            "- The user's Watch Later queue\n\n"
            "Transcription is async: `transcribe(url)` returns a `job_id`, poll "
            "`job_status(job_id)` until done, then `read_transcript(video_id)` for "
            "the text. Already-transcribed videos return instantly."
        ),
        "lifespan": lifespan,
        "middleware": [acl_mw] if acl_mw._enabled else [],  # noqa: SLF001
    }
    if auth_provider:
        mcp_kwargs["auth"] = auth_provider

    mcp = FastMCP(**mcp_kwargs)

    # ── Service API (gateway-facing REST routes) ──
    from cassandra_yt_mcp.service_api import register_service_api  # noqa: PLC0415

    register_service_api(mcp, settings, _state)

    # ── Health check ──

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(request):  # noqa: ANN001, ARG001
        from starlette.responses import JSONResponse  # noqa: PLC0415

        return JSONResponse({"ok": True, "service": "yt-mcp"})

    # ── Prometheus metrics ──

    @mcp.custom_route("/metrics", methods=["GET"])
    async def prometheus_metrics(request):  # noqa: ANN001, ARG001
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # noqa: PLC0415
        from starlette.responses import Response  # noqa: PLC0415

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ── REST API ──
    # Public job status endpoint — job ID acts as a capability token.

    @mcp.custom_route("/api/jobs/{job_id}", methods=["GET"])
    async def api_get_job(request):  # noqa: ANN001
        from starlette.responses import JSONResponse  # noqa: PLC0415

        runtime: AppRuntime = _state["runtime"]
        job_id = request.path_params["job_id"]
        result = runtime.get_job_status(job_id)
        if "error" in result and result["error"] == "job_not_found":
            return JSONResponse(result, status_code=404)
        return JSONResponse(result)

    # ── Tool: transcribe ──

    @mcp.tool(
        description=(
            "Queue a video for transcription. Supports any yt-dlp-compatible URL "
            "(YouTube, Twitch VODs/clips, Twitter/X, and 1000+ other sites). "
            "Also supports YouTube playlist URLs and live streams (grabs available "
            "content up to the current point)."
        ),
    )
    def transcribe(
        url: str,
        ctx: Context,
        token: AccessToken = CurrentAccessToken(),
    ) -> dict:
        runtime: AppRuntime = ctx.lifespan_context["runtime"]

        cookies_b64 = None
        if _is_youtube_url(url):
            cookies_b64 = _get_youtube_cookies(token)

        return runtime.enqueue_transcription(url, cookies_b64=cookies_b64)

    # ── Tool: job_status ──

    @mcp.tool(
        description=(
            "Get the status of a transcription job. Returns immediately with current status. "
            "If in progress, includes a retry_after hint (seconds) for when to poll again."
        ),
    )
    def job_status(
        job_id: str,
        ctx: Context,
        token: AccessToken = CurrentAccessToken(),
    ) -> dict:
        runtime: AppRuntime = ctx.lifespan_context["runtime"]

        return runtime.get_job_status(job_id)

    # ── Tool: search ──

    @mcp.tool(description="Search transcripts by content.")
    def search(
        query: str,
        ctx: Context,
        limit: int = 10,
        token: AccessToken = CurrentAccessToken(),
    ) -> dict:
        runtime: AppRuntime = ctx.lifespan_context["runtime"]

        limit = max(1, min(limit, 50))
        return {"query": query, "results": runtime.transcripts.search(query=query, limit=limit)}

    # ── Tool: list_transcripts ──

    @mcp.tool(description="List available transcripts.")
    def list_transcripts(
        ctx: Context,
        platform: str | None = None,
        channel: str | None = None,
        limit: int = 20,
        token: AccessToken = CurrentAccessToken(),
    ) -> dict:
        runtime: AppRuntime = ctx.lifespan_context["runtime"]

        limit = max(1, min(limit, 100))
        items = runtime.transcripts.list_transcripts(platform=platform, channel=channel, limit=limit)
        return {"count": len(items), "items": items}

    # ── Tool: read_transcript ──

    @mcp.tool(
        description=(
            "Read a transcript by video ID or URL. If not cached, automatically queues "
            "transcription and returns job info. Formats: compact (default, token-efficient "
            "``[MM:SS] S0: text``), markdown (full with metadata), text (plain), json (segments)."
        ),
    )
    def read_transcript(
        video_id: str,
        ctx: Context,
        format: str = "compact",
        offset: int = 0,
        limit: int | None = None,
        token: AccessToken = CurrentAccessToken(),
    ) -> dict:
        runtime: AppRuntime = ctx.lifespan_context["runtime"]

        transcript = runtime.transcripts.get_by_video_id(video_id)
        if transcript is None:
            # Auto-queue: treat video_id as URL if it looks like one, otherwise build YouTube URL
            url = video_id if video_id.startswith("http") else f"https://www.youtube.com/watch?v={video_id}"
            cookies_b64 = _get_youtube_cookies(token) if _is_youtube_url(url) else None
            job = runtime.enqueue_transcription(url, cookies_b64=cookies_b64)
            return {
                "queued": True,
                "message": f"Transcript not cached — queued for transcription. Use job_status with job_id to check progress.",
                **job,
            }

        import json  # noqa: PLC0415

        base = Path(str(transcript["path"]))

        if format == "compact":
            compact_path = base / "transcript.compact.txt"
            if compact_path.exists():
                full = compact_path.read_text(encoding="utf-8")
            else:
                # Fallback: generate on the fly from JSON
                payload = json.loads((base / "transcript.json").read_text(encoding="utf-8"))
                from cassandra_yt_mcp.services.storage import to_compact  # noqa: PLC0415
                from cassandra_yt_mcp.types import TranscriptResult as TR, TranscriptSegment as TS  # noqa: PLC0415

                result = TR(
                    text=payload.get("text", ""),
                    segments=[
                        TS(start=s["start"], end=s["end"], text=s.get("text", ""), speaker=s.get("speaker"))
                        for s in payload.get("segments", [])
                    ],
                    language=payload.get("language"),
                )
                full = to_compact(result)
            lines = full.splitlines(keepends=True)
            page = lines[offset:] if limit is None else lines[offset : offset + limit]
            return {
                "video_id": video_id,
                "content": "".join(page),
                "total_lines": len(lines),
                "offset": offset,
            }

        if format in ("markdown", "text"):
            ext = "md" if format == "markdown" else "txt"
            full = (base / f"transcript.{ext}").read_text(encoding="utf-8")
            lines = full.splitlines(keepends=True)
            page = lines[offset:] if limit is None else lines[offset : offset + limit]
            return {
                "video_id": video_id,
                "format": format,
                "content": "".join(page),
                "total_lines": len(lines),
                "offset": offset,
                "lines_returned": len(page),
            }

        payload = (base / "transcript.json").read_text(encoding="utf-8")
        document = json.loads(payload)
        segments = document.get("segments", document.get("utterances", []))
        page = segments[offset:] if limit is None else segments[offset : offset + limit]
        return {
            "video_id": video_id,
            "format": "json",
            "content": page,
            "total_segments": len(segments),
            "offset": offset,
            "segments_returned": len(page),
        }

    # ── Tool: yt_search ──

    @mcp.tool(description="Search YouTube for videos.")
    def yt_search(
        query: str,
        ctx: Context,
        limit: int = 10,
        token: AccessToken = CurrentAccessToken(),
    ) -> dict:
        runtime: AppRuntime = ctx.lifespan_context["runtime"]

        limit = max(1, min(limit, 25))
        cookies_file = _write_cookies_to_temp(token, ctx)
        t0 = time.monotonic()
        try:
            results = runtime.youtube_info.search(query=query, limit=limit, cookies_file=cookies_file)
        except RuntimeError as exc:
            elapsed = time.monotonic() - t0
            logger.error("yt_search failed after %.1fs | query=%r limit=%d | %s", elapsed, query, limit, exc)
            return {"error": "search_failed", "message": str(exc), "elapsed_seconds": round(elapsed, 1)}
        finally:
            if cookies_file and cookies_file.exists():
                cookies_file.unlink()
        elapsed = time.monotonic() - t0
        logger.info("yt_search ok in %.1fs | query=%r limit=%d results=%d", elapsed, query, limit, len(results))
        return {"query": query, "count": len(results), "results": results}

    # ── Tool: list_channel_videos ──

    @mcp.tool(
        description="List videos from a YouTube channel. Use tab parameter to browse shorts, videos, or streams.",
    )
    def list_channel_videos(
        url: str,
        ctx: Context,
        tab: str = "videos",
        limit: int = 20,
        token: AccessToken = CurrentAccessToken(),
    ) -> dict:
        runtime: AppRuntime = ctx.lifespan_context["runtime"]

        limit = max(1, min(limit, 50))
        if tab not in ("shorts", "videos", "streams"):
            tab = "videos"
        cookies_file = _write_cookies_to_temp(token, ctx)
        try:
            results = runtime.youtube_info.list_channel_videos(
                channel_url=url, limit=limit, tab=tab, cookies_file=cookies_file,
            )
        except RuntimeError as exc:
            return {"error": "channel_list_failed", "message": str(exc)}
        finally:
            if cookies_file and cookies_file.exists():
                cookies_file.unlink()
        return {"url": url, "tab": tab, "count": len(results), "results": results}

    # ── Tool: get_comments ──

    @mcp.tool(
        description="Get comments for a video. Comment sorting/limits are optimized for YouTube; other platforms return all available comments.",
    )
    def get_comments(
        url: str,
        ctx: Context,
        limit: int = 20,
        sort: str = "top",
        token: AccessToken = CurrentAccessToken(),
    ) -> dict:
        runtime: AppRuntime = ctx.lifespan_context["runtime"]

        limit = max(1, min(limit, 100))
        if sort not in ("top", "new"):
            sort = "top"
        cookies_file = _write_cookies_to_temp(token, ctx)
        try:
            comments = runtime.youtube_info.get_comments(
                url=url, limit=limit, sort=sort, cookies_file=cookies_file,
            )
        except RuntimeError as exc:
            return {"error": "comments_failed", "message": str(exc)}
        finally:
            if cookies_file and cookies_file.exists():
                cookies_file.unlink()
        return {"url": url, "count": len(comments), "sort": sort, "comments": comments}

    # ── Tool: watch_later_sync ──

    @mcp.tool(
        description=(
            "Sync your YouTube Watch Later playlist. Finds new videos, queues them for "
            "transcription, and tracks which ones have been seen. Requires YouTube cookies "
            "to be configured in the portal."
        ),
    )
    def watch_later_sync(
        ctx: Context,
        token: AccessToken = CurrentAccessToken(),
    ) -> dict:
        runtime: AppRuntime = ctx.lifespan_context["runtime"]

        cookies = _get_youtube_cookies(token)
        if not cookies:
            return {
                "error": "no_cookies",
                "message": "YouTube cookies not configured. Set them in the portal under yt-mcp service credentials.",
            }

        user_id = _get_email(token)
        runtime.watch_later_repo.register_user(user_id, cookies)
        try:
            return runtime.watch_later_service.sync(user_id, cookies)
        except RuntimeError as exc:
            return {"error": "sync_failed", "message": str(exc)}

    # ── Tool: watch_later_status ──

    @mcp.tool(description="Check the status of your Watch Later sync — seen videos, last sync time, etc.")
    def watch_later_status(
        ctx: Context,
        token: AccessToken = CurrentAccessToken(),
    ) -> dict:
        runtime: AppRuntime = ctx.lifespan_context["runtime"]

        user_id = _get_email(token)
        user = runtime.watch_later_repo.get_user(user_id)
        if user is None:
            return {"registered": False, "user_id": user_id}
        seen = runtime.watch_later_repo.list_seen(user_id, limit=20)
        return {
            "registered": True,
            "user_id": user_id,
            "enabled": bool(user["enabled"]),
            "interval_minutes": user["interval_minutes"],
            "last_sync_at": user["last_sync_at"],
            "last_error": user["last_error"],
            "seen_count": runtime.watch_later_repo.count_seen(user_id),
            "recent_seen": seen,
        }

    return mcp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_cookies_to_temp(token: AccessToken, ctx: Context | None = None) -> Path | None:
    """Decode YouTube cookies from token credentials to a temp file."""
    cookies_b64 = _get_youtube_cookies(token)
    if not cookies_b64:
        return None

    import base64  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        tmp.write(base64.b64decode(cookies_b64))
        tmp.close()
        return Path(tmp.name)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to decode cookies from token credentials")
        return None
