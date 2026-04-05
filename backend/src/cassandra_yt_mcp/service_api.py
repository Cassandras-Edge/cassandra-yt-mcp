"""REST service API for media/yt-mcp — gateway-facing, secured with X-Auth-Secret + X-User-Email.

These endpoints mirror the MCP tools but return JSON over HTTP so the gateway
can embed a thin MCP shim without needing AppRuntime/yt-dlp/GPU access.
"""

from __future__ import annotations

import logging

from fastmcp import FastMCP

from cassandra_yt_mcp.config import Settings

logger = logging.getLogger(__name__)


def _check_auth(request, settings: Settings) -> str | None:
    """Validate X-Auth-Secret and return email, or None on failure."""
    secret = request.headers.get("x-auth-secret", "")
    if not secret or secret != settings.auth_secret:
        return None
    return request.headers.get("x-user-email", "").strip() or None


def register_service_api(mcp: FastMCP, settings: Settings, state: dict) -> None:
    """Register /api/* REST routes on the FastMCP server."""

    def _get_runtime():
        from cassandra_yt_mcp.runtime import AppRuntime
        return state["runtime"]

    @mcp.custom_route("/api/transcribe", methods=["POST"])
    async def api_transcribe(request):
        from starlette.responses import JSONResponse

        email = _check_auth(request, settings)
        if not email:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        body = await request.json()
        url = body.get("url", "")
        cookies_b64 = body.get("cookies_b64")

        if not url:
            return JSONResponse({"error": "url is required"}, status_code=400)

        runtime = _get_runtime()
        result = runtime.enqueue_transcription(url, cookies_b64=cookies_b64)
        return JSONResponse(result)

    # /api/jobs/:id already exists as a custom_route in mcp_server.py — skip

    @mcp.custom_route("/api/search", methods=["GET"])
    async def api_search(request):
        from starlette.responses import JSONResponse

        email = _check_auth(request, settings)
        if not email:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        query = request.query_params.get("query", "")
        limit = min(int(request.query_params.get("limit", "10")), 50)

        if not query:
            return JSONResponse({"error": "query is required"}, status_code=400)

        runtime = _get_runtime()
        results = runtime.transcripts.search(query=query, limit=limit)
        return JSONResponse({"query": query, "results": results})

    @mcp.custom_route("/api/transcripts", methods=["GET"])
    async def api_list_transcripts(request):
        from starlette.responses import JSONResponse

        email = _check_auth(request, settings)
        if not email:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        platform = request.query_params.get("platform")
        channel = request.query_params.get("channel")
        limit = min(int(request.query_params.get("limit", "20")), 100)

        runtime = _get_runtime()
        items = runtime.transcripts.list_transcripts(platform=platform, channel=channel, limit=limit)
        return JSONResponse({"count": len(items), "items": items})

    @mcp.custom_route("/api/transcripts/{video_id}", methods=["GET"])
    async def api_read_transcript(request):
        import json
        from pathlib import Path

        from starlette.responses import JSONResponse

        email = _check_auth(request, settings)
        if not email:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        video_id = request.path_params["video_id"]
        fmt = request.query_params.get("format", "compact")
        offset = int(request.query_params.get("offset", "0"))
        limit_str = request.query_params.get("limit")
        limit = int(limit_str) if limit_str else None
        cookies_b64 = request.query_params.get("cookies_b64")

        runtime = _get_runtime()
        transcript = runtime.transcripts.get_by_video_id(video_id)
        if transcript is None:
            # Auto-queue
            url = video_id if video_id.startswith("http") else f"https://www.youtube.com/watch?v={video_id}"
            job = runtime.enqueue_transcription(url, cookies_b64=cookies_b64)
            return JSONResponse({"queued": True, "message": "Transcript not cached — queued.", **job})

        base = Path(str(transcript["path"]))

        if fmt == "compact":
            compact_path = base / "transcript.compact.txt"
            if compact_path.exists():
                full = compact_path.read_text(encoding="utf-8")
            else:
                payload = json.loads((base / "transcript.json").read_text(encoding="utf-8"))
                from cassandra_yt_mcp.services.storage import to_compact
                from cassandra_yt_mcp.types import TranscriptResult as TR, TranscriptSegment as TS

                result = TR(
                    text=payload.get("text", ""),
                    segments=[TS(start=s["start"], end=s["end"], text=s.get("text", ""), speaker=s.get("speaker")) for s in payload.get("segments", [])],
                    language=payload.get("language"),
                )
                full = to_compact(result)
            lines = full.splitlines(keepends=True)
            page = lines[offset:] if limit is None else lines[offset:offset + limit]
            return JSONResponse({"video_id": video_id, "content": "".join(page), "total_lines": len(lines), "offset": offset})

        if fmt in ("markdown", "text"):
            ext = "md" if fmt == "markdown" else "txt"
            full = (base / f"transcript.{ext}").read_text(encoding="utf-8")
            lines = full.splitlines(keepends=True)
            page = lines[offset:] if limit is None else lines[offset:offset + limit]
            return JSONResponse({"video_id": video_id, "format": fmt, "content": "".join(page), "total_lines": len(lines), "offset": offset, "lines_returned": len(page)})

        payload = (base / "transcript.json").read_text(encoding="utf-8")
        document = json.loads(payload)
        segments = document.get("segments", document.get("utterances", []))
        page = segments[offset:] if limit is None else segments[offset:offset + limit]
        return JSONResponse({"video_id": video_id, "format": "json", "content": page, "total_segments": len(segments), "offset": offset, "segments_returned": len(page)})

    @mcp.custom_route("/api/yt/search", methods=["POST"])
    async def api_yt_search(request):
        import time
        from pathlib import Path

        from starlette.responses import JSONResponse

        email = _check_auth(request, settings)
        if not email:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        body = await request.json()
        query = body.get("query", "")
        limit = min(body.get("limit", 10), 25)
        cookies_b64 = body.get("cookies_b64")

        if not query:
            return JSONResponse({"error": "query is required"}, status_code=400)

        runtime = _get_runtime()
        cookies_file = _write_cookies(cookies_b64)
        t0 = time.monotonic()
        try:
            results = runtime.youtube_info.search(query=query, limit=limit, cookies_file=cookies_file)
        except RuntimeError as exc:
            elapsed = time.monotonic() - t0
            return JSONResponse({"error": "search_failed", "message": str(exc), "elapsed_seconds": round(elapsed, 1)})
        finally:
            if cookies_file and cookies_file.exists():
                cookies_file.unlink()

        return JSONResponse({"query": query, "count": len(results), "results": results})

    @mcp.custom_route("/api/yt/channel", methods=["POST"])
    async def api_yt_channel(request):
        from starlette.responses import JSONResponse

        email = _check_auth(request, settings)
        if not email:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        body = await request.json()
        url = body.get("url", "")
        tab = body.get("tab", "videos")
        limit = min(body.get("limit", 20), 50)
        cookies_b64 = body.get("cookies_b64")

        if not url:
            return JSONResponse({"error": "url is required"}, status_code=400)
        if tab not in ("shorts", "videos", "streams"):
            tab = "videos"

        runtime = _get_runtime()
        cookies_file = _write_cookies(cookies_b64)
        try:
            results = runtime.youtube_info.list_channel_videos(channel_url=url, limit=limit, tab=tab, cookies_file=cookies_file)
        except RuntimeError as exc:
            return JSONResponse({"error": "channel_list_failed", "message": str(exc)})
        finally:
            if cookies_file and cookies_file.exists():
                cookies_file.unlink()

        return JSONResponse({"url": url, "tab": tab, "count": len(results), "results": results})

    @mcp.custom_route("/api/yt/comments", methods=["POST"])
    async def api_yt_comments(request):
        from starlette.responses import JSONResponse

        email = _check_auth(request, settings)
        if not email:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        body = await request.json()
        url = body.get("url", "")
        limit = min(body.get("limit", 20), 100)
        sort = body.get("sort", "top")
        cookies_b64 = body.get("cookies_b64")

        if not url:
            return JSONResponse({"error": "url is required"}, status_code=400)
        if sort not in ("top", "new"):
            sort = "top"

        runtime = _get_runtime()
        cookies_file = _write_cookies(cookies_b64)
        try:
            comments = runtime.youtube_info.get_comments(url=url, limit=limit, sort=sort, cookies_file=cookies_file)
        except RuntimeError as exc:
            return JSONResponse({"error": "comments_failed", "message": str(exc)})
        finally:
            if cookies_file and cookies_file.exists():
                cookies_file.unlink()

        return JSONResponse({"url": url, "count": len(comments), "sort": sort, "comments": comments})

    @mcp.custom_route("/api/watch-later/sync", methods=["POST"])
    async def api_watch_later_sync(request):
        from starlette.responses import JSONResponse

        email = _check_auth(request, settings)
        if not email:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        body = await request.json()
        cookies_b64 = body.get("cookies_b64")

        if not cookies_b64:
            return JSONResponse({"error": "no_cookies", "message": "YouTube cookies required."})

        runtime = _get_runtime()
        runtime.watch_later_repo.register_user(email, cookies_b64)
        try:
            result = runtime.watch_later_service.sync(email, cookies_b64)
            return JSONResponse(result)
        except RuntimeError as exc:
            return JSONResponse({"error": "sync_failed", "message": str(exc)})

    @mcp.custom_route("/api/watch-later/status/{user_id}", methods=["GET"])
    async def api_watch_later_status(request):
        from starlette.responses import JSONResponse

        email = _check_auth(request, settings)
        if not email:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        user_id = request.path_params["user_id"]
        # Only allow users to check their own status
        if user_id != email:
            return JSONResponse({"error": "forbidden"}, status_code=403)

        runtime = _get_runtime()
        user = runtime.watch_later_repo.get_user(user_id)
        if user is None:
            return JSONResponse({"registered": False, "user_id": user_id})

        seen = runtime.watch_later_repo.list_seen(user_id, limit=20)
        return JSONResponse({
            "registered": True, "user_id": user_id,
            "enabled": bool(user["enabled"]), "interval_minutes": user["interval_minutes"],
            "last_sync_at": user["last_sync_at"], "last_error": user["last_error"],
            "seen_count": runtime.watch_later_repo.count_seen(user_id),
            "recent_seen": seen,
        })


def _write_cookies(cookies_b64: str | None):
    """Write base64-encoded cookies to a temp file, return Path or None."""
    if not cookies_b64:
        return None
    import base64
    import tempfile
    from pathlib import Path

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        tmp.write(base64.b64decode(cookies_b64))
        tmp.close()
        return Path(tmp.name)
    except Exception:
        logger.exception("Failed to decode cookies")
        return None
