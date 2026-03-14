from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from cassandra_yt_mcp.config import Settings, load_settings
from cassandra_yt_mcp.metrics import api_request_duration_seconds, api_requests_total
from cassandra_yt_mcp.models.api import TranscribeRequest, WatchLaterSyncRequest
from cassandra_yt_mcp.runtime import AppRuntime

_SKIP_METRICS_PATHS = frozenset({"/metrics", "/healthz"})


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = AppRuntime(app_settings)
        runtime.start()
        app.state.runtime = runtime
        app.state.settings = app_settings
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(title="cassandra-yt-mcp-backend", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):  # noqa: ANN001
        path = request.url.path
        if path in _SKIP_METRICS_PATHS:
            return await call_next(request)
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        endpoint = path.split("/")[:4]  # collapse /api/transcripts/<id> → /api/transcripts
        endpoint_label = "/".join(endpoint[:4]) if len(endpoint) >= 4 else path
        method = request.method
        api_requests_total.labels(method=method, endpoint=endpoint_label, status=response.status_code).inc()
        api_request_duration_seconds.labels(method=method, endpoint=endpoint_label).observe(elapsed)
        return response

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    def get_runtime(request: Request) -> AppRuntime:
        return request.app.state.runtime  # type: ignore[no-any-return]

    def require_api_token(request: Request) -> None:
        configured = request.app.state.settings.backend_api_token
        if not configured:
            return
        auth_header = request.headers.get("authorization", "")
        expected = f"Bearer {configured}"
        if auth_header != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")

    @app.get("/healthz")
    def health(request: Request) -> dict[str, object]:
        runtime: AppRuntime = request.app.state.runtime
        settings: Settings = request.app.state.settings
        return {
            "ok": True,
            "worker_running": runtime.worker.is_running,
            "db_path": str(settings.database_path),
        }

    @app.post("/api/jobs/transcribe", dependencies=[Depends(require_api_token)])
    def transcribe_job(payload: TranscribeRequest, runtime: AppRuntime = Depends(get_runtime)) -> dict[str, object]:
        return runtime.enqueue_transcription(payload.url, cookies_b64=payload.cookies_b64)

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(require_api_token)])
    def job_status(job_id: str, runtime: AppRuntime = Depends(get_runtime)) -> dict[str, object]:
        return runtime.get_job_status(job_id)

    @app.get("/api/transcripts", dependencies=[Depends(require_api_token)])
    def list_transcripts(
        runtime: AppRuntime = Depends(get_runtime),
        platform: str | None = None,
        channel: str | None = None,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, object]:
        items = runtime.transcripts.list_transcripts(platform=platform, channel=channel, limit=limit)
        return {"count": len(items), "items": items}

    @app.get("/api/transcripts/search", dependencies=[Depends(require_api_token)])
    def search_transcripts(
        query: str,
        runtime: AppRuntime = Depends(get_runtime),
        limit: int = Query(default=10, ge=1, le=50),
    ) -> dict[str, object]:
        return {"query": query, "results": runtime.transcripts.search(query=query, limit=limit)}

    @app.get("/api/transcripts/{video_id}", dependencies=[Depends(require_api_token)])
    def read_transcript(
        video_id: str,
        runtime: AppRuntime = Depends(get_runtime),
        format: str = Query(default="markdown", pattern="^(markdown|text|json)$"),
        offset: int = Query(default=0, ge=0),
        limit: int | None = Query(default=None, ge=1),
    ) -> dict[str, object]:
        transcript = runtime.transcripts.get_by_video_id(video_id)
        if transcript is None:
            url = f"https://www.youtube.com/watch?v={video_id}"
            transcription = runtime.enqueue_transcription(url)
            return {
                "auto_transcribe": True,
                "video_id": video_id,
                "transcription": transcription,
            }

        warnings = _transcript_warnings(transcript)
        base = Path(str(transcript["path"]))
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
                "warnings": warnings,
            }

        payload = (base / "transcript.json").read_text(encoding="utf-8")
        import json  # noqa: PLC0415

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
            "warnings": warnings,
        }

    @app.get("/api/youtube/search", dependencies=[Depends(require_api_token)])
    def yt_search(
        query: str,
        runtime: AppRuntime = Depends(get_runtime),
        limit: int = Query(default=10, ge=1, le=25),
    ) -> dict[str, object]:
        try:
            results = runtime.youtube_info.search(query=query, limit=limit)
        except RuntimeError as exc:
            return {"error": "search_failed", "message": str(exc)}
        return {"query": query, "count": len(results), "results": results}

    @app.get("/api/youtube/metadata", dependencies=[Depends(require_api_token)])
    def get_metadata(url: str, runtime: AppRuntime = Depends(get_runtime)) -> dict[str, object]:
        try:
            return {"url": url, "metadata": runtime.youtube_info.get_metadata(url=url)}
        except RuntimeError as exc:
            return {"error": "metadata_failed", "message": str(exc)}

    @app.get("/api/youtube/comments", dependencies=[Depends(require_api_token)])
    def get_comments(
        url: str,
        runtime: AppRuntime = Depends(get_runtime),
        limit: int = Query(default=20, ge=1, le=100),
        sort: str = Query(default="top", pattern="^(top|new)$"),
    ) -> dict[str, object]:
        try:
            comments = runtime.youtube_info.get_comments(url=url, limit=limit, sort=sort)
        except RuntimeError as exc:
            return {"error": "comments_failed", "message": str(exc)}
        return {"url": url, "count": len(comments), "sort": sort, "comments": comments}

    @app.post("/api/watch-later/sync", dependencies=[Depends(require_api_token)])
    def watch_later_sync(
        payload: WatchLaterSyncRequest,
        runtime: AppRuntime = Depends(get_runtime),
    ) -> dict[str, object]:
        # Register/update user cookies for auto-sync
        runtime.watch_later_repo.register_user(payload.user_id, payload.cookies_b64)
        try:
            return runtime.watch_later_service.sync(payload.user_id, payload.cookies_b64)
        except RuntimeError as exc:
            return {"error": "sync_failed", "message": str(exc)}

    @app.get("/api/watch-later/status", dependencies=[Depends(require_api_token)])
    def watch_later_status(
        user_id: str,
        runtime: AppRuntime = Depends(get_runtime),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, object]:
        user = runtime.watch_later_repo.get_user(user_id)
        if user is None:
            return {"registered": False, "user_id": user_id}
        seen = runtime.watch_later_repo.list_seen(user_id, limit=limit)
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

    return app


def _transcript_warnings(transcript: dict[str, object]) -> list[str]:
    warnings: list[str] = [
        "Transcript was generated by local GPU (Parakeet TDT 0.6b). "
        "Accuracy may vary — expect minor errors in proper nouns, numbers, and homophones.",
    ]
    speaker_count = transcript.get("speaker_count")
    if speaker_count and int(str(speaker_count)) > 1:
        warnings.append(
            f"Speaker diarization detected {speaker_count} speakers (pyannote). "
            "Speaker labels (SPEAKER_00, etc.) are best-effort and may be "
            "inaccurate during overlapping speech or speaker transitions."
        )
    elif speaker_count and int(str(speaker_count)) == 1:
        warnings.append("Single speaker detected — speaker labels should be reliable.")
    confidence = transcript.get("confidence")
    if confidence is not None:
        score = float(str(confidence))
        if score < 0.7:
            warnings.append(
                f"Low confidence score ({score:.0%}). "
                "Audio quality may be poor or content may include music/non-speech."
            )
    return warnings
