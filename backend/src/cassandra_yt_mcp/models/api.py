from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TranscribeRequest(BaseModel):
    url: str = Field(min_length=1)
    cookies_b64: str | None = None


class WatchLaterSyncRequest(BaseModel):
    user_id: str = Field(min_length=1)
    cookies_b64: str = Field(min_length=1)


class HealthResponse(BaseModel):
    ok: bool
    worker_running: bool
    db_path: str


class ApiEnvelope(BaseModel):
    data: dict[str, Any]
