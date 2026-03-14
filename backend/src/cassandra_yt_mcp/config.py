from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    host: str
    port: int
    poll_interval_seconds: int
    data_dir: Path
    database_path: Path
    assemblyai_api_key: str | None
    huggingface_token: str | None
    max_workers: int
    backend_api_token: str | None
    enable_local_transcription: bool
    role: str  # "standalone" | "coordinator" | "worker"
    gpu_workers: list[str]  # URLs for remote GPU workers (coordinator mode)
    worker_port: int  # Port for worker HTTP server (worker mode)
    download_concurrency: int  # Number of concurrent downloads (downloader mode)
    downloader_port: int  # Port for downloader healthz (downloader mode)
    transcription_engine: str  # "onnx" | "nemo"


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def load_settings() -> Settings:
    load_dotenv()
    data_dir = Path(os.getenv("DATA_DIR", "/data")).resolve()
    database_path = Path(
        os.getenv("DATABASE_PATH", str(data_dir / "cassandra_yt_mcp.sqlite3"))
    ).resolve()

    role = os.getenv("ROLE", "standalone").lower()
    if role not in ("standalone", "coordinator", "worker", "downloader"):
        raise ValueError(f"ROLE must be standalone|coordinator|worker|downloader, got '{role}'")

    gpu_workers_raw = os.getenv("GPU_WORKERS", "").strip()
    gpu_workers = [u.strip() for u in gpu_workers_raw.split(",") if u.strip()] if gpu_workers_raw else []

    return Settings(
        host=os.getenv("HOST", "0.0.0.0"),
        port=_as_int("PORT", 3000),
        poll_interval_seconds=_as_int("POLL_INTERVAL_SECONDS", 5),
        data_dir=data_dir,
        database_path=database_path,
        assemblyai_api_key=os.getenv("ASSEMBLYAI_API_KEY", "").strip() or None,
        huggingface_token=os.getenv("HUGGINGFACE_TOKEN", "").strip() or None,
        max_workers=_as_int("MAX_WORKERS", 3),
        backend_api_token=os.getenv("BACKEND_API_TOKEN", "").strip() or None,
        enable_local_transcription=os.getenv("ENABLE_LOCAL_TRANSCRIPTION", "true").lower()
        in ("true", "1", "yes"),
        role=role,
        gpu_workers=gpu_workers,
        worker_port=_as_int("WORKER_PORT", 3001),
        download_concurrency=_as_int("DOWNLOAD_CONCURRENCY", 2),
        downloader_port=_as_int("DOWNLOADER_PORT", 3002),
        transcription_engine=os.getenv("TRANSCRIPTION_ENGINE", "onnx").lower(),
    )
