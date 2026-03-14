"""Minimal FastAPI app for GPU worker mode.

Receives audio files, runs ASR + diarization, returns TranscriptResult JSON.
No database, no storage, no job queue — purely stateless.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, status

from cassandra_yt_mcp.config import Settings, load_settings

logger = logging.getLogger(__name__)


def _create_transcriber(settings: Settings) -> object:
    """Instantiate transcriber based on TRANSCRIPTION_ENGINE setting."""
    engine = settings.transcription_engine
    if engine == "onnx":
        from cassandra_yt_mcp.services.onnx_transcriber import OnnxTranscriber  # noqa: PLC0415

        logger.info("Using ONNX transcription engine")
        return OnnxTranscriber(use_gpu=True)
    else:
        from cassandra_yt_mcp.services.local_transcriber import LocalTranscriber  # noqa: PLC0415

        logger.info("Using NeMo transcription engine")
        return LocalTranscriber(huggingface_token=settings.huggingface_token)


def create_worker_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        transcriber = _create_transcriber(app_settings)
        # Warm models into VRAM on startup
        transcriber._load_models()  # type: ignore[attr-defined]
        app.state.transcriber = transcriber
        logger.info("Worker ready — models loaded (%s engine)", app_settings.transcription_engine)
        try:
            yield
        finally:
            pass

    app = FastAPI(title="cassandra-yt-mcp-worker", version="0.1.0", lifespan=lifespan)

    @app.get("/worker/healthz")
    def healthz() -> dict[str, object]:
        transcriber = app.state.transcriber
        gpu_info: dict[str, object] = {"available": False}
        try:
            import torch  # noqa: PLC0415

            gpu_info["available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                gpu_info["device"] = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory
                gpu_info["vram_gb"] = round(vram / (1024**3), 1)
        except Exception:  # noqa: BLE001
            # torch not available or GPU query failed — try onnxruntime
            try:
                import onnxruntime as ort  # noqa: PLC0415

                providers = ort.get_available_providers()
                gpu_info["available"] = "CUDAExecutionProvider" in providers
            except Exception:  # noqa: BLE001
                pass

        model_loaded = getattr(transcriber, "_asr_model", None) is not None
        # OnnxTranscriber uses model_loaded property
        if hasattr(transcriber, "model_loaded"):
            model_loaded = transcriber.model_loaded  # type: ignore[union-attr]

        return {
            "ok": True,
            "engine": app_settings.transcription_engine,
            "gpu": gpu_info,
            "model_loaded": model_loaded,
        }

    @app.post("/worker/transcribe")
    async def transcribe(audio: UploadFile) -> dict[str, object]:
        if audio.filename is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No audio file provided",
            )

        transcriber = app.state.transcriber

        # Stream uploaded audio to temp file (avoid buffering entire file in RAM)
        suffix = Path(audio.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            while chunk := await audio.read(1024 * 1024):  # 1MB chunks
                tmp.write(chunk)
            tmp_path = Path(tmp.name)

        try:
            # Run in thread pool so uvicorn stays responsive to health checks
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, partial(transcriber.transcribe, tmp_path),
            )
            return {
                "text": result.text,
                "language": result.language,
                "segments": [
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "speaker": s.speaker,
                    }
                    for s in result.segments
                ],
            }
        finally:
            tmp_path.unlink(missing_ok=True)

    return app
