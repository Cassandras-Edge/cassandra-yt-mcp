"""ONNX-based transcriber using onnx-asr + ONNX diarization.

Replaces NeMo/PyTorch with pure ONNX inference. ~2GB RAM vs ~6GB.
Long audio is split into physical chunks on disk to bound memory.
"""

from __future__ import annotations

import gc
import logging
import subprocess
import unicodedata
from pathlib import Path

import soundfile as sf

from cassandra_yt_mcp.services.onnx_diarization import OnnxDiarization
from cassandra_yt_mcp.services.transcriber import UnsupportedLanguageError
from cassandra_yt_mcp.types import TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {
        "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
        "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
        "sl", "es", "sv", "ru", "uk",
    }
)
_EUROPEAN_SCRIPTS: frozenset[str] = frozenset({"LATIN", "CYRILLIC", "GREEK"})
_PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v3"

# Chunk long audio into segments of this duration (seconds).
# Each chunk is written to disk and processed independently.
_CHUNK_SECONDS = 600  # 10 minutes
_OVERLAP_SECONDS = 10  # overlap for diarization continuity


class OnnxTranscriber:
    """Transcriber using onnx-asr (Parakeet TDT) + ONNX diarization."""

    def __init__(self, *, use_gpu: bool = True) -> None:
        self._use_gpu = use_gpu
        self._asr_model: object | None = None
        self._diarization: OnnxDiarization | None = None

    def _load_models(self) -> None:
        if self._asr_model is not None:
            return

        import onnx_asr  # noqa: PLC0415

        providers: list | None = None
        if self._use_gpu:
            import os  # noqa: PLC0415

            trt_cache = os.environ.get("TRT_ENGINE_CACHE_PATH", "/data/trt_cache")
            os.makedirs(trt_cache, exist_ok=True)
            providers = [
                ("TensorrtExecutionProvider", {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": trt_cache,
                }),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

        logger.info("Loading onnx-asr model: %s (TensorRT FP16 preferred)", _PARAKEET_MODEL)
        vad = onnx_asr.load_vad("silero", providers=providers)
        model = onnx_asr.load_model(
            _PARAKEET_MODEL,
            providers=providers,
        )
        self._asr_model = model.with_timestamps().with_vad(vad)
        logger.info("onnx-asr model loaded")

        logger.info("Loading ONNX diarization models")
        # Diarization uses simpler models — CUDA is sufficient, TRT overhead not worth it
        diar_providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._use_gpu else ["CPUExecutionProvider"]
        self._diarization = OnnxDiarization(providers=diar_providers)
        logger.info("ONNX diarization models loaded")

    @property
    def model_loaded(self) -> bool:
        return self._asr_model is not None

    @staticmethod
    def _to_wav(audio_path: Path) -> Path:
        """Convert any audio format to mono 16kHz WAV via ffmpeg."""
        wav_path = audio_path.with_suffix(".wav")
        if audio_path.suffix.lower() in (".wav",):
            return audio_path
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(wav_path),
            ],
            check=True,
            capture_output=True,
        )
        return wav_path

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        self._load_models()

        # Convert to WAV first (m4a/webm not supported by soundfile)
        wav_path = self._to_wav(audio_path)
        cleanup_wav = wav_path != audio_path

        try:
            return self._transcribe_wav(wav_path)
        finally:
            if cleanup_wav:
                wav_path.unlink(missing_ok=True)

    def _transcribe_wav(self, wav_path: Path) -> TranscriptResult:
        # Ensure mono 16kHz
        info = sf.info(str(wav_path))
        if info.samplerate != 16000 or info.channels > 1:
            mono_path = wav_path.with_suffix(".mono.wav")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(wav_path),
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                    str(mono_path),
                ],
                check=True,
                capture_output=True,
            )
            cleanup_mono = True
        else:
            mono_path = wav_path
            cleanup_mono = False

        try:
            mono_info = sf.info(str(mono_path))
            duration_secs = mono_info.frames / mono_info.samplerate
            logger.info("Processing %.0fs audio in single pass", duration_secs)
            return self._process_single(mono_path)
        finally:
            if cleanup_mono:
                mono_path.unlink(missing_ok=True)

    def _process_single(self, mono_path: Path) -> TranscriptResult:
        """Process a short audio file (< ~10 min) in one pass."""
        # ASR
        raw_segments, text, detected_lang = self._run_asr(mono_path, offset_secs=0.0)
        gc.collect()

        self._check_language(detected_lang, text)

        # Diarization
        speaker_turns = self._diarization(str(mono_path))  # type: ignore[misc]
        gc.collect()

        return TranscriptResult(
            text=text,
            segments=_align_segments(raw_segments, speaker_turns),
            language=detected_lang or "en",
        )

    def _process_chunked(self, mono_path: Path, total_duration: float) -> TranscriptResult:
        """Process long audio in chunks: split to disk, ASR+diarize each, merge."""
        sr = 16000
        chunk_samples = _CHUNK_SECONDS * sr
        overlap_samples = _OVERLAP_SECONDS * sr
        stride_samples = chunk_samples - overlap_samples
        total_samples = int(total_duration * sr)

        all_raw_segments: list[dict[str, object]] = []
        all_speaker_turns: list[tuple[float, float, str]] = []
        detected_lang: str | None = None

        chunk_idx = 0
        offset_sample = 0

        while offset_sample < total_samples:
            end_sample = min(offset_sample + chunk_samples, total_samples)
            num_frames = end_sample - offset_sample
            offset_secs = offset_sample / sr

            # Write this chunk to a temp file on disk
            chunk_path = mono_path.with_suffix(f".chunk{chunk_idx}.wav")
            try:
                with sf.SoundFile(str(mono_path)) as f:
                    f.seek(offset_sample)
                    chunk_data = f.read(num_frames, dtype="int16")
                sf.write(str(chunk_path), chunk_data, sr, subtype="PCM_16")
                del chunk_data
                gc.collect()

                logger.info(
                    "Processing chunk %d (%.0fs–%.0fs)",
                    chunk_idx, offset_secs, end_sample / sr,
                )

                # ASR on this chunk
                chunk_segments, _, lang = self._run_asr(
                    chunk_path, offset_secs=offset_secs,
                )
                gc.collect()

                if not detected_lang and lang:
                    detected_lang = lang

                # Diarization on this chunk
                chunk_turns = self._diarization(str(chunk_path))  # type: ignore[misc]
                # Shift diarization timestamps to absolute positions
                chunk_turns = [
                    (s + offset_secs, e + offset_secs, spk)
                    for s, e, spk in chunk_turns
                ]
                gc.collect()

                all_raw_segments.extend(chunk_segments)
                all_speaker_turns.extend(chunk_turns)

            finally:
                chunk_path.unlink(missing_ok=True)

            chunk_idx += 1
            offset_sample += stride_samples

        text = " ".join(
            str(s.get("text", "")).strip()
            for s in all_raw_segments
            if str(s.get("text", "")).strip()
        )

        self._check_language(detected_lang, text)

        # Merge overlapping speaker turns from adjacent chunks
        all_speaker_turns.sort(key=lambda t: t[0])

        return TranscriptResult(
            text=text,
            segments=_align_segments(all_raw_segments, all_speaker_turns),
            language=detected_lang or "en",
        )

    def _run_asr(
        self, audio_path: Path, *, offset_secs: float,
    ) -> tuple[list[dict[str, object]], str, str | None]:
        """Run ASR on a single audio file. Returns (segments, text, detected_lang)."""
        segments_iter = self._asr_model.recognize(str(audio_path))  # type: ignore[union-attr]

        raw_segments: list[dict[str, object]] = []
        all_text_parts: list[str] = []
        detected_lang: str | None = None

        if not isinstance(segments_iter, list):
            segments_iter = list(segments_iter)

        for seg_result in segments_iter:
            seg_text = seg_result.text.strip() if seg_result.text else ""
            if not seg_text:
                continue
            all_text_parts.append(seg_text)

            start = float(getattr(seg_result, "start", 0.0)) + offset_secs
            end = float(getattr(seg_result, "end", 0.0)) + offset_secs

            raw_segments.append({
                "text": seg_text,
                "start": start,
                "end": end,
            })

            if detected_lang is None:
                lang = getattr(seg_result, "lang", None) or getattr(seg_result, "language", None)
                if lang and isinstance(lang, str):
                    detected_lang = lang.strip().lower()[:2] or None

        del segments_iter
        text = " ".join(all_text_parts)
        return raw_segments, text, detected_lang

    @staticmethod
    def _check_language(detected_lang: str | None, text: str) -> None:
        if detected_lang and detected_lang not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(detected_lang)
        if not detected_lang and not _text_uses_european_scripts(text):
            raise UnsupportedLanguageError("unknown")


def _text_uses_european_scripts(text: str) -> bool:
    if not text:
        return True
    european = 0
    total = 0
    for char in text:
        if not char.isalpha():
            continue
        total += 1
        if unicodedata.name(char, "").split(" ")[0] in _EUROPEAN_SCRIPTS:
            european += 1
    return total == 0 or (european / total) >= 0.7


def _find_best_speaker(
    seg_start: float,
    seg_end: float,
    speaker_turns: list[tuple[float, float, str]],
) -> str | None:
    best_speaker: str | None = None
    best_overlap = 0.0
    for turn_start, turn_end, speaker in speaker_turns:
        overlap = max(0.0, min(seg_end, turn_end) - max(seg_start, turn_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker


def _align_segments(
    raw_segments: list[dict[str, object]],
    speaker_turns: list[tuple[float, float, str]],
) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for segment in raw_segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        segments.append(
            TranscriptSegment(
                start=start,
                end=end,
                text=text,
                speaker=_find_best_speaker(start, end, speaker_turns),
            )
        )
    return segments
