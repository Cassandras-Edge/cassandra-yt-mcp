"""ONNX-based transcriber using onnx-asr + ONNX diarization.

Replaces NeMo/PyTorch with pure ONNX inference. ~2GB RAM vs ~6GB.
"""

from __future__ import annotations

import gc
import logging
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
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

        providers: list[str] | None = None
        if self._use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        logger.info("Loading onnx-asr model: %s", _PARAKEET_MODEL)
        vad = onnx_asr.load_vad("silero", providers=providers)
        model = onnx_asr.load_model(
            _PARAKEET_MODEL,
            providers=providers,
        )
        self._asr_model = model.with_timestamps().with_vad(vad)
        logger.info("onnx-asr model loaded")

        logger.info("Loading ONNX diarization models")
        diar_providers = providers or ["CPUExecutionProvider"]
        self._diarization = OnnxDiarization(providers=diar_providers)
        logger.info("ONNX diarization models loaded")

    @property
    def model_loaded(self) -> bool:
        return self._asr_model is not None

    @staticmethod
    def _to_wav(audio_path: Path) -> Path:
        """Convert any audio format to mono 16kHz WAV via ffmpeg.

        soundfile only supports WAV/FLAC/OGG — YouTube downloads are typically m4a/webm.
        """
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
        # Ensure mono 16kHz WAV on disk (ffmpeg already does this in _to_wav,
        # but handle raw .wav uploads that might be stereo/wrong sample rate)
        info = sf.info(str(wav_path))
        if info.samplerate != 16000 or info.channels > 1:
            mono_path = wav_path.with_suffix(".mono.wav")
            waveform, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
            if waveform.shape[1] > 1:
                waveform = waveform.mean(axis=1)
            else:
                waveform = waveform[:, 0]
            if sr != 16000:
                from scipy.signal import resample as scipy_resample  # noqa: PLC0415

                num_samples = int(len(waveform) * 16000 / sr)
                waveform = scipy_resample(waveform, num_samples).astype(np.float32)
            sf.write(str(mono_path), waveform, 16000)
            del waveform
            gc.collect()
            cleanup_mono = True
        else:
            mono_path = wav_path
            cleanup_mono = False

        try:
            # ASR with VAD chunking + timestamps (onnx-asr reads from disk)
            segments_iter = self._asr_model.recognize(str(mono_path))  # type: ignore[union-attr]

            all_text_parts: list[str] = []
            raw_segments: list[dict[str, object]] = []
            detected_lang: str | None = None

            if not isinstance(segments_iter, list):
                segments_iter = list(segments_iter)

            for seg_result in segments_iter:
                text = seg_result.text.strip() if seg_result.text else ""
                if not text:
                    continue
                all_text_parts.append(text)

                start = getattr(seg_result, "start", 0.0)
                end = getattr(seg_result, "end", 0.0)

                raw_segments.append({
                    "text": text,
                    "start": float(start),
                    "end": float(end),
                })

                if detected_lang is None:
                    lang = getattr(seg_result, "lang", None) or getattr(seg_result, "language", None)
                    if lang and isinstance(lang, str):
                        detected_lang = lang.strip().lower()[:2] or None

            del segments_iter
            text = " ".join(all_text_parts)
            gc.collect()

            if detected_lang and detected_lang not in SUPPORTED_LANGUAGES:
                raise UnsupportedLanguageError(detected_lang)
            if not detected_lang and not _text_uses_european_scripts(text):
                raise UnsupportedLanguageError("unknown")

            # Diarization — reads from disk, memory-bounded
            speaker_turns = self._diarization(str(mono_path))  # type: ignore[misc]
            gc.collect()

            return TranscriptResult(
                text=text,
                segments=_align_segments(raw_segments, speaker_turns),
                language=detected_lang or "en",
            )
        finally:
            if cleanup_mono:
                mono_path.unlink(missing_ok=True)


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
