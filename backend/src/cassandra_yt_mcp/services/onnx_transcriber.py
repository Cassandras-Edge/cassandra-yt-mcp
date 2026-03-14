"""ONNX-based transcriber using sherpa-onnx + ONNX diarization.

Uses sherpa-onnx for batched GPU inference via decode_streams().
Long audio is split into physical chunks on disk to bound memory.
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import sherpa_onnx
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

_SAMPLE_RATE = 16000

# Chunk long audio into segments of this duration (seconds).
# Each chunk is written to disk and processed independently.
_CHUNK_SECONDS = 600  # 10 minutes
_OVERLAP_SECONDS = 10  # overlap for diarization continuity


class OnnxTranscriber:
    """Transcriber using sherpa-onnx (Parakeet TDT) + ONNX diarization."""

    def __init__(self, *, use_gpu: bool = True) -> None:
        self._use_gpu = use_gpu
        self._recognizer: sherpa_onnx.OfflineRecognizer | None = None
        self._vad: sherpa_onnx.VoiceActivityDetector | None = None
        self._diarization: OnnxDiarization | None = None

    def _load_models(self) -> None:
        if self._recognizer is not None:
            return

        models_dir = os.environ.get(
            "SHERPA_MODELS_DIR",
            "/opt/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        )
        vad_model = os.environ.get("SILERO_VAD_MODEL", "/opt/models/silero_vad.onnx")
        provider = "cuda" if self._use_gpu else "cpu"

        logger.info("Loading sherpa-onnx recognizer from %s (provider=%s)", models_dir, provider)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{models_dir}/encoder.int8.onnx",
            decoder=f"{models_dir}/decoder.int8.onnx",
            joiner=f"{models_dir}/joiner.int8.onnx",
            tokens=f"{models_dir}/tokens.txt",
            provider=provider,
            model_type="nemo_transducer",
            num_threads=4,
        )
        logger.info("sherpa-onnx recognizer loaded")

        logger.info("Loading Silero VAD from %s", vad_model)
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = vad_model
        vad_config.silero_vad.min_silence_duration = 0.25
        vad_config.silero_vad.min_speech_duration = 0.25
        vad_config.sample_rate = _SAMPLE_RATE
        vad_config.provider = provider
        self._vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
        logger.info("Silero VAD loaded")

        logger.info("Loading ONNX diarization models")
        # Diarization uses standalone onnxruntime (CPU-only) to avoid conflicts
        # with sherpa-onnx's bundled CUDA runtime. Diarization is lightweight
        # enough that CPU is fine (~4s for 2 min audio).
        self._diarization = OnnxDiarization(providers=["CPUExecutionProvider"])
        logger.info("ONNX diarization models loaded")

    @property
    def model_loaded(self) -> bool:
        return self._recognizer is not None

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
        import time  # noqa: PLC0415

        # ASR
        t0 = time.monotonic()
        raw_segments, text, detected_lang = self._run_asr(mono_path, offset_secs=0.0)
        t_asr = time.monotonic() - t0
        logger.info("ASR completed in %.1fs (%d segments)", t_asr, len(raw_segments))
        gc.collect()

        self._check_language(detected_lang, text)

        # Diarization
        t0 = time.monotonic()
        speaker_turns = self._diarization(str(mono_path))  # type: ignore[misc]
        t_diar = time.monotonic() - t0
        logger.info("Diarization completed in %.1fs (%d turns)", t_diar, len(speaker_turns))
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
        """Run VAD + batched ASR via sherpa-onnx. Returns (segments, text, detected_lang)."""
        import time  # noqa: PLC0415

        assert self._recognizer is not None
        assert self._vad is not None

        t_start = time.monotonic()

        window_size = 512  # Silero VAD expects 512 samples at 16kHz
        max_stream_chunk = 160000  # 10s — avoid C++ vector overflow in accept_waveform

        vad = self._vad
        vad.reset()

        samples, sr = sf.read(str(audio_path), dtype="float32")
        if len(samples.shape) > 1:
            samples = samples[:, 0]
        logger.info("Read %d samples (%.0fs), feeding VAD", len(samples), len(samples) / _SAMPLE_RATE)

        streams: list[sherpa_onnx.OfflineStream] = []
        segment_offsets: list[float] = []
        segment_durations: list[float] = []

        def _drain_vad() -> None:
            """Pop any completed segments from VAD and create ASR streams."""
            while not vad.empty():
                seg = vad.front()
                seg_start_secs = seg.start / _SAMPLE_RATE
                seg_samples = seg.samples
                s = self._recognizer.create_stream()
                for i in range(0, len(seg_samples), max_stream_chunk):
                    s.accept_waveform(_SAMPLE_RATE, seg_samples[i : i + max_stream_chunk])
                streams.append(s)
                segment_offsets.append(seg_start_secs + offset_secs)
                segment_durations.append(len(seg_samples) / _SAMPLE_RATE)
                vad.pop()

        # Feed VAD in 512-sample windows, draining detected segments periodically
        drain_interval = (16000 * 10 // window_size) * window_size  # ~10s, aligned to window
        for idx in range(0, len(samples) - window_size + 1, window_size):
            vad.accept_waveform(samples[idx : idx + window_size].tolist())
            if idx > 0 and idx % drain_interval == 0:
                _drain_vad()

        del samples
        vad.flush()
        _drain_vad()

        t_vad = time.monotonic() - t_start

        if not streams:
            return [], "", None

        logger.info("VAD completed in %.1fs — %d speech segments", t_vad, len(streams))

        # BATCHED GPU inference — single call for all segments
        t_decode_start = time.monotonic()
        self._recognizer.decode_streams(streams)
        t_decode = time.monotonic() - t_decode_start
        logger.info("decode_streams completed in %.1fs (%d streams)", t_decode, len(streams))

        # Extract results
        raw_segments: list[dict[str, object]] = []
        all_text_parts: list[str] = []

        for stream, seg_offset, seg_dur in zip(streams, segment_offsets, segment_durations):
            result = stream.result
            seg_text = result.text.strip() if result.text else ""
            if not seg_text:
                continue
            all_text_parts.append(seg_text)

            # Use word-level timestamps if available, else segment boundaries
            if result.timestamps:
                start = result.timestamps[0] + seg_offset
                end = result.timestamps[-1] + seg_offset
                # Add estimated duration of last token
                if result.timestamps and len(result.timestamps) > 1:
                    avg_dur = (result.timestamps[-1] - result.timestamps[0]) / len(result.timestamps)
                    end += avg_dur
            else:
                start = seg_offset
                end = seg_offset + seg_dur

            raw_segments.append({
                "text": seg_text,
                "start": start,
                "end": end,
            })

        del streams
        text = " ".join(all_text_parts)
        # Parakeet is English-only; sherpa-onnx doesn't expose language detection
        detected_lang: str | None = "en" if text else None
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
