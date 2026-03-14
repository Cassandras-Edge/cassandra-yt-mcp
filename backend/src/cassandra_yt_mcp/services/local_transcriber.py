from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

from cassandra_yt_mcp.services.transcriber import UnsupportedLanguageError
from cassandra_yt_mcp.types import TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {
        "bg",
        "hr",
        "cs",
        "da",
        "nl",
        "en",
        "et",
        "fi",
        "fr",
        "de",
        "el",
        "hu",
        "it",
        "lv",
        "lt",
        "mt",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "es",
        "sv",
        "ru",
        "uk",
    }
)
_EUROPEAN_SCRIPTS: frozenset[str] = frozenset({"LATIN", "CYRILLIC", "GREEK"})
_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
_PYANNOTE_PIPELINE = "pyannote/speaker-diarization-3.1"
_CHUNK_SECONDS = 600  # 10-min chunks for ASR to avoid VRAM OOM
_OVERLAP_SECONDS = 30  # 30s overlap for reliable LCS merge at boundaries
_BATCH_SIZE = 32  # GPU batch inference — RTX 5080 16GB can handle ~32 safely


class LocalTranscriber:
    def __init__(self, *, huggingface_token: str | None = None) -> None:
        self._device = "cpu"
        try:
            import torch  # noqa: PLC0415

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            logger.info("torch not installed yet; local transcription models will load lazily")
        self._huggingface_token = huggingface_token
        self._asr_model: object | None = None
        self._diarization_pipeline: object | None = None

    def _load_models(self) -> None:
        if self._asr_model is None:
            import gc  # noqa: PLC0415

            import torch  # noqa: PLC0415
            import nemo.collections.asr as nemo_asr  # noqa: PLC0415

            logger.info("Loading Parakeet model: %s", _PARAKEET_MODEL)
            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=_PARAKEET_MODEL, map_location=self._device,
            )
            model = model.to(self._device)
            model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=[256, 256],
            )
            model.change_subsampling_conv_chunking_factor(1)  # auto-chunk conv to reduce VRAM
            model.eval()
            gc.collect()
            torch.cuda.empty_cache()
            self._asr_model = model

        if self._diarization_pipeline is None:
            import os  # noqa: PLC0415

            from pyannote.audio import Pipeline  # noqa: PLC0415

            # pyannote 3.x checkpoints use globals (TorchVersion, omegaconf
            # classes, etc.) that torch 2.6+ rejects under weights_only=True.
            # Models come from HuggingFace so this is safe.
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

            pipeline = Pipeline.from_pretrained(
                _PYANNOTE_PIPELINE,
                use_auth_token=self._huggingface_token,
            )
            if pipeline is None:
                raise RuntimeError(
                    f"Failed to load {_PYANNOTE_PIPELINE}. "
                    "Ensure you've accepted the model license on HuggingFace."
                )
            pipeline.to(self._device)
            self._diarization_pipeline = pipeline

    @staticmethod
    def _ensure_mono(audio_path: Path) -> tuple[Path, int]:
        """Downmix to mono 16kHz WAV if needed — Parakeet expects single-channel input.

        Returns (mono_path, duration_samples) so callers don't need to reload.
        """
        import gc  # noqa: PLC0415

        import torchaudio  # noqa: PLC0415

        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        duration_samples = waveform.shape[1]
        if waveform.shape[0] == 1 and sr == 16000 and audio_path.suffix in (".wav",):
            # Already correct format — still return duration
            del waveform
            gc.collect()
            return audio_path, duration_samples
        mono_path = audio_path.with_suffix(".mono.wav")
        torchaudio.save(str(mono_path), waveform, 16000)
        del waveform
        gc.collect()
        return mono_path, duration_samples

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        import gc  # noqa: PLC0415

        self._load_models()
        mono_path, duration_samples = self._ensure_mono(audio_path)
        audio_str = str(mono_path)
        sr = 16000
        duration_secs = duration_samples / sr

        if duration_secs <= _CHUNK_SECONDS + _OVERLAP_SECONDS:
            hypothesis = self._transcribe_file(audio_str)
            text = str(hypothesis.text).strip()
            detected_lang = self._detect_language(hypothesis, text)
            raw_segments = hypothesis.timestamp.get("segment", []) if hypothesis.timestamp else []
        else:
            text, raw_segments, detected_lang = self._transcribe_chunked(mono_path)

        gc.collect()

        if detected_lang and detected_lang not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(detected_lang)
        if not detected_lang and not self._text_uses_european_scripts(text):
            raise UnsupportedLanguageError("unknown")

        diarization = self._diarization_pipeline(audio_str)  # type: ignore[operator]
        speaker_turns = self._extract_speaker_turns(diarization)
        del diarization
        gc.collect()
        return TranscriptResult(
            text=text,
            segments=self._align_segments(raw_segments, speaker_turns),
            language=detected_lang or "en",
        )

    def _transcribe_file(self, audio_path: str) -> object:
        import torch  # noqa: PLC0415

        with torch.inference_mode():
            return self._asr_model.transcribe(  # type: ignore[union-attr]
                [audio_path], timestamps=True, batch_size=_BATCH_SIZE,
            )[0]

    def _transcribe_chunked(
        self, mono_path: Path,
    ) -> tuple[str, list[dict[str, object]], str | None]:
        """Transcribe long audio in chunks, loading each from disk to avoid
        holding the full waveform in memory."""
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415
        import torchaudio  # noqa: PLC0415

        sr = 16000
        chunk_samples = _CHUNK_SECONDS * sr
        overlap_samples = _OVERLAP_SECONDS * sr
        stride = chunk_samples - overlap_samples

        # Get total length without loading full audio into memory
        info = torchaudio.info(str(mono_path))
        total_samples = info.num_frames

        merged_segments: list[dict[str, object]] = []
        detected_lang: str | None = None
        chunk_idx = 0
        offset = 0

        while offset < total_samples:
            end = min(offset + chunk_samples, total_samples)
            num_frames = end - offset

            # Load only this chunk from disk
            chunk, _ = torchaudio.load(
                str(mono_path), frame_offset=offset, num_frames=num_frames,
            )
            chunk_path = mono_path.with_suffix(f".chunk{chunk_idx}.wav")
            torchaudio.save(str(chunk_path), chunk, sr)
            del chunk
            chunk_offset_secs = offset / sr

            logger.info(
                "Transcribing chunk %d (%.1fs – %.1fs)",
                chunk_idx, chunk_offset_secs, end / sr,
            )
            hyp = self._transcribe_file(str(chunk_path))

            # Shift timestamps to absolute positions
            chunk_segments: list[dict[str, object]] = []
            for seg in (hyp.timestamp.get("segment", []) if hyp.timestamp else []):
                seg["start"] = float(seg.get("start", 0.0)) + chunk_offset_secs
                seg["end"] = float(seg.get("end", 0.0)) + chunk_offset_secs
                chunk_segments.append(seg)

            if not detected_lang:
                detected_lang = self._detect_language(hyp, str(hyp.text).strip())

            # Merge with previous segments using LCS in the overlap region
            merged_segments = _merge_segments_lcs(
                merged_segments, chunk_segments, _OVERLAP_SECONDS,
            )

            del hyp
            torch.cuda.empty_cache()
            gc.collect()
            chunk_path.unlink(missing_ok=True)
            chunk_idx += 1
            offset += stride

        text = " ".join(
            str(s.get("segment") or s.get("text") or "").strip()
            for s in merged_segments
            if str(s.get("segment") or s.get("text") or "").strip()
        )
        return text, merged_segments, detected_lang

    @staticmethod
    def _detect_language(hypothesis: object, text: str) -> str | None:
        lang = getattr(hypothesis, "lang", None) or getattr(hypothesis, "language", None)
        if lang and isinstance(lang, str):
            code = lang.strip().lower()[:2]
            return code if code else None
        return None

    @staticmethod
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

    @staticmethod
    def _extract_speaker_turns(diarization: object) -> list[tuple[float, float, str]]:
        turns: list[tuple[float, float, str]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((turn.start, turn.end, str(speaker)))
        return turns

    @staticmethod
    def _align_segments(
        raw_segments: list[dict[str, object]],
        speaker_turns: list[tuple[float, float, str]],
    ) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        for segment in raw_segments:
            text = str(segment.get("segment") or segment.get("text") or "").strip()
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


def _seg_text(seg: dict[str, object]) -> str:
    return str(seg.get("segment") or seg.get("text") or "").strip().lower()


def _merge_segments_lcs(
    a: list[dict[str, object]],
    b: list[dict[str, object]],
    overlap_secs: float,
) -> list[dict[str, object]]:
    """Merge two segment lists using LCS in the overlap region.

    Adapted from parakeet-mlx's merge_longest_common_subsequence.
    Matches segments by text equality and timestamp proximity, then
    takes A before the match point and B after it.
    """
    if not a:
        return list(b)
    if not b:
        return list(a)

    a_end = float(a[-1].get("end", 0.0))
    b_start = float(b[0].get("start", 0.0))

    # No overlap — simple concatenation
    if a_end <= b_start:
        return a + b

    # Extract segments in the overlap zone
    overlap_a = [s for s in a if float(s.get("end", 0.0)) > b_start - overlap_secs]
    overlap_b = [s for s in b if float(s.get("start", 0.0)) < a_end + overlap_secs]

    if len(overlap_a) < 2 or len(overlap_b) < 2:
        # Not enough overlap — cut at midpoint
        cutoff = (a_end + b_start) / 2
        return [s for s in a if float(s.get("end", 0.0)) <= cutoff] + [
            s for s in b if float(s.get("start", 0.0)) >= cutoff
        ]

    # LCS dynamic programming on segment text + timestamp proximity
    dp = [[0] * (len(overlap_b) + 1) for _ in range(len(overlap_a) + 1)]
    for i in range(1, len(overlap_a) + 1):
        for j in range(1, len(overlap_b) + 1):
            sa, sb = overlap_a[i - 1], overlap_b[j - 1]
            text_match = _seg_text(sa) == _seg_text(sb)
            time_close = (
                abs(float(sa.get("start", 0.0)) - float(sb.get("start", 0.0)))
                < overlap_secs / 2
            )
            if text_match and time_close:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find matching pairs
    pairs: list[tuple[int, int]] = []
    i, j = len(overlap_a), len(overlap_b)
    while i > 0 and j > 0:
        sa, sb = overlap_a[i - 1], overlap_b[j - 1]
        text_match = _seg_text(sa) == _seg_text(sb)
        time_close = (
            abs(float(sa.get("start", 0.0)) - float(sb.get("start", 0.0)))
            < overlap_secs / 2
        )
        if text_match and time_close:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()

    if not pairs:
        cutoff = (a_end + b_start) / 2
        return [s for s in a if float(s.get("end", 0.0)) <= cutoff] + [
            s for s in b if float(s.get("start", 0.0)) >= cutoff
        ]

    # Build result: A up to first match, then B from first match onward
    a_offset = len(a) - len(overlap_a)
    first_a_idx = a_offset + pairs[0][0]
    first_b_idx = pairs[0][1]

    return a[:first_a_idx] + b[first_b_idx:]


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
