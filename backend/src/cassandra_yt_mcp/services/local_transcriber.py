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
            import nemo.collections.asr as nemo_asr  # noqa: PLC0415

            logger.info("Loading Parakeet model: %s", _PARAKEET_MODEL)
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=_PARAKEET_MODEL)
            model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=[256, 256],
            )
            model.eval()
            self._asr_model = model

        if self._diarization_pipeline is None:
            import torch.serialization  # noqa: PLC0415

            from pyannote.audio import Pipeline  # noqa: PLC0415

            # pyannote 3.x checkpoints contain TorchVersion which torch 2.6+
            # rejects under the default weights_only=True policy.
            torch.serialization.add_safe_globals([__import__("torch").torch_version.TorchVersion])

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
    def _ensure_mono(audio_path: Path) -> Path:
        """Downmix to mono 16kHz WAV if needed — Parakeet expects single-channel input."""
        import torchaudio  # noqa: PLC0415

        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] == 1 and sr == 16000:
            return audio_path
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        mono_path = audio_path.with_suffix(".mono.wav")
        torchaudio.save(str(mono_path), waveform, 16000)
        return mono_path

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        import torch  # noqa: PLC0415

        self._load_models()
        mono_path = self._ensure_mono(audio_path)
        audio_str = str(mono_path)
        with torch.amp.autocast("cuda"):
            hypothesis = self._asr_model.transcribe(  # type: ignore[union-attr]
                [audio_str], timestamps=True, batch_size=1,
            )[0]
        text = str(hypothesis.text).strip()
        detected_lang = self._detect_language(hypothesis, text)
        if detected_lang and detected_lang not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(detected_lang)
        if not detected_lang and not self._text_uses_european_scripts(text):
            raise UnsupportedLanguageError("unknown")
        diarization = self._diarization_pipeline(audio_str)  # type: ignore[operator]
        speaker_turns = self._extract_speaker_turns(diarization)
        raw_segments = hypothesis.timestamp.get("segment", []) if hypothesis.timestamp else []
        return TranscriptResult(
            text=text,
            segments=self._align_segments(raw_segments, speaker_turns),
            language=detected_lang or "en",
        )

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
