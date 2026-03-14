"""ONNX-based speaker diarization.

Adapted from pyannote-onnx-extended (MIT, Samson Woof).
Replaces librosa with scipy/numpy, pyannote.core with plain tuples,
and av with soundfile — eliminating torch/pyannote/numba deps entirely.

Memory-bounded: reads audio from disk in chunks, never holds the full
waveform in memory. Handles multi-hour audio without OOM.
"""

from __future__ import annotations

import gc
import logging
from itertools import permutations

import numpy as np
import onnxruntime as ort
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import get_window, stft as _scipy_stft  # noqa: F401
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)

_SEGMENTATION_REPO = "onnx-community/pyannote-segmentation-3.0"
_EMBEDDING_REPO = "onnx-community/wespeaker-voxceleb-resnet34-LM"

# Max segment duration (seconds) for embedding extraction.
# Longer segments are split into sub-segments to bound memory.
_MAX_EMB_SEGMENT_SECS = 30.0


# ── Mel spectrogram (replaces librosa.feature.melspectrogram) ────────────


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Build a Mel filterbank matrix (n_mels, n_fft//2+1)."""
    fmax = sr / 2.0
    mel_min = _hz_to_mel(np.array(0.0))
    mel_max = _hz_to_mel(np.array(fmax))
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hz = _mel_to_hz(mels)

    fft_freqs = np.linspace(0, fmax, n_fft // 2 + 1)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (fft_freqs - lo) / max(mid - lo, 1e-10)
        down = (hi - fft_freqs) / max(hi - mid, 1e-10)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return fb


def _melspectrogram(
    y: np.ndarray,
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
) -> np.ndarray:
    """Compute log-mel spectrogram using scipy STFT + numpy mel filterbank.

    Returns shape (n_mels, T).
    """
    window = get_window("hamming", n_fft, fftbins=True).astype(np.float64)
    _, _, zxx = _scipy_stft(
        y.astype(np.float64),
        fs=sr,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx).astype(np.float32) ** 2
    fb = _mel_filterbank(sr, n_fft, n_mels)
    return fb @ power


# ── Timeline merge (replaces pyannote.core.Timeline.support) ────────────


def _merge_segments(
    segments: list[tuple[float, float]], collar: float = 0.0,
) -> list[tuple[float, float]]:
    """Merge overlapping or near-touching segments (sorted by start)."""
    if not segments:
        return []
    segments = sorted(segments)
    merged: list[tuple[float, float]] = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + collar:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


# ── Annotation merge (replaces pyannote.core.Annotation.support) ────────


def _merge_annotation(
    turns: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Merge consecutive turns with the same speaker label."""
    if not turns:
        return []
    turns = sorted(turns, key=lambda t: (t[2], t[0]))
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for start, end, spk in turns:
        by_speaker.setdefault(spk, []).append((start, end))
    result: list[tuple[float, float, str]] = []
    for spk, segs in by_speaker.items():
        for start, end in _merge_segments(segs, collar=0.0):
            result.append((start, end, spk))
    result.sort(key=lambda t: t[0])
    return result


def _read_audio_chunk(
    audio_path: str, start_sample: int, num_samples: int,
) -> np.ndarray:
    """Read a chunk of audio from disk without loading the full file."""
    with sf.SoundFile(audio_path) as f:
        f.seek(start_sample)
        return f.read(num_samples, dtype="float32")


# ── Main diarization class ──────────────────────────────────────────────


class OnnxDiarization:
    """ONNX-based speaker diarization using pyannote segmentation + wespeaker embeddings.

    Reads audio from disk in chunks — memory stays bounded regardless of duration.
    """

    def __init__(
        self,
        *,
        segmentation_path: str | None = None,
        embedding_path: str | None = None,
        providers: list[str] | None = None,
        onset: float = 0.5,
        offset: float = 0.5,
        min_duration_on: float = 0.5,
        min_duration_off: float = 0.3,
    ) -> None:
        if providers is None:
            providers = ["CPUExecutionProvider"]

        if segmentation_path is None:
            segmentation_path = hf_hub_download(
                repo_id=_SEGMENTATION_REPO, filename="onnx/model.onnx",
            )
        if embedding_path is None:
            embedding_path = hf_hub_download(
                repo_id=_EMBEDDING_REPO, filename="onnx/model.onnx",
            )

        self._seg_session = ort.InferenceSession(segmentation_path, providers=providers)
        self._emb_session = ort.InferenceSession(embedding_path, providers=providers)
        self._sample_rate = 16000
        self._duration = 10.0  # segmentation model window (seconds)
        self._step = 0.5 * self._duration  # 50% overlap
        self._onset = onset
        self._offset = offset
        self._min_duration_on = min_duration_on
        self._min_duration_off = min_duration_off

    @staticmethod
    def _sample2frame(sample: int) -> int:
        return (sample - 721) // 270

    @staticmethod
    def _reorder(overlap_prob: np.ndarray, prob: np.ndarray) -> np.ndarray:
        """Reorder speaker columns to best match the overlap region."""
        perms = [np.array(perm).T for perm in permutations(prob.T)]
        perms_arr = np.array(perms)
        sum_perms = np.sum(
            perms_arr[:, : overlap_prob.shape[0], :] - overlap_prob, axis=1,
        )
        diffs = np.sum(np.abs(sum_perms), axis=1)
        return perms_arr[np.argmin(diffs)]

    def __call__(
        self,
        audio_path: str,
        *,
        num_speakers: int | None = None,
    ) -> list[tuple[float, float, str]]:
        """Run diarization on a mono 16kHz WAV file.

        Reads from disk in chunks — does not load full audio into memory.
        Returns list of (start, end, speaker_label) tuples.
        """
        info = sf.info(audio_path)
        total_samples = info.frames
        total_duration = total_samples / self._sample_rate

        # 1. Segmentation (reads from disk in sliding windows)
        segments = self._run_segmentation(audio_path, total_samples, total_duration)
        logger.info(
            "Diarization segmentation: %d segments from %.0fs audio",
            len(segments), total_duration,
        )

        # 2. Embedding extraction (reads each segment from disk individually)
        embeddings, valid_segments = self._extract_embeddings(audio_path, segments)
        gc.collect()

        if len(embeddings) == 0:
            return []

        # 3. Clustering
        labels = self._cluster_embeddings(embeddings, valid_segments, num_speakers=num_speakers)

        # 4. Build turns
        turns: list[tuple[float, float, str]] = []
        for seg, label in zip(valid_segments, labels):
            turns.append((seg[0], seg[1], f"SPEAKER_{label:02d}"))

        return _merge_annotation(turns)

    def _run_segmentation(
        self, audio_path: str, total_samples: int, total_duration: float,
    ) -> list[tuple[float, float]]:
        sr = self._sample_rate
        step_samples = int(self._step * sr)
        window_samples = int(self._duration * sr)

        num_frames = self._sample2frame(window_samples)
        seconds_per_frame = self._duration / num_frames
        total_frames = int(total_duration / seconds_per_frame) + 100

        global_scores = np.zeros((total_frames, 3), dtype=np.float32)
        overlap_frames = self._sample2frame(window_samples - step_samples)

        for i_sample in range(0, total_samples, step_samples):
            # Read just this window from disk
            remaining = total_samples - i_sample
            if remaining < window_samples // 2:
                break  # skip very short tail

            chunk = _read_audio_chunk(audio_path, i_sample, window_samples)

            # Pad if shorter than window (last chunk)
            if len(chunk) < window_samples:
                chunk = np.pad(chunk, (0, window_samples - len(chunk)), mode="constant")

            chunk_input = chunk[np.newaxis, np.newaxis, :].astype(np.float32)

            out = self._seg_session.run(None, {"input_values": chunk_input})[0][0]
            out = np.exp(out)  # (frames, 7) → probabilities

            # Combine overlapping speaker probabilities
            out[:, 1] += out[:, 4] + out[:, 5]
            out[:, 2] += out[:, 4] + out[:, 6]
            out[:, 3] += out[:, 5] + out[:, 6]
            speech_prob = out[:, 1:4]

            start_frame = int((i_sample / sr) / seconds_per_frame)
            end_frame = start_frame + len(speech_prob)

            if i_sample > 0:
                current_slice = global_scores[start_frame:end_frame]
                overlap_slice = current_slice[:overlap_frames]
                speech_prob = self._reorder(overlap_slice, speech_prob)

                overlap_end = start_frame + overlap_frames
                global_scores[start_frame:overlap_end] = (
                    overlap_slice + speech_prob[:overlap_frames]
                ) / 2
                global_scores[overlap_end:end_frame] = speech_prob[overlap_frames:]
            else:
                global_scores[start_frame:end_frame] = speech_prob

            del chunk, chunk_input, out
            # No need to gc.collect() every iteration — just free references

        # Hysteresis thresholding
        is_active = [False, False, False]
        start_ts = [0.0, 0.0, 0.0]
        speaker_segments: list[list[tuple[float, float]]] = [[], [], []]

        for f, scores in enumerate(global_scores):
            t = f * seconds_per_frame
            if t > total_duration:
                break
            for i in range(3):
                if not is_active[i]:
                    if scores[i] > self._onset:
                        is_active[i] = True
                        start_ts[i] = t
                else:
                    if scores[i] < self._offset:
                        is_active[i] = False
                        if t - start_ts[i] >= self._min_duration_on:
                            speaker_segments[i].append((start_ts[i], t))

        # Close any open segments
        for i in range(3):
            if is_active[i] and total_duration - start_ts[i] >= self._min_duration_on:
                speaker_segments[i].append((start_ts[i], total_duration))

        del global_scores
        gc.collect()

        # Merge per-speaker, then combine all
        all_segments: list[tuple[float, float]] = []
        for segs in speaker_segments:
            all_segments.extend(_merge_segments(segs, collar=self._min_duration_off))

        return all_segments

    def _extract_embeddings(
        self, audio_path: str, segments: list[tuple[float, float]],
    ) -> tuple[np.ndarray, list[tuple[float, float]]]:
        sr = self._sample_rate
        embeddings: list[np.ndarray] = []
        valid_segments: list[tuple[float, float]] = []

        for start, end in segments:
            seg_duration = end - start
            if seg_duration < 400 / sr:
                continue

            # For long segments, extract embedding from a representative sub-segment
            # to bound memory. Use the middle 30s (or full segment if shorter).
            if seg_duration > _MAX_EMB_SEGMENT_SECS:
                mid = (start + end) / 2
                emb_start = max(start, mid - _MAX_EMB_SEGMENT_SECS / 2)
                emb_end = min(end, mid + _MAX_EMB_SEGMENT_SECS / 2)
            else:
                emb_start, emb_end = start, end

            start_sample = int(emb_start * sr)
            num_samples = int((emb_end - emb_start) * sr)
            chunk = _read_audio_chunk(audio_path, start_sample, num_samples)

            if len(chunk) < 400:
                continue

            emb = self._compute_embedding(chunk)
            if emb is not None:
                embeddings.append(emb)
                valid_segments.append((start, end))

            del chunk

        if not embeddings:
            return np.array([]), valid_segments
        return np.array(embeddings), valid_segments

    def _compute_embedding(self, chunk: np.ndarray) -> np.ndarray | None:
        """Compute speaker embedding for an audio chunk."""
        sr = self._sample_rate
        melspec = _melspectrogram(
            chunk, sr=sr, n_fft=400, hop_length=160, n_mels=80,
        )
        log_mel = np.log(melspec + 1e-6)
        features = log_mel.T  # (T, n_mels)
        features = features - np.mean(features, axis=0)
        features = features[np.newaxis, :, :].astype(np.float32)

        del melspec, log_mel

        try:
            emb = self._emb_session.run(None, {"input_features": features})[0][0]
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb = emb / norm
            return emb
        except Exception:
            logger.debug("Embedding extraction failed", exc_info=True)
            return None

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        segments: list[tuple[float, float]],
        *,
        num_speakers: int | None = None,
    ) -> list[int]:
        if len(embeddings) == 0:
            return []
        if len(embeddings) == 1:
            return [0]

        # Two-stage clustering: long segments define speakers, short assigned to nearest
        total_duration = sum(end - start for start, end in segments)
        min_duration = float(np.clip(total_duration / 60, 2, 5))
        long_idx = [i for i, (s, e) in enumerate(segments) if (e - s) >= min_duration]
        short_idx = [i for i, (s, e) in enumerate(segments) if (e - s) < min_duration]

        if len(long_idx) < 2 or (num_speakers and num_speakers <= 1):
            return [0] * len(embeddings)

        long_emb = embeddings[long_idx]

        if num_speakers:
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers, metric="euclidean", linkage="ward",
            )
        else:
            distance_threshold = max((350 - total_duration) / 350, 0.73)
            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=distance_threshold,
                metric="euclidean", linkage="single",
            )

        clustering.fit(long_emb)
        long_labels = clustering.labels_

        labels = np.zeros(len(embeddings), dtype=int)
        labels[long_idx] = long_labels

        if short_idx:
            unique_labels = np.unique(long_labels)
            centroids = np.array([
                np.mean(long_emb[long_labels == lbl], axis=0) for lbl in unique_labels
            ])
            centroid_labels = list(unique_labels)

            short_emb = embeddings[short_idx]
            dists = cdist(short_emb, centroids, metric="euclidean")
            nearest = np.argmin(dists, axis=1)
            labels[short_idx] = [centroid_labels[i] for i in nearest]

        return list(labels)
