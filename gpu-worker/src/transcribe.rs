use std::path::Path;
use std::time::Instant;

use eyre::{Context, Result};
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer, SpeakerSegment};
use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetTDT, TimestampMode, Transcriber};
use tracing::info;

/// Max seconds per TDT chunk — the ONNX model's relative position bias is
/// exported with a fixed 1001-frame window. At 12.5 frames/sec (80ms/frame),
/// that's ~80s max. 75s = 937 frames, safely under the limit.
const MAX_CHUNK_SECS: f32 = 75.0;

const SAMPLE_RATE: u32 = 16000;

pub struct TranscribeResult {
    pub text: String,
    pub language: Option<String>,
    pub segments: Vec<Segment>,
}

pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub speaker: Option<String>,
}

pub struct TranscribeEngine {
    tdt: ParakeetTDT,
    sortformer: Sortformer,
}

impl TranscribeEngine {
    pub fn new(tdt_dir: &str, sortformer_path: &str) -> Result<Self> {
        // Try TensorRT first (INT8 model + TRT = best perf), fall back to CUDA EP (FP32)
        let trt_config = ExecutionConfig::new()
            .with_execution_provider(ExecutionProvider::TensorRT);
        let cuda_config = ExecutionConfig::new()
            .with_execution_provider(ExecutionProvider::Cuda);

        let tdt = ParakeetTDT::from_pretrained(tdt_dir, Some(trt_config))
            .or_else(|e| {
                tracing::warn!("TensorRT failed for TDT, falling back to CUDA FP32: {e}");
                // CUDA EP + INT8 is slow — swap back to FP32 preference
                // by loading from a dir where FP32 is tried first
                ParakeetTDT::from_pretrained(tdt_dir, Some(cuda_config.clone()))
            })
            .wrap_err("failed to load TDT model")?;

        let sortformer = Sortformer::with_config(
            sortformer_path,
            Some(cuda_config),
            DiarizationConfig::callhome(),
        )
        .wrap_err("failed to load Sortformer model")?;

        Ok(Self { tdt, sortformer })
    }

    pub fn transcribe(&mut self, wav_path: &Path) -> Result<TranscribeResult> {
        self.transcribe_with_options(wav_path, true)
    }

    pub fn transcribe_with_options(&mut self, wav_path: &Path, diarize: bool) -> Result<TranscribeResult> {
        let (audio, spec) = load_wav(wav_path)?;
        let duration_secs = audio.len() as f32 / SAMPLE_RATE as f32;
        info!(duration_secs = format!("{:.1}", duration_secs), diarize, "processing audio");

        let t0 = Instant::now();

        let sr = spec.sample_rate;
        let ch = spec.channels;

        // Step 1: Diarization (optional)
        let speaker_segments = if diarize {
            let audio_for_diar = audio.clone();
            let t_diar = Instant::now();
            let segments = self
                .sortformer
                .diarize(audio_for_diar, sr, ch)
                .wrap_err("diarization failed")?;
            info!(
                speaker_segments = segments.len(),
                elapsed_ms = t_diar.elapsed().as_millis(),
                "diarization complete"
            );
            segments
        } else {
            info!("diarization skipped");
            Vec::new()
        };

        // Step 2: Transcribe — chunk if needed
        let t_asr = Instant::now();
        let tdt_segments = if duration_secs <= MAX_CHUNK_SECS {
            self.transcribe_single(&audio, sr, ch)?
        } else {
            self.transcribe_chunked(&audio, sr, ch)?
        };
        info!(
            tdt_segments = tdt_segments.len(),
            elapsed_ms = t_asr.elapsed().as_millis(),
            "ASR complete"
        );

        // Step 3: Align TDT sentences with speaker segments
        let text = tdt_segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let segments = tdt_segments
            .into_iter()
            .map(|seg| {
                let speaker = find_best_speaker(seg.start, seg.end, &speaker_segments);
                Segment {
                    start: seg.start,
                    end: seg.end,
                    text: seg.text,
                    speaker,
                }
            })
            .collect();

        info!(
            total_elapsed_ms = t0.elapsed().as_millis(),
            rtf = format!("{:.1}x", duration_secs / t0.elapsed().as_secs_f32()),
            "transcription complete"
        );

        Ok(TranscribeResult {
            text,
            language: Some("en".into()),
            segments,
        })
    }

    fn transcribe_single(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        channels: u16,
    ) -> Result<Vec<TdtSegment>> {
        let result = self
            .tdt
            .transcribe_samples(audio.to_vec(), sample_rate, channels, Some(TimestampMode::Sentences))
            .wrap_err("TDT transcription failed")?;

        Ok(result
            .tokens
            .into_iter()
            .filter(|t| !t.text.trim().is_empty())
            .map(|t| TdtSegment {
                start: t.start,
                end: t.end,
                text: t.text,
            })
            .collect())
    }

    fn transcribe_chunked(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        channels: u16,
    ) -> Result<Vec<TdtSegment>> {
        let chunk_samples = (MAX_CHUNK_SECS * sample_rate as f32) as usize;
        let total_samples = audio.len();
        let mut all_segments = Vec::new();
        let mut offset: usize = 0;
        let mut chunk_idx = 0;

        while offset < total_samples {
            let end = (offset + chunk_samples).min(total_samples);
            // Don't process very short tails
            if end - offset < (sample_rate as usize) {
                break;
            }

            let chunk = &audio[offset..end];
            let offset_secs = offset as f32 / sample_rate as f32;

            info!(
                chunk = chunk_idx,
                start_secs = format!("{:.1}", offset_secs),
                end_secs = format!("{:.1}", end as f32 / sample_rate as f32),
                "transcribing chunk"
            );

            let result = self
                .tdt
                .transcribe_samples(
                    chunk.to_vec(),
                    sample_rate,
                    channels,
                    Some(TimestampMode::Sentences),
                )
                .wrap_err_with(|| format!("TDT failed on chunk {chunk_idx}"))?;

            for token in result.tokens {
                if token.text.trim().is_empty() {
                    continue;
                }
                all_segments.push(TdtSegment {
                    start: token.start + offset_secs,
                    end: token.end + offset_secs,
                    text: token.text,
                });
            }

            offset = end;
            chunk_idx += 1;
        }

        Ok(all_segments)
    }
}

struct TdtSegment {
    start: f32,
    end: f32,
    text: String,
}

/// Find the speaker with the most overlap for a given time range.
fn find_best_speaker(
    seg_start: f32,
    seg_end: f32,
    speaker_segments: &[SpeakerSegment],
) -> Option<String> {
    let mut best_speaker: Option<usize> = None;
    let mut best_overlap: f32 = 0.0;

    for spk_seg in speaker_segments {
        let spk_start = spk_seg.start as f32 / SAMPLE_RATE as f32;
        let spk_end = spk_seg.end as f32 / SAMPLE_RATE as f32;

        let overlap_start = seg_start.max(spk_start);
        let overlap_end = seg_end.min(spk_end);
        let overlap = (overlap_end - overlap_start).max(0.0);

        if overlap > best_overlap {
            best_overlap = overlap;
            best_speaker = Some(spk_seg.speaker_id);
        }
    }

    best_speaker.map(|id| format!("SPEAKER_{id:02}"))
}

struct AudioSpec {
    sample_rate: u32,
    channels: u16,
}

fn load_wav(path: &Path) -> Result<(Vec<f32>, AudioSpec)> {
    let mut reader =
        hound::WavReader::open(path).wrap_err_with(|| format!("failed to open {}", path.display()))?;
    let spec = reader.spec();

    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .wrap_err("failed to read float samples")?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<std::result::Result<Vec<_>, _>>()
            .wrap_err("failed to read int samples")?,
    };

    Ok((
        audio,
        AudioSpec {
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        },
    ))
}
