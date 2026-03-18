use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{DefaultBodyLimit, Multipart, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use eyre::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{error, info};

mod transcribe;

use transcribe::TranscribeEngine;

struct AppState {
    engine: Mutex<TranscribeEngine>,
}

#[derive(Serialize)]
struct HealthResponse {
    ok: bool,
    engine: &'static str,
    model_loaded: bool,
}

#[derive(Serialize)]
struct TranscribeResponse {
    text: String,
    language: Option<String>,
    segments: Vec<SegmentResponse>,
}

#[derive(Serialize)]
struct SegmentResponse {
    start: f32,
    end: f32,
    text: String,
    speaker: Option<String>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

async fn healthz() -> Json<HealthResponse> {
    // No lock needed — models are loaded at startup and always available.
    // Locking here would block healthz during transcription, killing probes.
    Json(HealthResponse {
        ok: true,
        engine: "parakeet-rs",
        model_loaded: true,
    })
}

#[derive(Deserialize)]
struct TranscribeParams {
    #[serde(default = "default_true")]
    diarize: bool,
}

fn default_true() -> bool { true }

async fn transcribe(
    State(state): State<Arc<AppState>>,
    Query(params): Query<TranscribeParams>,
    mut multipart: Multipart,
) -> Result<Json<TranscribeResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Extract audio file from multipart
    let tmp_path = loop {
        let field = multipart
            .next_field()
            .await
            .map_err(|e| bad_request(format!("multipart error: {e}")))?;

        let Some(field) = field else {
            return Err(bad_request("no audio file provided".into()));
        };

        if field.name() == Some("audio") {
            let file_name = field.file_name().unwrap_or("audio.wav").to_string();
            let suffix = std::path::Path::new(&file_name)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("wav");

            let tmp_dir = tempfile::tempdir()
                .map_err(|e| internal(format!("failed to create temp dir: {e}")))?;
            let dir_path = tmp_dir.path().to_path_buf();
            let _ = tmp_dir.keep();
            let tmp_path = dir_path.join(format!("input.{suffix}"));

            let bytes = field
                .bytes()
                .await
                .map_err(|e| bad_request(format!("failed to read upload: {e}")))?;

            tokio::fs::write(&tmp_path, &bytes)
                .await
                .map_err(|e| internal(format!("failed to write temp file: {e}")))?;

            break tmp_path;
        }
    };

    // Convert to WAV if needed
    let wav_path = ensure_wav(&tmp_path)
        .await
        .map_err(|e| internal(format!("audio conversion failed: {e}")))?;

    let t0 = Instant::now();

    // Run transcription in blocking task (holds the mutex)
    let diarize = params.diarize;
    let result = {
        let wav = wav_path.clone();
        let state = Arc::clone(&state);
        tokio::task::spawn_blocking(move || {
            let mut engine = state.engine.blocking_lock();
            engine.transcribe_with_options(&wav, diarize)
        })
        .await
        .map_err(|e| internal(format!("task join error: {e}")))?
        .map_err(|e| internal(format!("transcription failed: {e}")))?
    };

    let elapsed = t0.elapsed();
    info!(
        segments = result.segments.len(),
        elapsed_secs = format!("{:.1}", elapsed.as_secs_f32()),
        "transcription complete"
    );

    // Cleanup
    let _ = tokio::fs::remove_file(&tmp_path).await;
    if wav_path != tmp_path {
        let _ = tokio::fs::remove_file(&wav_path).await;
    }
    // Clean up parent temp dir
    if let Some(parent) = tmp_path.parent() {
        let _ = tokio::fs::remove_dir(parent).await;
    }

    Ok(Json(TranscribeResponse {
        text: result.text,
        language: result.language,
        segments: result
            .segments
            .into_iter()
            .map(|s| SegmentResponse {
                start: s.start,
                end: s.end,
                text: s.text,
                speaker: s.speaker,
            })
            .collect(),
    }))
}

/// Convert any audio format to mono 16kHz WAV via ffmpeg.
async fn ensure_wav(path: &std::path::Path) -> Result<PathBuf> {
    if path.extension().and_then(|e| e.to_str()) == Some("wav") {
        // Check if already 16kHz mono
        let output = tokio::process::Command::new("ffprobe")
            .args([
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels",
                "-of", "csv=p=0",
            ])
            .arg(path)
            .output()
            .await
            .wrap_err("ffprobe failed")?;

        let info = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = info.trim().split(',').collect();
        if parts.len() == 2 && parts[0] == "16000" && parts[1] == "1" {
            return Ok(path.to_path_buf());
        }
    }

    let wav_path = path.with_extension("converted.wav");
    let status = tokio::process::Command::new("ffmpeg")
        .args([
            "-y", "-i",
        ])
        .arg(path)
        .args([
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        ])
        .arg(&wav_path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .wrap_err("ffmpeg failed")?;

    if !status.success() {
        eyre::bail!("ffmpeg exited with status {}", status);
    }

    Ok(wav_path)
}

fn bad_request(msg: String) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse { error: msg }),
    )
}

fn internal(msg: String) -> (StatusCode, Json<ErrorResponse>) {
    error!("{}", msg);
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error: msg }),
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let port: u16 = std::env::var("WORKER_PORT")
        .unwrap_or_else(|_| "3001".into())
        .parse()
        .wrap_err("invalid WORKER_PORT")?;

    let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "/models".into());
    let tdt_dir = std::env::var("TDT_MODEL_DIR")
        .unwrap_or_else(|_| format!("{model_dir}/tdt"));
    let sortformer_path = std::env::var("SORTFORMER_MODEL_PATH")
        .unwrap_or_else(|_| format!("{model_dir}/sortformer/diar_streaming_sortformer_4spk-v2.1.onnx"));

    info!(tdt_dir = %tdt_dir, sortformer_path = %sortformer_path, "loading models");
    let engine = TranscribeEngine::new(&tdt_dir, &sortformer_path)
        .wrap_err("failed to load models")?;
    info!("models loaded, ready to serve");

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
    });

    let app = Router::new()
        .route("/worker/healthz", get(healthz))
        .route("/worker/transcribe", post(transcribe))
        .layer(DefaultBodyLimit::max(512 * 1024 * 1024)) // 512MB for large audio files
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
