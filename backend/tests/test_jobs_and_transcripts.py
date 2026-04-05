from pathlib import Path

from cassandra_yt_mcp.config import Settings
from cassandra_yt_mcp.db.database import Database
from cassandra_yt_mcp.db.jobs import JobsRepository
from cassandra_yt_mcp.db.transcripts import TranscriptsRepository
from cassandra_yt_mcp.runtime import AppRuntime


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=3003,
        poll_interval_seconds=5,
        data_dir=tmp_path,
        database_path=tmp_path / "test.sqlite3",
        fluidaudio_url="http://localhost:8420",
        max_workers=1,
        auth_url="",
        auth_secret="",
        auth_yaml_path="/app/acl.yaml",
        base_url="",
        workos_client_id="",
        workos_authkit_domain="",
        code_mode=True,
    )


def test_job_lifecycle(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.sqlite3")
    jobs = JobsRepository(db)
    created = jobs.enqueue("https://example.com/v/1", "https://example.com/v/1")
    assert created["status"] == "queued"

    claimed = jobs.claim_next()
    assert claimed is not None
    assert claimed["status"] == "downloading"

    jobs.set_status(str(claimed["id"]), "transcribing")
    transcribing = jobs.get(str(claimed["id"]))
    assert transcribing is not None
    assert transcribing["status"] == "transcribing"

    jobs.mark_completed(str(claimed["id"]), "video1", "/tmp/video1")
    completed = jobs.get(str(claimed["id"]))
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["video_id"] == "video1"
    db.close()


def test_transcripts_search(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.sqlite3")
    repo = TranscriptsRepository(db)
    repo.upsert(
        video_id="vid1",
        normalized_url="https://example.com/1",
        url="https://example.com/1",
        path="/tmp/vid1",
        transcript_text="hello this is a transcription test",
        title="Demo title",
        channel="demo channel",
        platform="youtube",
        duration=12.3,
        upload_date="20250101",
        description="description text",
        thumbnail=None,
        view_count=1,
        speaker_count=1,
        word_count=6,
        confidence=None,
    )
    results = repo.search("transcription", limit=5)
    assert len(results) == 1
    assert results[0]["video_id"] == "vid1"
    db.close()


def test_enqueue_transcription_deduplicates_completed_transcripts(tmp_path: Path) -> None:
    runtime = AppRuntime(make_settings(tmp_path))
    runtime.transcripts.upsert(
        video_id="vid1",
        normalized_url="https://youtube.com/watch?v=vid1",
        url="https://youtube.com/watch?v=vid1",
        path="/tmp/vid1",
        transcript_text="cached transcript",
        title="Demo title",
        channel="demo channel",
        platform="youtube",
        duration=None,
        upload_date=None,
        description=None,
        thumbnail=None,
        view_count=None,
        speaker_count=None,
        word_count=2,
        confidence=None,
    )

    try:
        result = runtime.enqueue_transcription("https://www.youtube.com/watch?v=vid1&t=30")
    finally:
        runtime.close()

    assert result["status"] == "completed"
    assert result["deduplicated"] is True
    assert result["video_id"] == "vid1"
    assert result["transcript_path"] == "/tmp/vid1"
    assert "metadata" in result
    assert result["metadata"]["title"] == "Demo title"


def test_enqueue_transcription_deduplicates_active_jobs(tmp_path: Path) -> None:
    runtime = AppRuntime(make_settings(tmp_path))
    queued = runtime.jobs.enqueue(
        url="https://www.youtube.com/watch?v=vid2",
        normalized_url="https://youtube.com/watch?v=vid2",
    )

    try:
        result = runtime.enqueue_transcription("https://youtu.be/vid2")
    finally:
        runtime.close()

    assert result["job_id"] == queued["id"]
    assert result["status"] == "queued"
    assert result["deduplicated"] is True


def test_get_job_status_increments_poll_count_and_shapes_backoff(tmp_path: Path) -> None:
    runtime = AppRuntime(make_settings(tmp_path))
    queued = runtime.jobs.enqueue(
        url="https://www.youtube.com/watch?v=vid3",
        normalized_url="https://youtube.com/watch?v=vid3",
    )
    runtime.jobs.mark_failed(str(queued["id"]), "temporary failure", attempt=0)

    try:
        status = runtime.get_job_status(str(queued["id"]))
    finally:
        runtime.close()

    assert status["status"] == "queued"
    assert status["poll_count"] == 1
    assert status["attempt"] == 1
    assert status["retry_after"] == 5
    assert status["waiting_until"] is not None


def test_mark_failed_retries_then_marks_terminal_failure(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.sqlite3")
    jobs = JobsRepository(db)
    created = jobs.enqueue("https://example.com/v/retry", "https://example.com/v/retry")

    jobs.mark_failed(str(created["id"]), "first failure", attempt=0)
    first_retry = jobs.get(str(created["id"]))
    assert first_retry is not None
    assert first_retry["status"] == "queued"
    assert first_retry["attempt"] == 1
    assert first_retry["retry_after"] is not None

    jobs.mark_failed(str(created["id"]), "final failure", attempt=2)
    failed = jobs.get(str(created["id"]))
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["error"] == "final failure"
    db.close()
