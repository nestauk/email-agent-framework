"""Tests for storage module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_pool():
    """Create a mock connection pool."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn
    return mock_pool, mock_conn, mock_cursor


@pytest.fixture
def db(mock_pool, monkeypatch):
    """Create AgentDatabase with mocked pool."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")

    pool_instance, mock_conn, mock_cursor = mock_pool

    with patch("email_agent.agent_api.storage.pool.ThreadedConnectionPool", return_value=pool_instance):
        from email_agent.agent_api.storage import AgentDatabase

        database = AgentDatabase()
        database._mock_cursor = mock_cursor
        database._mock_conn = mock_conn
        yield database


# ── Job Tests ─────────────────────────────────────────────────────────────────


def test_record_job_inserts_correctly(db):
    """record_job inserts a job with correct values."""
    db.record_job(
        job_id="test-job-123",
        user_id=42,
        tool_name="send_email_tool",
        payload={"test": "data"},
        run_handle="thread-abc",
    )

    # Verify INSERT was called
    cursor = db._mock_cursor
    cursor.execute.assert_called()
    call_args = cursor.execute.call_args
    sql = call_args[0][0]
    params = call_args[0][1]

    assert "INSERT INTO jobs" in sql
    assert params[0] == "test-job-123"  # job_id
    assert params[1] == 42  # user_id
    assert params[2] == "send_email_tool"  # tool_name
    # status is hardcoded as 'pending' in SQL, not a param
    assert '"test": "data"' in params[3]  # payload JSON
    assert params[4] == "thread-abc"  # run_handle


def test_get_job_returns_none_when_not_found(db):
    """get_job returns None for non-existent job."""
    db._mock_cursor.fetchone.return_value = None

    result = db.get_job("nonexistent")

    assert result is None


def test_get_job_returns_job_dict(db):
    """get_job returns job as dictionary."""
    db._mock_cursor.description = [
        ("job_id",),
        ("user_id",),
        ("tool_name",),
        ("status",),
        ("payload",),
        ("created_at",),
    ]
    db._mock_cursor.fetchone.return_value = (
        "job-123",
        42,
        "send_email_tool",
        "pending",
        '{"test": "data"}',
        "2024-01-01T00:00:00",
    )

    result = db.get_job("job-123")

    assert result is not None
    assert result["job_id"] == "job-123"
    assert result["user_id"] == 42
    assert result["tool_name"] == "send_email_tool"
    assert result["status"] == "pending"


def test_list_pending_jobs_returns_empty_list(db):
    """list_pending_jobs returns empty list when no jobs."""
    db._mock_cursor.fetchall.return_value = []

    result = db.list_pending_jobs()

    assert result == []


def test_list_pending_jobs_returns_jobs(db):
    """list_pending_jobs returns list of job dicts."""
    db._mock_cursor.description = [
        ("job_id",),
        ("user_id",),
        ("tool_name",),
        ("status",),
        ("payload",),
        ("created_at",),
    ]
    db._mock_cursor.fetchall.return_value = [
        ("job-1", 1, "send_email_tool", "pending", "{}", "2024-01-01"),
        ("job-2", 2, "Question", "pending", "{}", "2024-01-02"),
    ]

    result = db.list_pending_jobs(limit=10)

    assert len(result) == 2
    assert result[0]["job_id"] == "job-1"
    assert result[1]["job_id"] == "job-2"


# ── Attachment Tests ──────────────────────────────────────────────────────────


def test_save_email_attachment_inserts(db):
    """save_email_attachment inserts attachment record."""
    db.save_email_attachment(
        job_id="job-123",
        field_path="pdf_attachment_0",
        base64_data="SGVsbG8gV29ybGQ=",
        filename="test.pdf",
        content_type="application/pdf",
        summary="Test PDF document",
    )

    cursor = db._mock_cursor
    cursor.execute.assert_called()
    call_args = cursor.execute.call_args
    sql = call_args[0][0]
    params = call_args[0][1]

    # Order: job_id, field_path, filename, content_type, base64_data, summary, created_at
    assert "INSERT INTO email_attachments" in sql
    assert params[0] == "job-123"
    assert params[1] == "pdf_attachment_0"
    assert params[2] == "test.pdf"
    assert params[3] == "application/pdf"
    assert params[4] == "SGVsbG8gV29ybGQ="


def test_get_email_attachment_returns_none_when_not_found(db):
    """get_email_attachment returns None for non-existent attachment."""
    db._mock_cursor.fetchone.return_value = None

    result = db.get_email_attachment("job-123", "pdf_attachment_0")

    assert result is None


def test_get_email_attachment_returns_attachment(db):
    """get_email_attachment returns attachment dict."""
    db._mock_cursor.description = [
        ("id",),
        ("job_id",),
        ("field_path",),
        ("filename",),
        ("content_type",),
        ("base64_data",),
        ("summary",),
        ("created_at",),
    ]
    db._mock_cursor.fetchone.return_value = (
        1,
        "job-123",
        "pdf_attachment_0",
        "test.pdf",
        "application/pdf",
        "SGVsbG8=",
        "Test doc",
        "2024-01-01",
    )

    result = db.get_email_attachment("job-123", "pdf_attachment_0")

    assert result is not None
    assert result["filename"] == "test.pdf"
    assert result["content_type"] == "application/pdf"
    assert result["base64_data"] == "SGVsbG8="


def test_get_email_attachments_returns_list(db):
    """get_email_attachments returns list of attachments for job."""
    db._mock_cursor.description = [
        ("id",),
        ("job_id",),
        ("field_path",),
        ("filename",),
        ("content_type",),
        ("base64_data",),
        ("summary",),
        ("created_at",),
    ]
    db._mock_cursor.fetchall.return_value = [
        (1, "job-123", "pdf_0", "doc1.pdf", "application/pdf", "data1", "Summary 1", "2024-01-01"),
        (2, "job-123", "pdf_1", "doc2.pdf", "application/pdf", "data2", "Summary 2", "2024-01-01"),
    ]

    result = db.get_email_attachments("job-123")

    assert len(result) == 2
    assert result[0]["filename"] == "doc1.pdf"
    assert result[1]["filename"] == "doc2.pdf"


def test_get_email_attachments_returns_empty_list(db):
    """get_email_attachments returns empty list when no attachments."""
    db._mock_cursor.fetchall.return_value = []

    result = db.get_email_attachments("job-123")

    assert result == []


# ── User Tests ────────────────────────────────────────────────────────────────


def test_get_user_returns_none_when_not_found(db):
    """get_user returns None for non-existent user."""
    db._mock_cursor.fetchone.return_value = None

    result = db.get_user(999)

    assert result is None


def test_list_active_users_returns_users(db):
    """list_active_users returns list of active user dicts."""
    db._mock_cursor.description = [
        ("user_id",),
        ("email_to_monitor",),
        ("status",),
    ]
    db._mock_cursor.fetchall.return_value = [
        (1, "user1@example.com", "active"),
        (2, "user2@example.com", "active"),
    ]

    result = db.list_active_users()

    assert len(result) == 2
    assert result[0]["user_id"] == 1
    assert result[1]["email_to_monitor"] == "user2@example.com"


# ── Worker Event Tests ────────────────────────────────────────────────────────


def test_enqueue_worker_event_returns_id(db):
    """enqueue_worker_event inserts event and returns ID."""
    db._mock_cursor.fetchone.return_value = (42,)

    result = db.enqueue_worker_event("register", {"user_id": 123})

    assert result == 42
    cursor = db._mock_cursor
    call_args = cursor.execute.call_args
    sql = call_args[0][0]
    assert "INSERT INTO worker_events" in sql
