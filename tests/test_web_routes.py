"""Tests for web frontend routes."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def mock_db():
    """Mock database with default empty responses."""
    db = Mock()
    db.list_pending_jobs.return_value = []
    db.list_active_users.return_value = []
    db.get_job.return_value = None
    db.get_email_attachments.return_value = []
    db.save_user = Mock()
    db.enqueue_worker_event = Mock(return_value=1)
    return db


@pytest.fixture
def client(mock_db, monkeypatch):
    """Test client with mocked database."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
    monkeypatch.setenv("AGENT_API_KEY", "test-key")

    # Create a new app with a lifespan that uses our mock_db
    @asynccontextmanager
    async def mock_lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.db = mock_db
        yield

    # Import routes after patching env vars
    from email_agent.agent_api.schemas import MessageResponse
    from email_agent.agent_api.web_routes import router as web_router

    app = FastAPI(lifespan=mock_lifespan)

    # Add the health endpoint for completeness
    @app.get("/health", response_model=MessageResponse)
    async def healthcheck() -> MessageResponse:
        return MessageResponse(message="healthy")

    # Include web router
    app.include_router(web_router)

    with TestClient(app) as c:
        yield c


# ── Dashboard Tests ────────────────────────────────────────────────────────────


def test_dashboard_returns_html(client):
    """Dashboard returns HTML with proper content type."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Inbox" in response.text


def test_dashboard_shows_no_jobs_message(client, mock_db):
    """Dashboard shows message when no pending jobs."""
    mock_db.list_pending_jobs.return_value = []
    response = client.get("/")
    assert response.status_code == 200
    assert "No emails to review" in response.text


def test_dashboard_shows_jobs(client, mock_db):
    """Dashboard displays jobs with sender and subject."""
    mock_db.list_pending_jobs.return_value = [
        {
            "job_id": "test-job-123456",
            "tool_name": "send_email_tool",
            "user_id": 1,
            "status": "pending",
            "created_at": "2024-01-01T00:00:00",
            "payload": json.dumps({
                "email": {
                    "from_email": "John Doe <john@example.com>",
                    "subject": "Meeting tomorrow",
                }
            }),
        }
    ]
    response = client.get("/")
    assert response.status_code == 200
    assert "John Doe" in response.text
    assert "Meeting tomorrow" in response.text


# ── Job List Partial Tests ─────────────────────────────────────────────────────


def test_job_list_partial_returns_fragment(client, mock_db):
    """HTMX partial returns fragment only, not full HTML."""
    mock_db.list_pending_jobs.return_value = []
    response = client.get("/partials/job-list")
    assert response.status_code == 200
    # Should NOT have full HTML structure
    assert "<!DOCTYPE" not in response.text
    assert "No emails to review" in response.text


def test_job_list_partial_shows_jobs(client, mock_db):
    """Job list partial renders job cards."""
    mock_db.list_pending_jobs.return_value = [
        {
            "job_id": "hitl-abc123",
            "tool_name": "send_email_tool",
            "user_id": 42,
            "status": "pending",
            "created_at": "2024-06-15T10:30:00",
            "payload": json.dumps({
                "email": {
                    "from_email": "sender@example.com",
                    "subject": "Test subject",
                }
            }),
        }
    ]
    response = client.get("/partials/job-list")
    assert response.status_code == 200
    assert "hitl-abc123" in response.text
    assert "Review draft" in response.text  # Action label for send_email_tool


# ── Review Page Tests ──────────────────────────────────────────────────────────


def test_review_nonexistent_job_returns_404(client, mock_db):
    """Reviewing missing job returns 404."""
    mock_db.get_job.return_value = None
    response = client.get("/jobs/nonexistent-id")
    assert response.status_code == 404


def test_review_job_renders_details(client, mock_db):
    """Review page shows job details."""
    mock_db.get_job.return_value = {
        "job_id": "test-123",
        "tool_name": "send_email_tool",
        "status": "pending",
        "user_id": 1,
        "payload": json.dumps({
            "request": {
                "description": "Email from sender@example.com",
                "config": {
                    "allow_accept": True,
                    "allow_edit": True,
                    "allow_respond": True,
                    "allow_ignore": True,
                },
                "action_request": {
                    "action": "send_email_tool",
                    "args": {
                        "email_address": "recipient@example.com",
                        "response_text": "Hello, this is a test response.",
                    },
                },
            }
        }),
        "created_at": "2024-01-01T00:00:00",
    }
    mock_db.get_email_attachments.return_value = []

    response = client.get("/jobs/test-123")
    assert response.status_code == 200
    assert "recipient@example.com" in response.text
    assert "Hello, this is a test response" in response.text
    assert "Send as written" in response.text  # Action button for send_email_tool


def test_review_completed_job_shows_message(client, mock_db):
    """Review page shows message for non-pending jobs."""
    mock_db.get_job.return_value = {
        "job_id": "test-123",
        "tool_name": "send_email_tool",
        "status": "completed",
        "user_id": 1,
        "payload": "{}",
        "created_at": "2024-01-01T00:00:00",
    }
    mock_db.get_email_attachments.return_value = []

    response = client.get("/jobs/test-123")
    assert response.status_code == 200
    assert "not pending" in response.text.lower() or "completed" in response.text.lower()


# ── Action Submission Tests ────────────────────────────────────────────────────


def test_action_submit_job_not_found(client, mock_db):
    """Action submission for missing job returns error fragment."""
    mock_db.get_job.return_value = None
    response = client.post("/jobs/nonexistent/action", data={"action_type": "accept"})
    assert response.status_code == 200
    assert "Job not found" in response.text


def test_action_submit_accept_success(client, mock_db, monkeypatch):
    """Action submission calls /toolCompleted and returns success."""
    mock_db.get_job.return_value = {
        "job_id": "test-123",
        "tool_name": "send_email_tool",
        "status": "pending",
        "payload": json.dumps({"request": {"config": {}}}),
    }

    # Mock httpx.AsyncClient
    mock_response = Mock()
    mock_response.raise_for_status = Mock()

    async def mock_post(*args, **kwargs):
        return mock_response

    with patch("email_agent.agent_api.web_routes.httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.post = mock_post
        mock_client.return_value.__aenter__.return_value = mock_instance

        response = client.post("/jobs/test-123/action", data={"action_type": "accept"})

    assert response.status_code == 200
    assert "success" in response.text.lower()


def test_action_submit_timeout_error(client, mock_db, monkeypatch):
    """Action submission handles timeout gracefully."""
    import httpx

    mock_db.get_job.return_value = {
        "job_id": "test-123",
        "tool_name": "send_email_tool",
        "status": "pending",
        "payload": json.dumps({"request": {"config": {}}}),
    }

    async def mock_post(*args, **kwargs):
        raise httpx.TimeoutException("Connection timed out")

    with patch("email_agent.agent_api.web_routes.httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.post = mock_post
        mock_client.return_value.__aenter__.return_value = mock_instance

        response = client.post("/jobs/test-123/action", data={"action_type": "accept"})

    assert response.status_code == 200
    assert "timed out" in response.text.lower()


# ── Setup Page Tests ───────────────────────────────────────────────────────────


def test_setup_page_renders(client):
    """Setup page renders HTML."""
    response = client.get("/setup")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Gmail Setup" in response.text


def test_setup_page_shows_not_configured(client, monkeypatch):
    """Setup page shows not configured when GMAIL_CLIENT_ID missing."""
    monkeypatch.delenv("GMAIL_CLIENT_ID", raising=False)
    monkeypatch.delenv("GMAIL_CLIENT_SECRET", raising=False)

    response = client.get("/setup")
    assert response.status_code == 200
    assert "not configured" in response.text.lower()


def test_setup_page_shows_connect_button(client, monkeypatch):
    """Setup page shows connect button when Gmail is configured."""
    monkeypatch.setenv("GMAIL_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("GMAIL_CLIENT_SECRET", "test-client-secret")

    response = client.get("/setup")
    assert response.status_code == 200
    assert "Connect Gmail" in response.text


def test_setup_page_shows_success(client):
    """Setup page shows success message."""
    response = client.get("/setup?success=1&email=test@example.com")
    assert response.status_code == 200
    assert "Successful" in response.text


def test_setup_page_shows_error(client):
    """Setup page shows error message."""
    response = client.get("/setup?error=invalid_state")
    assert response.status_code == 200
    assert "expired" in response.text.lower() or "invalid" in response.text.lower()


# ── OAuth Flow Tests ───────────────────────────────────────────────────────��───


def test_oauth_start_redirects_to_google(client, monkeypatch):
    """OAuth start sets state cookie and redirects to Google."""
    monkeypatch.setenv("GMAIL_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("GMAIL_CLIENT_SECRET", "test-client-secret")

    response = client.get("/setup/oauth/start", follow_redirects=False)
    assert response.status_code == 307
    assert "accounts.google.com" in response.headers["location"]
    assert "state=" in response.headers["location"]
    assert "oauth_state" in response.cookies


def test_oauth_start_fails_without_client_id(client, monkeypatch):
    """OAuth start returns 500 when GMAIL_CLIENT_ID not set."""
    monkeypatch.delenv("GMAIL_CLIENT_ID", raising=False)

    response = client.get("/setup/oauth/start", follow_redirects=False)
    assert response.status_code == 500


def test_oauth_callback_invalid_state(client, mock_db):
    """OAuth callback rejects invalid state."""
    # No state cookie set, so any state will be invalid
    response = client.get("/setup/oauth/callback?code=test&state=invalid", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert "invalid_state" in response.headers["location"]


def test_oauth_callback_error_from_google(client, mock_db):
    """OAuth callback handles error from Google."""
    # Set a valid state cookie
    client.cookies.set("oauth_state", "valid-state")
    response = client.get(
        "/setup/oauth/callback?error=access_denied&state=valid-state",
        follow_redirects=False,
    )
    assert response.status_code in (302, 307)
    assert "access_denied" in response.headers["location"]


# ── Status Page Tests ──────────────────────────────────────────────────────────


def test_status_page_renders(client):
    """Status page renders HTML."""
    response = client.get("/status")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "System Status" in response.text


def test_status_page_shows_env_checks(client, monkeypatch):
    """Status page shows environment variable checks."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setenv("AGENT_API_KEY", "test-key")
    monkeypatch.delenv("GMAIL_CLIENT_ID", raising=False)

    response = client.get("/status")
    assert response.status_code == 200
    assert "DATABASE_URL" in response.text
    assert "AGENT_API_KEY" in response.text
    assert "GMAIL_CLIENT_ID" in response.text


def test_status_page_shows_db_connected(client, mock_db):
    """Status page shows database connected with counts."""
    mock_db.list_active_users.return_value = [{"user_id": 1}, {"user_id": 2}]
    mock_db.list_pending_jobs.return_value = [{"job_id": "test-1"}]

    response = client.get("/status")
    assert response.status_code == 200
    assert "Connected" in response.text
    assert "2" in response.text  # 2 users
    assert "1" in response.text  # 1 pending job


def test_status_page_handles_db_failure(client, mock_db, monkeypatch):
    """Status page handles DB failure gracefully."""
    mock_db.list_active_users.side_effect = Exception("Connection refused")
    monkeypatch.delenv("GMAIL_CLIENT_ID", raising=False)

    response = client.get("/status")
    # Should return 200, not 500
    assert response.status_code == 200
    assert "Not Connected" in response.text or "Unknown" in response.text


# ── Health Endpoint Test ───────────────────────────────────────────────────────


def test_health_endpoint(client):
    """Health endpoint returns healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["message"] == "healthy"
