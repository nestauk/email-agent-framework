"""Web frontend routes for the email agent."""

from __future__ import annotations

import json
import os
import secrets
from typing import Annotated, Any
from urllib.parse import urlencode

import base64

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from email_agent.agent_api.storage import AgentDatabase

router = APIRouter(tags=["web"])

# Template setup - path relative to this file
_templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=_templates_dir)

# OAuth state cookie name
STATE_COOKIE_NAME = "oauth_state"


def get_db(request: Request) -> AgentDatabase:
    """Dependency that exposes the AgentDatabase instance."""
    return request.app.state.db


# ── Dashboard ──────────────────────────────────────────────────────────────────


def _parse_job_for_list(job: dict[str, Any]) -> dict[str, Any]:
    """Parse job payload to extract display-friendly info for the job list."""
    from datetime import datetime

    result = {
        "job_id": job.get("job_id"),
        "tool_name": job.get("tool_name"),
        "created_at": job.get("created_at"),
        "from_name": None,
        "from_email": None,
        "subject": None,
        "action_label": "Review",
    }

    # Parse payload for email info
    payload_str = job.get("payload", "{}")
    try:
        payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
        email_meta = payload.get("email", {})
        result["from_email"] = email_meta.get("from_email", "")
        result["subject"] = email_meta.get("subject", "")

        # Extract name from "Name <email>" format
        from_email = result["from_email"] or ""
        if "<" in from_email and ">" in from_email:
            result["from_name"] = from_email.split("<")[0].strip()
        else:
            result["from_name"] = from_email.split("@")[0] if "@" in from_email else from_email
    except (json.JSONDecodeError, TypeError):
        pass

    # Friendly action labels
    tool_name = job.get("tool_name", "")
    if tool_name == "send_email_tool":
        result["action_label"] = "Review draft"
    elif tool_name == "Question":
        result["action_label"] = "Answer"
    else:
        result["action_label"] = "Review"

    # Relative time
    created_at = job.get("created_at")
    if created_at:
        try:
            if isinstance(created_at, str):
                # Handle ISO format
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            else:
                dt = created_at
            now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
            delta = now - dt
            minutes = int(delta.total_seconds() / 60)
            if minutes < 1:
                result["time_ago"] = "just now"
            elif minutes < 60:
                result["time_ago"] = f"{minutes}m ago"
            elif minutes < 1440:
                result["time_ago"] = f"{minutes // 60}h ago"
            else:
                result["time_ago"] = f"{minutes // 1440}d ago"
        except (ValueError, TypeError):
            result["time_ago"] = ""

    return result


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: AgentDatabase = Depends(get_db)) -> HTMLResponse:  # noqa: B008
    """Render the dashboard with pending jobs list."""
    raw_jobs = db.list_pending_jobs(limit=50)
    jobs = [_parse_job_for_list(job) for job in raw_jobs]
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"jobs": jobs},
    )


@router.get("/partials/job-list", response_class=HTMLResponse)
async def job_list_partial(request: Request, db: AgentDatabase = Depends(get_db)) -> HTMLResponse:  # noqa: B008
    """HTMX partial: render just the job list table."""
    raw_jobs = db.list_pending_jobs(limit=50)
    jobs = [_parse_job_for_list(job) for job in raw_jobs]
    return templates.TemplateResponse(
        request,
        "partials/job_list.html",
        {"jobs": jobs},
    )


# ── Review Page ────────────────────────────────────────────────────────────────


def _parse_question_options(question_text: str) -> dict[str, Any]:
    """Parse a question with multiple choice options like (A), (B), (C)."""
    import re

    # Try to split on "Do you want to:" or similar
    context = question_text
    options_text = ""

    # Look for "Do you want to:" pattern
    do_you_match = re.search(r"[.?!]\s*Do you want to:?\s*", question_text, re.IGNORECASE)
    if do_you_match:
        context = question_text[: do_you_match.start() + 1].strip()
        options_text = question_text[do_you_match.end() :].strip()

    # Parse options - look for pattern like "- (A) text - (B) text" or "(A) text (B) text"
    options: list[dict[str, str]] = []

    if options_text:
        # Split by "- (X)" pattern where X is A, B, C, etc.
        parts = re.split(r"\s*-\s*\(([A-Z])\)\s*", options_text)
        # parts will be: ['', 'A', 'text for A', 'B', 'text for B', ...]

        if len(parts) > 1:
            # Skip first empty element, then take pairs of (letter, text)
            i = 1
            while i < len(parts) - 1:
                letter = parts[i]
                text = parts[i + 1].strip().rstrip("—").rstrip("-").rstrip(".").strip()
                if letter and text:
                    options.append({"letter": letter.upper(), "text": text})
                i += 2

    # Fallback: try splitting by "(A)" pattern without dash
    if not options and options_text:
        parts = re.split(r"\s*\(([A-Z])\)\s*", options_text)
        if len(parts) > 1:
            i = 1
            while i < len(parts) - 1:
                letter = parts[i]
                text = parts[i + 1].strip().rstrip("—").rstrip("-").rstrip(".").strip()
                if letter and text:
                    options.append({"letter": letter.upper(), "text": text})
                i += 2

    return {
        "context": context,
        "options": options,
        "has_options": len(options) > 0,
    }


def _parse_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    """Parse and extract useful info from job payload."""
    import re

    payload = json.loads(job.get("payload") or "{}") if isinstance(job.get("payload"), str) else job.get("payload", {})
    request_data = payload.get("request", {})
    config = request_data.get("config", {})
    action_request = request_data.get("action_request", {})

    # Clean up description (remove tool call JSON and question sections)
    description = request_data.get("description", "")
    if description:
        # Remove "Tool Call: ..." section
        clean_desc = re.split(r"\n-+\n\s*Tool Call:", description)[0].strip()
        # Remove "# Question for User" section
        clean_desc = re.split(r"\n-+\n\s*#\s*Question for User", clean_desc)[0].strip()
        description = clean_desc

    # Get the question content
    args = action_request.get("args", {}) if action_request else {}
    question_content = args.get("content") or args.get("question", "")

    # Parse question for multiple choice options
    question_parsed = _parse_question_options(question_content) if question_content else {
        "context": question_content,
        "options": [],
        "has_options": False,
    }

    # Parse email summary from description
    email_summary = _parse_email_summary(description)

    return {
        "description": description,
        "config": config,
        "action_request": action_request,
        "action": action_request.get("action", "unknown") if action_request else "unknown",
        "args": args,
        "question": question_parsed,
        "email_summary": email_summary,
    }


def _parse_email_summary(description: str) -> dict[str, str]:
    """Extract email metadata from description for summary display."""
    import re

    summary: dict[str, str] = {
        "from_name": "",
        "from_email": "",
        "subject": "",
        "preview": "",
    }

    if not description:
        return summary

    # Extract From: "Simon Wisdom <simon.wisdom@nesta.org.uk>"
    from_match = re.search(r"\*\*From\*\*:\s*([^<\n]+)?<?([^>\n]+@[^>\n]+)?>?", description)
    if from_match:
        summary["from_name"] = (from_match.group(1) or "").strip()
        summary["from_email"] = (from_match.group(2) or "").strip()

    # Extract Subject
    subject_match = re.search(r"\*\*Subject\*\*:\s*([^\n]+)", description)
    if subject_match:
        summary["subject"] = subject_match.group(1).strip()

    # Extract body preview - text after the metadata, before signatures
    # Look for content after the metadata block
    lines = description.split("\n")
    body_lines = []
    in_body = False
    for line in lines:
        # Skip metadata lines
        if line.strip().startswith("**") and "**:" in line:
            continue
        # Skip empty lines at start
        if not in_body and not line.strip():
            continue
        in_body = True
        # Stop at signature indicators
        if line.strip().lower() in ["regards", "thanks", "best", "cheers"] or line.strip().startswith("--"):
            break
        body_lines.append(line.strip())

    preview = " ".join(body_lines)[:150]
    if len(" ".join(body_lines)) > 150:
        preview += "..."
    summary["preview"] = preview

    return summary


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def review_job(
    request: Request,
    job_id: str,
    db: AgentDatabase = Depends(get_db),  # noqa: B008
) -> HTMLResponse:
    """Render the review page for a specific job."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    parsed = _parse_job_payload(job)
    attachments = db.get_email_attachments(job_id)

    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "job": job,
            "parsed": parsed,
            "attachments": attachments,
        },
    )


@router.post("/jobs/{job_id}/action", response_class=HTMLResponse)
async def submit_action(  # noqa: C901
    request: Request,
    job_id: str,
    action_type: Annotated[str, Form()],
    response_text: Annotated[str | None, Form()] = None,
    edited_body: Annotated[str | None, Form()] = None,
    db: AgentDatabase = Depends(get_db),  # noqa: B008
) -> HTMLResponse:
    """Handle action submission for a job."""
    job = db.get_job(job_id)
    if not job:
        return templates.TemplateResponse(
            request,
            "partials/action_result.html",
            {"success": False, "message": "Job not found"},
        )

    # Build tool_output based on action_type
    tool_output: dict[str, Any] = {"type": action_type}

    if action_type == "response" and response_text:
        tool_output["args"] = {"response": response_text}
    elif action_type == "edit" and edited_body:
        # Get original args and merge with edited body
        parsed = _parse_job_payload(job)
        original_args = parsed.get("args", {})
        edited_args = dict(original_args)

        # For send_email_tool, update the response_text
        if parsed.get("action") == "send_email_tool":
            edited_args["response_text"] = edited_body
        else:
            # Generic: try to update body/content/response_text
            for key in ["response_text", "body", "content"]:
                if key in edited_args:
                    edited_args[key] = edited_body
                    break

        tool_output["args"] = {"args": edited_args}

    # Call /toolCompleted via httpx
    api_url = os.getenv("AGENT_API_URL", "http://localhost:8000")
    api_key = os.getenv("AGENT_API_KEY", "")

    payload = {
        "jobId": job_id,
        "tool": job.get("tool_name") or "unknown",
        "toolOutputData": tool_output,
    }

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{api_url}/toolCompleted",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
            return templates.TemplateResponse(
                request,
                "partials/action_result.html",
                {
                    "success": True,
                    "message": f"Action '{action_type}' submitted successfully!",
                },
            )
        except httpx.TimeoutException:
            return templates.TemplateResponse(
                request,
                "partials/action_result.html",
                {
                    "success": False,
                    "message": "Request timed out. Please try again.",
                },
            )
        except httpx.HTTPStatusError as e:
            return templates.TemplateResponse(
                request,
                "partials/action_result.html",
                {
                    "success": False,
                    "message": f"Server error: {e.response.status_code}",
                },
            )
        except httpx.RequestError as e:
            return templates.TemplateResponse(
                request,
                "partials/action_result.html",
                {
                    "success": False,
                    "message": f"Connection error: {e}",
                },
            )


# ── Attachment Download ───────────────────────────────────────────────────────


@router.get("/jobs/{job_id}/attachments/{field_path}")
async def download_attachment(
    request: Request,
    job_id: str,
    field_path: str,
    db: AgentDatabase = Depends(get_db),
) -> Response:
    """Download an attachment file."""
    from urllib.parse import quote

    attachment = db.get_email_attachment(job_id, field_path)
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")

    base64_data = attachment.get("base64_data")
    if not base64_data:
        raise HTTPException(status_code=404, detail="Attachment data not available")

    # Decode base64 to bytes
    file_bytes = base64.b64decode(base64_data)
    filename = attachment.get("filename", "attachment.pdf")
    content_type = attachment.get("content_type", "application/octet-stream")

    # RFC 5987 encoding for Unicode filenames
    filename_ascii = filename.encode("ascii", "ignore").decode("ascii") or "attachment.pdf"
    filename_utf8 = quote(filename, safe="")

    return Response(
        content=file_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f"inline; filename=\"{filename_ascii}\"; filename*=UTF-8''{filename_utf8}",
        },
    )


# ── Setup Page with OAuth ───────────────────────────────────────────���──────────


@router.get("/setup", response_class=HTMLResponse)
async def setup_page(
    request: Request,
    success: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    """Render the OAuth setup page."""
    # Check if Gmail credentials are configured
    has_gmail_config = bool(os.getenv("GMAIL_CLIENT_ID")) and bool(os.getenv("GMAIL_CLIENT_SECRET"))

    return templates.TemplateResponse(
        request,
        "setup.html",
        {
            "has_gmail_config": has_gmail_config,
            "success": success,
            "error": error,
        },
    )


@router.get("/setup/oauth/start")
async def oauth_start(request: Request) -> RedirectResponse:
    """Start the OAuth flow by redirecting to Google."""
    client_id = os.getenv("GMAIL_CLIENT_ID")
    if not client_id:
        raise HTTPException(status_code=500, detail="GMAIL_CLIENT_ID not configured")

    # Generate random state
    state = secrets.token_urlsafe(32)

    # Build redirect URI from request
    redirect_uri = str(request.url_for("oauth_callback"))

    # Build Google OAuth URL
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/gmail.modify",
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"

    # Set state in cookie, redirect to Google
    response = RedirectResponse(url=auth_url, status_code=307)
    response.set_cookie(
        STATE_COOKIE_NAME,
        state,
        httponly=True,
        secure=request.url.scheme == "https",
        max_age=600,  # 10 minutes
        samesite="lax",
    )
    return response


@router.get("/setup/oauth/callback")
async def oauth_callback(  # noqa: C901
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    db: AgentDatabase = Depends(get_db),  # noqa: B008
) -> RedirectResponse:
    """Handle the OAuth callback from Google."""
    # Validate state from cookie
    stored_state = request.cookies.get(STATE_COOKIE_NAME)
    if not stored_state or stored_state != state:
        return RedirectResponse("/setup?error=invalid_state")

    if error:
        return RedirectResponse(f"/setup?error={error}")

    if not code:
        return RedirectResponse("/setup?error=no_code")

    # Get credentials
    client_id = os.getenv("GMAIL_CLIENT_ID")
    client_secret = os.getenv("GMAIL_CLIENT_SECRET")
    if not client_id or not client_secret:
        return RedirectResponse("/setup?error=missing_credentials")

    # Build redirect URI (same as start)
    redirect_uri = str(request.url_for("oauth_callback"))

    # Exchange code for tokens
    token_url = "https://oauth2.googleapis.com/token"
    token_data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }

    async with httpx.AsyncClient() as client:
        try:
            token_resp = await client.post(token_url, data=token_data, timeout=30.0)
            token_resp.raise_for_status()
            tokens = token_resp.json()
        except httpx.HTTPStatusError:
            response = RedirectResponse("/setup?error=token_exchange_failed")
            response.delete_cookie(STATE_COOKIE_NAME)
            return response
        except httpx.RequestError:
            response = RedirectResponse("/setup?error=token_exchange_error")
            response.delete_cookie(STATE_COOKIE_NAME)
            return response

    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    expires_in = tokens.get("expires_in", 3600)

    if not access_token or not refresh_token:
        response = RedirectResponse("/setup?error=missing_tokens")
        response.delete_cookie(STATE_COOKIE_NAME)
        return response

    # Get user profile from Gmail API
    async with httpx.AsyncClient() as client:
        try:
            profile_resp = await client.get(
                "https://www.googleapis.com/gmail/v1/users/me/profile",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=30.0,
            )
            profile_resp.raise_for_status()
            profile = profile_resp.json()
        except (httpx.HTTPStatusError, httpx.RequestError):
            response = RedirectResponse("/setup?error=profile_fetch_failed")
            response.delete_cookie(STATE_COOKIE_NAME)
            return response

    email_address = profile.get("emailAddress")
    if not email_address:
        response = RedirectResponse("/setup?error=no_email")
        response.delete_cookie(STATE_COOKIE_NAME)
        return response

    # Generate user ID from email hash
    user_id = abs(hash(email_address)) % 1_000_000

    # Calculate token expiry
    import time

    expires_at = int(time.time()) + expires_in

    # Register user with the API
    from email_agent.agent_api.schemas import RegisterUserRequest

    user_payload = RegisterUserRequest(
        user_id=user_id,
        email_to_monitor=email_address,
        email_api_provider="google",
        email_api_access_token=access_token,
        email_api_access_token_expires_at=expires_at,
        email_api_refresh_token=refresh_token,
        email_api_refresh_token_expires_in=expires_in,
        display_name=email_address.split("@")[0],
    )

    try:
        db.save_user(user_payload)
        db.enqueue_worker_event("register", {"user_id": user_id})
    except Exception:
        response = RedirectResponse("/setup?error=registration_failed")
        response.delete_cookie(STATE_COOKIE_NAME)
        return response

    # Success - clear state cookie and redirect
    response = RedirectResponse(f"/setup?success=1&email={email_address}")
    response.delete_cookie(STATE_COOKIE_NAME)
    return response


# ── Status Page ────────────────────────────────────────────────────────────────


@router.get("/status", response_class=HTMLResponse)
async def status_page(request: Request, db: AgentDatabase = Depends(get_db)) -> HTMLResponse:  # noqa: B008
    """Render the system status page."""
    # Environment checks
    env_checks = {
        "DATABASE_URL": bool(os.getenv("DATABASE_URL")),
        "AGENT_API_KEY": bool(os.getenv("AGENT_API_KEY")),
        "GMAIL_CLIENT_ID": bool(os.getenv("GMAIL_CLIENT_ID")),
        "GMAIL_CLIENT_SECRET": bool(os.getenv("GMAIL_CLIENT_SECRET")),
        "AZURE_OPENAI_ENDPOINT_MINI": bool(os.getenv("AZURE_OPENAI_ENDPOINT_MINI")),
    }

    # Database checks with graceful failure
    db_status: dict[str, Any] = {"connected": False, "active_users": "Unknown", "pending_jobs": "Unknown"}
    try:
        users = db.list_active_users()
        jobs = db.list_pending_jobs()
        db_status = {
            "connected": True,
            "active_users": len(users),
            "pending_jobs": len(jobs),
        }
    except Exception as e:
        db_status["error"] = str(e)

    return templates.TemplateResponse(
        request,
        "status.html",
        {
            "env_checks": env_checks,
            "db_status": db_status,
        },
    )
