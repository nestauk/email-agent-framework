"""FastAPI application exposing the email agent API endpoints."""

from __future__ import annotations

import base64
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request

from fastapi.staticfiles import StaticFiles

from email_agent.agent_api.auth import verify_agent_api_key
from email_agent.agent_api.logging_utils import configure_logging
from email_agent.agent_api.schemas import (
    MessageResponse,
    RegisterUserRequest,
    ToolCompletedRequest,
    UnregisterUserRequest,
)
from email_agent.agent_api.storage import AgentDatabase
from email_agent.agent_api.web_routes import router as web_router

configure_logging(os.getenv("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


# ── Base64 Attachment Detection and Extraction ────────────────────────────────


def _detect_base64_content(value: str) -> tuple[bool, str | None, str | None]:
    """Check if a string is likely base64-encoded binary content.

    Returns (is_base64, content_type, cleaned_data).
    Handles both raw base64 and data URI format (data:application/pdf;base64,...).
    """
    if len(value) < 500:
        return (False, None, None)

    # Handle data URI format: data:application/pdf;base64,JVBERi0...
    if value.startswith("data:"):
        try:
            # Parse data URI: data:<mime>;base64,<data>
            header, data = value.split(",", 1)
            if ";base64" in header:
                mime = header.replace("data:", "").replace(";base64", "")
                return (True, mime, data)
        except ValueError:
            pass
        return (False, None, None)

    # Check for no spaces in first 100 chars (base64 doesn't have spaces)
    if " " in value[:100]:
        return (False, None, None)

    # Check magic bytes for common file types
    if value.startswith("JVBERi0"):  # %PDF-
        return (True, "application/pdf", value)
    if value.startswith("/9j/"):  # JPEG
        return (True, "image/jpeg", value)
    if value.startswith("iVBORw0"):  # PNG
        return (True, "image/png", value)

    return (False, None, None)


def _summarize_pdf_for_storage(base64_data: str, field_name: str) -> str:
    """Extract and summarize a base64-encoded PDF for storage.

    Returns a human-readable summary string.
    """
    try:
        from email_agent.agent.utils import extract_text_from_pdf_bytes, summarise_pdf

        pdf_bytes = base64.b64decode(base64_data)
        text = extract_text_from_pdf_bytes(pdf_bytes)

        if not text:
            return f"[ATTACHMENT: {field_name} - PDF received but could not extract text]"

        summary = summarise_pdf(text, field_name)
        if summary.unreadable_flag:
            return f"[ATTACHMENT: {field_name} - PDF received but unreadable]"

        parts = [f"[ATTACHMENT: {field_name}]"]
        if summary.key_points:
            parts.append(f"Key points: {', '.join(summary.key_points)}")
        if summary.actions:
            parts.append(f"Actions: {', '.join(summary.actions)}")
        if summary.deadlines:
            parts.append(f"Deadlines: {', '.join(summary.deadlines)}")
        return " | ".join(parts)
    except Exception as e:
        logger.warning("Failed to summarize PDF %s: %s", field_name, e)
        return f"[ATTACHMENT: {field_name} - PDF received]"


def _extract_and_store_attachments(  # noqa: C901
    db: AgentDatabase,
    job_id: str,
    data: dict[str, Any],
    path_prefix: str = "",
) -> dict[str, Any]:
    """Recursively scan data for base64 content, store separately, replace with references.

    Args:
        db: Database instance for storing attachments
        job_id: Job ID to associate attachments with
        data: The data dict to scan (toolOutputData)
        path_prefix: Current path for nested fields (e.g., "nested.field")

    Returns:
        Modified data dict with base64 replaced by placeholder references
    """
    result: dict[str, Any] = {}

    for key, value in data.items():
        field_path = f"{path_prefix}.{key}" if path_prefix else key

        if isinstance(value, str):
            is_base64, content_type, cleaned_data = _detect_base64_content(value)
            if is_base64 and cleaned_data:
                # Extract filename from nested dict if this is a file object
                filename = None
                if isinstance(data.get(key), dict):
                    filename = data[key].get("name") or data[key].get("filename")

                # Compute summary for PDFs
                summary = None
                if content_type == "application/pdf":
                    summary = _summarize_pdf_for_storage(cleaned_data, field_path)
                else:
                    summary = f"[ATTACHMENT: {field_path} - {content_type or 'binary'} file]"

                # Store in database
                db.save_email_attachment(
                    job_id=job_id,
                    field_path=field_path,
                    base64_data=cleaned_data,
                    filename=filename,
                    content_type=content_type,
                    summary=summary,
                )
                logger.info("Stored attachment %s for job %s (%d bytes)", field_path, job_id, len(cleaned_data))

                # Replace with placeholder reference
                result[key] = {
                    "_attachment_ref": field_path,
                    "content_type": content_type,
                    "summary": summary,
                }
            else:
                result[key] = value

        elif isinstance(value, dict):
            # Check if this is a file object with base64Content field
            base64_content = value.get("base64Content") or value.get("content_base64")
            if base64_content and isinstance(base64_content, str):
                is_base64, content_type, cleaned_data = _detect_base64_content(base64_content)
                if is_base64 and cleaned_data:
                    filename = value.get("name") or value.get("filename")
                    file_type = value.get("type") or content_type

                    # Compute summary
                    summary = None
                    if file_type == "application/pdf" or (filename and filename.lower().endswith(".pdf")):
                        summary = _summarize_pdf_for_storage(cleaned_data, field_path)
                    else:
                        summary = f"[ATTACHMENT: {field_path} - {file_type or 'binary'} file]"

                    # Store in database
                    db.save_email_attachment(
                        job_id=job_id,
                        field_path=field_path,
                        base64_data=cleaned_data,
                        filename=filename,
                        content_type=file_type,
                        summary=summary,
                    )
                    logger.info("Stored attachment %s for job %s (%d bytes)", field_path, job_id, len(cleaned_data))

                    # Replace with placeholder (keep non-base64 metadata)
                    result[key] = {
                        "_attachment_ref": field_path,
                        "filename": filename,
                        "content_type": file_type,
                        "summary": summary,
                    }
                else:
                    # Recurse into dict
                    result[key] = _extract_and_store_attachments(db, job_id, value, field_path)
            else:
                # Recurse into dict
                result[key] = _extract_and_store_attachments(db, job_id, value, field_path)

        elif isinstance(value, list):
            result[key] = []
            for i, item in enumerate(value):
                item_path = f"{field_path}[{i}]"
                if isinstance(item, dict):
                    result[key].append(_extract_and_store_attachments(db, job_id, item, item_path))
                elif isinstance(item, str):
                    is_base64, content_type, cleaned_data = _detect_base64_content(item)
                    if is_base64 and cleaned_data:
                        summary = None
                        if content_type == "application/pdf":
                            summary = _summarize_pdf_for_storage(cleaned_data, item_path)
                        else:
                            summary = f"[ATTACHMENT: {item_path} - {content_type or 'binary'} file]"

                        db.save_email_attachment(
                            job_id=job_id,
                            field_path=item_path,
                            base64_data=cleaned_data,
                            content_type=content_type,
                            summary=summary,
                        )
                        logger.info("Stored attachment %s for job %s (%d bytes)", item_path, job_id, len(cleaned_data))

                        result[key].append(
                            {
                                "_attachment_ref": item_path,
                                "content_type": content_type,
                                "summary": summary,
                            }
                        )
                    else:
                        result[key].append(item)
                else:
                    result[key].append(item)
        else:
            result[key] = value

    return result


def build_database() -> AgentDatabase:
    """Instantiate the PostgreSQL store using DATABASE_URL."""
    database_url = os.getenv("DATABASE_URL")
    return AgentDatabase(database_url=database_url)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Attach the database to the FastAPI app."""
    app.state.db = build_database()
    logger.info("Agent API database initialised at %s", app.state.db.database_url)
    yield


app = FastAPI(
    title="Email Agent API",
    description="Endpoints for email agent registration and async tool callbacks",
    version="0.1.0",
    lifespan=lifespan,
)


def get_db(request: Request) -> AgentDatabase:
    """Dependency that exposes the AgentDatabase instance."""
    return request.app.state.db


@app.get("/health", response_model=MessageResponse)
async def healthcheck() -> MessageResponse:
    """Simple readiness endpoint."""
    return MessageResponse(message="healthy")


@app.post("/registerUser", response_model=MessageResponse)
async def register_user(
    payload: RegisterUserRequest,
    _api_key: str = Depends(verify_agent_api_key),  # noqa: B008
    db: AgentDatabase = Depends(get_db),  # noqa: B008
) -> MessageResponse:
    """Store user credentials and kick off inbox monitoring."""
    db.save_user(payload)
    logger.info("Registered/updated user %s (%s)", payload.user_id, payload.email_to_monitor)

    db.enqueue_worker_event(
        "register",
        {"user_id": payload.user_id},
    )

    return MessageResponse(message="success")


@app.post("/unregisterUser", response_model=MessageResponse)
async def unregister_user(
    payload: UnregisterUserRequest,
    _api_key: str = Depends(verify_agent_api_key),  # noqa: B008
    db: AgentDatabase = Depends(get_db),  # noqa: B008
) -> MessageResponse:
    """Disable inbox monitoring for a user."""
    user = db.get_user(payload.user_id)
    if not user:
        logger.warning(
            "Unregister request for user %s that is not in the local DB; treating as success",
            payload.user_id,
        )
        message = f"User {payload.user_id} already inactive locally"
    else:
        db.deactivate_user(payload.user_id)
        logger.info("Unregistered user %s", payload.user_id)
        message = "success"

    db.enqueue_worker_event(
        "unregister",
        {"user_id": payload.user_id},
    )

    return MessageResponse(message=message)


def _summarize_tool_output(data: object) -> object:
    """Summarize tool output for logging, truncating base64 content."""
    if isinstance(data, str):
        if len(data) > 200:
            return f"{data[:50]}... ({len(data)} chars)"
        return data
    if isinstance(data, list):
        return [_summarize_tool_output(item) for item in data]
    if isinstance(data, dict):
        summary = {}
        for key, value in data.items():
            summary[key] = _summarize_tool_output(value)
        return summary
    return data


@app.post("/toolCompleted", response_model=MessageResponse)
async def tool_completed(
    request: Request,
    _api_key: str = Depends(verify_agent_api_key),  # noqa: B008
    db: AgentDatabase = Depends(get_db),  # noqa: B008
) -> MessageResponse:
    """Receive asynchronous tool output data."""
    raw_body = await request.body()

    try:
        raw_json = json.loads(raw_body)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # Parse with Pydantic
    payload = ToolCompletedRequest.model_validate(raw_json)

    # Extract base64 attachments, store separately, replace with summaries
    # This keeps large files out of LLM context while preserving them for forwarding
    processed_output = _extract_and_store_attachments(
        db=db,
        job_id=str(payload.job_id),
        data=payload.tool_output_data,
    )

    db.record_job_callback(payload.job_id, processed_output)

    # Log tool completion with output summary (skip if empty)
    output_summary = _summarize_tool_output(raw_json.get("toolOutputData", {}))
    if output_summary:
        logger.info("Tool completion for job %s: %s", payload.job_id, output_summary)

    db.enqueue_worker_event("resume", {"job_id": payload.job_id})

    return MessageResponse(message="success")


# ── Web Frontend Routes ────────────────────────────────────────────────────────

app.include_router(web_router)
