"""Centralized logging utilities for the agent API.

Provides:
- Box formatting for major action headers
- Color support for terminal output
- Centralized logging configuration
- Third-party log suppression
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal

_COLOR_ENABLED = sys.stdout.isatty()
_COLOR_MAP = {
    "reset": "\033[0m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "bold": "\033[1m",
}

ColorName = Literal["red", "green", "yellow", "blue", "magenta", "cyan", "white", "bold"]

# Maximum width for boxes to prevent overly wide output
MAX_BOX_WIDTH = 60

# Third-party loggers to suppress at INFO level (always)
NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "googleapiclient",
    "googleapiclient.discovery",
    "googleapiclient.discovery_cache",
    "google_auth_httplib2",
    "google.auth.transport.requests",
    "urllib3",
    "asyncio",  # Suppress "Future exception was never retrieved" for expected GraphInterrupt
]

# Additional loggers to suppress in quiet mode (LOG_QUIET=true)
# Used in Docker to reduce noise while keeping verbose output for local dev
QUIET_MODE_LOGGERS = [
    "uvicorn",  # Startup messages
    "uvicorn.access",  # Health check spam
    "uvicorn.error",  # Uvicorn error logger
    "transformers",  # Model loading messages
    "transformers.tokenization_utils_base",  # Tokenizer warnings
    "sentence_transformers",  # Embedding model messages
    "huggingface_hub",  # Model download messages
]


def colorize(text: str, color: ColorName) -> str:
    """Apply ANSI color to text if terminal supports it."""
    if not _COLOR_ENABLED or color not in _COLOR_MAP:
        return text
    return f"{_COLOR_MAP[color]}{text}{_COLOR_MAP['reset']}"


def _truncate_line(line: str, max_len: int) -> str:
    """Truncate a line if it exceeds max length, adding ellipsis."""
    if len(line) <= max_len:
        return line
    return line[: max_len - 3] + "..."


def format_box(
    title: str,
    lines: list[str] | None = None,
    color: ColorName = "cyan",
    width: int | None = None,
) -> str:
    """Format text in a Unicode box for visual prominence.

    Args:
        title: Header text for the box
        lines: Optional list of content lines
        color: Color to apply to the box
        width: Explicit width (auto-calculated if None, capped at MAX_BOX_WIDTH)

    Returns:
        Formatted box string ready for logging
    """
    all_lines = [title] + (lines or [])
    if width is None:
        # Calculate width but cap at MAX_BOX_WIDTH
        width = min(max(len(line) for line in all_lines) + 4, MAX_BOX_WIDTH)

    # Content area width (excluding borders and padding)
    content_width = width - 4

    top = "+" + "-" * (width - 2) + "+"
    # Truncate and center title
    display_title = _truncate_line(title, content_width)
    title_line = "| " + display_title.center(content_width) + " |"
    bottom = "+" + "-" * (width - 2) + "+"

    result_lines = [top, title_line]
    if lines:
        result_lines.append("+" + "-" * (width - 2) + "+")
        for line in lines:
            # Truncate long lines
            display_line = _truncate_line(line, content_width)
            result_lines.append("| " + display_line.ljust(content_width) + " |")
    result_lines.append(bottom)

    # Leading newline ensures box is visually separated from previous log output
    return "\n" + colorize("\n".join(result_lines), color)


# ---------------------------------------------------------------------------
# Pre-built box formatters for common worker events
# ---------------------------------------------------------------------------


def worker_start_box(db_path: str, user_count: int = 0) -> str:
    """Format worker startup header."""
    lines = [f"Database: {db_path}"]
    if user_count > 0:
        lines.append(f"Active users: {user_count}")
    return format_box("LangGraph Worker Started", lines, color="green")


def worker_stop_box() -> str:
    """Format worker shutdown header."""
    return format_box("LangGraph Worker Stopped", color="yellow")


def email_processing_box(
    user_id: int,
    email_id: str,
    subject: str,
    from_email: str,
) -> str:
    """Format email processing start header."""
    # Truncate long subjects
    display_subject = subject[:50] + "..." if len(subject) > 50 else subject
    return format_box(
        "Processing Email",
        [
            f"User: {user_id}",
            f"Email ID: {email_id}",
            f"From: {from_email}",
            f"Subject: {display_subject}",
        ],
        color="cyan",
    )


def email_complete_box(
    user_id: int,
    email_id: str,
    subject: str,
) -> str:
    """Format email processing complete header."""
    display_subject = subject[:50] + "..." if len(subject) > 50 else subject
    return format_box(
        "Email Processing Complete",
        [
            f"User: {user_id}",
            f"Email ID: {email_id}",
            f"Subject: {display_subject}",
        ],
        color="green",
    )


def thread_pause_box(thread_id: str, job_id: str, action: str) -> str:
    """Format LangGraph thread pause header."""
    # Truncate long thread IDs for display
    display_thread = thread_id if len(thread_id) <= 35 else thread_id[:32] + "..."
    return format_box(
        "Thread Paused",
        [
            f"Thread: {display_thread}",
            f"Job: {job_id}",
            f"Action: {action}",
        ],
        color="yellow",
    )


def thread_resume_box(thread_id: str, job_id: str, user_id: int | None) -> str:
    """Format LangGraph thread resume header."""
    display_thread = thread_id if len(thread_id) <= 35 else thread_id[:32] + "..."
    return format_box(
        "Thread Resuming",
        [
            f"Thread: {display_thread}",
            f"Job: {job_id}",
            f"User: {user_id or 'N/A'}",
        ],
        color="green",
    )


def job_awaiting_box(thread_id: str, job_id: str, tool_name: str) -> str:
    """Format job awaiting callback header."""
    display_thread = thread_id if len(thread_id) <= 35 else thread_id[:32] + "..."
    return format_box(
        "Awaiting Callback",
        [
            f"Thread: {display_thread}",
            f"Job: {job_id}",
            f"Tool: {tool_name}",
        ],
        color="magenta",
    )


def gmail_poll_summary(
    user_id: int,
    fetched: int,
    processed: int,
    skipped: int,
) -> str:
    """Format Gmail polling summary as single line."""
    return colorize(
        f"Gmail Poll (User {user_id}): Fetched {fetched}, Processed {processed}, Skipped {skipped}",
        "cyan",
    )


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


def is_quiet_mode() -> bool:
    """Check if quiet logging mode is enabled."""
    return os.environ.get("LOG_QUIET", "").lower() == "true"


def log_expected_error(
    logger: logging.Logger,
    message: str,
    *args: object,
    exc_info: BaseException | None = None,
) -> None:
    """Log an expected error - one-liner in quiet mode, full traceback otherwise.

    Use this for errors that are expected/recoverable (OAuth failures, network issues).
    In quiet mode (Docker), logs a clean one-liner. In verbose mode (local dev),
    logs full traceback for debugging.

    Args:
        logger: The logger to use
        message: Log message (can include %s placeholders)
        *args: Arguments for message formatting
        exc_info: Optional exception to include (uses current exception if None)
    """
    if is_quiet_mode():
        # One-liner with just the exception type and message
        if exc_info is None:
            import sys

            exc_info = sys.exc_info()[1]
        if exc_info:
            exc_type = type(exc_info).__name__
            exc_msg = str(exc_info)[:100]  # Truncate long messages
            full_message = f"{message} [{exc_type}: {exc_msg}]"
        else:
            full_message = message
        logger.error(full_message, *args)
    else:
        # Full traceback for local debugging
        logger.exception(message, *args)


def configure_logging(level: str = "INFO") -> None:
    """Configure logging with third-party suppression.

    Args:
        level: Log level for application loggers (DEBUG, INFO, WARNING, ERROR)

    Environment:
        LOG_QUIET: Set to 'true' to enable quiet mode (suppresses more loggers).
                   Used in Docker for cleaner output.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    quiet_mode = os.environ.get("LOG_QUIET", "").lower() == "true"

    # Configure root logger format - simpler format without module prefix clutter
    log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_format = "%H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        force=True,  # Override any existing basicConfig
    )

    # Suppress noisy third-party loggers to WARNING
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # In quiet mode, suppress additional loggers (health checks, model loading, etc.)
    if quiet_mode:
        for logger_name in QUIET_MODE_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
