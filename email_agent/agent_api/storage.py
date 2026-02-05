"""PostgreSQL-backed persistence for users and tool jobs."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

import psycopg2
from psycopg2 import pool

from email_agent.agent_api.schemas import RegisterUserRequest


def utc_now() -> str:
    """Return the current UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class AgentDatabase:
    """PostgreSQL wrapper for user + job persistence."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        """Configure the database connection and ensure schemas exist."""
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")

        # Parse the URL for connection parameters
        parsed = urlparse(self.database_url)
        self._conn_params = {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip("/"),
            "user": parsed.username,
            "password": parsed.password,
        }

        # Create a connection pool
        self._pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            **self._conn_params,
        )
        self._lock = threading.Lock()
        self._initialize()

    def _get_conn(self) -> psycopg2.extensions.connection:
        """Get a connection from the pool."""
        return self._pool.getconn()

    def _put_conn(self, conn: psycopg2.extensions.connection) -> None:
        """Return a connection to the pool."""
        self._pool.putconn(conn)

    def _initialize(self) -> None:
        """Create tables if they don't exist."""
        schema_users = """
        CREATE TABLE IF NOT EXISTS users (
            user_id BIGINT PRIMARY KEY,
            email_to_monitor TEXT NOT NULL,
            email_api_provider TEXT NOT NULL,
            email_api_access_token TEXT NOT NULL,
            email_api_access_token_expires_at BIGINT NOT NULL,
            email_api_refresh_token TEXT NOT NULL,
            email_api_refresh_token_expires_in BIGINT NOT NULL,
            display_name TEXT NOT NULL,
            main_contact TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
        schema_jobs = """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            user_id BIGINT,
            tool_name TEXT,
            status TEXT NOT NULL,
            payload TEXT,
            run_handle TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_callback_at TEXT,
            callback_payload TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
        schema_worker_events = """
        CREATE TABLE IF NOT EXISTS worker_events (
            id SERIAL PRIMARY KEY,
            event_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            processing_started_at TEXT,
            processed_at TEXT,
            processing_error TEXT
        )
        """
        schema_processed_messages = """
        CREATE TABLE IF NOT EXISTS processed_messages (
            user_id BIGINT NOT NULL,
            gmail_message_id TEXT NOT NULL,
            thread_id TEXT,
            processed_at TEXT NOT NULL,
            PRIMARY KEY (user_id, gmail_message_id),
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
        schema_email_attachments = """
        CREATE TABLE IF NOT EXISTS email_attachments (
            id SERIAL PRIMARY KEY,
            job_id TEXT NOT NULL,
            field_path TEXT NOT NULL,
            filename TEXT,
            content_type TEXT,
            base64_data TEXT,
            summary TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES jobs(job_id),
            UNIQUE(job_id, field_path)
        )
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(schema_users)
                cur.execute(schema_jobs)
                cur.execute(schema_worker_events)
                cur.execute(schema_processed_messages)
                cur.execute(schema_email_attachments)
                self._ensure_column(cur, "jobs", "run_handle", "TEXT")
            conn.commit()
        finally:
            self._put_conn(conn)

    @staticmethod
    def _ensure_column(cur: psycopg2.extensions.cursor, table: str, column: str, ddl: str) -> None:
        """Add a column to `table` if it's missing (idempotent)."""
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = %s AND column_name = %s
            """,
            (table, column),
        )
        if not cur.fetchone():
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def save_user(self, payload: RegisterUserRequest) -> None:
        """Insert or update a user row."""
        timestamp = utc_now()
        data = {
            "user_id": payload.user_id,
            "email_to_monitor": payload.email_to_monitor,
            "email_api_provider": payload.email_api_provider,
            "email_api_access_token": payload.email_api_access_token,
            "email_api_access_token_expires_at": payload.email_api_access_token_expires_at,
            "email_api_refresh_token": payload.email_api_refresh_token,
            "email_api_refresh_token_expires_in": payload.email_api_refresh_token_expires_in,
            "display_name": payload.display_name,
            "main_contact": payload.main_contact,
            "status": "active",
            "created_at": timestamp,
            "updated_at": timestamp,
        }

        conn = self._get_conn()
        try:
            with self._lock, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (
                        user_id, email_to_monitor, email_api_provider, email_api_access_token,
                        email_api_access_token_expires_at, email_api_refresh_token,
                        email_api_refresh_token_expires_in, display_name, main_contact,
                        status, created_at, updated_at
                    ) VALUES (
                        %(user_id)s, %(email_to_monitor)s, %(email_api_provider)s, %(email_api_access_token)s,
                        %(email_api_access_token_expires_at)s, %(email_api_refresh_token)s,
                        %(email_api_refresh_token_expires_in)s, %(display_name)s, %(main_contact)s,
                        %(status)s, %(created_at)s, %(updated_at)s
                    )
                    ON CONFLICT(user_id) DO UPDATE SET
                        email_to_monitor=EXCLUDED.email_to_monitor,
                        email_api_provider=EXCLUDED.email_api_provider,
                        email_api_access_token=EXCLUDED.email_api_access_token,
                        email_api_access_token_expires_at=EXCLUDED.email_api_access_token_expires_at,
                        email_api_refresh_token=EXCLUDED.email_api_refresh_token,
                        email_api_refresh_token_expires_in=EXCLUDED.email_api_refresh_token_expires_in,
                        display_name=EXCLUDED.display_name,
                        main_contact=EXCLUDED.main_contact,
                        status='active',
                        updated_at=%(updated_at)s
                    """,
                    data,
                )
            conn.commit()
        finally:
            self._put_conn(conn)

    def deactivate_user(self, user_id: int) -> bool:
        """Mark a user as inactive. Returns True if a row was updated."""
        timestamp = utc_now()
        conn = self._get_conn()
        try:
            with self._lock, conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE users
                    SET status='inactive', updated_at=%s
                    WHERE user_id=%s AND status!='inactive'
                    """,
                    (timestamp, user_id),
                )
                rowcount = cur.rowcount
            conn.commit()
            return rowcount > 0
        finally:
            self._put_conn(conn)

    def get_user(self, user_id: int) -> Optional[dict[str, Any]]:
        """Fetch user row as dict."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE user_id=%s", (user_id,))
                row = cur.fetchone()
                if row:
                    columns = [desc[0] for desc in cur.description]
                    return dict(zip(columns, row, strict=True))
                return None
        finally:
            self._put_conn(conn)

    def list_active_users(self) -> list[dict[str, Any]]:
        """Return all users currently marked as active."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE status='active' ORDER BY user_id")
                rows = cur.fetchall()
                if rows:
                    columns = [desc[0] for desc in cur.description]
                    return [dict(zip(columns, row, strict=True)) for row in rows]
                return []
        finally:
            self._put_conn(conn)

    def record_job(
        self,
        job_id: str,
        user_id: int | None,
        tool_name: str,
        payload: dict[str, Any] | None,
        run_handle: str | None = None,
    ) -> None:
        """Persist a pending job created by the agent."""
        timestamp = utc_now()
        conn = self._get_conn()
        try:
            with self._lock, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO jobs (
                        job_id, user_id, tool_name, status, payload, run_handle, created_at, updated_at
                    ) VALUES (%s, %s, %s, 'pending', %s, %s, %s, %s)
                    ON CONFLICT(job_id) DO UPDATE SET
                        user_id=EXCLUDED.user_id,
                        tool_name=EXCLUDED.tool_name,
                        payload=EXCLUDED.payload,
                        run_handle=COALESCE(EXCLUDED.run_handle, jobs.run_handle),
                        status='pending',
                        updated_at=%s
                    """,
                    (
                        job_id,
                        user_id,
                        tool_name,
                        json.dumps(payload or {}),
                        run_handle,
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
            conn.commit()
        finally:
            self._put_conn(conn)

    def link_job_to_run(self, job_id: str, user_id: int | None, run_handle: str) -> None:
        """Associate a job id with a LangGraph run identifier."""
        self.record_job(job_id, user_id, tool_name="", payload={}, run_handle=run_handle)

    def get_job_run_handle(self, job_id: str) -> Optional[str]:
        """Return the LangGraph run handle for a job, if stored."""
        job_id_str = str(job_id)  # Ensure string
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT run_handle FROM jobs WHERE job_id=%s", (job_id_str,))
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            self._put_conn(conn)

    def record_job_callback(self, job_id: str, tool_output_data: dict[str, Any]) -> None:
        """Store the completion payload for a job (creating the job if necessary)."""
        timestamp = utc_now()
        conn = self._get_conn()
        try:
            with self._lock, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO jobs (
                        job_id, status, payload, created_at, updated_at, last_callback_at, callback_payload
                    ) VALUES (%s, 'completed', %s, %s, %s, %s, %s)
                    ON CONFLICT(job_id) DO UPDATE SET
                        status='completed',
                        last_callback_at=%s,
                        callback_payload=%s,
                        updated_at=%s
                    """,
                    (
                        job_id,
                        json.dumps({}),
                        timestamp,
                        timestamp,
                        timestamp,
                        json.dumps(tool_output_data),
                        timestamp,
                        json.dumps(tool_output_data),
                        timestamp,
                    ),
                )
            conn.commit()
        finally:
            self._put_conn(conn)

    def enqueue_worker_event(self, event_type: str, payload: dict[str, Any]) -> int:
        """Persist an event for the LangGraph worker to consume."""
        timestamp = utc_now()
        conn = self._get_conn()
        try:
            with self._lock, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO worker_events (event_type, payload, created_at)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (event_type, json.dumps(payload), timestamp),
                )
                result = cur.fetchone()
                event_id = result[0] if result else 0
            conn.commit()
            return event_id
        finally:
            self._put_conn(conn)

    def claim_worker_events(self, limit: int = 10) -> list["WorkerEventRecord"]:
        """Fetch up to `limit` pending events and mark them as in-progress."""
        timestamp = utc_now()
        events: list[WorkerEventRecord] = []
        conn = self._get_conn()
        try:
            with self._lock, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, event_type, payload FROM worker_events
                    WHERE processed_at IS NULL AND processing_started_at IS NULL
                    ORDER BY id
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()

                for row in rows:
                    cur.execute(
                        """
                        UPDATE worker_events
                        SET processing_started_at=%s
                        WHERE id=%s
                        """,
                        (timestamp, row[0]),
                    )
                    events.append(
                        WorkerEventRecord(
                            id=row[0],
                            event_type=row[1],
                            payload=json.loads(row[2]),
                        )
                    )
            conn.commit()
        finally:
            self._put_conn(conn)

        return events

    def mark_worker_event_processed(self, event_id: int, error: str | None = None) -> None:
        """Mark an event as processed, storing an error if one occurred."""
        timestamp = utc_now()
        conn = self._get_conn()
        try:
            with self._lock, conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE worker_events
                    SET processed_at=%s,
                        processing_error=%s
                    WHERE id=%s
                    """,
                    (timestamp, error, event_id),
                )
            conn.commit()
        finally:
            self._put_conn(conn)

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Fetch a stored job record."""
        job_id_str = str(job_id)  # Ensure string
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM jobs WHERE job_id=%s", (job_id_str,))
                row = cur.fetchone()
                if row:
                    columns = [desc[0] for desc in cur.description]
                    return dict(zip(columns, row, strict=True))
                return None
        finally:
            self._put_conn(conn)

    def get_job_callback_payload(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return the parsed callback payload for a completed job."""
        job = self.get_job(job_id)
        if not job or not job.get("callback_payload"):
            return None
        return json.loads(job["callback_payload"])

    def list_pending_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List all jobs with status='pending', ordered by creation time."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT job_id, user_id, tool_name, status, payload, run_handle,
                           created_at, updated_at
                    FROM jobs
                    WHERE status = 'pending'
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row, strict=True)) for row in rows]
        finally:
            self._put_conn(conn)

    def has_processed_email(self, user_id: int, gmail_message_id: str) -> bool:
        """Return True if we've already handled this Gmail message for the user."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM processed_messages
                    WHERE user_id=%s AND gmail_message_id=%s
                    """,
                    (user_id, gmail_message_id),
                )
                return cur.fetchone() is not None
        finally:
            self._put_conn(conn)

    def get_processed_message_ids(self, user_id: int) -> set[str]:
        """Return all processed Gmail message IDs for a user."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT gmail_message_id
                    FROM processed_messages
                    WHERE user_id=%s
                    """,
                    (user_id,),
                )
                return {row[0] for row in cur.fetchall()}
        finally:
            self._put_conn(conn)

    def record_processed_email(
        self,
        user_id: int,
        gmail_message_id: str,
        thread_id: str | None,
    ) -> None:
        """Persist that we've kicked off processing for a Gmail message."""
        timestamp = utc_now()
        conn = self._get_conn()
        try:
            with self._lock, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO processed_messages (user_id, gmail_message_id, thread_id, processed_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT(user_id, gmail_message_id) DO UPDATE SET processed_at=%s
                    """,
                    (user_id, gmail_message_id, thread_id, timestamp, timestamp),
                )
            conn.commit()
        finally:
            self._put_conn(conn)

    def close(self) -> None:
        """Close all connections in the pool."""
        self._pool.closeall()

    # ── Email Attachments ────────────────────────────────────────────────────

    def save_email_attachment(
        self,
        job_id: str,
        field_path: str,
        base64_data: str,
        filename: str | None = None,
        content_type: str | None = None,
        summary: str | None = None,
    ) -> int:
        """Store a base64 attachment for a job, returning the row id."""
        timestamp = utc_now()
        conn = self._get_conn()
        try:
            with self._lock, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO email_attachments
                        (job_id, field_path, filename, content_type, base64_data, summary, created_at)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(job_id, field_path) DO UPDATE SET
                        filename=EXCLUDED.filename,
                        content_type=EXCLUDED.content_type,
                        base64_data=EXCLUDED.base64_data,
                        summary=EXCLUDED.summary
                    RETURNING id
                    """,
                    (job_id, field_path, filename, content_type, base64_data, summary, timestamp),
                )
                result = cur.fetchone()
                row_id = result[0] if result else 0
            conn.commit()
            return row_id
        finally:
            self._put_conn(conn)

    def get_email_attachment(self, job_id: str, field_path: str) -> Optional[dict[str, Any]]:
        """Retrieve a single attachment by job_id and field_path."""
        job_id_str = str(job_id)  # Ensure string
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, job_id, field_path, filename, content_type, base64_data, summary, created_at
                    FROM email_attachments
                    WHERE job_id=%s AND field_path=%s
                    """,
                    (job_id_str, field_path),
                )
                row = cur.fetchone()
                if row:
                    columns = [desc[0] for desc in cur.description]
                    return dict(zip(columns, row, strict=True))
                return None
        finally:
            self._put_conn(conn)

    def get_email_attachments(self, job_id: str) -> list[dict[str, Any]]:
        """Retrieve all attachments for a job."""
        job_id_str = str(job_id)  # Ensure string
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, job_id, field_path, filename, content_type, base64_data, summary, created_at
                    FROM email_attachments
                    WHERE job_id=%s
                    ORDER BY id
                    """,
                    (job_id_str,),
                )
                rows = cur.fetchall()
                if rows:
                    columns = [desc[0] for desc in cur.description]
                    return [dict(zip(columns, row, strict=True)) for row in rows]
                return []
        finally:
            self._put_conn(conn)

    def delete_email_attachments(self, job_id: str) -> int:
        """Delete all attachments for a job, returning count deleted."""
        job_id_str = str(job_id)  # Ensure string
        conn = self._get_conn()
        try:
            with self._lock, conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM email_attachments WHERE job_id=%s",
                    (job_id_str,),
                )
                rowcount = cur.rowcount
            conn.commit()
            return rowcount
        finally:
            self._put_conn(conn)


@dataclass(slots=True)
class WorkerEventRecord:
    """Typed mapping for events popped from the worker queue."""

    id: int
    event_type: str
    payload: dict[str, Any]
