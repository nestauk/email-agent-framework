"""Lightweight LangGraph worker that consumes events from PostgreSQL."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import logging
import os
import signal
import uuid
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from email_agent.agent_api.logging_utils import (
    configure_logging,
    email_complete_box,
    email_processing_box,
    gmail_poll_summary,
    log_expected_error,
    thread_pause_box,
    worker_start_box,
    worker_stop_box,
)
from email_agent.agent_api.storage import AgentDatabase, WorkerEventRecord
from email_agent.agent.tools.gmail.gmail_tools import fetch_group_emails

configure_logging(os.getenv("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

logging.getLogger("email_agent.agent.tools.gmail.gmail_tools").setLevel(logging.WARNING)


def _load_email_assistant() -> ModuleType:
    """Import the LangGraph workflow for the Gmail HITL agent."""
    try:
        from email_agent.agent import graph as assistant_module
    except ImportError as exc:  # pragma: no cover - validated at runtime
        raise RuntimeError(
            "Could not import email_agent.agent.graph. Ensure the email_agent package is installed."
        ) from exc

    return assistant_module


WorkerEventType = Literal["register", "unregister", "resume"]


class LangGraphWorker:
    """Background worker that translates DB queue events into LangGraph runs."""

    def __init__(
        self,
        database_url: str | None = None,
        poll_interval: float = 1.0,
    ) -> None:
        """Initialize the worker, LangGraph runtime, and polling options."""
        self._db = AgentDatabase(database_url=database_url)
        self._poll_interval = poll_interval
        self._stop_event = asyncio.Event()
        self._user_lock = asyncio.Lock()
        self._user_tasks: dict[int, asyncio.Task[Any]] = {}
        self._user_stop_signals: dict[int, asyncio.Event] = {}
        self._seen_message_ids: defaultdict[int, set[str]] = defaultdict(set)
        self._inbox_poll_interval = float(os.getenv("WORKER_GMAIL_POLL_SECONDS", "30"))
        self._gmail_minutes_since = int(os.getenv("WORKER_GMAIL_LOOKBACK_MINUTES", "120"))
        self._auto_accept_interrupts = os.getenv("WORKER_AUTO_ACCEPT_INTERRUPTS", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        self._gmail_client_id = os.getenv("GMAIL_CLIENT_ID")
        self._gmail_client_secret = os.getenv("GMAIL_CLIENT_SECRET")

        self._assistant_module = _load_email_assistant()
        self._store = InMemoryStore()
        # Path for persistent checkpoints (initialized async in run())
        # Use environment variable, db_path (SQLite mode), or default to /tmp
        checkpoint_dir = os.getenv("LANGGRAPH_CHECKPOINT_DIR")
        if checkpoint_dir:
            self._checkpoint_db_path = Path(checkpoint_dir) / "langgraph_checkpoints.db"
        elif hasattr(self._db, "db_path"):
            self._checkpoint_db_path = self._db.db_path.parent / "langgraph_checkpoints.db"
        else:
            # PostgreSQL mode - use /tmp or a data directory
            self._checkpoint_db_path = Path("/tmp") / "langgraph_checkpoints.db"
        self._checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_conn: aiosqlite.Connection | None = None
        self._graph = None  # Compiled in run() after async checkpointer init

    async def run(self) -> None:
        """Continuously poll the worker_events table until stopped."""
        # Initialize async checkpointer for persistent state across restarts
        self._checkpoint_conn = await aiosqlite.connect(str(self._checkpoint_db_path))
        checkpointer = AsyncSqliteSaver(self._checkpoint_conn)
        self._graph = self._assistant_module.overall_workflow.compile(
            checkpointer=checkpointer,
            store=self._store,
        )

        # Suppress GraphInterrupt "Future exception was never retrieved" warnings
        # These are expected when LangGraph pauses for async callbacks
        loop = asyncio.get_running_loop()
        original_handler = loop.get_exception_handler()

        def _graph_interrupt_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
            exception = context.get("exception")
            if isinstance(exception, GraphInterrupt):
                return  # Silently ignore - this is expected behavior
            if original_handler:
                original_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(_graph_interrupt_handler)

        users = self._db.list_active_users()
        db_display = getattr(self._db, "db_path", None) or self._db.database_url or "PostgreSQL"
        logger.info("\n%s\n", worker_start_box(str(db_display), len(users)))
        await self._bootstrap_active_users()
        while not self._stop_event.is_set():
            events = self._db.claim_worker_events(limit=10)
            if not events:
                await asyncio.sleep(self._poll_interval)
                continue

            for event in events:
                await self._handle_event(event)

        logger.info("\n%s\n", worker_stop_box())

    async def stop(self) -> None:
        """Stop the worker loop."""
        self._stop_event.set()
        user_ids = list(self._user_tasks.keys())
        for user_id in user_ids:
            await self._stop_user_polling(user_id)
        if self._checkpoint_conn:
            await self._checkpoint_conn.close()

    async def _handle_event(self, event: WorkerEventRecord) -> None:
        """Route events to their handlers and mark completion."""
        handler_map = {
            "register": self._handle_register,
            "unregister": self._handle_unregister,
            "resume": self._handle_resume,
        }

        handler = handler_map.get(event.event_type)
        if not handler:
            logger.warning("Unknown worker event type %s", event.event_type)
            self._db.mark_worker_event_processed(event.id, error=f"Unknown event {event.event_type}")
            return

        try:
            await handler(event.payload)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Worker event %s failed", event.id)
            self._db.mark_worker_event_processed(event.id, error=str(exc))
        else:
            self._db.mark_worker_event_processed(event.id)

    async def _bootstrap_active_users(self) -> None:
        """Ensure polling tasks exist for every active user on startup."""
        users = self._db.list_active_users()
        if not users:
            logger.info("No active users to monitor yet")
            return

        logger.info("Starting Gmail polling for %s user(s)", len(users))
        for user in users:
            await self._start_user_polling(user)

    async def _start_user_polling(self, user: dict[str, Any]) -> None:
        """Start (or restart) the Gmail polling loop for a user."""
        user_id = user["user_id"]
        await self._stop_user_polling(user_id)

        stop_event = asyncio.Event()
        task = asyncio.create_task(self._poll_user_loop(user_id, stop_event))

        async with self._user_lock:
            self._user_stop_signals[user_id] = stop_event
            self._user_tasks[user_id] = task
            self._seen_message_ids.setdefault(user_id, set())

    async def _stop_user_polling(self, user_id: int) -> None:
        """Cancel any active polling loop for the user."""
        async with self._user_lock:
            stop_event = self._user_stop_signals.pop(user_id, None)
            task = self._user_tasks.pop(user_id, None)
            self._seen_message_ids.pop(user_id, None)

        if stop_event:
            stop_event.set()

        if task:
            with suppress(asyncio.CancelledError):
                await task

    async def _poll_user_loop(self, user_id: int, stop_event: asyncio.Event) -> None:
        """Continuously fetch new emails for a user and invoke the graph."""
        try:
            while not stop_event.is_set():
                logger.debug("Poll loop: fetching user %s from DB", user_id)
                user = self._db.get_user(user_id)
                logger.debug("Poll loop: got user, status=%s", user.get("status") if user else None)
                if not self._is_user_active(user):
                    logger.info("User %s missing or inactive; stopping inbox polling", user_id)
                    break

                logger.debug("Poll loop: starting Gmail fetch for user %s", user_id)
                emails = await self._fetch_emails_async(user)
                logger.debug("Poll loop: Gmail fetch returned %d emails", len(emails))
                await self._handle_email_batch(user, emails)
                await self._wait_for_next_poll(stop_event)
        except asyncio.CancelledError:
            # Registration handler logs the stop message
            raise

    @staticmethod
    def _is_user_active(user: dict[str, Any] | None) -> bool:
        """Return True when the user exists and is marked active."""
        return bool(user and user.get("status") == "active")

    async def _fetch_emails_async(self, user: dict[str, Any]) -> list[dict[str, Any]]:
        """Fetch recent emails in a thread pool, shielding the event loop."""
        try:
            return await asyncio.to_thread(self._fetch_recent_emails_for_user, user)
        except Exception:
            log_expected_error(logger, "Failed to fetch Gmail messages for user %s", user["user_id"])
            return []

    async def _handle_email_batch(self, user: dict[str, Any], emails: list[dict[str, Any]]) -> None:
        """Deduplicate and feed each Gmail message into LangGraph."""
        user_id = user["user_id"]
        processed = 0
        skipped = 0
        for email in emails:
            message_id = email.get("id")
            if not self._should_process_email(user_id, message_id, email):
                skipped += 1
                continue

            if message_id:
                self._seen_message_ids[user_id].add(message_id)
            thread_id = email.get("thread_id")
            try:
                self._db.record_processed_email(user_id, message_id, thread_id)
            except Exception:
                logger.exception(
                    "Failed to persist processed flag for user %s message %s",
                    user_id,
                    message_id,
                )
            try:
                await self._process_email(user, email)
            except Exception:
                logger.exception(
                    "LangGraph run failed for user %s message %s",
                    user_id,
                    message_id,
                )
            processed += 1

        # Only log poll summary when there's activity (avoid noisy empty polls)
        if len(emails) > 0 or processed > 0:
            logger.info(gmail_poll_summary(user_id, len(emails), processed, skipped))

    def _should_process_email(
        self,
        user_id: int,
        message_id: str | None,
        email: dict[str, Any],
    ) -> bool:
        """Return True if the message is new and requires processing."""
        if not message_id:
            return False
        if email.get("user_respond"):
            return False
        if self._db.has_processed_email(user_id, message_id):
            return False
        if message_id in self._seen_message_ids[user_id]:
            return False
        return True

    async def _wait_for_next_poll(self, stop_event: asyncio.Event) -> None:
        """Sleep until the next poll unless the stop event fires."""
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=self._inbox_poll_interval)
        except asyncio.TimeoutError:
            return

    def _fetch_recent_emails_for_user(self, user: dict[str, Any]) -> list[dict[str, Any]]:
        """Fetch unread Gmail messages for the user using their OAuth tokens."""
        user_id = user["user_id"]
        token_payload = self._build_gmail_token_payload(user)
        email_address = user["email_to_monitor"]

        # Get already-processed message IDs to skip PDF extraction for them
        processed_ids = self._db.get_processed_message_ids(user_id)
        processed_ids.update(self._seen_message_ids.get(user_id, set()))

        emails = list(
            fetch_group_emails(
                email_address,
                minutes_since=self._gmail_minutes_since,
                gmail_token=token_payload,
                skip_pdf_for_ids=processed_ids,
            )
        )
        return emails

    def _build_gmail_token_payload(self, user: dict[str, Any]) -> dict[str, Any] | None:
        """Map user OAuth fields into the shape expected by gmail_tools."""
        access_token = user.get("email_api_access_token")
        refresh_token = user.get("email_api_refresh_token")
        if not access_token:
            return None

        token_payload: dict[str, Any] = {
            "token": access_token,
            "refresh_token": refresh_token,
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": self._gmail_client_id,
            "client_secret": self._gmail_client_secret,
            "scopes": ["https://www.googleapis.com/auth/gmail.modify"],
        }
        return {k: v for k, v in token_payload.items() if v}

    async def _process_email(self, user: dict[str, Any], email: dict[str, Any]) -> None:
        """Kick off the LangGraph flow for a single Gmail message."""
        user_id = user["user_id"]
        thread_id = self._build_thread_id(user_id, email)

        # Clear any stale checkpoint for this thread to ensure fresh start
        # (Resumes go through _handle_resume, not here, so this won't affect them)
        config = {"configurable": {"thread_id": thread_id}}
        if self._graph and self._graph.checkpointer:
            try:
                existing = await self._graph.checkpointer.aget_tuple(config)
                if existing:
                    await self._graph.checkpointer.adelete_thread(thread_id)
                    logger.debug("Cleared stale checkpoint for thread %s", thread_id)
            except Exception:
                logger.debug("No checkpoint to clear for thread %s", thread_id)

        email_input = self._build_email_input(email)
        email_input["user_id"] = user_id
        email_input["gmail_token"] = self._build_gmail_token_payload(user)

        metadata = {
            "gmail_message_id": email.get("id"),
            "gmail_thread_id": email.get("thread_id"),
            "subject": email.get("subject"),
            "from_email": email.get("from_email"),
            "pdf_attachments": email.get("pdf_attachments", []),
        }

        # Generate run_id for LangSmith tracing
        run_id = uuid.uuid4()

        logger.info(
            "\n%s\n",
            email_processing_box(
                user_id,
                email.get("id", ""),
                email.get("subject", ""),
                email.get("from_email", ""),
            ),
        )

        # Log LangSmith trace URL for easy terminal navigation
        if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
            trace_url = self._build_langsmith_url(run_id)
            if trace_url:
                logger.info("Trace: %s", trace_url)

        await self._consume_graph_stream(
            {"configurable": {"thread_id": thread_id, "user_id": user_id}, "run_id": run_id},
            {"email_input": email_input},
            user_id=user_id,
            email_metadata=metadata,
        )

    async def _consume_graph_stream(
        self,
        config: dict[str, Any],
        graph_input: dict[str, Any] | Command,
        user_id: int | None = None,
        email_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Stream LangGraph execution, pausing when interrupts occur."""
        thread_id = config.get("configurable", {}).get("thread_id")
        try:
            async for chunk in self._graph.astream(graph_input, config=config):
                interrupts = chunk.get("__interrupt__")
                if interrupts:
                    await self._handle_interrupts(
                        interrupts,
                        user_id=user_id,
                        thread_id=thread_id,
                        email_metadata=email_metadata or {},
                    )
                    return
            # Graph finished without interrupt - email processing complete
            if email_metadata and user_id:
                logger.info(
                    "\n%s\n",
                    email_complete_box(
                        user_id,
                        email_metadata.get("gmail_message_id", ""),
                        email_metadata.get("subject", ""),
                    ),
                )
        except Exception:
            logger.exception("LangGraph execution failed for user=%s thread=%s", user_id, thread_id)

    async def _handle_interrupts(
        self,
        interrupts: list[Any],
        user_id: int | None,
        thread_id: str | None,
        email_metadata: dict[str, Any],
    ) -> None:
        """Persist interrupt payloads as async jobs or auto-respond if configured."""
        if not thread_id:
            logger.warning("Interrupt received without a thread_id; skipping persistence")
            return

        for interrupt_obj in interrupts:
            requests = getattr(interrupt_obj, "value", []) or []
            if not requests:
                continue
            request = requests[0]

            if self._auto_accept_interrupts:
                action = request.get("action_request", {}).get("action", "")

                # Question tool: send synthetic response (accept is not valid)
                # Tool name defined in TOOL_CONFIG["question_tool_name"] (configuration.py)
                if action == "Question":
                    logger.warning(
                        "Auto-accept: Question tool received synthetic response for user=%s thread=%s. "
                        "Set WORKER_AUTO_ACCEPT_INTERRUPTS=false for real HITL.",
                        user_id,
                        thread_id,
                    )
                    resume_cmd = Command(resume=[{"type": "response", "args": "Proceed with your best judgement"}])

                # Triage notify: log the skipped notification
                # Prefix constructed in graph.py triage_interrupt_handler ("Email Assistant: {decision}")
                elif action.startswith("Email Assistant:"):
                    logger.info(
                        "Auto-accept: skipping notify interrupt for user=%s thread=%s action=%s",
                        user_id,
                        thread_id,
                        action,
                    )
                    resume_cmd = Command(resume=[{"type": "ignore"}])

                # All other tools (send_email, schedule_meeting): accept as before
                else:
                    logger.info(
                        "Auto-accepting interrupt for user=%s thread=%s action=%s",
                        user_id,
                        thread_id,
                        action,
                    )
                    resume_cmd = Command(resume=[{"type": "accept"}])

                try:
                    await self._consume_graph_stream(
                        {"configurable": {"thread_id": thread_id, "user_id": user_id}},
                        resume_cmd,
                        user_id=user_id,
                        email_metadata=email_metadata,
                    )
                finally:
                    pass
                continue

            job_id = self._register_interrupt_job(
                user_id=user_id,
                thread_id=thread_id,
                request=request,
                email_metadata=email_metadata,
            )
            action = request.get("action_request", {}).get("action", "interrupt")
            logger.info("\n%s\n", thread_pause_box(thread_id, job_id, action))
            logger.debug("Call POST /toolCompleted with jobId=%s once the async action finishes", job_id)
            break

    def _register_interrupt_job(
        self,
        user_id: int | None,
        thread_id: str,
        request: dict[str, Any],
        email_metadata: dict[str, Any],
    ) -> str:
        """Create a job row so /toolCompleted can resume this thread."""
        job_id = f"hitl-{uuid.uuid4().hex[:10]}"
        tool_name = request.get("action_request", {}).get("action", "interrupt")
        payload = {
            "request": request,
            "email": email_metadata,
        }
        self._db.record_job(
            job_id,
            user_id,
            tool_name=tool_name,
            payload=payload,
            run_handle=thread_id,
        )

        # Store PDF attachments for web UI display and download
        pdf_attachments = email_metadata.get("pdf_attachments", [])
        for i, att in enumerate(pdf_attachments):
            self._db.save_email_attachment(
                job_id=job_id,
                field_path=f"pdf_attachment_{i}",
                base64_data=att.get("base64_data"),
                filename=att.get("filename", f"attachment_{i}.pdf"),
                content_type="application/pdf",
                summary=att.get("content", "")[:500] if att.get("content") else None,
            )

        return job_id

    @staticmethod
    def _build_langsmith_url(run_id: uuid.UUID) -> str | None:
        """Construct LangSmith trace URL if org and project IDs are configured."""
        org_id = os.getenv("LANGSMITH_ORG_ID")
        project_id = os.getenv("LANGSMITH_PROJECT_ID")
        if org_id and project_id:
            return f"https://smith.langchain.com/o/{org_id}/projects/p/{project_id}?peek={run_id}"
        # Fallback: just log run_id if IDs not configured
        return None

    def _build_thread_id(self, user_id: int, email: dict[str, Any]) -> str:
        """Construct a deterministic LangGraph thread id per Gmail conversation."""
        gmail_thread_id = email.get("thread_id") or email.get("id")
        return f"user-{user_id}-{gmail_thread_id}"

    @staticmethod
    def _build_email_input(email: dict[str, Any]) -> dict[str, Any]:
        """Translate Gmail metadata into the shape the graph expects."""
        required_keys = ["from_email", "to_email", "subject", "page_content", "id"]
        missing = [key for key in required_keys if key not in email]
        if missing:
            raise ValueError(f"Gmail payload missing keys: {missing}")

        pdf_attachments = email.get("pdf_attachments", [])

        return {
            "from": email["from_email"],
            "to": email["to_email"],
            "subject": email["subject"],
            "body": email["page_content"],
            "id": email["id"],
            "pdf_attachments": pdf_attachments,
        }

    async def _handle_register(self, payload: dict[str, Any]) -> None:
        user_id = payload["user_id"]
        user = self._db.get_user(user_id)
        if not user:
            logger.warning("Received register event for unknown user %s", user_id)
            return

        if user.get("status") == "inactive":
            logger.info("User %s is inactive; skipping polling start", user_id)
            return

        await self._start_user_polling(user)
        logger.info(
            "User %s (%s) registered – Gmail polling active",
            user_id,
            user["email_to_monitor"],
        )

    async def _handle_unregister(self, payload: dict[str, Any]) -> None:
        user_id = payload["user_id"]
        logger.info("User %s unregistered – stopping Gmail polling", user_id)
        await self._stop_user_polling(user_id)

    async def _handle_resume(self, payload: dict[str, Any]) -> None:  # noqa: C901
        job_id = str(payload["job_id"])
        job = self._db.get_job(job_id)
        if not job:
            logger.warning("Resume requested for unknown job %s", job_id)
            return

        run_handle = job.get("run_handle")
        if not run_handle:
            logger.warning("Job %s has no run handle stored", job_id)
            return

        callback_payload = self._db.get_job_callback_payload(job_id)
        if callback_payload is None:
            logger.warning("Job %s has no callback payload recorded yet", job_id)
            return

        raw_payload = job.get("payload")
        payload_data: dict[str, Any] = {}
        if raw_payload:
            with suppress(json.JSONDecodeError):
                payload_data = json.loads(raw_payload)

        is_hitl_job = isinstance(payload_data, dict) and "request" in payload_data
        if is_hitl_job:
            if not isinstance(callback_payload, dict):
                raise ValueError(f"Job {job_id} callback payload must be an object")
            if "type" not in callback_payload:
                raise ValueError(f"Job {job_id} callback payload missing 'type'; expected HITL resume data")

        # For async jobs: payload already has summaries (base64 extracted at /toolCompleted)
        # Pass job_id so tools can fetch full attachments from DB when needed for forwarding
        if isinstance(callback_payload, dict) and not is_hitl_job:
            callback_payload = {
                "_llm_summary": callback_payload,  # Already has summaries, no base64
                "_job_id": job_id,  # For attachment lookup by tools
            }

        # Extract email metadata from job payload for completion logging
        email_metadata = payload_data.get("email") if isinstance(payload_data, dict) else None

        # Generate run_id for LangSmith tracing (resumed runs get new trace)
        run_id = uuid.uuid4()
        if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
            trace_url = self._build_langsmith_url(run_id)
            if trace_url:
                logger.info("Trace (resume): %s", trace_url)

        # Inject Gmail token into environment for send_email_tool
        # (The tool can't access LangGraph state, so we pass via env var)
        user_id = job.get("user_id")
        user = self._db.get_user(user_id) if user_id else None
        token_payload = self._build_gmail_token_payload(user) if user else None
        old_gmail_token = os.environ.get("GMAIL_TOKEN")
        if token_payload:
            os.environ["GMAIL_TOKEN"] = json.dumps(token_payload)

        command = Command(resume=[callback_payload])
        try:
            await self._consume_graph_stream(
                {"configurable": {"thread_id": run_handle, "user_id": user_id}, "run_id": run_id},
                command,
                user_id=user_id,
                email_metadata=email_metadata,
            )
        finally:
            # Restore previous env var state
            if token_payload:
                if old_gmail_token is not None:
                    os.environ["GMAIL_TOKEN"] = old_gmail_token
                else:
                    os.environ.pop("GMAIL_TOKEN", None)


async def _run_forever(worker: LangGraphWorker) -> None:
    """Helper to run the worker until cancelled (CLI entrypoint)."""
    loop = asyncio.get_running_loop()
    stop_future = loop.create_future()

    def _signal_handler() -> None:
        logger.info("Received shutdown signal")
        if not stop_future.done():
            stop_future.set_result(True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    worker_task = asyncio.create_task(worker.run())
    await stop_future
    await worker.stop()
    await worker_task


def build_worker_from_env() -> LangGraphWorker:
    """Helper so server + CLI share the same construction logic."""
    database_url = os.getenv("DATABASE_URL")
    return LangGraphWorker(database_url=database_url)


if __name__ == "__main__":  # pragma: no cover - manual execution path
    asyncio.run(_run_forever(build_worker_from_env()))
