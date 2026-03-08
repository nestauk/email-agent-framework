"""CLI tool for reviewing and responding to pending agent interrupts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from email_agent.agent_api.storage import AgentDatabase

console = Console()

# Config file location for storing user setup
CONFIG_DIR = Path.home() / ".config" / "email-agent"
CONFIG_FILE = CONFIG_DIR / "config.json"

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def get_database() -> AgentDatabase:
    """Get database connection from environment."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        console.print("[red]ERROR: DATABASE_URL environment variable not set[/red]")
        sys.exit(1)
    return AgentDatabase(database_url=database_url)


def get_api_url() -> str:
    """Get the agent API URL from environment."""
    return os.getenv("AGENT_API_URL", "http://localhost:8000")


def get_api_key() -> str | None:
    """Get the agent API key from environment."""
    return os.getenv("AGENT_API_KEY")


def load_config() -> dict[str, Any]:
    """Load config from file."""
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {}


def save_config(config: dict[str, Any]) -> None:
    """Save config to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def setup_gmail() -> None:
    """Set up Gmail OAuth and register user with the agent."""
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    console.print(Panel("[bold]Gmail Setup[/bold]\nThis will connect your Gmail account to the email agent."))

    # Check for required environment variables
    client_id = os.getenv("GMAIL_CLIENT_ID")
    client_secret = os.getenv("GMAIL_CLIENT_SECRET")

    if not client_id or not client_secret:
        console.print("[red]ERROR: GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET must be set[/red]")
        console.print("\nTo get these credentials:")
        console.print("  1. Go to https://console.cloud.google.com/")
        console.print("  2. Create a project and enable the Gmail API")
        console.print("  3. Create OAuth credentials (Desktop app)")
        console.print("  4. Add GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET to your .env file")
        sys.exit(1)

    # Check API is reachable
    api_url = get_api_url()
    try:
        health = httpx.get(f"{api_url}/health", timeout=5.0)
        health.raise_for_status()
    except httpx.RequestError:
        console.print(f"[red]ERROR: Cannot reach API at {api_url}[/red]")
        console.print("Make sure the API server is running: uv run uvicorn email_agent.agent_api.server:app")
        sys.exit(1)

    # Create OAuth flow
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }

    console.print("\n[bold]Step 1:[/bold] Opening browser for Gmail authorization...")
    console.print("[dim]Please sign in with the Gmail account you want to monitor.[/dim]\n")

    try:
        flow = InstalledAppFlow.from_client_config(client_config, GMAIL_SCOPES)
        credentials = flow.run_local_server(port=8080, prompt="consent")
    except Exception as e:
        console.print(f"[red]OAuth flow failed: {e}[/red]")
        sys.exit(1)

    console.print("[green]Authorization successful![/green]\n")

    # Get user's email address from Gmail API
    console.print("[bold]Step 2:[/bold] Fetching your Gmail profile...")
    try:
        service = build("gmail", "v1", credentials=credentials)
        profile = service.users().getProfile(userId="me").execute()
        email_address = profile["emailAddress"]
        console.print(f"  Email: [cyan]{email_address}[/cyan]")
    except Exception as e:
        console.print(f"[red]Failed to get Gmail profile: {e}[/red]")
        sys.exit(1)

    # Prompt for display name
    display_name = Prompt.ask("\n[bold]Step 3:[/bold] Enter your display name", default=email_address.split("@")[0])

    # Register with API
    console.print("\n[bold]Step 4:[/bold] Registering with the email agent...")

    api_key = get_api_key()
    if not api_key:
        console.print("[yellow]Warning: AGENT_API_KEY not set, registration may fail[/yellow]")

    # Generate a user ID based on email hash (or use existing from config)
    config = load_config()
    user_id = config.get("user_id") or abs(hash(email_address)) % 1_000_000

    payload = {
        "userId": user_id,
        "emailToMonitor": email_address,
        "emailAPIProvider": "google",
        "emailAPIAccessToken": credentials.token,
        "emailAPIAccessTokenExpiresAt": int(credentials.expiry.timestamp()) if credentials.expiry else 0,
        "emailAPIRefreshToken": credentials.refresh_token,
        "emailAPIRefreshTokenExpiresIn": 3600,
        "displayName": display_name,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        response = httpx.post(
            f"{api_url}/registerUser",
            json=payload,
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Registration failed: {e.response.status_code} - {e.response.text}[/red]")
        sys.exit(1)
    except httpx.RequestError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        sys.exit(1)

    # Save config
    config["user_id"] = user_id
    config["email"] = email_address
    config["display_name"] = display_name
    save_config(config)

    console.print("\n[bold green]Setup complete![/bold green]")
    console.print(
        Panel(
            f"[bold]Email:[/bold] {email_address}\n"
            f"[bold]User ID:[/bold] {user_id}\n"
            f"[bold]Config saved to:[/bold] {CONFIG_FILE}",
            title="Registration Successful",
            border_style="green",
        )
    )

    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. The worker will now start monitoring your inbox")
    console.print("  2. Run [cyan]email-agent-review watch[/cyan] to review emails as they come in")
    console.print("  3. Or run [cyan]email-agent-review list[/cyan] to see pending items")


def status_check() -> None:
    """Check the status of the email agent setup."""
    config = load_config()

    console.print(Panel("[bold]Email Agent Status[/bold]"))

    # Check config
    if config:
        console.print(f"[green]✓[/green] Config file: {CONFIG_FILE}")
        console.print(f"  Email: [cyan]{config.get('email', 'N/A')}[/cyan]")
        console.print(f"  User ID: {config.get('user_id', 'N/A')}")
    else:
        console.print(f"[yellow]○[/yellow] No config file found at {CONFIG_FILE}")
        console.print("  Run [cyan]email-agent-review setup[/cyan] to get started")

    # Check environment
    console.print("")
    env_vars = [
        ("DATABASE_URL", bool(os.getenv("DATABASE_URL"))),
        ("AGENT_API_KEY", bool(os.getenv("AGENT_API_KEY"))),
        ("GMAIL_CLIENT_ID", bool(os.getenv("GMAIL_CLIENT_ID"))),
        ("GMAIL_CLIENT_SECRET", bool(os.getenv("GMAIL_CLIENT_SECRET"))),
    ]

    for var, is_set in env_vars:
        status = "[green]✓[/green]" if is_set else "[red]✗[/red]"
        console.print(f"{status} {var}")

    # Check API health
    console.print("")
    api_url = get_api_url()
    try:
        response = httpx.get(f"{api_url}/health", timeout=5.0)
        response.raise_for_status()
        console.print(f"[green]✓[/green] API server: {api_url}")
    except httpx.RequestError:
        console.print(f"[red]✗[/red] API server: {api_url} (not reachable)")

    # Check database
    try:
        db = get_database()
        users = db.list_active_users() if hasattr(db, "list_active_users") else []
        console.print("[green]✓[/green] Database connected")
        if users:
            console.print(f"  Active users: {len(users)}")
    except Exception:
        console.print("[red]✗[/red] Database: connection failed")


def list_pending_jobs(db: AgentDatabase) -> None:
    """List all pending jobs awaiting human review."""
    jobs = db.list_pending_jobs()

    if not jobs:
        console.print("[yellow]No pending jobs found.[/yellow]")
        return

    table = Table(title="Pending Jobs Awaiting Review")
    table.add_column("Job ID", style="cyan", no_wrap=True)
    table.add_column("Tool", style="green")
    table.add_column("User ID", style="dim")
    table.add_column("Created", style="dim")

    for job in jobs:
        table.add_row(
            job["job_id"],
            job["tool_name"] or "unknown",
            str(job["user_id"] or "-"),
            job["created_at"][:19] if job["created_at"] else "-",
        )

    console.print(table)
    console.print("\n[dim]Use 'review <job_id>' to review a specific job[/dim]")


def show_job_details(db: AgentDatabase, job_id: str) -> dict[str, Any] | None:
    """Display details of a specific job."""
    job = db.get_job(job_id)

    if not job:
        console.print(f"[red]Job '{job_id}' not found[/red]")
        return None

    if job["status"] != "pending":
        console.print(f"[yellow]Job '{job_id}' is not pending (status: {job['status']})[/yellow]")
        return None

    # Parse payload
    payload = json.loads(job["payload"]) if job.get("payload") else {}
    request = payload.get("request", {})

    # Show job info
    console.print(Panel(f"[bold]Job ID:[/bold] {job_id}\n[bold]Tool:[/bold] {job['tool_name']}", title="Job Details"))

    # Show the proposed action
    action_request = request.get("action_request", {})
    action = action_request.get("action", "unknown") if action_request else "unknown"
    args = action_request.get("args", {}) if action_request else {}

    # Show description (contains original email context)
    # Clean up the description by removing the raw tool call JSON and Question sections
    description = request.get("description", "")
    if description:
        import re

        # Remove the "Tool Call: ..." section that contains raw JSON
        clean_desc = re.split(r"\n-+\n\s*Tool Call:", description)[0].strip()
        # Remove "# Question for User" section (shown separately below)
        clean_desc = re.split(r"\n-+\n\s*#\s*Question for User", clean_desc)[0].strip()
        if clean_desc:
            console.print(Panel(Markdown(clean_desc), title="Original Email"))

    if action_request:
        console.print(f"\n[bold]Proposed Action:[/bold] {action}")

        if action == "send_email_tool":
            # send_email_tool uses: email_address, response_text, email_id, additional_recipients
            to_addr = args.get("email_address") or args.get("to", "N/A")
            body = args.get("response_text") or args.get("body", "")
            additional = args.get("additional_recipients", [])
            recipients = f"{to_addr}" + (f" + {additional}" if additional else "")
            console.print(
                Panel(
                    f"[bold]To:[/bold] {recipients}\n"
                    f"[bold]Reply to message:[/bold] {args.get('email_id', 'N/A')}\n\n"
                    f"{body}",
                    title="Draft Response",
                    border_style="green",
                )
            )
        elif action == "Question":
            # Question tool uses "content" key, not "question"
            question_text = args.get("content") or args.get("question", "No question provided")
            console.print(Panel(question_text, title="Question for User", border_style="blue"))
        else:
            console.print(Panel(json.dumps(args, indent=2, ensure_ascii=False), title=f"Args for {action}"))

    # Show allowed actions
    config = request.get("config", {})
    allowed = []
    if config.get("allow_accept"):
        allowed.append("[green]accept[/green]")
    if config.get("allow_edit"):
        allowed.append("[yellow]edit[/yellow]")
    if config.get("allow_respond"):
        allowed.append("[blue]respond[/blue]")
    if config.get("allow_ignore"):
        allowed.append("[red]ignore[/red]")

    console.print(f"\n[bold]Allowed actions:[/bold] {', '.join(allowed)}")

    return job


def submit_response(
    job_id: str, response_type: str, args: dict[str, Any] | None = None, tool_name: str = "unknown"
) -> bool:
    """Submit a response to the agent API."""
    api_url = get_api_url()
    api_key = get_api_key()

    payload: dict[str, Any] = {
        "jobId": job_id,
        "tool": tool_name,
        "toolOutputData": {
            "type": response_type,
        },
    }

    if args:
        payload["toolOutputData"]["args"] = args

    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
        headers["authorization"] = api_key

    try:
        response = httpx.post(
            f"{api_url}/toolCompleted",
            json=payload,
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        console.print("[green]Response submitted successfully![/green]")
        return True
    except httpx.HTTPStatusError as e:
        console.print(f"[red]API error: {e.response.status_code} - {e.response.text}[/red]")
        return False
    except httpx.RequestError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        return False


def interactive_review(db: AgentDatabase, job_id: str) -> None:
    """Interactively review and respond to a job."""
    job = show_job_details(db, job_id)
    if not job:
        return

    tool_name = job.get("tool_name", "unknown")
    payload = json.loads(job["payload"]) if job.get("payload") else {}
    request = payload.get("request", {})
    config = request.get("config", {})
    action_request = request.get("action_request", {})

    console.print("\n[bold]Choose an action:[/bold]")

    choices = []
    num_to_choice: dict[str, str] = {}
    num = 1
    if config.get("allow_accept"):
        choices.append("accept")
        num_to_choice[str(num)] = "accept"
        console.print(f"  [green]{num}. accept[/green] - Approve and execute as-is")
        num += 1
    if config.get("allow_edit"):
        choices.append("edit")
        num_to_choice[str(num)] = "edit"
        console.print(f"  [yellow]{num}. edit[/yellow] - Modify before executing")
        num += 1
    if config.get("allow_respond"):
        choices.append("respond")
        num_to_choice[str(num)] = "respond"
        console.print(f"  [blue]{num}. respond[/blue] - Provide a text response")
        num += 1
    if config.get("allow_ignore"):
        choices.append("ignore")
        num_to_choice[str(num)] = "ignore"
        console.print(f"  [red]{num}. ignore[/red] - Skip this action")
        num += 1
    choices.append("cancel")
    num_to_choice[str(num)] = "cancel"
    console.print(f"  [dim]{num}. cancel[/dim] - Exit without responding")

    # Simple number input
    raw_choice = Prompt.ask(f"\nYour choice [1-{num}]", default="1")
    choice = num_to_choice.get(raw_choice, raw_choice)
    if choice not in choices:
        console.print(f"[red]Invalid choice: {raw_choice}[/red]")
        return

    if choice == "cancel":
        console.print("[dim]Cancelled[/dim]")
        return

    if choice == "accept":
        if Confirm.ask("Confirm: Execute the action as proposed?"):
            submit_response(job_id, "accept", tool_name=tool_name)

    elif choice == "ignore":
        if Confirm.ask("Confirm: Ignore this action?"):
            submit_response(job_id, "ignore", tool_name=tool_name)

    elif choice == "respond":
        response_text = Prompt.ask("Enter your response")
        if response_text and Confirm.ask("Submit this response?"):
            submit_response(job_id, "response", {"response": response_text}, tool_name=tool_name)

    elif choice == "edit":
        action = action_request.get("action", "")
        args = action_request.get("args", {})

        if action == "send_email_tool":
            # send_email_tool uses: email_address, response_text, email_id, additional_recipients
            current_to = args.get("email_address") or args.get("to", "")
            current_body = args.get("response_text") or args.get("body", "")

            console.print("\n[bold]Edit the email (press Enter to keep current value):[/bold]")
            console.print(f"[dim]Current response:[/dim]\n{current_body}\n")
            new_body = Prompt.ask("New response (or Enter to keep)", default="")

            edited_args = {
                "email_id": args.get("email_id"),
                "email_address": current_to,
                "response_text": new_body if new_body else current_body,
                "additional_recipients": args.get("additional_recipients", []),
            }

            console.print(
                Panel(
                    f"[bold]To:[/bold] {edited_args['email_address']}\n"
                    f"[bold]Reply to:[/bold] {edited_args['email_id']}\n\n"
                    f"{edited_args['response_text']}",
                    title="Edited Email",
                    border_style="yellow",
                )
            )

            if Confirm.ask("Submit this edited email?"):
                submit_response(job_id, "edit", {"args": edited_args}, tool_name=tool_name)
        else:
            console.print("[yellow]Editing not fully supported for this tool type yet.[/yellow]")
            console.print(f"Current args: {json.dumps(args, indent=2)}")
            new_args_str = Prompt.ask("Enter new args as JSON (or press Enter to cancel)")
            if new_args_str:
                try:
                    new_args = json.loads(new_args_str)
                    if Confirm.ask("Submit these edited args?"):
                        submit_response(job_id, "edit", {"args": new_args}, tool_name=tool_name)
                except json.JSONDecodeError:
                    console.print("[red]Invalid JSON[/red]")


def watch_mode(db: AgentDatabase, interval: int = 5) -> None:
    """Continuously poll for new pending jobs."""
    import time

    console.print(f"[bold]Watching for pending jobs (poll every {interval}s)...[/bold]")
    console.print("[dim]Press Ctrl+C to exit[/dim]\n")

    seen_jobs: set[str] = set()

    try:
        while True:
            jobs = db.list_pending_jobs()
            new_jobs = [j for j in jobs if j["job_id"] not in seen_jobs]

            for job in new_jobs:
                seen_jobs.add(job["job_id"])
                console.print(f"\n[bold green]New job:[/bold green] {job['job_id']} ({job['tool_name']})")
                if Confirm.ask("Review now?", default=True):
                    interactive_review(db, job["job_id"])

            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching[/dim]")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Review and respond to pending email agent interrupts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s setup                   Connect your Gmail account
  %(prog)s status                  Check setup status
  %(prog)s list                    List all pending jobs
  %(prog)s review hitl-abc123      Review a specific job
  %(prog)s watch                   Watch for new jobs interactively
  %(prog)s accept hitl-abc123      Quick accept a job
  %(prog)s ignore hitl-abc123      Quick ignore a job
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Setup command
    subparsers.add_parser("setup", help="Connect your Gmail account to the email agent")

    # Status command
    subparsers.add_parser("status", help="Check the status of the email agent setup")

    # List command
    subparsers.add_parser("list", help="List all pending jobs")

    # Review command
    review_parser = subparsers.add_parser("review", help="Review a specific job interactively")
    review_parser.add_argument("job_id", help="The job ID to review")

    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch for new jobs and review interactively")
    watch_parser.add_argument("--interval", type=int, default=5, help="Poll interval in seconds")

    # Quick accept command
    accept_parser = subparsers.add_parser("accept", help="Accept a job without review")
    accept_parser.add_argument("job_id", help="The job ID to accept")

    # Quick ignore command
    ignore_parser = subparsers.add_parser("ignore", help="Ignore a job without review")
    ignore_parser.add_argument("job_id", help="The job ID to ignore")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Commands that don't need database
    if args.command == "setup":
        setup_gmail()
        return
    elif args.command == "status":
        status_check()
        return

    db = get_database()

    if args.command == "list":
        list_pending_jobs(db)
    elif args.command == "review":
        interactive_review(db, args.job_id)
    elif args.command == "watch":
        watch_mode(db, args.interval)
    elif args.command == "accept":
        job = db.get_job(args.job_id)
        if not job:
            console.print(f"[red]Job '{args.job_id}' not found[/red]")
            return
        submit_response(args.job_id, "accept", tool_name=job.get("tool_name", "unknown"))
    elif args.command == "ignore":
        job = db.get_job(args.job_id)
        if not job:
            console.print(f"[red]Job '{args.job_id}' not found[/red]")
            return
        submit_response(args.job_id, "ignore", tool_name=job.get("tool_name", "unknown"))


if __name__ == "__main__":
    main()
