"""Tool management module for email assistant."""

import logging

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def get_tools(
    tool_names: list[str] | None = None,
    include_gmail: bool = False,
    include_rag: bool = False,
) -> list[BaseTool]:
    """Get specified tools or all tools if tool_names is None.

    Args:
        tool_names: Optional list of tool names to include. If None, returns all tools.
        include_gmail: Whether to include Gmail tools. Defaults to False.
        include_rag: Whether to include RAG tools. Defaults to False.

    Returns:
        List of tool objects
    """
    # Import default tools
    from .default.calendar_tools import (
        check_calendar_availability,
        schedule_meeting,
    )
    from .default.email_tools import Done, Question, write_email

    # Base tools dictionary
    all_tools = {
        "write_email": write_email,
        "Done": Done,
        "Question": Question,
        "schedule_meeting": schedule_meeting,
        "check_calendar_availability": check_calendar_availability,
    }

    # Add Gmail tools if requested
    if include_gmail:
        try:
            from .gmail.gmail_tools import (
                check_calendar_tool,
                fetch_emails_tool,
                schedule_meeting_tool,
                send_email_tool,
            )

            all_tools.update(
                {
                    "fetch_emails_tool": fetch_emails_tool,
                    "send_email_tool": send_email_tool,
                    "check_calendar_tool": check_calendar_tool,
                    "schedule_meeting_tool": schedule_meeting_tool,
                }
            )
        except ImportError:
            # If Gmail tools aren't available, continue without them
            pass
    # Add RAG tools if requested
    if include_rag:
        try:
            from .rag.rag_tools import search_guidance_tool

            all_tools.update(
                {
                    "search_guidance_tool": search_guidance_tool,
                }
            )
        except ImportError as e:
            # If RAG tools aren't available, continue without them
            logger.warning(f"Failed to import RAG tools: {e}")

    if tool_names is None:
        return list(all_tools.values())

    return [all_tools[name] for name in tool_names if name in all_tools]


def get_tools_by_name(tools: list[BaseTool] | None = None) -> dict[str, BaseTool]:
    """Get a dictionary of tools mapped by name."""
    if tools is None:
        tools = get_tools()

    return {tool.name: tool for tool in tools}
