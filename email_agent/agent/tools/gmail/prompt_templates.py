"""Tool prompt templates for Gmail integration."""

from __future__ import annotations

from textwrap import dedent


def get_tools_prompt(question_tool_name: str, send_tool_name: str) -> str:
    """Return the tools prompt for the email assistant."""
    return dedent(
        f"""
        1. {question_tool_name}(content) - Ask the user for missing approvals, decisions, or information before drafting a reply
        2. search_guidance_tool(query, max_results) - Search the operational knowledge base for relevant guidance (call at most once per workflow)
        3. write_email(to, subject, content) - Draft emails to specified recipients
        4. {send_tool_name}(email_id, response_text, email_address, additional_recipients) - Send a reply to an email thread only after all approvals and information are confirmed
        5. Done - E-mail has been sent

        When escalating with the {question_tool_name} tool:
        - Start with one short sentence that names the decision or information gap
        - If there are multiple options, follow with "Do you want to:" and up to three hyphen bullets labelled (A), (B), (C) describing each option in plain language along with the immediate requirement for that option
        - If there is only one outstanding action, skip the "Do you want to:" line and use at most three dot-point bullets, each a single short sentence phrased as a polite question or "please" request focused on one ask
        """
    ).strip()


# Backwards compatibility exports (legacy callers expect these names)
GMAIL_TOOLS_PROMPT = get_tools_prompt("Question", "send_email_tool")
COMBINED_TOOLS_PROMPT = GMAIL_TOOLS_PROMPT
