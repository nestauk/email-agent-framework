"""Email assistant agent with human-in-the-loop and Gmail integration."""

import logging
import os
import sys
from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt

from .configuration import TOOL_CONFIG
from .prompts import (
    MEMORY_UPDATE_INSTRUCTIONS,
    MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT,
    agent_system_prompt_hitl_memory,
    default_background,
    default_response_preferences,
    default_triage_instructions,
    triage_system_prompt,
    triage_user_prompt,
)
from .schemas import (
    RouterSchema,
    State,
    StateInput,
    UserPreferences,
)
from .tools import get_tools, get_tools_by_name
from .tools.gmail.gmail_tools import mark_as_read
from .tools.gmail.prompt_templates import get_tools_prompt
from .utils import (
    format_for_display,
    format_gmail_markdown,
    get_llm,
    parse_gmail,
    summarise_pdf,
)

logger = logging.getLogger(__name__)

load_dotenv(".env")

_COLOR_ENABLED = sys.stdout.isatty()
_COLOR_MAP = {
    "reset": "\033[0m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
}


def _colorize(text: str, color: str) -> str:
    if not _COLOR_ENABLED or color not in _COLOR_MAP:
        return text
    return f"{_COLOR_MAP[color]}{text}{_COLOR_MAP['reset']}"


QUESTION_TOOL_NAME = TOOL_CONFIG["question_tool_name"]
SEND_EMAIL_TOOL_NAME = TOOL_CONFIG["send_email_tool_name"]

# RAG can be disabled via env var to avoid Qdrant lock conflicts when running multiple processes
INCLUDE_RAG = os.getenv("INCLUDE_RAG", "true").lower() in ("true", "1", "yes")

logger.info("Email assistant initialized (RAG=%s)", INCLUDE_RAG)


def _format_system_prompt(prompt: str) -> str:
    """Swap tool names inside the prompt so instructions match the active mode."""
    updated = prompt
    if QUESTION_TOOL_NAME != "Question":
        updated = updated.replace("Question tool", f"{QUESTION_TOOL_NAME} tool")
    if SEND_EMAIL_TOOL_NAME != "send_email_tool":
        updated = updated.replace("send_email_tool", SEND_EMAIL_TOOL_NAME)
    return updated


# Get tools with Gmail tools
tools = get_tools(
    TOOL_CONFIG["tool_names"],
    include_gmail=True,
    include_rag=INCLUDE_RAG,
)
tools_by_name = get_tools_by_name(tools)


def get_memory(store, namespace, default_content=None):
    """Get memory from the store or initialize with default if it doesn't exist.

    Args:
        store: LangGraph BaseStore instance to search for existing memory
        namespace: Tuple defining the memory namespace, e.g. ("email_assistant", "triage_preferences")
        default_content: Default content to use if memory doesn't exist

    Returns:
        str: The content of the memory profile, either from existing memory or the default
    """
    # Search for existing memory with namespace and key
    user_preferences = store.get(namespace, "user_preferences")

    # If memory exists, return its content (the value)
    if user_preferences:
        return user_preferences.value

    # If memory doesn't exist, add it to the store and return the default content
    else:
        # Namespace, key, value
        store.put(namespace, "user_preferences", default_content)
        user_preferences = default_content

    # Return the default content
    return user_preferences


def update_memory(store, namespace, messages):
    """Update memory profile in the store.

    Args:
        store: LangGraph BaseStore instance to update memory
        namespace: Tuple defining the memory namespace, e.g. ("email_assistant", "triage_preferences")
        messages: List of messages to update the memory with
    """
    # Get the existing memory
    user_preferences = store.get(namespace, "user_preferences")
    # Update the memory
    llm = get_llm("gpt-5-nano").with_structured_output(UserPreferences)
    result = llm.invoke(
        [
            {
                "role": "system",
                "content": MEMORY_UPDATE_INSTRUCTIONS.format(
                    current_profile=user_preferences.value, namespace=namespace
                ),
            },
        ]
        + messages
    )
    # Save the updated memory to the store
    store.put(namespace, "user_preferences", result.user_preferences)


# Nodes
def triage_router(
    state: State, store: BaseStore
) -> Command[Literal["triage_interrupt_handler", "llm_call", "__end__"]]:
    """Analyze email content to decide if we should respond, notify, or ignore.

    The triage step prevents the assistant from wasting time on:
    - Marketing emails and spam
    - Company-wide announcements
    - Messages meant for other teams
    """
    # Parse the email input
    author, to, subject, email_thread, email_id, pdf_attachments = parse_gmail(state["email_input"])

    # Build user prompt with PDF attachments if present
    pdf_context = ""
    summary_lines = []
    pdf_summaries: dict[str, dict] = {}  # Store summaries to persist in state
    if pdf_attachments:
        logger.info("PDF summarization starting: %d attachment(s)", len(pdf_attachments))
        for pdf in pdf_attachments:
            summary_obj = summarise_pdf(pdf["content"], pdf["filename"])
            summary_dict = summary_obj.model_dump()
            pdf_summaries[pdf["filename"]] = summary_dict  # Store for state
            logger.info("PDF summarization complete: %s", summary_obj.filename)
            # Add things to summary line
            if summary_obj.unreadable_flag:
                summary_lines.append(f"{summary_obj.filename}: UNREADABLE")
            else:
                summary_lines.append(
                    f"{summary_obj.filename} | Key: {', '.join(summary_obj.key_points) or 'None'} | "
                    f"Actions: {', '.join(summary_obj.actions) or 'None'} | "
                    f"Deadlines: {', '.join(summary_obj.deadlines) or 'None'}"
                )
    if summary_lines:
        pdf_context = "\n\nPDF Attachment Summaries:\n" + "\n".join(f"- {line}" for line in summary_lines)

    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread + pdf_context
    )

    # Search for existing triage_preferences memory
    triage_instructions = get_memory(store, ("email_assistant", "triage_preferences"), default_triage_instructions)

    # Format system prompt with background and triage instructions
    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=triage_instructions,
    )

    # Run the router LLM
    llm = get_llm("gpt-5-mini")
    llm_router = llm.with_structured_output(RouterSchema)
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Decision
    classification = result.classification

    # Process the classification decision
    if classification == "respond":
        logger.info("Classification: RESPOND - This email requires a response")
        # Next node (llm_call is the entry point of the response agent flow)
        goto = "llm_call"

        # Create email markdown with PDF context for LLM
        email_markdown_with_pdfs = format_gmail_markdown(subject, author, to, email_thread + pdf_context, email_id)

        # Update the state (with PDF context for LLM)
        update = {
            "classification_decision": result.classification,
            "pdf_summaries": pdf_summaries,
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email: {email_markdown_with_pdfs}",
                }
            ],
        }

    elif classification == "ignore":
        logger.info("Classification: IGNORE - This email can be safely ignored")

        # Next node
        goto = END
        # Update the state
        update = {
            "classification_decision": classification,
            "pdf_summaries": pdf_summaries,
        }

    elif classification == "notify":
        logger.info("Classification: NOTIFY - This email contains important information")

        # Next node
        goto = "triage_interrupt_handler"
        # Update the state
        update = {
            "classification_decision": classification,
            "pdf_summaries": pdf_summaries,
        }

    else:
        raise ValueError(f"Invalid classification: {classification}")

    return Command(goto=goto, update=update)


def triage_interrupt_handler(state: State, store: BaseStore) -> Command[Literal["llm_call", "__end__"]]:
    """Handle interrupts from the triage step."""
    # Parse the email input
    author, to, subject, email_thread, email_id, _ = parse_gmail(state["email_input"])

    # Build PDF context from summaries stored in state (computed in triage_router)
    pdf_context = ""
    pdf_summaries = state.get("pdf_summaries", {})
    if pdf_summaries:
        summary_lines = []
        for filename, summary in pdf_summaries.items():
            if summary.get("unreadable_flag"):
                summary_lines.append(f"{filename}: UNREADABLE")
            else:
                key_points = ", ".join(summary.get("key_points", [])) or "None"
                actions = ", ".join(summary.get("actions", [])) or "None"
                deadlines = ", ".join(summary.get("deadlines", [])) or "None"
                summary_lines.append(f"{filename} | Key: {key_points} | Actions: {actions} | Deadlines: {deadlines}")
        if summary_lines:
            pdf_context = "\n\nPDF Attachment Summaries:\n" + "\n".join(f"- {line}" for line in summary_lines)

    # Create email markdown for Agent Inbox (WITHOUT PDF content for display)
    email_markdown = format_gmail_markdown(subject, author, to, email_thread, email_id)

    # Create email markdown with PDF context for LLM
    email_markdown_with_pdfs = format_gmail_markdown(subject, author, to, email_thread + pdf_context, email_id)

    # Create messages (with PDF context for LLM)
    messages = [
        {
            "role": "user",
            "content": f"Email to notify user about: {email_markdown_with_pdfs}",
        }
    ]

    # Create interrupt for Agent Inbox
    request = {
        "action_request": {
            "action": f"Email Assistant: {state['classification_decision']}",
            "args": {},
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": False,
        },
        # Email to show in Agent Inbox
        "description": email_markdown,
    }

    # Send to Agent Inbox and wait for response
    response = interrupt([request])[0]

    # If user provides feedback, go to response agent and use feedback to respond to email
    if response["type"] == "response":
        # Add feedback to messages
        user_input = response["args"]
        messages.append(
            {
                "role": "user",
                "content": f"User wants to reply to the email. Use this feedback to respond: {user_input}",
            }
        )
        # Update memory with feedback
        update_memory(
            store,
            ("email_assistant", "triage_preferences"),
            [
                {
                    "role": "user",
                    "content": "The user decided to respond to the email, so update the triage preferences to capture this.",
                }
            ]
            + messages,
        )

        goto = "llm_call"  # Entry point of the response agent flow

    # If user ignores email, go to END
    elif response["type"] == "ignore":
        # Make note of the user's decision to ignore the email
        messages.append(
            {
                "role": "user",
                "content": "The user decided to ignore the email even though it was classified as notify. Update triage preferences to capture this.",
            }
        )
        # Update memory with feedback
        update_memory(store, ("email_assistant", "triage_preferences"), messages)
        goto = END

    # Catch all other responses
    else:
        raise ValueError(f"Invalid response: {response}")

    # Update the state
    update = {
        "messages": messages,
    }

    return Command(goto=goto, update=update)


def llm_call(state: State, store: BaseStore):
    """Call LLM to decide whether to call a tool or not."""
    # Search for existing response_preferences memory
    response_preferences = get_memory(store, ("email_assistant", "response_preferences"), default_response_preferences)

    tools_prompt = get_tools_prompt(
        QUESTION_TOOL_NAME,
        SEND_EMAIL_TOOL_NAME,
    )
    system_prompt = agent_system_prompt_hitl_memory.format(
        tools_prompt=tools_prompt,
        background=default_background,
        response_preferences=response_preferences,
    )
    system_prompt = _format_system_prompt(system_prompt)

    llm = get_llm("gpt-5-mini")
    llm_with_tools = llm.bind_tools(tools, tool_choice="required")

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {
                        "role": "system",
                        "content": system_prompt,
                    }
                ]
                + state["messages"]
            )
        ]
    }


def interrupt_handler(state: State, store: BaseStore) -> Command[Literal["llm_call", "__end__"]]:
    """Create an interrupt for human review of tool calls."""
    # Store messages
    result = []

    # Go to the LLM call node next
    goto = "llm_call"

    # Track tool call usage for lightweight rate limiting
    tool_call_counts = dict(state.get("tool_call_counts", {}))

    hitl_tools = TOOL_CONFIG["hitl_tools"]

    # Iterate over the tool calls in the last message
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    for tool_call in tool_calls:
        # If tool is not in our HITL list, execute it directly without interruption
        if tool_call["name"] not in hitl_tools:
            tool_name = tool_call["name"]
            tool = tools_by_name[tool_name]

            if tool_name == "search_guidance_tool":
                current_count = tool_call_counts.get(tool_name, 0)
                if current_count >= 1:
                    result.append(
                        {
                            "role": "tool",
                            "content": "search_guidance_tool limit reached; no further searches executed.",
                            "tool_call_id": tool_call["id"],
                        }
                    )
                    continue

            observation = tool.invoke(tool_call["args"])
            if tool_name == "search_guidance_tool":
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

            result.append(
                {
                    "role": "tool",
                    "content": observation,
                    "tool_call_id": tool_call["id"],
                }
            )
            continue

        # Get original email from email_input in state
        email_input = state["email_input"]
        author, to, subject, email_thread, email_id, pdf_attachments = parse_gmail(email_input)
        # For display in Agent Inbox, we don't include PDF content (keeps UI clean)
        original_email_markdown = format_gmail_markdown(subject, author, to, email_thread, email_id)

        # Format tool call for display and prepend the original email
        tool_display = format_for_display(tool_call)
        description = original_email_markdown + tool_display

        # Configure what actions are allowed in Agent Inbox
        if tool_call["name"] == SEND_EMAIL_TOOL_NAME:
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            }
        elif tool_call["name"] == "schedule_meeting_tool":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            }
        elif tool_call["name"] == QUESTION_TOOL_NAME:
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": False,
                "allow_accept": False,
            }
        else:
            raise ValueError(f"Invalid tool call: {tool_call['name']}")

        # Create the interrupt request
        request = {
            "action_request": {"action": tool_call["name"], "args": tool_call["args"]},
            "config": config,
            "description": description,
        }

        # Send to Agent Inbox and wait for response
        response = interrupt([request])[0]

        # Handle the responses
        if response["type"] == "accept":
            # Execute the tool with original args
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(
                {
                    "role": "tool",
                    "content": observation,
                    "tool_call_id": tool_call["id"],
                }
            )

        elif response["type"] == "edit":
            # Tool selection
            tool = tools_by_name[tool_call["name"]]
            initial_tool_call = tool_call["args"]

            # Get edited args from Agent Inbox
            edited_args = response["args"]["args"]

            # Update the AI message's tool call with edited content (reference to the message in the state)
            ai_message = state["messages"][-1]  # Get the most recent message from the state
            current_id = tool_call["id"]  # Store the ID of the tool call being edited

            # Create a new list of tool calls by filtering out the one being edited and adding the updated version
            ai_tool_calls = getattr(ai_message, "tool_calls", None) or []
            updated_tool_calls = [tc for tc in ai_tool_calls if tc["id"] != current_id] + [
                {
                    "type": "tool_call",
                    "name": tool_call["name"],
                    "args": edited_args,
                    "id": current_id,
                }
            ]

            # Create a new copy of the message with updated tool calls
            result.append(ai_message.model_copy(update={"tool_calls": updated_tool_calls}))

            # Save feedback in memory
            if tool_call["name"] == SEND_EMAIL_TOOL_NAME:
                # Execute the tool with edited args
                observation = tool.invoke(edited_args)

                # Add only the tool response message
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})

                # Update the memory
                update_memory(
                    store,
                    ("email_assistant", "response_preferences"),
                    [
                        {
                            "role": "user",
                            "content": f"User edited the email response. Here is the initial email generated by the assistant: {initial_tool_call}. Here is the edited email: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            elif tool_call["name"] == "schedule_meeting_tool":
                # Execute the tool with edited args
                observation = tool.invoke(edited_args)

                # Add only the tool response message
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})

                # Update the memory
                update_memory(
                    store,
                    ("email_assistant", "cal_preferences"),
                    [
                        {
                            "role": "user",
                            "content": f"User edited the calendar invitation. Here is the initial calendar invitation generated by the assistant: {initial_tool_call}. Here is the edited calendar invitation: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "ignore":
            if tool_call["name"] == SEND_EMAIL_TOOL_NAME:
                # Don't execute the tool
                result.append(
                    {
                        "role": "tool",
                        "content": "User ignored this email draft. Ignore this email and end the workflow.",
                        "tool_call_id": tool_call["id"],
                    }
                )
                # Go to END
                goto = END
                # Update the memory
                update_memory(
                    store,
                    ("email_assistant", "triage_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"The user ignored the email draft. That means they did not want to respond to the email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            elif tool_call["name"] == "schedule_meeting_tool":
                # Don't execute the tool
                result.append(
                    {
                        "role": "tool",
                        "content": "User ignored this calendar meeting draft. Ignore this email and end the workflow.",
                        "tool_call_id": tool_call["id"],
                    }
                )
                # Go to END
                goto = END
                # Update the memory
                update_memory(
                    store,
                    ("email_assistant", "triage_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"The user ignored the calendar meeting draft. That means they did not want to schedule a meeting for this email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            elif tool_call["name"] == QUESTION_TOOL_NAME:
                # Don't execute the tool
                result.append(
                    {
                        "role": "tool",
                        "content": "User ignored this question. Ignore this email and end the workflow.",
                        "tool_call_id": tool_call["id"],
                    }
                )
                # Go to END
                goto = END
                # Update the memory
                update_memory(
                    store,
                    ("email_assistant", "triage_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"The user ignored the {QUESTION_TOOL_NAME}. That means they did not want to answer the question or deal with this email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "response":
            # User provided feedback
            user_feedback = response["args"]
            if tool_call["name"] == SEND_EMAIL_TOOL_NAME:
                # Don't execute the tool, add feedback message
                result.append(
                    {
                        "role": "tool",
                        "content": f"User gave feedback, which can we incorporate into the email. Feedback: {user_feedback}",
                        "tool_call_id": tool_call["id"],
                    }
                )
                # Update the memory
                update_memory(
                    store,
                    ("email_assistant", "response_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"User gave feedback, which we can use to update the response preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            elif tool_call["name"] == "schedule_meeting_tool":
                # Don't execute the tool, add feedback message
                result.append(
                    {
                        "role": "tool",
                        "content": f"User gave feedback, which can we incorporate into the meeting request. Feedback: {user_feedback}",
                        "tool_call_id": tool_call["id"],
                    }
                )
                # Update the memory
                update_memory(
                    store,
                    ("email_assistant", "cal_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"User gave feedback, which we can use to update the calendar preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            elif tool_call["name"] == QUESTION_TOOL_NAME:
                # Don't execute the tool, add feedback message
                result.append(
                    {
                        "role": "tool",
                        "content": f"User answered the question, which can we can use for any follow up actions. Feedback: {user_feedback}",
                        "tool_call_id": tool_call["id"],
                    }
                )

            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

    # Update the state
    update = {
        "messages": result,
        "tool_call_counts": tool_call_counts,
    }

    return Command(goto=goto, update=update)


# Conditional edge function
def should_continue(state: State, store: BaseStore) -> Literal["interrupt_handler", "mark_as_read_node"]:
    """Route to tool handler, or end if Done tool called."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if tool_calls:
        for tool_call in tool_calls:
            if tool_call["name"] == "Done":
                return "mark_as_read_node"
            else:
                return "interrupt_handler"


def mark_as_read_node(state: State):
    """Mark email as read in Gmail."""
    email_input = state["email_input"]
    email_id = email_input.get("id")
    if email_id:
        # Get Gmail token from state (context vars don't persist across node boundaries)
        gmail_token = email_input.get("gmail_token")
        mark_as_read(email_id, gmail_token=gmail_token)


# Build flattened workflow
overall_workflow = StateGraph(State, input_schema=StateInput)

# Triage nodes
overall_workflow.add_node(triage_router)
overall_workflow.add_node(triage_interrupt_handler)

# Response agent nodes (previously in nested subgraph)
overall_workflow.add_node("llm_call", llm_call)
overall_workflow.add_node("interrupt_handler", interrupt_handler)
overall_workflow.add_node("mark_as_read_node", mark_as_read_node)

# Triage flow entry point
overall_workflow.add_edge(START, "triage_router")
# Note: triage_router uses Command(goto=...) to route to:
#   - "llm_call" (respond)
#   - "triage_interrupt_handler" (notify)
#   - END (ignore)
# triage_interrupt_handler uses Command(goto=...) to route to:
#   - "llm_call" (user wants to respond)
#   - END (user ignores)

# Response agent flow (previously the inner subgraph edges)
overall_workflow.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "interrupt_handler": "interrupt_handler",
        "mark_as_read_node": "mark_as_read_node",
    },
)
# interrupt_handler uses Command(goto=...) to route back to llm_call or END
overall_workflow.add_edge("mark_as_read_node", END)


email_assistant = overall_workflow.compile()
