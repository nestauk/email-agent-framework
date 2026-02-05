"""Pydantic models and TypedDicts for email assistant agent."""

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import Literal, NotRequired, TypedDict


class RouterSchema(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(description="Step-by-step reasoning behind the classification.")
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )


class StateInput(TypedDict):
    """Input data for the email assistant state."""

    # This is the input to the state
    email_input: dict


class State(MessagesState):
    """Email assistant state with messages and classification."""

    # This state class has the messages key build in
    email_input: dict
    classification_decision: Literal["ignore", "respond", "notify"]
    tool_call_counts: NotRequired[dict[str, int]]
    pdf_summaries: NotRequired[dict[str, dict]]  # filename -> summary dict


class UserPreferences(BaseModel):
    """Updated user preferences based on user's feedback."""

    chain_of_thought: str = Field(description="Reasoning about which user preferences need to add/update if required")
    user_preferences: str = Field(description="Updated user preferences")


class PDFSummary(BaseModel):
    """Summary of PDF attachment content."""

    filename: str
    original_length: int
    key_points: list[str]
    actions: list[str]
    deadlines: list[str]
    unreadable_flag: bool = False
