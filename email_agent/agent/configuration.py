"""Define the configurable parameters for the agent."""

import os
from dataclasses import dataclass, fields
from typing import Any

from langchain_core.runnables import RunnableConfig

TOOL_CONFIG = {
    "tool_names": [
        "send_email_tool",
        "Question",
        "search_guidance_tool",
        "Done",
    ],
    "hitl_tools": ["send_email_tool", "Question"],
    "question_tool_name": "Question",
    "send_email_tool_name": "send_email_tool",
}


@dataclass(kw_only=True)
class Configuration:
    """Placeholder for configuration."""

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config["configurable"] if config and "configurable" in config else {}
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name)) for f in fields(cls) if f.init
        }

        return cls(**{k: v for k, v in values.items() if v})
