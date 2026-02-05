"""Header-based authentication for the agent API."""

import os

from fastapi import Header, HTTPException, status


def get_agent_api_key() -> str:
    """Get the AGENT_API_KEY from environment, failing if not set."""
    key = os.getenv("AGENT_API_KEY")
    if not key:
        raise RuntimeError("AGENT_API_KEY environment variable is required. Set it in your .env file or environment.")
    return key


async def verify_agent_api_key(
    x_api_key: str = Header(..., alias="X-API-Key", description="AGENT_API_KEY value"),
) -> str:
    """Validate the X-API-Key header matches the configured AGENT_API_KEY."""
    expected_key = get_agent_api_key()

    if x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Invalid API key"},
        )

    return expected_key
