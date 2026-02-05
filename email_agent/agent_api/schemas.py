"""Pydantic models for the agent-facing API."""

from __future__ import annotations

from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import BaseModel, BeforeValidator, EmailStr, Field


def _coerce_to_str(v: int | str) -> str:
    """Coerce int or str to str for job_id fields."""
    return str(v)


class MessageResponse(BaseModel):
    """Simple success/error message wrapper."""

    message: str = Field(..., description="Human readable status message")


class RegisterUserRequest(BaseModel):
    """Payload for POST /registerUser."""

    user_id: int = Field(..., alias="userId")
    email_to_monitor: EmailStr = Field(..., alias="emailToMonitor")
    email_api_provider: Literal["google", "microsoft"] = Field(..., alias="emailAPIProvider")
    email_api_access_token: str = Field(..., alias="emailAPIAccessToken")
    email_api_access_token_expires_at: int = Field(..., alias="emailAPIAccessTokenExpiresAt")
    email_api_refresh_token: str = Field(..., alias="emailAPIRefreshToken")
    email_api_refresh_token_expires_in: int = Field(..., alias="emailAPIRefreshTokenExpiresIn")
    display_name: str = Field(..., alias="displayName")
    main_contact: Optional[str] = Field(None, alias="mainContact")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "userId": 42,
                    "emailToMonitor": "user@example.com",
                    "emailAPIProvider": "google",
                    "emailAPIAccessToken": "ya29.a0AR...",
                    "emailAPIAccessTokenExpiresAt": 1_732_208_000,
                    "emailAPIRefreshToken": "1//0f-oauth-refresh",
                    "emailAPIRefreshTokenExpiresIn": 3_600,
                    "displayName": "Example User",
                    "mainContact": "Jane Doe",
                }
            ]
        },
    }


class UnregisterUserRequest(BaseModel):
    """Payload for POST /unregisterUser."""

    user_id: int = Field(..., alias="userId")

    model_config = {"populate_by_name": True}


class ToolCompletedRequest(BaseModel):
    """Payload for POST /toolCompleted."""

    job_id: Annotated[str, BeforeValidator(_coerce_to_str)] = Field(..., alias="jobId")
    tool: str = Field(..., description="The tool name that was completed")
    tool_output_data: Dict[str, Any] = Field(..., alias="toolOutputData")

    model_config = {"populate_by_name": True}


class UserRecord(BaseModel):
    """Response model representing a stored user."""

    user_id: int
    email_to_monitor: EmailStr
    display_name: str
    email_api_provider: str
    status: str
