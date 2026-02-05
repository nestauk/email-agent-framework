#!/usr/bin/env python3
"""Script to get Gmail OAuth tokens for the email agent."""

import os
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def main():
    client_id = os.getenv("GMAIL_CLIENT_ID")
    client_secret = os.getenv("GMAIL_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("ERROR: Set GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET in your environment")
        sys.exit(1)

    # Create OAuth flow from client secrets
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)

    print("Opening browser for Gmail authorization...")
    print("(Make sure to log in with the Gmail account you want to monitor)\n")

    credentials = flow.run_local_server(port=8080)

    print("\n" + "=" * 60)
    print("SUCCESS! Here are your tokens:")
    print("=" * 60)
    print(f"\nAccess Token:\n{credentials.token}\n")
    print(f"Refresh Token:\n{credentials.refresh_token}\n")
    print(f"Token Expiry:\n{credentials.expiry}\n")
    print("=" * 60)

    print("\nTo register this user, run:")
    print(f"""
curl -X POST http://localhost:8000/registerUser \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: $AGENT_API_KEY" \\
  -d '{{
    "userId": 1,
    "emailToMonitor": "YOUR_EMAIL@gmail.com",
    "emailAPIProvider": "google",
    "emailAPIAccessToken": "{credentials.token}",
    "emailAPIAccessTokenExpiresAt": {int(credentials.expiry.timestamp()) if credentials.expiry else 0},
    "emailAPIRefreshToken": "{credentials.refresh_token}",
    "emailAPIRefreshTokenExpiresIn": 3600,
    "displayName": "Your Name"
  }}'
""")


if __name__ == "__main__":
    main()
