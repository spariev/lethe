"""OAuth login flow for OpenAI Codex (ChatGPT Plus/Pro subscription).

Uses PKCE (Proof Key for Code Exchange) OAuth flow with a local callback server:
1. Generate code verifier + challenge
2. Start local HTTP server on port 1455 for callback
3. Open browser to OpenAI's OAuth authorize URL
4. Callback server receives the auth code from redirect
5. Exchange code for access + refresh tokens
6. Save tokens to ~/.lethe/codex_tokens.json

Fallback: if the callback server fails (port in use), prompts user to
paste the full redirect URL manually.

Usage:
    uv run lethe codex-login
"""

import asyncio
import base64
import hashlib
import json
import os
import secrets
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs

import httpx

from lethe.memory.codex_oauth import (
    CLIENT_ID,
    AUTHORIZE_URL,
    TOKEN_URL,
    REDIRECT_URI,
    CALLBACK_PORT,
    SCOPES,
    TOKEN_FILE,
)


def _generate_pkce() -> tuple:
    """Generate PKCE code verifier and challenge.

    Returns:
        (verifier, challenge) tuple
    """
    # 43-128 chars of URL-safe base64
    verifier = secrets.token_urlsafe(32)

    # S256 challenge = base64url(sha256(verifier))
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    return verifier, challenge


def _build_authorize_url(challenge: str, state: str) -> str:
    """Build the OpenAI OAuth authorization URL."""
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        # Required OpenAI-specific params
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "codex_cli_rs",
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


class _CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth callback code."""

    auth_code = None
    auth_state = None
    error = None

    def do_GET(self):
        """Handle the OAuth redirect callback."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "error" in params:
            _CallbackHandler.error = params["error"][0]
            self._send_response("Authentication failed. You can close this window.")
            return

        if "code" in params:
            _CallbackHandler.auth_code = params["code"][0]
            _CallbackHandler.auth_state = params.get("state", [None])[0]
            self._send_response(
                "Authentication successful! You can close this window and return to the terminal."
            )
        else:
            self._send_response("No authorization code received. Please try again.")

    def _send_response(self, message: str):
        """Send an HTML response to the browser."""
        html = f"""<!DOCTYPE html>
<html>
<head><title>Lethe - Codex Login</title></head>
<body style="font-family: sans-serif; text-align: center; padding: 50px;">
<h2>{message}</h2>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        """Suppress default HTTP server logging."""
        pass


def _start_callback_server(timeout: int = 120) -> tuple:
    """Start a local HTTP server to receive the OAuth callback.

    Args:
        timeout: Seconds to wait for the callback before giving up.

    Returns:
        (code, state) tuple, or (None, None) on failure.
    """
    # Reset class-level state
    _CallbackHandler.auth_code = None
    _CallbackHandler.auth_state = None
    _CallbackHandler.error = None

    try:
        server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _CallbackHandler)
    except OSError as e:
        # Port in use
        return None, None

    server.timeout = timeout

    # Run in a thread so we can handle the single request
    def serve():
        server.handle_request()  # Handle exactly one request
        server.server_close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    thread.join(timeout=timeout + 5)

    if _CallbackHandler.error:
        raise RuntimeError(f"OAuth error: {_CallbackHandler.error}")

    return _CallbackHandler.auth_code, _CallbackHandler.auth_state


async def _exchange_code(code: str, verifier: str) -> dict:
    """Exchange authorization code for tokens.

    Args:
        code: The authorization code
        verifier: The PKCE code verifier

    Returns:
        Token response dict with access_token, refresh_token, expires_in
    """
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "code_verifier": verifier,
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            TOKEN_URL,
            data=data,
            headers={"content-type": "application/x-www-form-urlencoded"},
        )

    if response.status_code != 200:
        raise RuntimeError(f"Token exchange failed: {response.status_code} {response.text}")

    return response.json()


def _extract_code_from_url(url: str) -> tuple:
    """Extract auth code and state from a redirect URL.

    Args:
        url: The full redirect URL

    Returns:
        (code, state) tuple
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    code = params.get("code", [None])[0]
    state = params.get("state", [None])[0]
    return code, state


def run_codex_login():
    """Run the interactive Codex OAuth login flow."""
    print("\nOpenAI Codex OAuth Login (ChatGPT Plus/Pro Subscription)\n")
    print("This will open your browser to sign in with your OpenAI/ChatGPT account.")
    print("After signing in, you'll be redirected back automatically.\n")

    # Generate PKCE
    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(16)
    url = _build_authorize_url(challenge, state)

    # Try to start callback server
    print(f"Starting callback server on port {CALLBACK_PORT}...")

    # Start the callback server in a thread
    server_code = [None]
    server_state = [None]
    server_error = [None]
    server_done = threading.Event()

    def run_server():
        try:
            code, st = _start_callback_server(timeout=120)
            server_code[0] = code
            server_state[0] = st
        except Exception as e:
            server_error[0] = str(e)
        finally:
            server_done.set()

    # Try to start server
    try:
        test_server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _CallbackHandler)
        test_server.server_close()
        use_callback = True
    except OSError:
        use_callback = False
        print(f"Port {CALLBACK_PORT} is in use. Falling back to manual flow.\n")

    if use_callback:
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        print(f"Opening browser to:\n{url}\n")

        try:
            webbrowser.open(url)
        except Exception:
            print("(Could not open browser automatically - please open the URL above manually)")

        print("Waiting for authentication callback...")
        print("(If the browser doesn't open, copy the URL above and paste it in your browser)\n")

        server_done.wait(timeout=130)

        if server_error[0]:
            print(f"Error: {server_error[0]}")
            sys.exit(1)

        code = server_code[0]

        if not code:
            # Fallback to manual
            print("Callback not received. Falling back to manual flow.\n")
            use_callback = False

    if not use_callback:
        # Manual flow: user pastes the redirect URL
        print(f"Open this URL in your browser:\n{url}\n")

        try:
            webbrowser.open(url)
        except Exception:
            pass

        print("After authenticating, you'll be redirected to a localhost URL.")
        print("Copy the FULL redirect URL from your browser's address bar and paste it below.\n")

        redirect_url = input("Redirect URL: ").strip()

        if not redirect_url:
            print("No URL provided. Aborting.")
            sys.exit(1)

        code, _ = _extract_code_from_url(redirect_url)

        if not code:
            print(f"Could not extract authorization code from URL: {redirect_url}")
            sys.exit(1)

    print("\nExchanging code for tokens...")

    try:
        result = asyncio.run(_exchange_code(code, verifier))
    except Exception as e:
        print(f"Token exchange failed: {e}")
        sys.exit(1)

    access_token = result.get("access_token")
    refresh_token = result.get("refresh_token")
    expires_in = result.get("expires_in", 3600)

    if not access_token:
        print(f"No access token in response: {result}")
        sys.exit(1)

    # Save tokens
    token_data = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": time.time() + expires_in,
    }

    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps(token_data, indent=2))
    TOKEN_FILE.chmod(0o600)

    print(f"\nOAuth tokens saved to {TOKEN_FILE}")
    print(f"   Access token: {access_token[:20]}...")
    print(f"   Refresh token: {'yes' if refresh_token else 'no'}")
    print(f"   Expires in: {expires_in}s")
    print(f"\nYou can now start Lethe with Codex authentication.")
    print(f"Set LLM_PROVIDER=codex in your .env (no API key needed).")
