"""OpenAI Codex OAuth client for ChatGPT Plus/Pro subscription auth.

Bypasses litellm to make direct API calls to the ChatGPT backend using
the Responses API format. Tokens are refreshed automatically.

Mirrors the architecture of anthropic_oauth.py but targets the Codex
backend (https://chatgpt.com/backend-api/codex/responses).
"""

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Codex backend
CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_RESPONSES_PATH = "/codex/responses"

# OAuth config
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
CALLBACK_PORT = 1455
SCOPES = "openid profile email offline_access"

# Token file location
TOKEN_FILE = Path(os.environ.get("LETHE_CODEX_TOKENS", "~/.lethe/codex_tokens.json")).expanduser()

# Tool name mapping - pass through as-is for Responses API.
# Populated only if specific renames are discovered necessary at runtime.
TOOL_NAME_TO_CODEX: Dict[str, str] = {}
TOOL_NAME_FROM_CODEX: Dict[str, str] = {}

# Error codes that indicate usage limits (returned as 404 by backend)
USAGE_LIMIT_CODES = {"usage_limit_reached", "usage_not_included", "rate_limit_exceeded"}

# Reasoning effort by model
MODEL_REASONING_SUPPORT = {
    "gpt-5.2": {"efforts": ["none", "low", "medium", "high", "xhigh"]},
    "gpt-5.2-codex": {"efforts": ["none", "low", "medium", "high", "xhigh"]},
    "gpt-5.1-codex-max": {"efforts": ["none", "low", "medium", "high", "xhigh"]},
    "gpt-5.1-codex": {"efforts": ["none", "low", "medium", "high"]},
    "gpt-5.1": {"efforts": ["none", "low", "medium", "high"]},
    "gpt-5.1-codex-mini": {"efforts": ["medium", "high"]},
    "codex-mini-latest": {"efforts": ["medium", "high"]},
}


def _map_tool_name_to_codex(name: str) -> str:
    """Map our tool name to Codex format (pass-through)."""
    return TOOL_NAME_TO_CODEX.get(name, name)


def _map_tool_name_from_codex(name: str) -> str:
    """Map Codex tool name back to ours (pass-through)."""
    return TOOL_NAME_FROM_CODEX.get(name, name)


def _decode_jwt_payload(token: str) -> dict:
    """Decode the payload (middle segment) of a JWT without verification."""
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    # Add padding
    padding = 4 - len(payload) % 4
    if padding != 4:
        payload += "=" * padding
    try:
        decoded = base64.urlsafe_b64decode(payload)
        return json.loads(decoded)
    except Exception:
        return {}


def _get_reasoning_config(model: str) -> dict:
    """Get reasoning configuration for a model."""
    effort = "high"  # default

    # Validate effort against model capabilities
    model_info = MODEL_REASONING_SUPPORT.get(model)
    if model_info:
        supported = model_info["efforts"]
        if effort not in supported:
            # Clamp to nearest supported level
            effort = supported[-1]  # highest supported

    return {"effort": effort, "summary": "auto"}


class CodexOAuth:
    """Direct Codex API client using OAuth tokens (ChatGPT Plus/Pro subscription).

    Handles:
    - Token storage, loading, and auto-refresh
    - Codex-specific headers (originator, openai-beta, account ID from JWT)
    - Message format transform (litellm Chat Completions -> Responses API)
    - SSE response parsing (backend always returns event streams)
    - Error mapping (404 usage limits -> 429 for retry logic)
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        expires_at: Optional[float] = None,
    ):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = expires_at or 0
        self._client: Optional[httpx.AsyncClient] = None
        self._account_id: Optional[str] = None

        # Try loading from env or file if not provided
        if not self.access_token:
            self._load_tokens()

        # Extract account ID from JWT
        if self.access_token:
            self._account_id = self._extract_account_id(self.access_token)

    def _load_tokens(self):
        """Load tokens from env var or token file."""
        # Check env first (access token only - no refresh possible)
        env_token = os.environ.get("CODEX_AUTH_TOKEN")
        if env_token:
            self.access_token = env_token
            logger.info("Codex OAuth: loaded access token from CODEX_AUTH_TOKEN env")
            return

        # Check token file
        if TOKEN_FILE.exists():
            try:
                data = json.loads(TOKEN_FILE.read_text())
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                self.expires_at = data.get("expires_at", 0)
                logger.info(f"Codex OAuth: loaded tokens from {TOKEN_FILE}")
            except Exception as e:
                logger.error(f"Codex OAuth: failed to load tokens from {TOKEN_FILE}: {e}")

    def save_tokens(self):
        """Persist tokens to file."""
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(json.dumps({
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
        }, indent=2))
        TOKEN_FILE.chmod(0o600)
        logger.info(f"Codex OAuth: saved tokens to {TOKEN_FILE}")

    @property
    def is_available(self) -> bool:
        """Check if OAuth is configured (has tokens)."""
        return bool(self.access_token)

    async def ensure_access(self):
        """Refresh the access token if expired."""
        if not self.refresh_token:
            return

        # Refresh 60s before expiry
        if self.expires_at > time.time() + 60:
            return

        logger.info("Codex OAuth: refreshing access token")
        client = await self._get_client()

        response = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "client_id": CLIENT_ID,
            },
            headers={"content-type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise RuntimeError(f"Codex OAuth token refresh failed: {response.status_code} {response.text}")

        data = response.json()
        self.access_token = data["access_token"]
        self.refresh_token = data.get("refresh_token", self.refresh_token)
        self.expires_at = time.time() + data.get("expires_in", 3600)
        self._account_id = self._extract_account_id(self.access_token)
        self.save_tokens()
        logger.info("Codex OAuth: token refreshed successfully")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=600.0)
        return self._client

    def _extract_account_id(self, token: str) -> Optional[str]:
        """Extract ChatGPT account ID from JWT claim."""
        payload = _decode_jwt_payload(token)
        auth_claim = payload.get("https://api.openai.com/auth", {})
        account_id = None
        if isinstance(auth_claim, dict):
            account_id = auth_claim.get("chatgpt_account_id")
        if account_id:
            logger.debug(f"Codex OAuth: extracted account_id from JWT")
        return account_id

    def _build_headers(self, cache_key: Optional[str] = None) -> dict:
        """Build Codex-specific headers."""
        headers = {
            "authorization": f"Bearer {self.access_token}",
            "openai-beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "accept": "text/event-stream",
            "content-type": "application/json",
        }

        if self._account_id:
            headers["chatgpt-account-id"] = self._account_id

        # Prompt caching headers
        if cache_key:
            headers["session_id"] = cache_key
            headers["conversation_id"] = cache_key

        return headers

    def _normalize_model(self, model: str) -> str:
        """Normalize model name - strip litellm provider prefixes."""
        # Strip prefixes like "openai/", "openrouter/", etc.
        if "/" in model:
            model = model.split("/")[-1]
        return model

    def _normalize_tools(self, tools: List[Dict]) -> List[Dict]:
        """Transform litellm tool schemas to Responses API function format."""
        normalized = []
        for tool in tools:
            if "function" in tool:
                func = tool["function"]
                name = _map_tool_name_to_codex(func.get("name", ""))
                normalized.append({
                    "type": "function",
                    "name": name,
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                # Already in some other format - pass through
                t = tool.copy()
                if "name" in t:
                    t["name"] = _map_tool_name_to_codex(t["name"])
                normalized.append(t)
        return normalized

    def _normalize_messages(self, messages: List[Dict]) -> tuple:
        """Convert litellm Chat Completions format to Responses API input items.

        Returns:
            (instructions, input_items) - system prompt extracted as instructions
        """
        instructions = None
        input_items = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                # Extract system prompt as instructions
                if isinstance(content, list):
                    # Structured content - extract text parts
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    instructions = "\n\n".join(text_parts)
                elif isinstance(content, str):
                    instructions = content
                continue

            if role == "tool":
                # Tool results -> function_call_output items
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": str(content),
                })
                continue

            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    # Text content before tool calls
                    if content:
                        input_items.append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": str(content)}],
                        })
                    # Tool calls -> function_call items
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        name = _map_tool_name_to_codex(func.get("name", ""))
                        input_items.append({
                            "type": "function_call",
                            "name": name,
                            "arguments": func.get("arguments", "{}"),
                            "call_id": tc.get("id", ""),
                        })
                else:
                    if content:
                        input_items.append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": str(content)}],
                        })
                continue

            # User messages
            if isinstance(content, list):
                # Multimodal content
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append({"type": "input_text", "text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            parts.append({"type": "input_image", "image_url": url})
                    elif isinstance(part, str):
                        parts.append({"type": "input_text", "text": part})
                if parts:
                    input_items.append({
                        "type": "message",
                        "role": "user",
                        "content": parts,
                    })
            else:
                input_items.append({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": str(content)}],
                })

        return instructions, input_items

    def _parse_response(self, data: dict) -> dict:
        """Convert Responses API output to litellm-compatible response dict."""
        output_items = data.get("output", [])

        text_parts = []
        tool_calls = []

        for item in output_items:
            item_type = item.get("type", "")

            if item_type == "message":
                # Extract text from message content
                for block in item.get("content", []):
                    if block.get("type") in ("output_text", "text"):
                        text_parts.append(block.get("text", ""))

            elif item_type == "function_call":
                name = _map_tool_name_from_codex(item.get("name", ""))
                tool_calls.append({
                    "id": item.get("call_id", ""),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": item.get("arguments", "{}"),
                    },
                })

            elif item_type == "reasoning":
                # Pass through encrypted reasoning content - will be included
                # in subsequent turns' input automatically via the output items
                pass

        # Determine finish reason
        status = data.get("status", "completed")
        if tool_calls:
            finish_reason = "tool_calls"
        elif status == "completed":
            finish_reason = "stop"
        elif status == "incomplete":
            finish_reason = "length"
        else:
            finish_reason = "stop"

        # Build litellm-compatible response
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        usage = data.get("usage", {})

        return {
            "id": data.get("id", ""),
            "object": "chat.completion",
            "model": data.get("model", ""),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": (
                    usage.get("input_tokens", 0) +
                    usage.get("output_tokens", 0)
                ),
            },
            # Store raw output for passing encrypted reasoning in next turn
            "_codex_output": output_items,
        }

    async def _parse_sse_response(self, response: httpx.Response) -> dict:
        """Consume SSE stream and extract the final response object.

        The Codex backend always returns SSE (text/event-stream). We look
        for the response.done or response.completed event which contains
        the complete Responses API response object.
        """
        final_data = None

        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue

            # SSE format: "data: {...}"
            if line.startswith("data: "):
                json_str = line[6:]
                if json_str == "[DONE]":
                    break
                try:
                    event_data = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                event_type = event_data.get("type", "")

                if event_type in ("response.done", "response.completed"):
                    # The response field contains the complete response
                    final_data = event_data.get("response", event_data)
                    break

            # Also handle "event: " prefix format
            elif line.startswith("event: "):
                # Next data line will contain the payload
                continue

        if final_data is None:
            raise RuntimeError("Codex SSE stream ended without response.done event")

        return final_data

    async def call_responses(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: str = "gpt-5.2",
        **kwargs,
    ) -> dict:
        """Make a Codex API call via the Responses API.

        Args:
            messages: litellm-format messages (system/user/assistant/tool)
            tools: litellm-format tool schemas (optional)
            model: model name

        Returns:
            litellm-compatible response dict
        """
        await self.ensure_access()

        # Normalize
        model = self._normalize_model(model)
        instructions, input_items = self._normalize_messages(messages)

        # Guard against empty input after normalization
        if not input_items:
            logger.warning("Codex OAuth: no input items after normalization, adding placeholder")
            input_items = [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "[Continue]"}],
            }]

        normalized_tools = self._normalize_tools(tools) if tools else []

        # Build request body
        body: Dict[str, Any] = {
            "model": model,
            "input": input_items,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "reasoning": _get_reasoning_config(model),
            "text": {"format": {"type": "text"}},
        }

        if instructions:
            body["instructions"] = instructions

        if normalized_tools:
            body["tools"] = normalized_tools

        # Do NOT include max_output_tokens or max_completion_tokens (backend rejects them)

        # Build headers
        headers = self._build_headers()

        # Make request via httpx streaming (backend returns SSE)
        url = f"{CODEX_BASE_URL}{CODEX_RESPONSES_PATH}"
        client = await self._get_client()

        logger.info(f"Codex API call: model={model}, input_items={len(input_items)}, tools={len(normalized_tools)}")

        async with client.stream("POST", url, headers=headers, json=body) as stream_response:
            # Handle error responses
            if stream_response.status_code == 404:
                body_text = ""
                async for chunk in stream_response.aiter_bytes():
                    body_text += chunk.decode("utf-8", errors="replace")
                    if len(body_text) > 1000:
                        break
                # Error mapping: 404 with usage-limit codes -> treat as rate limit
                try:
                    error_data = json.loads(body_text)
                    error_code = error_data.get("error", {}).get("code", "")
                    if error_code in USAGE_LIMIT_CODES or "usage limit" in body_text.lower():
                        raise RuntimeError(
                            f"Codex API rate limit (429 mapped from 404): {error_code}"
                        )
                except (json.JSONDecodeError, RuntimeError) as e:
                    if isinstance(e, RuntimeError):
                        raise
                raise RuntimeError(f"Codex API error: 404 - {body_text[:500]}")

            if stream_response.status_code != 200:
                body_text = ""
                async for chunk in stream_response.aiter_bytes():
                    body_text += chunk.decode("utf-8", errors="replace")
                    if len(body_text) > 1000:
                        break
                logger.error(f"Codex API error: {stream_response.status_code} {body_text[:500]}")
                raise RuntimeError(
                    f"Codex API error: {stream_response.status_code} - {body_text[:500]}"
                )

            data = await self._parse_sse_response(stream_response)

        usage = data.get("usage", {})
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        logger.info(f"Codex API response: {in_tok} in + {out_tok} out tokens")

        return self._parse_response(data)

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


def is_codex_oauth_available() -> bool:
    """Check if Codex OAuth tokens are available (env or file)."""
    if os.environ.get("CODEX_AUTH_TOKEN"):
        return True
    if TOKEN_FILE.exists():
        try:
            data = json.loads(TOKEN_FILE.read_text())
            return bool(data.get("access_token"))
        except Exception:
            pass
    return False
