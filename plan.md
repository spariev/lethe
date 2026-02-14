# Implementation Plan: OpenAI Codex Subscription OAuth Support

## Summary

Add OpenAI Codex subscription support using the same architectural pattern as the
upstream [Anthropic OAuth commit](https://github.com/atemerev/lethe/commit/64cdf6be)
(`anthropic_oauth.py` + `oauth_login.py`). That commit **bypasses litellm** and
makes direct API calls with provider-specific headers and request/response transforms.
We follow the same approach for the ChatGPT/Codex backend, using `codex` as a
**distinct provider** (not overloading the existing `openai` provider).

Reference plugin: [opencode-openai-codex-auth](https://github.com/numman-ali/opencode-openai-codex-auth)

---

## Key Technical Details

### Upstream Anthropic OAuth Pattern (commit 64cdf6be)

The upstream commit introduces:

| File | Purpose |
|---|---|
| `src/lethe/memory/anthropic_oauth.py` | `AnthropicOAuth` class: token mgmt, direct `httpx` calls to `/v1/messages`, request/response transforms (litellm format <-> Anthropic native), tool name mapping (snake_case <-> PascalCase), Claude Code-compatible headers |
| `src/lethe/tools/oauth_login.py` | PKCE OAuth CLI flow: `uv run lethe oauth-login` opens browser, user pastes code, tokens saved to `~/.lethe/oauth_tokens.json` |
| `src/lethe/memory/llm.py` (diff) | Import + `is_oauth_available()` check; `AsyncLLMClient.__init__` creates `_oauth` instance; `_call_api()` and `_call_api_no_tools()` route through `_call_api_oauth()` when `_oauth` is set |
| `src/lethe/main.py` (diff) | Adds `oauth-login` subcommand via `argparse` subparsers |

Key design decisions in upstream:
- **Bypasses litellm entirely** — direct `httpx.AsyncClient.post()` to Anthropic API
- **Returns litellm-compatible response dicts** — so the rest of `AsyncLLMClient` (tool execution, context management, persistence) works unchanged
- **Token file**: `~/.lethe/oauth_tokens.json` (env override: `LETHE_OAUTH_TOKENS`)
- **Env shortcut**: `ANTHROPIC_AUTH_TOKEN` for access-token-only mode (no refresh)
- **No temperature, no tool_choice, no cache_control** in OAuth requests
- **`metadata.user_id`** injected from `~/.claude.json`

### OpenAI Codex Details (from opencode-openai-codex-auth)

| Parameter | Value |
|---|---|
| API base URL | `https://chatgpt.com/backend-api` |
| API path | `/codex/responses` (Responses API, not Chat Completions) |
| Originator header | `codex_cli_rs` |
| OpenAI-Beta header | `responses=experimental` |
| Accept header | `text/event-stream` |
| store parameter | Must be `false` (backend rejects `true`) |
| stream parameter | `true` (backend returns SSE; see SSE Response Parsing below) |
| Account ID | Extracted from JWT claim `https://api.openai.com/auth` → `chatgpt_account_id` |
| Session ID headers | `session_id` and `conversation_id` (for prompt caching, sent if cache key present) |
| OAuth Client ID | `app_EMoamEEZ73f0CkXaXp7hrann` |
| Authorize URL | `https://auth.openai.com/oauth/authorize` |
| Token URL | `https://auth.openai.com/oauth/token` |
| Redirect URI | `http://localhost:1455/auth/callback` (local callback server) |
| Scope | `openid profile email offline_access` |
| Extra authorize params | `id_token_add_organizations=true`, `codex_cli_simplified_flow=true`, `originator=codex_cli_rs` |
| PKCE method | S256 |
| Models (canonical) | `gpt-5.2`, `gpt-5.2-codex`, `gpt-5.1-codex-max`, `gpt-5.1-codex`, `gpt-5.1-codex-mini`, `gpt-5.1`, `codex-mini-latest` |
| include field | Must always contain `["reasoning.encrypted_content"]` (for multi-turn reasoning) |
| Reasoning config | `{"effort": "high", "summary": "auto"}` (model-specific; see Reasoning section) |
| max_output_tokens | Must be **omitted** (backend rejects it; remove `max_output_tokens` and `max_completion_tokens`) |

### Key Difference: Responses API vs Chat Completions

The ChatGPT/Codex backend uses OpenAI's **Responses API** format, not the Chat
Completions format that litellm expects. This is the same reason the Anthropic
OAuth bypasses litellm (Anthropic uses its own Messages API). We need to:

1. Transform litellm-format messages into Responses API input items
2. Transform Responses API output back into litellm-compatible response dicts
3. Map tool schemas and tool call formats between the two
4. Parse SSE event streams (backend always returns `text/event-stream`)
5. Include `reasoning.encrypted_content` for multi-turn reasoning continuity
6. Omit fields the backend rejects (`max_output_tokens`, `store: true`)

---

## Files to Create / Modify

### 1. NEW: `src/lethe/memory/codex_oauth.py` — Codex OAuth + API Client

**Mirrors `anthropic_oauth.py` structure exactly.** Single file containing:

#### Constants
```python
CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_RESPONSES_PATH = "/codex/responses"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
CALLBACK_PORT = 1455
SCOPES = "openid profile email offline_access"
TOKEN_FILE = Path(os.environ.get("LETHE_CODEX_TOKENS", "~/.lethe/codex_tokens.json")).expanduser()
```

#### Tool Name Mapping — **Not needed initially**

Unlike the Anthropic OAuth path (which maps snake_case ↔ PascalCase for Claude
Code compatibility), the Codex backend uses the Responses API where tool names are
standard function names passed through as-is. The reference plugin does **not**
perform explicit tool name remapping.

**Approach:** Pass tool names through unchanged. If the Codex backend rejects
specific names at runtime, add targeted mappings then. Log any unrecognized tool
names for iterative debugging.

```python
# No TOOL_NAME_TO_CODEX mapping needed — names pass through as function names.
# If specific renames are discovered necessary, add them here.
TOOL_NAME_TO_CODEX: Dict[str, str] = {}  # empty; populated only if needed
TOOL_NAME_FROM_CODEX: Dict[str, str] = {}
```

#### `CodexOAuth` class
Mirrors `AnthropicOAuth`:

- `__init__(access_token, refresh_token, expires_at)` — same pattern
- `_load_tokens()` — check `CODEX_AUTH_TOKEN` env, then `TOKEN_FILE`
- `save_tokens()` — persist to file with `0o600` permissions
- `is_available` property — check if tokens exist
- `ensure_access()` — refresh token if expired (60s buffer)
- `_get_client()` — shared `httpx.AsyncClient`
- `_build_headers(cache_key=None)` — Codex-specific headers:
  ```python
  headers = {
      "authorization": f"Bearer {self.access_token}",
      "openai-beta": "responses=experimental",
      "originator": "codex_cli_rs",
      "chatgpt-account-id": self._account_id,  # from JWT
      "accept": "text/event-stream",
      "content-type": "application/json",
  }
  # Prompt caching headers (sent when cache key is available)
  if cache_key:
      headers["session_id"] = cache_key
      headers["conversation_id"] = cache_key
  ```
  Note: The `x-api-key` header must NOT be sent (delete it if present from httpx defaults).
- `_extract_account_id(token)` — decode JWT middle segment, extract
  `https://api.openai.com/auth` claim for account ID
- `_normalize_messages(messages)` — transform litellm Chat Completions
  format into Responses API input items:
  - system messages -> `instructions` parameter
  - user/assistant messages -> input items with appropriate types
  - tool calls/results -> function_call / function_call_output items
  - Strip message IDs (Codex backend rejects them with `store:false`)
- `_normalize_tools(tools)` — transform litellm tool schemas to
  Responses API function format (names passed through as-is; see Tool Name Mapping above)
- `_parse_response(data)` — transform Responses API output back to
  litellm-compatible dict (same structure as `AnthropicOAuth._parse_response`):
  ```python
  {
      "id": ...,
      "object": "chat.completion",
      "model": ...,
      "choices": [{"index": 0, "message": {...}, "finish_reason": ...}],
      "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...},
  }
  ```
- `_normalize_model(model)` — normalize model names:
  - Strip litellm provider prefixes like `openai/...`, `openrouter/...` (mirror upstream fix `89849cf`)

- `call_responses(messages, tools, model)` — main API call:
  1. `await self.ensure_access()`
  2. Normalize model, messages, tools
     - **Guard against empty input** after normalization (mirror upstream fix `e210524`)
  3. Build request body:
     ```python
     body = {
         "model": normalized_model,
         "input": input_items,            # from _normalize_messages()
         "instructions": system_prompt,    # extracted from messages
         "tools": normalized_tools,        # from _normalize_tools()
         "store": False,                   # required: backend rejects True
         "stream": True,                   # backend returns SSE events
         "include": ["reasoning.encrypted_content"],  # required for multi-turn reasoning
         "reasoning": {                    # see Reasoning Configuration below
             "effort": "high",             # default; model-specific override possible
             "summary": "auto",
         },
         "text": {"format": {"type": "text"}},
     }
     # Do NOT include max_output_tokens or max_completion_tokens (backend rejects them)
     ```
  4. POST to `{CODEX_BASE_URL}{CODEX_RESPONSES_PATH}`
  5. **Parse SSE response** (see SSE Response Parsing below)
  6. Return litellm-compatible response dict

- `_parse_sse_response(response)` — **new**: consume SSE stream, extract final
  `response.done` or `response.completed` event's JSON payload, pass to
  `_parse_response()`. See SSE Response Parsing section below.

- `close()` — close httpx client

#### Reasoning Configuration

The Codex backend supports model-specific reasoning parameters:

| Model | Supported Effort Levels | Notes |
|---|---|---|
| `gpt-5.2`, `gpt-5.2-codex`, `gpt-5.1-codex-max` | `none`, `low`, `medium`, `high`, `xhigh` | Full range |
| `gpt-5.1-codex`, `gpt-5.1` | `none`, `low`, `medium`, `high` | No `xhigh` |
| `gpt-5.1-codex-mini`, `codex-mini-latest` | `medium`, `high` | No `none`/`low`/`xhigh` |

Default: `{"effort": "high", "summary": "auto"}`. The `summary` field controls
whether the model produces a plaintext summary of its reasoning chain (useful
since encrypted reasoning content is opaque).

#### SSE Response Parsing

The Codex backend always returns SSE (`text/event-stream`), even for non-streaming
logical requests. The response is a stream of `data:` lines. To get the final
response object:

1. Read the SSE stream line by line
2. Parse each `data:` line as JSON
3. Look for events with `type` = `response.done` or `response.completed`
4. The JSON payload of that event is the complete Responses API response object
5. Pass to `_parse_response()` for litellm-compatible transform

This avoids buffering partial deltas and is simpler than true streaming support.
If the stream ends without a `response.done` event, raise an error.

#### Error Mapping: 404 → 429

The Codex backend returns **404** (not 429) for usage-limit errors. Map these
to 429 so the retry logic in `_call_api_codex()` handles them correctly:

```python
# Error codes that indicate usage limits (returned as 404 by backend)
USAGE_LIMIT_CODES = {"usage_limit_reached", "usage_not_included", "rate_limit_exceeded"}

if response.status_code == 404:
    error_data = response.json()
    error_code = error_data.get("error", {}).get("code", "")
    if error_code in USAGE_LIMIT_CODES or "usage limit" in str(error_data).lower():
        # Treat as 429 for retry logic
        raise RateLimitError(...)
```

#### `is_codex_oauth_available()` function
Same pattern as `is_oauth_available()` in upstream:
```python
def is_codex_oauth_available() -> bool:
    if os.environ.get("CODEX_AUTH_TOKEN"):
        return True
    if TOKEN_FILE.exists():
        ...
```

### 2. NEW: `src/lethe/tools/codex_login.py` — PKCE OAuth Login CLI

**Mirrors `oauth_login.py` structure** but uses a **local callback server** instead
of code-paste, because OpenAI's OAuth flow redirects to a URI rather than displaying
a code for the user to copy. Provides `run_codex_login()`:

- `_generate_pkce()` — reuse same PKCE implementation (or import from
  `oauth_login.py` if we extract to shared util)
- `_build_authorize_url(verifier, challenge)` — OpenAI OAuth authorize URL
  with PKCE params and required extra parameters:
  ```python
  params = {
      "response_type": "code",
      "client_id": CLIENT_ID,
      "redirect_uri": REDIRECT_URI,
      "scope": SCOPES,
      "code_challenge": challenge,
      "code_challenge_method": "S256",
      "state": state,
      # Required OpenAI-specific params:
      "id_token_add_organizations": "true",
      "codex_cli_simplified_flow": "true",
      "originator": "codex_cli_rs",
  }
  ```
- `_start_callback_server()` — start a local HTTP server on `localhost:1455`
  that listens for the OAuth callback, extracts the `code` parameter from the
  redirect, and signals the main flow. Server shuts down after receiving the
  callback or after a timeout (e.g., 120s).
- `_exchange_code(code, verifier)` — POST to `https://auth.openai.com/oauth/token`
  with `Content-Type: application/x-www-form-urlencoded`:
  ```python
  data = {
      "grant_type": "authorization_code",
      "code": code,
      "code_verifier": verifier,
      "client_id": CLIENT_ID,
      "redirect_uri": REDIRECT_URI,
  }
  ```
- `run_codex_login()` — interactive CLI flow:
  1. Generate PKCE pair + random state
  2. Start local callback server on port 1455
  3. Open browser to authorize URL
  4. Wait for callback server to receive the redirect with auth code
  5. Exchange code for tokens
  6. Save to `~/.lethe/codex_tokens.json`
  7. **Fallback**: if the callback server fails (port in use), print the
     authorize URL and prompt the user to paste the full redirect URL manually

### 3. MODIFY: `src/lethe/memory/llm.py` — Route Codex Through OAuth

Use `codex` as a **distinct provider** (not overloading `openai`). This avoids
ambiguity when a user has both `OPENAI_API_KEY` and Codex tokens, and makes the
config explicit.

Follow the same routing patterns as the upstream Anthropic OAuth fixes:
- `14dad51`: log which auth mode is used.
- `209791b`: **Route ALL calls through OAuth when active** (including no-tools calls).

Implementation details:

- **Import**: `from lethe.memory.codex_oauth import CodexOAuth, is_codex_oauth_available`

- **Add `"codex"` to `PROVIDERS` dict** (in llm.py):
  ```python
  "codex": {
      "env_key": "",  # no API key needed — uses OAuth tokens
      "model_prefix": "",
      "default_model": "gpt-5.2",
      "default_model_aux": "codex-mini-latest",
  },
  ```

- **`AsyncLLMClient.__init__`**:

  Add a Codex OAuth client when provider is `codex`:

  ```python
  # Codex OAuth (ChatGPT subscription — bypasses litellm)
  self._codex_oauth: Optional[CodexOAuth] = None
  if self.config.provider == "codex":
      if is_codex_oauth_available():
          self._codex_oauth = CodexOAuth()
          logger.info("Auth: using Codex OAuth token (ChatGPT subscription)")
      else:
          raise ValueError(
              "LLM_PROVIDER=codex requires OAuth tokens. "
              "Run 'uv run lethe codex-login' to authenticate, "
              "or set CODEX_AUTH_TOKEN in env."
          )
  ```

- **Routing:**
  - `_call_api()` must route through Codex OAuth whenever `self._codex_oauth` is set.
  - `_call_api_no_tools()` must ALSO route through Codex OAuth whenever `self._codex_oauth` is set.

  Do **not** mix modes within a single session.

- **New `_call_api_codex()` method**:
  Same structure as `_call_with_retry_oauth()` in upstream:
  - delegates to `self._codex_oauth.call_responses(...)`
  - debug logging
  - retries for 429/5xx **and 404 usage-limit errors** (see Error Mapping above)
  - tracks usage

- **`LLMConfig.__post_init__()` / config validation**:
  When provider is `"codex"`, skip API key requirement check entirely.
  (Match upstream: OAuth is a complete alternative auth mode.)

- **`LLMConfig` auto-detection fallback**:
  Add `CODEX_AUTH_TOKEN` to the env-based provider detection:
  ```python
  if os.environ.get("CODEX_AUTH_TOKEN") or TOKEN_FILE.exists():
      return "codex"
  ```
  This goes **after** the `ANTHROPIC_AUTH_TOKEN` check but **before** the
  `OPENAI_API_KEY` check, so explicit Codex tokens are detected correctly.

### 4. MODIFY: `src/lethe/main.py` — Add `codex-login` Subcommand

Same pattern as the `oauth-login` subcommand in upstream:
```python
subparsers.add_parser("codex-login", help="Login with OpenAI OAuth (ChatGPT Plus/Pro)")

if args.command == "codex-login":
    from lethe.tools.codex_login import run_codex_login
    run_codex_login()
    return
```

### 5. MODIFY: `install.sh` — Add Codex Provider Option

`codex` is a **distinct provider** (not an alias for `openai`). This keeps the
config explicit and avoids conflicts with `OPENAI_API_KEY`-based setups.

- Add `"codex"` to PROVIDERS array:
  ```bash
  ["codex"]="OpenAI Codex (ChatGPT Plus/Pro subscription, no API key - uses OAuth)"
  ```
- Add to PROVIDER_MODELS / PROVIDER_MODELS_AUX:
  ```bash
  ["codex"]="gpt-5.2" / "codex-mini-latest"
  ```
- Add to PROVIDER_KEYS with empty value (`["codex"]=""`) — no API key needed
- Add to PROVIDER_URLS:
  ```bash
  ["codex"]="https://chatgpt.com"
  ```
- Skip `prompt_api_key()` when `SELECTED_PROVIDER=codex`
- Generated config sets `LLM_PROVIDER=codex` (not `openai`)
- After install, print post-setup instructions:
  ```
  ══════════════════════════════════════════════════════════════
  Codex OAuth Setup Required:
    Run: uv run lethe codex-login
    This will open your browser to authenticate with your ChatGPT account.
  ══════════════════════════════════════════════════════════════
  ```

### 6. MODIFY: `.env.example` — Document Codex Option

Add comments:
```bash
# For ChatGPT Plus/Pro subscription (Codex), no API key needed:
# 1. Set: LLM_PROVIDER=codex
# 2. Run: uv run lethe codex-login
# Tokens stored in ~/.lethe/codex_tokens.json (auto-refreshed)
# Or set CODEX_AUTH_TOKEN=<token> for access-token-only mode (no refresh)
```

---

## Architecture Decisions

### Why bypass litellm (same as upstream)?
The ChatGPT/Codex backend uses the **Responses API** format, not Chat Completions.
litellm doesn't support this format natively. The upstream Anthropic OAuth takes
the same approach — bypass litellm, make direct httpx calls, and return
litellm-compatible response dicts so the rest of the codebase works unchanged.

### Why separate files from Anthropic OAuth?
Different provider, different API format (Responses vs Messages), different
endpoints, different auth headers, different tool name mappings. Keeping them
separate follows the upstream pattern and avoids coupling.

### Why `codex` as a distinct provider (not overloading `openai`)?

Unlike the Anthropic case (where `anthropic` provider auto-detects OAuth tokens
vs API key), we use a **separate `codex` provider** because:

1. **Different API format** — Codex uses the Responses API, while `openai` uses
   Chat Completions via litellm. These are fundamentally different code paths.
2. **No ambiguity** — a user with both `OPENAI_API_KEY` and Codex tokens won't
   get surprising behavior; they explicitly choose which path to use.
3. **Cleaner config** — `LLM_PROVIDER=codex` is self-documenting; no need to
   reason about implicit precedence rules.

Provider behavior:
- `LLM_PROVIDER=codex` → Codex OAuth path (requires tokens; bypasses litellm)
- `LLM_PROVIDER=openai` → standard OpenAI API key (litellm) path (unchanged)
- Auto-detection: if `CODEX_AUTH_TOKEN` or `codex_tokens.json` exists and no
  other provider is detected, auto-select `codex`

### Token lifecycle
Same as upstream Anthropic pattern:
- `uv run lethe codex-login` → PKCE flow → tokens saved to `~/.lethe/codex_tokens.json`
- Or `CODEX_AUTH_TOKEN=<token>` in env (no refresh, simpler)
- `ensure_access()` refreshes 60s before expiry (same buffer as upstream)
- `save_tokens()` uses `0o600` permissions

### Message format transform (Responses API)
The most complex part. The Responses API uses "input items" instead of messages:
- System prompt → `instructions` parameter (not an input item)
- User messages → `{"type": "message", "role": "user", "content": [...]}`
- Assistant messages → `{"type": "message", "role": "assistant", "content": [...]}`
- Tool calls → `{"type": "function_call", "name": ..., "arguments": ..., "call_id": ...}`
- Tool results → `{"type": "function_call_output", "call_id": ..., "output": ...}`
- All IDs must be stripped (store:false means no server-side state)
- `item_reference` constructs must be filtered

### `store:false` requirement
The ChatGPT backend explicitly rejects `store:true`. This means:
- Full message history must be sent with each request
- No server-side conversation persistence
- Encrypted reasoning content returned by model **must** be included in subsequent
  turns — always send `"include": ["reasoning.encrypted_content"]` in the request
  body so the backend returns encrypted reasoning, and pass it back in the next
  turn's input items

---

## Implementation Order

1. Create `src/lethe/memory/codex_oauth.py` (OAuth client + API transforms)
2. Create `src/lethe/tools/codex_login.py` (PKCE login CLI)
3. Modify `src/lethe/memory/llm.py` (route through Codex OAuth)
4. Modify `src/lethe/main.py` (add `codex-login` subcommand)
5. Modify `install.sh` (add codex provider option)
6. Modify `.env.example` (document codex setup)

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Responses API format mismatch | Study opencode-openai-codex-auth's request-transformer.ts closely; test with debug logging (`LLM_DEBUG=true`) |
| ChatGPT backend rate limits / usage limits | Map 429s appropriately; also map **404 usage-limit errors** to 429 (see Error Mapping section above); respect Plus/Pro tier limits |
| JWT parsing edge cases | Standard base64url decode with padding fix; no external library needed |
| Tool names rejected by backend | Start with pass-through (no remapping); log any rejections and add targeted mappings iteratively |
| Encrypted reasoning content | Always include `"reasoning.encrypted_content"` in the `include` field; pass through encrypted content in subsequent turns |
| SSE response parsing | Backend always returns SSE even for logical non-streaming; consume stream and extract `response.done` event (see SSE section above) |
| Port 1455 conflict on callback server | Fallback to manual flow: print authorize URL, user pastes redirect URL. Log a clear message when port is in use. |
| store:false context growth | Already handled by Lethe's `ContextWindow` compaction — messages are managed locally |
| `max_output_tokens` rejected by backend | Omit `max_output_tokens` and `max_completion_tokens` from request body entirely |
| OpenAI blocks third-party OAuth | Same risk as Anthropic (see `install.sh` line 43: "Anthropic blocked third-party OAuth in Jan 2026"). No mitigation beyond monitoring; document the risk to users |
| Model-specific reasoning constraints | Validate reasoning `effort` against model capabilities (see Reasoning Configuration table); clamp invalid values to nearest supported level |
