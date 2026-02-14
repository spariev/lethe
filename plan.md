# Implementation Plan: OpenAI Codex Subscription OAuth Support

## Summary

Add OpenAI Codex subscription support using the same architectural pattern as the
upstream [Anthropic OAuth commit](https://github.com/atemerev/lethe/commit/64cdf6be)
(`anthropic_oauth.py` + `oauth_login.py`). That commit **bypasses litellm** and
makes direct API calls with provider-specific headers, request/response transforms,
and tool name mapping. We follow the same approach for the ChatGPT/Codex backend.

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
| store parameter | Must be `false` (backend rejects `true`) |
| Account ID | Extracted from JWT claim `https://api.openai.com/auth` |
| Session ID header | `session_id` (for prompt caching) |
| OAuth Client ID | Codex CLI's public client ID |
| Authorize URL | OpenAI's OAuth authorize endpoint |
| Token URL | OpenAI's OAuth token endpoint |
| Redirect URI | `https://auth.openai.com/oauth/callback` (code-paste flow) |
| Scope | `openid profile email offline_access` |
| PKCE method | S256 |
| Models | `gpt-5.2`, `gpt-5.2-mini`, `codex-mini-latest`, `gpt-5.2-codex` |

### Key Difference: Responses API vs Chat Completions

The ChatGPT/Codex backend uses OpenAI's **Responses API** format, not the Chat
Completions format that litellm expects. This is the same reason the Anthropic
OAuth bypasses litellm (Anthropic uses its own Messages API). We need to:

1. Transform litellm-format messages into Responses API input items
2. Transform Responses API output back into litellm-compatible response dicts
3. Map tool schemas and tool call formats between the two

---

## Files to Create / Modify

### 1. NEW: `src/lethe/memory/codex_oauth.py` — Codex OAuth + API Client

**Mirrors `anthropic_oauth.py` structure exactly.** Single file containing:

#### Constants
```python
CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_RESPONSES_PATH = "/codex/responses"
CLIENT_ID = "<codex CLI public client_id>"  # from upstream Codex CLI source
TOKEN_URL = "<openai oauth token endpoint>"
TOKEN_FILE = Path(os.environ.get("LETHE_CODEX_TOKENS", "~/.lethe/codex_tokens.json")).expanduser()
```

#### Tool Name Mapping (like `TOOL_NAME_TO_CLAUDE` in upstream)
```python
# Our snake_case → Codex expected names
TOOL_NAME_TO_CODEX = {
    "bash": "shell",
    "read_file": "read_file",
    "write_file": "create_file",
    "edit_file": "apply_patch",  # or "edit" depending on Codex expectations
    ...
}
TOOL_NAME_FROM_CODEX = {v: k for k, v in TOOL_NAME_TO_CODEX.items()}
```

#### `CodexOAuth` class
Mirrors `AnthropicOAuth`:

- `__init__(access_token, refresh_token, expires_at)` — same pattern
- `_load_tokens()` — check `CODEX_AUTH_TOKEN` env, then `TOKEN_FILE`
- `save_tokens()` — persist to file with `0o600` permissions
- `is_available` property — check if tokens exist
- `ensure_access()` — refresh token if expired (60s buffer)
- `_get_client()` — shared `httpx.AsyncClient`
- `_build_headers()` — Codex-specific headers:
  ```python
  {
      "authorization": f"Bearer {self.access_token}",
      "openai-beta": "responses=experimental",
      "originator": "codex_cli_rs",
      "chatgpt-account-id": self._account_id,  # from JWT
      "content-type": "application/json",
  }
  ```
- `_extract_account_id(token)` — decode JWT middle segment, extract
  `https://api.openai.com/auth` claim for account ID
- `_normalize_messages(messages)` — transform litellm Chat Completions
  format into Responses API input items:
  - system messages -> `instructions` parameter
  - user/assistant messages -> input items with appropriate types
  - tool calls/results -> function_call / function_call_output items
  - Strip message IDs (Codex backend rejects them with `store:false`)
- `_normalize_tools(tools)` — transform litellm tool schemas to
  Responses API function format + name mapping
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

- `call_responses(messages, tools, model, max_tokens)` — main API call:
  1. `await self.ensure_access()`
  2. Normalize model, messages, tools
     - **Guard against empty input** after normalization (mirror upstream fix `e210524`)
  3. Build request body with `store: false`, `stream: false`
  4. POST to `{CODEX_BASE_URL}{CODEX_RESPONSES_PATH}`
  5. Parse and return litellm-compatible response
- `close()` — close httpx client

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

**Mirrors `oauth_login.py` exactly.** Provides `run_codex_login()`:

- `_generate_pkce()` — reuse same PKCE implementation (or import from
  `oauth_login.py` if we extract to shared util)
- `_build_authorize_url(verifier, challenge)` — OpenAI OAuth authorize URL
  with PKCE params
- `_exchange_code(code, verifier)` — POST to OpenAI token endpoint
- `run_codex_login()` — interactive CLI flow:
  1. Generate PKCE pair
  2. Open browser to authorize URL
  3. User pastes authorization code
  4. Exchange code for tokens
  5. Save to `~/.lethe/codex_tokens.json`

### 3. MODIFY: `src/lethe/memory/llm.py` — Route Codex Through OAuth (apply upstream decisions)

Follow the same pattern as the upstream Anthropic OAuth fixes, especially:
- `14dad51`: **OAuth takes priority over API key**, and log which auth mode is used.
- `209791b`: **Route ALL calls through OAuth when active** (including no-tools calls).

Implementation details:

- **Import**: `from lethe.memory.codex_oauth import CodexOAuth, is_codex_oauth_available`

- **`AsyncLLMClient.__init__`**:

  Add a codex OAuth client when configured.

  Auth precedence rule (match upstream):
  - If Codex OAuth tokens are present → **use Codex OAuth even if `OPENAI_API_KEY` is also set**.
  - Else fall back to standard OpenAI API key (litellm) path.

  Example logic:
  ```python
  # Codex OAuth (ChatGPT subscription — bypasses litellm)
  self._codex_oauth: Optional[CodexOAuth] = None
  if self.config.provider == "openai" and is_codex_oauth_available():
      self._codex_oauth = CodexOAuth()
      has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
      if has_api_key:
          logger.info("Auth: Codex OAuth token AND OPENAI_API_KEY both present — using OAuth (subscription)")
      else:
          logger.info("Auth: using Codex OAuth token (ChatGPT subscription)")
  elif self.config.provider == "openai":
      logger.info("Auth: using OpenAI API key")
  ```

- **Routing:**
  - `_call_api()` must route through Codex OAuth whenever `self._codex_oauth` is set.
  - `_call_api_no_tools()` must ALSO route through Codex OAuth whenever `self._codex_oauth` is set.

  Do **not** mix modes within a single session.

- **New `_call_api_codex()` method**:
  Same structure as `_call_with_retry_oauth()` in upstream:
  - delegates to `self._codex_oauth.call_responses(...)`
  - debug logging
  - retries for 429/5xx
  - tracks usage

- **`LLMConfig.__post_init__()` / config validation**:
  When Codex OAuth is available and provider is `"openai"`, skip the OpenAI API key requirement check.
  (Match upstream: OAuth is a complete alternative auth mode.)

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

Keep the user-facing install option as `codex`, but apply upstream auth precedence (`14dad51`) and avoid ambiguous mixed modes.

- Add `"codex"` to PROVIDERS array:
  ```bash
  ["codex"]="OpenAI Codex (ChatGPT subscription, no API key - uses OAuth)"
  ```
- Add to PROVIDER_MODELS / PROVIDER_MODELS_AUX:
  ```bash
  ["codex"]="gpt-5.2" / "codex-mini-latest"  # validate actual IDs during spike
  ```
- Add to PROVIDER_KEYS with empty value (no API key needed)
- Skip `prompt_api_key()` when `SELECTED_PROVIDER=codex`
- After install, print instructions: `Run 'uv run lethe codex-login' to authenticate`
- **Config suggestion:** either:
  - set `LLM_PROVIDER=codex` (preferred; explicit mode), OR
  - set `LLM_PROVIDER=openai` but ensure runtime selection prefers OAuth tokens when present.

### 6. MODIFY: `.env.example` — Document Codex Option

Add comments:
```bash
# For ChatGPT Plus/Pro subscription (Codex), no API key needed:
# 1. Run: uv run lethe codex-login
# 2. Set: LLM_PROVIDER=openai (tokens auto-detected from ~/.lethe/codex_tokens.json)
# Or set CODEX_AUTH_TOKEN=<token> for access-token-only mode
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

### How does the Codex provider interact with the existing `"openai"` provider?
Apply upstream precedence decision (`14dad51`): **OAuth (subscription) takes priority over API key**.

- `LLM_PROVIDER=openai` with `CODEX_AUTH_TOKEN` or `codex_tokens.json` → Codex OAuth path (bypasses litellm)
- `LLM_PROVIDER=openai` with only `OPENAI_API_KEY` set → standard litellm path (unchanged)
- If both OAuth tokens and `OPENAI_API_KEY` are present → **use OAuth** and log which auth mode is used

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
- Encrypted reasoning content returned by model should be included in subsequent turns

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
| ChatGPT backend rate limits | Map 429s appropriately; respect usage limits of Plus/Pro tier |
| JWT parsing edge cases | Standard base64url decode with padding fix; no external library needed |
| Tool name mapping incomplete | Start with core tools (bash/file/search); log unmapped names for iterative expansion |
| Encrypted reasoning content | Pass through `reasoning.encrypted_content` in subsequent turns (follow Codex CLI pattern) |
| Port conflict on callback server | Use code-paste flow (no local server needed, same as upstream `oauth_login.py`) |
| store:false context growth | Already handled by Lethe's `ContextWindow` compaction — messages are managed locally |
