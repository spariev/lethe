import json

from lethe.memory import openai_oauth
from lethe.memory.llm import AsyncLLMClient, LLMConfig
from lethe.memory.openai_oauth import OpenAIOAuth, is_oauth_available_openai


def test_is_oauth_available_openai_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_AUTH_TOKEN", "test-openai-oauth-token")
    assert is_oauth_available_openai()


def test_is_oauth_available_openai_from_token_file(monkeypatch, tmp_path):
    token_file = tmp_path / "openai_tokens.json"
    token_file.write_text(json.dumps({"access_token": "token-from-file"}))

    monkeypatch.delenv("OPENAI_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(openai_oauth, "TOKEN_FILE", token_file)

    assert is_oauth_available_openai()


def test_openai_oauth_headers_use_openclaw_identity_profile():
    oauth = OpenAIOAuth(access_token="access-token", account_id="acct_123")
    headers = oauth._build_headers()

    assert headers["authorization"] == "Bearer access-token"
    assert headers["chatgpt-account-id"] == "acct_123"
    assert headers["originator"] == "pi"
    assert headers["OpenAI-Beta"] == "responses=experimental"
    assert headers["user-agent"].startswith("pi (")


def test_openai_oauth_parse_response_maps_tools_and_usage():
    oauth = OpenAIOAuth(access_token="access-token")
    payload = {
        "id": "resp_abc",
        "model": "gpt-5.3-codex",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "hello "},
                    {"type": "output_text", "text": "world"},
                ],
            },
            {
                "type": "function_call",
                "id": "call_123",
                "name": "bash",
                "arguments": "{\"cmd\":\"ls\"}",
            },
        ],
        "stop_reason": "tool_call",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 7,
            "input_tokens_details": {"cached_tokens": 3},
        },
    }

    parsed = oauth._parse_response(payload)

    assert parsed["choices"][0]["message"]["content"] == "hello world"
    assert parsed["choices"][0]["finish_reason"] == "tool_calls"
    assert parsed["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash"
    assert parsed["usage"]["prompt_tokens"] == 12
    assert parsed["usage"]["completion_tokens"] == 7
    assert parsed["usage"]["prompt_tokens_details"]["cached_tokens"] == 3


def test_llm_config_accepts_openai_oauth_without_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_AUTH_TOKEN", "test-openai-oauth-token")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    config = LLMConfig(provider="openai")
    assert config.provider == "openai"


def test_llm_config_auto_detects_openai_oauth_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_AUTH_TOKEN", "test-openai-oauth-token")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    config = LLMConfig()
    assert config.provider == "openai"


def test_async_llm_client_prefers_openai_oauth_over_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_AUTH_TOKEN", "test-openai-oauth-token")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    config = LLMConfig(provider="openai", model="gpt-5.3-codex")
    client = AsyncLLMClient(config=config, system_prompt="test")
    assert isinstance(client._oauth, OpenAIOAuth)
