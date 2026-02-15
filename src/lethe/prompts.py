"""Prompt template loading utilities.

Resolution order:
1. Workspace templates (editable working memory): $WORKSPACE_DIR/prompts/<name>.md
2. Config templates (versioned defaults): $LETHE_CONFIG_DIR/prompts/<name>.md
3. Legacy config workspace templates: $LETHE_CONFIG_DIR/workspace/prompts/<name>.md
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def _workspace_dir() -> Path:
    raw = os.environ.get("WORKSPACE_DIR", "")
    if raw:
        return Path(raw).expanduser()
    return Path("workspace")


def _config_dir() -> Path:
    raw = os.environ.get("LETHE_CONFIG_DIR", "")
    if raw:
        return Path(raw).expanduser()
    return Path("config")


def _candidate_paths(name: str) -> list[Path]:
    base = name if name.endswith(".md") else f"{name}.md"
    w = _workspace_dir() / "prompts" / base
    c = _config_dir()
    return [
        w,
        c / "prompts" / base,
        c / "workspace" / "prompts" / base,
    ]


def load_prompt_template(name: str, fallback: str = "") -> str:
    """Load prompt template text by name (without extension)."""
    for path in _candidate_paths(name):
        try:
            if path.exists():
                text = path.read_text().strip()
                if text:
                    return text
        except Exception:
            continue
    return fallback


def render_prompt_template(name: str, variables: Dict[str, object], fallback: str = "") -> str:
    """Load and format a prompt template using str.format mapping."""
    template = load_prompt_template(name, fallback=fallback)
    if not template:
        return ""
    return template.format(**variables)

