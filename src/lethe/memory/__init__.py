"""Lethe Memory Layer - local memory and LLM utilities.

The package lazily resolves exported symbols so importing ``lethe.memory`` does
not force native LanceDB imports unless those symbols are actually used.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "MemoryStore",
    "BlockManager",
    "ArchivalMemory",
    "MessageHistory",
    "AsyncLLMClient",
    "LLMConfig",
    "Hippocampus",
]

_SYMBOL_TO_MODULE = {
    "MemoryStore": "lethe.memory.store",
    "BlockManager": "lethe.memory.blocks",
    "ArchivalMemory": "lethe.memory.archival",
    "MessageHistory": "lethe.memory.messages",
    "AsyncLLMClient": "lethe.memory.llm",
    "LLMConfig": "lethe.memory.llm",
    "Hippocampus": "lethe.memory.hippocampus",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if not module_name:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
