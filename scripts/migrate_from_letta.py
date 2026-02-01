#!/usr/bin/env python3
"""Migrate memory from Letta Cloud to local LanceDB storage.

Extracts:
- Memory blocks (persona, human, project, etc.)
- Archival memory (passages)
- Message history

Usage:
    python scripts/migrate_from_letta.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lethe.memory import MemoryStore


# Letta API config
LETTA_BASE_URL = os.environ.get("LETTA_BASE_URL", "https://api.letta.com")
LETTA_API_KEY = os.environ.get("LETTA_API_KEY")
AGENT_NAME = os.environ.get("LETHE_AGENT_NAME", "lethe")

# Local storage
LOCAL_MEMORY_DIR = os.environ.get("LETHE_MEMORY_DIR", "./data/memory")


class LettaClient:
    """Simple Letta API client for migration."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.client = httpx.Client(
            base_url=self.base_url,
            headers=self.headers,
            timeout=60.0,
        )
    
    def get_agents(self) -> list:
        """List all agents."""
        resp = self.client.get("/v1/agents")
        resp.raise_for_status()
        return resp.json()
    
    def get_agent_by_name(self, name: str) -> dict | None:
        """Find agent by name."""
        agents = self.get_agents()
        for agent in agents:
            if agent.get("name") == name:
                return agent
        return None
    
    def get_agent_memory(self, agent_id: str) -> dict:
        """Get agent's core memory."""
        resp = self.client.get(f"/v1/agents/{agent_id}/memory")
        resp.raise_for_status()
        return resp.json()
    
    def get_agent_blocks(self, agent_id: str) -> list:
        """Get agent's memory blocks from agent detail."""
        resp = self.client.get(f"/v1/agents/{agent_id}")
        resp.raise_for_status()
        data = resp.json()
        memory = data.get("memory", {})
        if isinstance(memory, dict):
            return memory.get("blocks", [])
        return []
    
    def get_agent_passages(self, agent_id: str, limit: int = 1000) -> list:
        """Get agent's archival passages."""
        all_passages = []
        cursor = None
        
        while True:
            params = {"limit": min(limit, 100)}
            if cursor:
                params["cursor"] = cursor
            
            resp = self.client.get(f"/v1/agents/{agent_id}/archival-memory", params=params)
            resp.raise_for_status()
            data = resp.json()
            
            passages = data if isinstance(data, list) else data.get("passages", [])
            all_passages.extend(passages)
            
            # Check for pagination
            if isinstance(data, dict) and data.get("cursor"):
                cursor = data["cursor"]
            else:
                break
            
            if len(all_passages) >= limit:
                break
        
        return all_passages[:limit]
    
    def get_agent_messages(self, agent_id: str, limit: int = 10000) -> list:
        """Get agent's message history."""
        all_messages = []
        cursor = None
        
        while True:
            params = {"limit": min(limit, 100)}
            if cursor:
                params["cursor"] = cursor
            
            resp = self.client.get(f"/v1/agents/{agent_id}/messages", params=params)
            resp.raise_for_status()
            data = resp.json()
            
            messages = data if isinstance(data, list) else data.get("messages", [])
            all_messages.extend(messages)
            
            if isinstance(data, dict) and data.get("cursor"):
                cursor = data["cursor"]
            else:
                break
            
            if len(all_messages) >= limit:
                break
        
        return all_messages[:limit]
    
    def close(self):
        self.client.close()


def extract_text_from_message(msg: dict) -> str:
    """Extract text content from a Letta message."""
    # Handle different message formats
    content = msg.get("content", "")
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif "text" in item:
                    texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts)
    
    return str(content) if content else ""


def migrate():
    """Run the migration."""
    if not LETTA_API_KEY:
        print("Error: LETTA_API_KEY not set")
        sys.exit(1)
    
    print(f"Connecting to Letta at {LETTA_BASE_URL}...")
    client = LettaClient(LETTA_BASE_URL, LETTA_API_KEY)
    
    # Find agent
    print(f"Looking for agent '{AGENT_NAME}'...")
    agent = client.get_agent_by_name(AGENT_NAME)
    
    if not agent:
        print(f"Error: Agent '{AGENT_NAME}' not found")
        print("Available agents:")
        for a in client.get_agents():
            print(f"  - {a.get('name')} (id: {a.get('id')})")
        sys.exit(1)
    
    agent_id = agent["id"]
    print(f"Found agent: {agent['name']} (id: {agent_id})")
    
    # Initialize local storage
    print(f"\nInitializing local storage at {LOCAL_MEMORY_DIR}...")
    Path(LOCAL_MEMORY_DIR).mkdir(parents=True, exist_ok=True)
    store = MemoryStore(data_dir=LOCAL_MEMORY_DIR)
    
    # Migrate memory blocks
    print("\n=== Migrating Memory Blocks ===")
    try:
        blocks = client.get_agent_blocks(agent_id)
        print(f"Found {len(blocks)} blocks")
        
        for block in blocks:
            label = block.get("label", "unknown")
            value = block.get("value", "")
            description = block.get("description", "")
            
            # Check if block exists
            existing = store.blocks.get_by_label(label)
            if existing:
                store.blocks.update(label, value=value, description=description)
                print(f"  Updated: {label} ({len(value)} chars)")
            else:
                store.blocks.create(label=label, value=value, description=description)
                print(f"  Created: {label} ({len(value)} chars)")
    except Exception as e:
        print(f"  Error migrating blocks: {e}")
    
    # Migrate archival memory
    print("\n=== Migrating Archival Memory ===")
    try:
        passages = client.get_agent_passages(agent_id)
        print(f"Found {len(passages)} passages")
        
        migrated = 0
        for passage in passages:
            text = passage.get("text", "")
            if not text:
                continue
            
            # Extract metadata
            metadata = {}
            if passage.get("created_at"):
                metadata["source_created_at"] = passage["created_at"]
            if passage.get("id"):
                metadata["letta_id"] = passage["id"]
            
            store.archival.add(text, metadata=metadata)
            migrated += 1
        
        print(f"  Migrated {migrated} passages")
    except Exception as e:
        print(f"  Error migrating archival: {e}")
    
    # Migrate message history
    print("\n=== Migrating Message History ===")
    try:
        messages = client.get_agent_messages(agent_id)
        print(f"Found {len(messages)} messages")
        
        migrated = 0
        skipped = 0
        
        for msg in messages:
            # Letta uses message_type field
            msg_type = msg.get("message_type", msg.get("role", ""))
            
            # Map Letta message types to standard roles
            role_mapping = {
                "user_message": "user",
                "assistant_message": "assistant",
                "system_message": "system",
                "tool_call_message": "assistant",
                "tool_return_message": "tool",
                # Legacy format
                "user": "user",
                "assistant": "assistant",
                "system": "system",
                "tool": "tool",
            }
            
            role = role_mapping.get(msg_type)
            if not role:
                skipped += 1
                continue
            
            content = extract_text_from_message(msg)
            if not content:
                skipped += 1
                continue
            
            metadata = {}
            if msg.get("created_at"):
                metadata["source_created_at"] = msg["created_at"]
            if msg.get("id"):
                metadata["letta_id"] = msg["id"]
            
            store.messages.add(role, content, metadata=metadata)
            migrated += 1
        
        print(f"  Migrated {migrated} messages, skipped {skipped}")
    except Exception as e:
        print(f"  Error migrating messages: {e}")
    
    # Summary
    print("\n=== Migration Complete ===")
    print(f"Memory blocks: {len(store.blocks.list_blocks())}")
    print(f"Archival memories: {store.archival.count()}")
    print(f"Message history: {store.messages.count()}")
    
    client.close()
    print("\nDone!")


if __name__ == "__main__":
    # Load .env if present
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        print(f"Loading {env_path}...")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key, value)
    
    migrate()
