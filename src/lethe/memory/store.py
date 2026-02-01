"""Main memory store coordinating all memory subsystems."""

import os
from pathlib import Path
from typing import Optional
import lancedb
import logging

logger = logging.getLogger(__name__)

from lethe.memory.blocks import BlockManager
from lethe.memory.archival import ArchivalMemory
from lethe.memory.messages import MessageHistory


class MemoryStore:
    """Unified memory store using LanceDB.
    
    Provides:
    - blocks: Core memory (persona, human, project, etc.)
    - archival: Long-term semantic memory with hybrid search
    - messages: Conversation history
    """
    
    def __init__(self, data_dir: str = "data/memory"):
        """Initialize memory store.
        
        Args:
            data_dir: Directory for storing memory data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Connect to LanceDB
        self.db = lancedb.connect(str(self.data_dir / "lancedb"))
        logger.info(f"Connected to LanceDB at {self.data_dir / 'lancedb'}")
        
        # Initialize subsystems
        self.blocks = BlockManager(self.db)
        self.archival = ArchivalMemory(self.db)
        self.messages = MessageHistory(self.db)
        
        logger.info("Memory store initialized")
    
    def get_context_for_prompt(self, max_tokens: int = 8000) -> str:
        """Get formatted memory context for LLM prompt.
        
        Matches Letta's context engineering format:
        - Memory blocks with description, metadata, value
        - Memory metadata with timestamps and counts
        
        Args:
            max_tokens: Approximate max tokens for context
            
        Returns:
            Formatted string with all memory blocks
        """
        from datetime import datetime, timezone
        
        sections = []
        
        # Build memory blocks section (Letta-style)
        blocks = self.blocks.list_blocks()
        if blocks:
            block_lines = ["<memory_blocks>"]
            block_lines.append("The following memory blocks are currently engaged in your core memory unit:\n")
            
            for i, block in enumerate(blocks):
                if block.get("hidden"):
                    continue
                    
                label = block["label"]
                value = block["value"] or ""
                description = block.get("description") or ""
                limit = block.get("limit") or 20000
                
                block_lines.append(f"<{label}>")
                block_lines.append("<description>")
                block_lines.append(description)
                block_lines.append("</description>")
                block_lines.append("<metadata>")
                block_lines.append(f"- chars_current={len(value)}")
                block_lines.append(f"- chars_limit={limit}")
                block_lines.append("</metadata>")
                block_lines.append("<warning>")
                block_lines.append("# NOTE: Line numbers shown below (with arrows like '1→') are to help during editing. Do NOT include line number prefixes in your memory edit tool calls.")
                block_lines.append("</warning>")
                block_lines.append("<value>")
                # Add line numbers like Letta does for Anthropic
                for line_num, line in enumerate(value.split("\n"), 1):
                    block_lines.append(f"{line_num}→ {line}")
                block_lines.append("</value>")
                block_lines.append(f"</{label}>")
                
                if i < len(blocks) - 1:
                    block_lines.append("")
            
            block_lines.append("\n</memory_blocks>")
            sections.append("\n".join(block_lines))
        
        # Build memory metadata section
        now = datetime.now(timezone.utc)
        message_count = self.messages.count()
        archival_count = self.archival.count()
        
        # Get last modified time from blocks
        last_modified = now  # Default to now
        for block in blocks:
            if block.get("updated_at"):
                block_time = datetime.fromisoformat(block["updated_at"].replace("Z", "+00:00"))
                if block_time > last_modified:
                    last_modified = block_time
        
        metadata_lines = [
            "<memory_metadata>",
            f"- The current system date is: {now.strftime('%B %d, %Y')}",
            f"- Memory blocks were last modified: {last_modified.strftime('%Y-%m-%d %I:%M:%S %p')} UTC{last_modified.strftime('%z')}",
            f"- {message_count} previous messages between you and the user are stored in recall memory (use tools to access them)",
        ]
        
        if archival_count > 0:
            metadata_lines.append(f"- {archival_count} total memories you created are stored in archival memory (use tools to access them)")
        
        metadata_lines.append("</memory_metadata>")
        sections.append("\n".join(metadata_lines))
        
        return "\n\n".join(sections)
    
    def search(
        self,
        query: str,
        limit: int = 10,
        search_type: str = "hybrid"
    ) -> list[dict]:
        """Search across archival memory.
        
        Args:
            query: Search query
            limit: Max results
            search_type: "hybrid", "vector", or "fts"
            
        Returns:
            List of matching passages
        """
        return self.archival.search(query, limit=limit, search_type=search_type)
    
    def add_memory(self, text: str, metadata: Optional[dict] = None) -> str:
        """Add a memory to archival storage.
        
        Args:
            text: Memory text
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        return self.archival.add(text, metadata=metadata)
    
    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        """Get recent conversation messages.
        
        Args:
            limit: Max messages to return
            
        Returns:
            List of messages
        """
        return self.messages.get_recent(limit=limit)
    
    def add_message(self, role: str, content: str, metadata: Optional[dict] = None) -> str:
        """Add a message to history.
        
        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Message ID
        """
        return self.messages.add(role, content, metadata=metadata)
