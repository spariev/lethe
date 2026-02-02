"""Hippocampus - Autoassociative memory retrieval.

Inspired by the biological hippocampus that consolidates and retrieves memories.
On each user message, searches archival and conversation history for relevant context.
Summarizes retrieved memories to compress context while preserving reference data.
"""

import logging
from typing import Optional, Callable, Awaitable
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Max lines of recalled memories before summarization
MAX_RECALL_LINES = 50

# Minimum score threshold for including memories
MIN_SCORE_THRESHOLD = 0.3

# Summarization prompt - preserves reference data
SUMMARIZE_PROMPT = """Summarize these recalled memories concisely for context. 

CRITICAL: Preserve ALL of the following exactly as-is (do not paraphrase or omit):
- URLs, links, file paths
- Credentials, API keys, tokens
- IDs, reference numbers
- Dates and times
- Names of people, projects, tools
- Code snippets, commands
- Specific numbers and measurements

Strip out filler, redundancy, and conversational fluff. Keep facts dense.

Memories to summarize:
{memories}

Summary (preserve all reference data):"""


class Hippocampus:
    """Retrieves and summarizes relevant memories to augment user messages.
    
    Uses the conversation context + new message as a search query,
    returning relevant archival memories and past conversations.
    Optionally summarizes to compress context.
    """
    
    def __init__(
        self, 
        memory_store, 
        summarizer: Optional[Callable[[str], Awaitable[str]]] = None,
        enabled: bool = True,
    ):
        """Initialize hippocampus.
        
        Args:
            memory_store: MemoryStore instance with archival and messages
            summarizer: Optional async function to summarize memories
            enabled: Whether to enable memory recall
        """
        self.memory = memory_store
        self.summarizer = summarizer
        self.enabled = enabled
        logger.info(f"Hippocampus initialized (enabled={enabled}, summarizer={summarizer is not None})")
    
    async def recall(
        self,
        message: str,
        recent_messages: Optional[list[dict]] = None,
        max_lines: int = MAX_RECALL_LINES,
    ) -> Optional[str]:
        """Recall relevant memories for a user message.
        
        Args:
            message: The new user message
            recent_messages: Recent conversation context (optional)
            max_lines: Maximum lines of memories before summarization
            
        Returns:
            Formatted (and optionally summarized) memory recall string
        """
        if not self.enabled:
            return None
        
        # Build search query from message + recent context
        query = self._build_query(message, recent_messages)
        
        # Search archival memory
        archival_results = self._search_archival(query)
        
        # Search conversation history
        conversation_results = self._search_conversations(query, exclude_recent=5)
        
        # Combine and format results
        memories = self._format_memories(archival_results, conversation_results, max_lines)
        
        if not memories:
            return None
        
        # Summarize if we have a summarizer, otherwise wrap in recall block
        if self.summarizer:
            return await self._summarize(memories)
        else:
            return (
                "[Associative memory recall]\n"
                + memories
                + "\n[End of recall]"
            )
    
    def _build_query(
        self,
        message: str,
        recent_messages: Optional[list[dict]] = None,
    ) -> str:
        """Build search query from message and recent context.
        
        Uses the new message as primary query, with recent context for
        semantic enrichment.
        """
        # Primary query is the user message
        query_parts = [message]
        
        # Add recent user messages for context (last 3)
        if recent_messages:
            recent_user = [
                m["content"][:200] 
                for m in recent_messages[-5:]
                if m.get("role") == "user"
            ][-3:]
            query_parts.extend(recent_user)
        
        return " ".join(query_parts)
    
    def _search_archival(self, query: str, limit: int = 5) -> list[dict]:
        """Search archival memory."""
        try:
            results = self.memory.archival.search(
                query,
                limit=limit,
                search_type="hybrid"
            )
            # Filter by score threshold
            return [r for r in results if r.get("score", 0) >= MIN_SCORE_THRESHOLD]
        except Exception as e:
            logger.warning(f"Archival search failed: {e}")
            return []
    
    def _search_conversations(
        self,
        query: str,
        limit: int = 5,
        exclude_recent: int = 5,
    ) -> list[dict]:
        """Search conversation history, excluding very recent messages."""
        try:
            results = self.memory.messages.search(query, limit=limit + exclude_recent)
            # Skip the most recent messages (they're already in context)
            return results[exclude_recent:] if len(results) > exclude_recent else []
        except Exception as e:
            logger.warning(f"Conversation search failed: {e}")
            return []
    
    def _format_memories(
        self,
        archival: list[dict],
        conversations: list[dict],
        max_lines: int,
    ) -> Optional[str]:
        """Format retrieved memories into a context block."""
        if not archival and not conversations:
            return None
        
        sections = []
        total_lines = 0
        
        # Format archival memories
        if archival:
            archival_lines = []
            for mem in archival:
                if total_lines >= max_lines:
                    break
                    
                text = mem.get("text", "")
                created = mem.get("created_at", "")[:10]  # Just date
                
                archival_lines.append(f"- [{created}] {text}")
                total_lines += 1
            
            if archival_lines:
                sections.append("**From long-term memory:**\n" + "\n".join(archival_lines))
        
        # Format conversation memories
        if conversations and total_lines < max_lines:
            conv_lines = []
            for msg in conversations:
                if total_lines >= max_lines:
                    break
                    
                role = msg.get("role", "?")
                content = msg.get("content", "")
                created = msg.get("created_at", "")[:16].replace("T", " ")
                
                conv_lines.append(f"- [{created}] {role}: {content}")
                total_lines += 1
            
            if conv_lines:
                sections.append("**From past conversations:**\n" + "\n".join(conv_lines))
        
        if not sections:
            return None
        
        return "\n\n".join(sections)
    
    async def _summarize(self, memories: str) -> str:
        """Summarize memories using the configured summarizer."""
        try:
            prompt = SUMMARIZE_PROMPT.format(memories=memories)
            summary = await self.summarizer(prompt)
            
            if summary:
                logger.info(f"Summarized {len(memories)} -> {len(summary)} chars")
                return (
                    "[Associative memory recall (summarized)]\n"
                    + summary.strip()
                    + "\n[End of recall]"
                )
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
        
        # Fallback to unsummarized
        return (
            "[Associative memory recall]\n"
            + memories
            + "\n[End of recall]"
        )
    
    async def augment_message(
        self,
        message: str,
        recent_messages: Optional[list[dict]] = None,
    ) -> str:
        """Augment a user message with recalled memories.
        
        Args:
            message: The user message
            recent_messages: Recent conversation context
            
        Returns:
            Original message, possibly with appended memory context
        """
        recall = await self.recall(message, recent_messages)
        
        if recall:
            logger.info(f"Hippocampus recalled {len(recall)} chars of context")
            return f"{message}\n\n{recall}"
        
        return message
