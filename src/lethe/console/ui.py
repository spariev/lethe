"""NiceGUI-based console UI."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from nicegui import ui, app

from . import get_state, ConsoleState

logger = logging.getLogger(__name__)

# Refresh interval in seconds
REFRESH_INTERVAL = 1.0


class ConsoleUI:
    """Mind state visualization console."""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the UI layout."""
        
        @ui.page("/")
        async def main_page():
            state = get_state()
            
            # Dark theme
            ui.dark_mode().enable()
            
            # Header
            with ui.header().classes("bg-primary"):
                ui.label("Lethe Console").classes("text-h5 text-white")
                ui.space()
                self.status_label = ui.label().classes("text-white")
            
            # Main content
            with ui.row().classes("w-full h-full"):
                # Left panel: Chat messages
                with ui.column().classes("w-1/2 p-4"):
                    ui.label("💬 Messages").classes("text-h6")
                    self.messages_container = ui.column().classes(
                        "w-full h-96 overflow-auto bg-gray-900 rounded p-2"
                    )
                
                # Right panel: Memory blocks
                with ui.column().classes("w-1/2 p-4"):
                    ui.label("🧠 Memory Blocks").classes("text-h6")
                    self.blocks_container = ui.column().classes(
                        "w-full h-96 overflow-auto bg-gray-900 rounded p-2"
                    )
            
            # Bottom panel: Context sent to LLM
            with ui.row().classes("w-full p-4"):
                with ui.column().classes("w-full"):
                    with ui.row().classes("items-center"):
                        ui.label("📤 Context Sent to LLM").classes("text-h6")
                        self.context_info = ui.label().classes("text-caption ml-4")
                    
                    with ui.expansion("View full context", icon="code").classes("w-full"):
                        self.context_display = ui.code(language="json").classes(
                            "w-full max-h-96 overflow-auto"
                        )
            
            # Start refresh timer
            ui.timer(REFRESH_INTERVAL, self._refresh_ui)
            
            # Initial refresh
            await self._refresh_ui()
        
        @ui.page("/api/state")
        async def api_state():
            """API endpoint for state (for external tools)."""
            state = get_state()
            return {
                "status": state.status,
                "current_tool": state.current_tool,
                "memory_blocks": list(state.memory_blocks.keys()),
                "message_count": len(state.messages),
                "last_context_tokens": state.last_context_tokens,
            }
    
    async def _refresh_ui(self):
        """Refresh UI with current state."""
        state = get_state()
        
        # Update status
        status_text = f"Status: {state.status}"
        if state.current_tool:
            status_text += f" ({state.current_tool})"
        status_text += f" | Messages: {state.total_messages} | Archival: {state.archival_count}"
        self.status_label.text = status_text
        
        # Update messages
        self.messages_container.clear()
        with self.messages_container:
            for msg in state.messages[-50:]:  # Last 50 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                # Truncate long content
                if isinstance(content, str) and len(content) > 500:
                    content = content[:500] + "..."
                elif isinstance(content, list):
                    content = "[multipart content]"
                
                role_colors = {
                    "user": "text-blue-400",
                    "assistant": "text-green-400",
                    "tool": "text-yellow-400",
                    "system": "text-gray-400",
                }
                color = role_colors.get(role, "text-white")
                
                with ui.row().classes("w-full mb-2"):
                    ui.label(f"[{role}]").classes(f"{color} font-bold mr-2")
                    ui.label(str(content)[:200]).classes("text-sm")
        
        # Update memory blocks
        self.blocks_container.clear()
        with self.blocks_container:
            # Identity block
            if state.identity:
                with ui.expansion("identity", icon="person").classes("w-full"):
                    ui.code(state.identity[:1000], language="markdown").classes("text-xs")
            
            # Summary
            if state.summary:
                with ui.expansion("summary", icon="summarize").classes("w-full"):
                    ui.code(state.summary[:1000], language="markdown").classes("text-xs")
            
            # Other blocks
            for label, block in state.memory_blocks.items():
                value = block.get("value", "")
                chars = len(value)
                limit = block.get("limit", 20000)
                
                icon = "memory"
                if "persona" in label:
                    icon = "psychology"
                elif "human" in label:
                    icon = "person"
                elif "project" in label:
                    icon = "folder"
                elif "task" in label:
                    icon = "task"
                elif "tool" in label:
                    icon = "build"
                
                with ui.expansion(
                    f"{label} ({chars}/{limit} chars)", 
                    icon=icon
                ).classes("w-full"):
                    ui.code(value[:2000], language="markdown").classes("text-xs")
        
        # Update context display
        if state.last_context_time:
            time_str = state.last_context_time.strftime("%H:%M:%S")
            self.context_info.text = f"{state.last_context_tokens} tokens @ {time_str}"
        
        if state.last_context:
            # Format context for display
            formatted = []
            for msg in state.last_context:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                # Truncate for display
                if isinstance(content, str) and len(content) > 1000:
                    content = content[:1000] + "...[truncated]"
                elif isinstance(content, list):
                    # System message with content blocks
                    content = [
                        {**c, "text": c.get("text", "")[:500] + "..." if len(c.get("text", "")) > 500 else c.get("text", "")}
                        if c.get("type") == "text" else c
                        for c in content
                    ]
                
                formatted.append({"role": role, "content": content})
            
            self.context_display.content = json.dumps(formatted, indent=2, default=str)
    
    def run(self):
        """Run the console server."""
        logger.info(f"Starting Lethe Console on port {self.port}")
        ui.run(
            port=self.port,
            title="Lethe Console",
            favicon="🧠",
            show=False,  # Don't open browser
            reload=False,
        )


async def run_console(port: int = 8080):
    """Run console in background."""
    console = ConsoleUI(port=port)
    # NiceGUI runs its own event loop, so we need to run it in a thread
    import threading
    thread = threading.Thread(target=console.run, daemon=True)
    thread.start()
    logger.info(f"Lethe Console started on http://localhost:{port}")
