"""Heartbeat - Periodic check-ins for proactive agent behavior.

Sends periodic internal messages to the agent, allowing it to:
- Review pending tasks and reminders
- Surface important information proactively
- Maintain continuity even when user is away
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Awaitable, Optional

logger = logging.getLogger(__name__)

# Default interval in seconds (15 minutes)
DEFAULT_HEARTBEAT_INTERVAL = 15 * 60

HEARTBEAT_MESSAGE = """[System Heartbeat - {timestamp}]

Periodic check-in. Review your memory blocks (persona, human, project) for pending tasks.

IMPORTANT: Do NOT use tools during heartbeat. Just review what you already know from memory.

Respond in one of two ways:
1. **"ok"** - Nothing urgent right now.
2. **Brief message** - ONLY if there's something immediately actionable:
   - A task/reminder that is DUE NOW
   - Time-sensitive information requiring immediate action

Do NOT: test tools, verify capabilities, do exploratory actions, or send "status updates".
Think: "Would I text my boss about this right now?" If not urgent, respond "ok".
"""


class Heartbeat:
    """Manages periodic heartbeat check-ins."""
    
    def __init__(
        self,
        process_callback: Callable[[str], Awaitable[Optional[str]]],
        send_callback: Callable[[str], Awaitable[None]],
        interval: int = DEFAULT_HEARTBEAT_INTERVAL,
        enabled: bool = True,
    ):
        """Initialize heartbeat.
        
        Args:
            process_callback: Async function to process heartbeat message through agent
                             Returns response string or None
            send_callback: Async function to send response to user (e.g., Telegram)
            interval: Seconds between heartbeats
            enabled: Whether heartbeats are enabled
        """
        self.process_callback = process_callback
        self.send_callback = send_callback
        self.interval = interval
        self.enabled = enabled
        self._task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"Heartbeat initialized (interval={interval}s, enabled={enabled})")
    
    async def start(self):
        """Start the heartbeat loop."""
        if not self.enabled:
            logger.info("Heartbeat disabled, not starting")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Heartbeat started")
    
    async def stop(self):
        """Stop the heartbeat loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Heartbeat stopped")
    
    async def _heartbeat_loop(self):
        """Main heartbeat loop."""
        # Initial heartbeat on startup
        await self._send_heartbeat()
        
        while self._running:
            try:
                # Wait for interval
                await asyncio.sleep(self.interval)
                
                if not self._running:
                    break
                
                # Send heartbeat
                await self._send_heartbeat()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Heartbeat error: {e}")
                # Continue running despite errors
                await asyncio.sleep(60)  # Brief pause before retry
    
    async def _send_heartbeat(self):
        """Send a heartbeat message to the agent."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        message = HEARTBEAT_MESSAGE.format(timestamp=timestamp)
        
        logger.info(f"Sending heartbeat at {timestamp}")
        
        try:
            # Process through agent
            response = await self.process_callback(message)
            
            # Response is already saved to history via agent.chat()
            if response and response.strip():
                # "ok" means nothing urgent - thoughts saved to history but not sent
                if response.strip().lower() == "ok":
                    logger.debug("Heartbeat: nothing urgent (saved to history)")
                else:
                    logger.info(f"Heartbeat sending: {response[:100]}...")
                    await self.send_callback(response)
            else:
                logger.debug("No heartbeat response")
                
        except Exception as e:
            logger.exception(f"Heartbeat processing failed: {e}")
    
    async def trigger(self):
        """Manually trigger a heartbeat (for testing)."""
        await self._send_heartbeat()
