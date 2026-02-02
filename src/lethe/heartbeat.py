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

Periodic check-in. Review your memory (human block, project block, tasks) and consider:
- What are the user's current goals? Is there anything you can proactively help with?
- Any pending tasks or reminders that are due?
- Any follow-ups from recent conversations worth surfacing?
- Any time-sensitive information the user should know?

Be proactive about advancing the user's goals - but don't be annoying. If there's genuinely nothing useful to say, respond with just "ok" and nothing will be sent. Only message the user if you have something valuable to tell them.
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
            
            # If agent has something to say, send it
            if response and response.strip():
                # "ok" means nothing to report
                if response.strip().lower() == "ok":
                    logger.debug("Heartbeat: nothing to report")
                else:
                    logger.info(f"Heartbeat response: {response[:100]}...")
                    await self.send_callback(response)
            else:
                logger.debug("No heartbeat response")
                
        except Exception as e:
            logger.exception(f"Heartbeat processing failed: {e}")
    
    async def trigger(self):
        """Manually trigger a heartbeat (for testing)."""
        await self._send_heartbeat()
