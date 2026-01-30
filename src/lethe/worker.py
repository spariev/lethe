"""Worker that processes conversations with interrupt support."""

import asyncio
import logging
from typing import Optional, TYPE_CHECKING

from aiogram.utils.chat_action import ChatActionSender

from lethe.agent import AgentManager
from lethe.config import Settings, get_settings
from lethe.conversation import ConversationManager
from lethe.telegram import TelegramBot
from lethe.tools.telegram_tools import set_telegram_context, clear_telegram_context
from lethe.tasks.tools import set_task_context, clear_task_context

if TYPE_CHECKING:
    from lethe.tasks import TaskManager

logger = logging.getLogger(__name__)


class Worker:
    """Processes conversations using the Letta agent with interrupt support.
    
    Unlike the old queue-based worker, this supports:
    - Interrupting current processing when new messages arrive
    - Combining multiple messages sent during processing
    - More conversational, less transactional
    """

    def __init__(
        self,
        conversation_manager: ConversationManager,
        agent_manager: AgentManager,
        telegram_bot: TelegramBot,
        task_manager: Optional["TaskManager"] = None,
        settings: Optional[Settings] = None,
    ):
        self.conversation_manager = conversation_manager
        self.agent_manager = agent_manager
        self.telegram_bot = telegram_bot
        self.task_manager = task_manager
        self.settings = settings or get_settings()
        self._running = False

    async def start(self):
        """Start the worker (initialize agent)."""
        self._running = True
        logger.info("Worker started")
        
        # Ensure agent is initialized
        await self.agent_manager.get_or_create_agent()

    async def stop(self):
        """Stop the worker."""
        self._running = False
        logger.info("Worker stopped")

    async def process_message(
        self,
        chat_id: int,
        user_id: int,
        message: str,
        metadata: Optional[dict] = None,
        interrupt_check: Optional[callable] = None,
    ):
        """Process a message (or combined messages) from a conversation.
        
        This is called by ConversationManager when ready to process.
        
        Args:
            chat_id: Telegram chat ID
            user_id: Telegram user ID
            message: Message content (may be combined from multiple messages)
            metadata: Metadata like username, attachments
            interrupt_check: Callable that returns True if interrupted
        """
        try:
            # Set contexts for tools
            set_telegram_context(self.telegram_bot.bot, chat_id)
            
            if self.task_manager:
                set_task_context(
                    self.task_manager,
                    telegram_bot=self.telegram_bot.bot,
                    chat_id=chat_id,
                )
            
            # Track messages sent
            messages_sent = []
            
            async def on_message(content: str):
                """Callback for streaming messages."""
                messages_sent.append(content)
                await self.telegram_bot.send_message(
                    chat_id=chat_id,
                    text=content,
                )
            
            # Build context
            context = metadata.copy() if metadata else {}
            context["_interrupt_check"] = interrupt_check
            context["_original_request"] = message
            
            # Show typing indicator while processing
            async with ChatActionSender.typing(
                bot=self.telegram_bot.bot,
                chat_id=chat_id,
                interval=4.0,
            ):
                response = await self.agent_manager.send_message(
                    message=message,
                    context=context,
                    on_message=on_message,
                )
            
            # Handle interrupt
            if response == "[INTERRUPTED]":
                logger.info(f"Chat {chat_id}: Processing interrupted, new messages pending")
                # Don't send anything - will process combined messages next
                return
            
            # Send final response if no messages were streamed
            if not messages_sent and response:
                await self.telegram_bot.send_message(
                    chat_id=chat_id,
                    text=response,
                )
            
            logger.info(f"Chat {chat_id}: Completed processing")
            
        except asyncio.CancelledError:
            logger.info(f"Chat {chat_id}: Processing cancelled")
            raise
        except Exception as e:
            logger.exception(f"Chat {chat_id}: Processing failed: {e}")
            await self.telegram_bot.send_message(
                chat_id=chat_id,
                text=f"❌ Error: {e}",
            )
        finally:
            clear_telegram_context()
            clear_task_context()

    def is_processing(self, chat_id: int) -> bool:
        """Check if a chat is currently being processed."""
        return self.conversation_manager.is_processing(chat_id)


class HeartbeatWorker:
    """Sends periodic heartbeat messages to the agent."""

    def __init__(
        self,
        agent_manager: AgentManager,
        telegram_bot: TelegramBot,
        chat_id: int,
        interval_minutes: int = 15,
        identity_refresh_hours: int = 2,
        enabled: bool = True,
    ):
        self.agent_manager = agent_manager
        self.telegram_bot = telegram_bot
        self.chat_id = chat_id
        self.interval_minutes = interval_minutes
        self.identity_refresh_hours = identity_refresh_hours
        self.enabled = enabled
        self._running = False
        self._heartbeat_count = 0
        self._identity_refresh_interval = (identity_refresh_hours * 60) // interval_minutes

    async def start(self):
        """Start the heartbeat loop."""
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return

        self._running = True
        logger.info(f"Heartbeat started (interval: {self.interval_minutes} min)")

        while self._running:
            try:
                await asyncio.sleep(self.interval_minutes * 60)

                if not self._running:
                    break

                from datetime import datetime
                now = datetime.now()
                date_str = now.strftime("%A, %B %d, %Y")
                time_str = now.strftime("%H:%M")
                
                self._heartbeat_count += 1
                should_refresh_identity = (self._heartbeat_count % self._identity_refresh_interval) == 0
                
                if should_refresh_identity:
                    logger.info(f"Sending heartbeat ({time_str}) + identity refresh...")
                else:
                    logger.info(f"Sending heartbeat ({time_str})...")

                messages_sent = []
                
                async def on_message(content: str):
                    if content and "[NO_NOTIFY]" not in content:
                        messages_sent.append(content)
                        await self.telegram_bot.send_message(
                            chat_id=self.chat_id,
                            text=f"🕐 {content}",
                        )

                identity_instruction = ""
                if should_refresh_identity:
                    identity_instruction = """
IDENTITY REFRESH: It's been 2 hours. Please re-read config/identity.md to refresh your persona and instructions. Use read_file to load it, then update your persona memory block if needed.
"""

                response = await self.agent_manager.send_message(
                    message=f"""[HEARTBEAT]

Current time: {time_str}
Current date: {date_str}
{identity_instruction}
Periodic check-in. Do this:

1. Call `todo_remind_check()` to see if any tasks are due for a reminder
2. If tasks are due: remind the user, then call `todo_reminded(id)` for each
3. Check memory blocks for anything else time-sensitive

IMPORTANT: 
- Only notify if you have something genuinely useful
- After reminding about a todo, ALWAYS call `todo_reminded(id)` to prevent spam
- If nothing to report, respond with just "[NO_NOTIFY]" """,
                    on_message=on_message,
                )

                if not messages_sent:
                    logger.info("Heartbeat: nothing to notify")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Heartbeat error: {e}")

        logger.info("Heartbeat stopped")

    async def stop(self):
        """Stop the heartbeat loop."""
        self._running = False
