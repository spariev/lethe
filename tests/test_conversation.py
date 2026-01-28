"""Tests for ConversationManager with interrupt support."""

import asyncio
import pytest
from lethe.conversation import ConversationManager, ConversationState, PendingMessage


class TestPendingMessage:
    """Tests for PendingMessage dataclass."""

    def test_creates_with_defaults(self):
        msg = PendingMessage(content="hello")
        assert msg.content == "hello"
        assert msg.metadata == {}
        assert msg.created_at is not None

    def test_creates_with_metadata(self):
        msg = PendingMessage(content="hello", metadata={"user": "test"})
        assert msg.metadata == {"user": "test"}


class TestConversationState:
    """Tests for ConversationState."""

    def test_add_message_when_not_processing(self):
        state = ConversationState(chat_id=123, user_id=456)
        
        was_interrupted = state.add_message("hello")
        
        assert not was_interrupted
        assert len(state.pending_messages) == 1
        assert state.pending_messages[0].content == "hello"

    def test_add_message_when_processing_triggers_interrupt(self):
        state = ConversationState(chat_id=123, user_id=456)
        state.is_processing = True
        
        was_interrupted = state.add_message("hello")
        
        assert was_interrupted
        assert state.interrupt_event.is_set()

    def test_get_combined_message_single(self):
        state = ConversationState(chat_id=123, user_id=456)
        state.add_message("hello", {"user": "test"})
        
        content, metadata = state.get_combined_message()
        
        assert content == "hello"
        assert metadata == {"user": "test"}
        assert len(state.pending_messages) == 0  # Cleared

    def test_get_combined_message_multiple(self):
        state = ConversationState(chat_id=123, user_id=456)
        state.add_message("first", {"a": 1})
        state.add_message("second", {"b": 2})
        state.add_message("third", {"a": 3})  # Overrides a
        
        content, metadata = state.get_combined_message()
        
        assert "first" in content
        assert "second" in content
        assert "third" in content
        assert "[Additional message while processing:]" in content
        assert metadata == {"a": 3, "b": 2}  # Merged, later overrides
        assert len(state.pending_messages) == 0

    def test_get_combined_message_empty(self):
        state = ConversationState(chat_id=123, user_id=456)
        
        content, metadata = state.get_combined_message()
        
        assert content == ""
        assert metadata == {}

    def test_check_interrupt_clears_event(self):
        state = ConversationState(chat_id=123, user_id=456)
        state.interrupt_event.set()
        
        assert state.check_interrupt() is True
        assert not state.interrupt_event.is_set()
        assert state.check_interrupt() is False


class TestConversationManager:
    """Tests for ConversationManager."""

    def test_get_or_create_state(self):
        manager = ConversationManager()
        
        state1 = manager.get_or_create_state(123, 456)
        state2 = manager.get_or_create_state(123, 456)
        state3 = manager.get_or_create_state(789, 456)
        
        assert state1 is state2  # Same chat
        assert state1 is not state3  # Different chat

    @pytest.mark.asyncio
    async def test_add_message_starts_processing(self):
        manager = ConversationManager()
        processed = []
        
        async def process_callback(chat_id, user_id, message, metadata, interrupt_check):
            processed.append(message)
        
        await manager.add_message(
            chat_id=123,
            user_id=456,
            content="hello",
            process_callback=process_callback,
        )
        
        # Wait for processing to complete
        await asyncio.sleep(0.1)
        
        assert processed == ["hello"]
        assert not manager.is_processing(123)

    @pytest.mark.asyncio
    async def test_add_message_during_processing_interrupts(self):
        manager = ConversationManager()
        processed = []
        interrupt_count = [0]
        
        async def process_callback(chat_id, user_id, message, metadata, interrupt_check):
            processed.append(message)
            # Simulate some work
            for _ in range(10):
                await asyncio.sleep(0.05)
                if interrupt_check():
                    interrupt_count[0] += 1
                    return  # Exit on interrupt
        
        # Start first message
        await manager.add_message(
            chat_id=123,
            user_id=456,
            content="first",
            process_callback=process_callback,
        )
        
        # Wait a bit then send second message
        await asyncio.sleep(0.1)
        await manager.add_message(
            chat_id=123,
            user_id=456,
            content="second",
            process_callback=process_callback,
        )
        
        # Wait for all processing to complete
        await asyncio.sleep(1)
        
        # First message was interrupted, second (or combined) was processed
        assert len(processed) >= 2
        assert interrupt_count[0] >= 1

    @pytest.mark.asyncio
    async def test_cancel_stops_processing(self):
        manager = ConversationManager()
        processed = []
        
        async def process_callback(chat_id, user_id, message, metadata, interrupt_check):
            processed.append("started")
            await asyncio.sleep(10)  # Long operation
            processed.append("finished")  # Should not reach
        
        # Start processing
        await manager.add_message(
            chat_id=123,
            user_id=456,
            content="hello",
            process_callback=process_callback,
        )
        
        await asyncio.sleep(0.1)
        assert manager.is_processing(123)
        
        # Cancel
        cancelled = await manager.cancel(123)
        
        assert cancelled
        await asyncio.sleep(0.1)
        assert not manager.is_processing(123)
        assert "started" in processed
        assert "finished" not in processed

    @pytest.mark.asyncio
    async def test_multiple_rapid_messages_combine(self):
        manager = ConversationManager()
        processed_messages = []
        
        async def process_callback(chat_id, user_id, message, metadata, interrupt_check):
            # Simulate slow processing
            await asyncio.sleep(0.5)
            processed_messages.append(message)
        
        # Send multiple messages rapidly
        await manager.add_message(123, 456, "one", process_callback=process_callback)
        await asyncio.sleep(0.05)
        await manager.add_message(123, 456, "two", process_callback=process_callback)
        await asyncio.sleep(0.05)
        await manager.add_message(123, 456, "three", process_callback=process_callback)
        
        # Wait for all processing
        await asyncio.sleep(2)
        
        # Messages should be combined (exact behavior depends on timing)
        total_content = " ".join(processed_messages)
        assert "one" in total_content or any("one" in m for m in processed_messages)

    def test_is_processing_false_when_not_started(self):
        manager = ConversationManager()
        assert not manager.is_processing(123)

    def test_get_pending_count(self):
        manager = ConversationManager()
        state = manager.get_or_create_state(123, 456)
        state.add_message("one")
        state.add_message("two")
        
        assert manager.get_pending_count(123) == 2
        assert manager.get_pending_count(999) == 0  # Non-existent chat

    @pytest.mark.asyncio
    async def test_error_in_callback_does_not_crash_loop(self):
        manager = ConversationManager()
        call_count = [0]
        
        async def process_callback(chat_id, user_id, message, metadata, interrupt_check):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Intentional error")
            # Second call should succeed
        
        # Add message that will fail
        state = manager.get_or_create_state(123, 456)
        state.add_message("first")
        
        # Manually trigger processing (should handle the error)
        await manager._process_loop(state, process_callback)
        
        # The loop should have been called once and handled the error gracefully
        assert call_count[0] == 1
        assert not state.is_processing  # Loop finished


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
