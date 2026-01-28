"""Tests for ConversationManager with debounce and interrupt support."""

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
        
        interrupted_proc, interrupted_debounce = state.add_message("hello")
        
        assert not interrupted_proc
        assert not interrupted_debounce
        assert len(state.pending_messages) == 1
        assert state.pending_messages[0].content == "hello"

    def test_add_message_when_processing_triggers_interrupt(self):
        state = ConversationState(chat_id=123, user_id=456)
        state.is_processing = True
        
        interrupted_proc, _ = state.add_message("hello")
        
        assert interrupted_proc
        assert state.interrupt_event.is_set()

    def test_add_message_when_debouncing_triggers_debounce_event(self):
        state = ConversationState(chat_id=123, user_id=456)
        state.is_debouncing = True
        
        _, interrupted_debounce = state.add_message("hello")
        
        assert interrupted_debounce
        assert state.debounce_event.is_set()

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


class TestConversationManagerDebounce:
    """Tests for ConversationManager debounce behavior."""

    @pytest.mark.asyncio
    async def test_debounce_waits_before_processing(self):
        manager = ConversationManager(debounce_seconds=0.2)
        processed = []
        process_started_at = []
        
        async def process_callback(chat_id, user_id, message, metadata, interrupt_check):
            process_started_at.append(asyncio.get_event_loop().time())
            processed.append(message)
        
        start_time = asyncio.get_event_loop().time()
        
        await manager.add_message(
            chat_id=123,
            user_id=456,
            content="hello",
            process_callback=process_callback,
        )
        
        # Should be debouncing now, not processed yet
        assert manager.is_debouncing(123)
        assert not manager.is_processing(123)
        assert len(processed) == 0
        
        # Wait for debounce + processing to complete
        await asyncio.sleep(0.5)
        
        assert processed == ["hello"]
        # Processing should have started after debounce period
        assert process_started_at[0] - start_time >= 0.19  # Allow small timing variance

    @pytest.mark.asyncio
    async def test_debounce_resets_on_new_message(self):
        manager = ConversationManager(debounce_seconds=0.2)
        processed = []
        
        async def process_callback(chat_id, user_id, message, metadata, interrupt_check):
            processed.append(message)
        
        # Send first message
        await manager.add_message(123, 456, "first", process_callback=process_callback)
        
        # Wait less than debounce
        await asyncio.sleep(0.1)
        
        # Send second message - should reset debounce
        await manager.add_message(123, 456, "second", process_callback=process_callback)
        
        # Should still be debouncing
        assert manager.is_debouncing(123)
        assert len(processed) == 0
        
        # Wait for debounce to complete
        await asyncio.sleep(0.3)
        
        # Both messages should be combined
        assert len(processed) == 1
        assert "first" in processed[0]
        assert "second" in processed[0]

    @pytest.mark.asyncio
    async def test_multiple_rapid_messages_batch_together(self):
        manager = ConversationManager(debounce_seconds=0.3)
        processed = []
        
        async def process_callback(chat_id, user_id, message, metadata, interrupt_check):
            processed.append(message)
        
        # Send multiple messages rapidly
        await manager.add_message(123, 456, "one", process_callback=process_callback)
        await asyncio.sleep(0.05)
        await manager.add_message(123, 456, "two", process_callback=process_callback)
        await asyncio.sleep(0.05)
        await manager.add_message(123, 456, "three", process_callback=process_callback)
        
        # All should batch
        await asyncio.sleep(0.5)
        
        assert len(processed) == 1
        assert "one" in processed[0]
        assert "two" in processed[0]
        assert "three" in processed[0]

    @pytest.mark.asyncio
    async def test_no_debounce_when_disabled(self):
        manager = ConversationManager(debounce_seconds=0)
        processed = []
        
        async def process_callback(chat_id, user_id, message, metadata, interrupt_check):
            processed.append(message)
        
        await manager.add_message(123, 456, "hello", process_callback=process_callback)
        
        # Should start processing immediately
        assert not manager.is_debouncing(123)
        
        await asyncio.sleep(0.1)
        assert processed == ["hello"]


class TestConversationManagerInterrupt:
    """Tests for ConversationManager interrupt behavior."""

    @pytest.mark.asyncio
    async def test_interrupt_during_processing_triggers_debounce(self):
        manager = ConversationManager(debounce_seconds=0.2)
        processed = []
        
        async def process_callback(chat_id, user_id, message, metadata, interrupt_check):
            processed.append(f"start:{message[:10]}")
            # Simulate work with interrupt checking
            for _ in range(20):
                await asyncio.sleep(0.05)
                if interrupt_check():
                    processed.append("interrupted")
                    return
            processed.append(f"done:{message[:10]}")
        
        # Start first message (will go through debounce first)
        await manager.add_message(123, 456, "first_msg", process_callback=process_callback)
        
        # Wait for debounce
        await asyncio.sleep(0.3)
        
        # Now processing, send another message
        await asyncio.sleep(0.1)  # Let processing start
        await manager.add_message(123, 456, "second_msg", process_callback=process_callback)
        
        # Wait for everything to complete
        await asyncio.sleep(1)
        
        # First should be interrupted, second should complete
        assert "start:first_msg" in processed
        assert "interrupted" in processed

    @pytest.mark.asyncio
    async def test_cancel_stops_debounce_and_processing(self):
        manager = ConversationManager(debounce_seconds=0.5)
        processed = []
        
        async def process_callback(chat_id, user_id, message, metadata, interrupt_check):
            processed.append("started")
            await asyncio.sleep(10)
            processed.append("finished")
        
        await manager.add_message(123, 456, "hello", process_callback=process_callback)
        
        await asyncio.sleep(0.1)
        assert manager.is_debouncing(123)
        
        # Cancel during debounce
        cancelled = await manager.cancel(123)
        
        assert cancelled
        assert not manager.is_debouncing(123)
        assert not manager.is_processing(123)


class TestConversationManagerBasic:
    """Basic tests for ConversationManager."""

    def test_get_or_create_state(self):
        manager = ConversationManager()
        
        state1 = manager.get_or_create_state(123, 456)
        state2 = manager.get_or_create_state(123, 456)
        state3 = manager.get_or_create_state(789, 456)
        
        assert state1 is state2  # Same chat
        assert state1 is not state3  # Different chat

    def test_is_processing_false_when_not_started(self):
        manager = ConversationManager()
        assert not manager.is_processing(123)
        assert not manager.is_debouncing(123)

    def test_get_pending_count(self):
        manager = ConversationManager()
        state = manager.get_or_create_state(123, 456)
        state.add_message("one")
        state.add_message("two")
        
        assert manager.get_pending_count(123) == 2
        assert manager.get_pending_count(999) == 0  # Non-existent chat

    @pytest.mark.asyncio
    async def test_error_in_callback_does_not_crash_loop(self):
        manager = ConversationManager(debounce_seconds=0)  # No debounce for simplicity
        call_count = [0]
        
        async def process_callback(chat_id, user_id, message, metadata, interrupt_check):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Intentional error")
        
        state = manager.get_or_create_state(123, 456)
        state.add_message("first")
        
        await manager._process_loop(state, process_callback)
        
        assert call_count[0] == 1
        assert not state.is_processing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
