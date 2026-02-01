"""Tests for file-based memory blocks."""

import tempfile
import os
from pathlib import Path

import pytest

from lethe.memory.blocks import BlockManager


class TestBlockManager:
    """Tests for BlockManager with temp directories."""
    
    @pytest.fixture
    def blocks_dir(self):
        """Create a temp directory for blocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def manager(self, blocks_dir):
        """Create a BlockManager with temp directory."""
        return BlockManager(blocks_dir)
    
    def test_create_block(self, manager, blocks_dir):
        """Should create a block file."""
        manager.create("test", value="Hello world", description="Test block")
        
        # Check file exists
        assert (Path(blocks_dir) / "test.md").exists()
        assert (Path(blocks_dir) / "test.meta.json").exists()
        
        # Check content
        content = (Path(blocks_dir) / "test.md").read_text()
        assert content == "Hello world"
    
    def test_get_block(self, manager):
        """Should retrieve a block."""
        manager.create("myblock", value="content here", description="My block")
        
        block = manager.get_by_label("myblock")
        
        assert block is not None
        assert block["label"] == "myblock"
        assert block["value"] == "content here"
        assert block["description"] == "My block"
    
    def test_get_nonexistent_block(self, manager):
        """Should return None for nonexistent block."""
        block = manager.get_by_label("nonexistent")
        assert block is None
    
    def test_update_block(self, manager):
        """Should update block content."""
        manager.create("updateme", value="original")
        
        manager.update("updateme", value="updated")
        
        block = manager.get_by_label("updateme")
        assert block["value"] == "updated"
    
    def test_update_description(self, manager):
        """Should update block description."""
        manager.create("desc", value="content", description="old desc")
        
        manager.update("desc", description="new desc")
        
        block = manager.get_by_label("desc")
        assert block["description"] == "new desc"
        assert block["value"] == "content"  # Content unchanged
    
    def test_delete_block(self, manager, blocks_dir):
        """Should delete block files."""
        manager.create("deleteme", value="bye")
        
        result = manager.delete("deleteme")
        
        assert result is True
        assert not (Path(blocks_dir) / "deleteme.md").exists()
        assert not (Path(blocks_dir) / "deleteme.meta.json").exists()
    
    def test_list_blocks(self, manager):
        """Should list all blocks."""
        manager.create("alpha", value="a")
        manager.create("beta", value="b")
        manager.create("gamma", value="c")
        
        blocks = manager.list_blocks()
        
        labels = [b["label"] for b in blocks]
        assert "alpha" in labels
        assert "beta" in labels
        assert "gamma" in labels
    
    def test_list_excludes_hidden(self, manager):
        """Should exclude hidden blocks by default."""
        manager.create("visible", value="see me")
        manager.create("hidden", value="hide me", hidden=True)
        
        blocks = manager.list_blocks()
        labels = [b["label"] for b in blocks]
        
        assert "visible" in labels
        assert "hidden" not in labels
    
    def test_list_includes_hidden_when_requested(self, manager):
        """Should include hidden blocks when requested."""
        manager.create("visible", value="see me")
        manager.create("hidden", value="hide me", hidden=True)
        
        blocks = manager.list_blocks(include_hidden=True)
        labels = [b["label"] for b in blocks]
        
        assert "visible" in labels
        assert "hidden" in labels
    
    def test_str_replace(self, manager):
        """Should replace text in block."""
        manager.create("replace", value="Hello World")
        
        manager.str_replace("replace", "World", "Python")
        
        block = manager.get_by_label("replace")
        assert block["value"] == "Hello Python"
    
    def test_append(self, manager):
        """Should append text to block."""
        manager.create("appendme", value="Start")
        
        manager.append("appendme", " End")
        
        block = manager.get_by_label("appendme")
        assert block["value"] == "Start End"
    
    def test_read_only_block(self, manager):
        """Should prevent updates to read-only blocks."""
        manager.create("readonly", value="locked", read_only=True)
        
        with pytest.raises(ValueError, match="read-only"):
            manager.update("readonly", value="new value")
    
    def test_duplicate_block_error(self, manager):
        """Should error on duplicate block creation."""
        manager.create("unique", value="first")
        
        with pytest.raises(ValueError, match="already exists"):
            manager.create("unique", value="second")
    
    def test_limit_enforced(self, manager):
        """Should enforce character limit."""
        manager.create("limited", value="x", limit=10)
        
        with pytest.raises(ValueError, match="exceeds limit"):
            manager.update("limited", value="x" * 100)
    
    def test_external_file_edit(self, manager, blocks_dir):
        """Should read externally edited files."""
        manager.create("external", value="original")
        
        # Edit file directly
        (Path(blocks_dir) / "external.md").write_text("edited externally")
        
        block = manager.get_by_label("external")
        assert block["value"] == "edited externally"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
