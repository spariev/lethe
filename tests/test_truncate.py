"""Tests for truncation utilities."""

import pytest
from lethe.tools.truncate import (
    truncate_head,
    truncate_tail,
    truncate_line,
    format_truncation_notice,
    format_size,
    DEFAULT_MAX_LINES,
    DEFAULT_MAX_BYTES,
)


class TestFormatSize:
    """Tests for format_size."""
    
    def test_bytes(self):
        assert format_size(100) == "100B"
        assert format_size(1023) == "1023B"
    
    def test_kilobytes(self):
        assert format_size(1024) == "1.0KB"
        assert format_size(2048) == "2.0KB"
        assert format_size(1536) == "1.5KB"
    
    def test_megabytes(self):
        assert format_size(1024 * 1024) == "1.0MB"
        assert format_size(2 * 1024 * 1024) == "2.0MB"


class TestTruncateHead:
    """Tests for truncate_head (for file reads)."""
    
    def test_no_truncation_needed(self):
        """Should return content unchanged when within limits."""
        content = "line 1\nline 2\nline 3"
        result = truncate_head(content)
        
        assert result.content == content
        assert result.truncated is False
        assert result.truncated_by is None
    
    def test_truncate_by_lines(self):
        """Should truncate when line limit exceeded."""
        lines = [f"line {i}" for i in range(100)]
        content = '\n'.join(lines)
        
        result = truncate_head(content, max_lines=50)
        
        assert result.truncated is True
        assert result.truncated_by == "lines"
        assert result.output_lines == 50
        assert "line 0" in result.content
        assert "line 49" in result.content
        assert "line 50" not in result.content
    
    def test_truncate_by_bytes(self):
        """Should truncate when byte limit exceeded."""
        # Create content where bytes limit hits before lines
        lines = ["x" * 100 for _ in range(100)]
        content = '\n'.join(lines)
        
        result = truncate_head(content, max_bytes=500, max_lines=1000)
        
        assert result.truncated is True
        assert result.truncated_by == "bytes"
        assert result.output_bytes <= 500
    
    def test_first_line_exceeds_limit(self):
        """Should handle first line exceeding byte limit."""
        content = "x" * 10000  # Single line > 1KB
        
        result = truncate_head(content, max_bytes=1000)
        
        assert result.truncated is True
        assert result.first_line_exceeds_limit is True
        assert result.content == ""
    
    def test_preserves_complete_lines(self):
        """Should never return partial lines (except when first line exceeds)."""
        content = "short\n" + "x" * 5000  # Second line is long
        
        result = truncate_head(content, max_bytes=100)
        
        # Should only include first line, not partial second
        assert "short" in result.content
        assert result.content.count('\n') == 0  # No newline = just first line


class TestTruncateTail:
    """Tests for truncate_tail (for bash output)."""
    
    def test_no_truncation_needed(self):
        """Should return content unchanged when within limits."""
        content = "line 1\nline 2\nline 3"
        result = truncate_tail(content)
        
        assert result.content == content
        assert result.truncated is False
    
    def test_keeps_end_of_content(self):
        """Should keep the END of content, not beginning."""
        lines = [f"line {i}" for i in range(100)]
        content = '\n'.join(lines)
        
        result = truncate_tail(content, max_lines=10)
        
        assert result.truncated is True
        # Should have last 10 lines
        assert "line 99" in result.content
        assert "line 90" in result.content
        # Should NOT have early lines
        assert "line 0" not in result.content
        assert "line 89" not in result.content
    
    def test_truncate_by_lines(self):
        """Should truncate when line limit exceeded."""
        lines = [f"line {i}" for i in range(100)]
        content = '\n'.join(lines)
        
        result = truncate_tail(content, max_lines=20)
        
        assert result.truncated is True
        assert result.truncated_by == "lines"
        assert result.output_lines == 20
    
    def test_partial_last_line(self):
        """Should handle case where last line alone exceeds limit."""
        # Single very long line
        content = "x" * 10000
        
        result = truncate_tail(content, max_bytes=500)
        
        assert result.truncated is True
        assert result.last_line_partial is True
        assert len(result.content.encode('utf-8')) <= 500


class TestTruncateLine:
    """Tests for truncate_line (for grep results)."""
    
    def test_short_line_unchanged(self):
        """Should return short lines unchanged."""
        line = "short line"
        result, was_truncated = truncate_line(line)
        
        assert result == line
        assert was_truncated is False
    
    def test_long_line_truncated(self):
        """Should truncate long lines with suffix."""
        line = "x" * 600
        result, was_truncated = truncate_line(line)
        
        assert was_truncated is True
        assert len(result) < len(line)
        assert "[truncated]" in result
    
    def test_custom_max_chars(self):
        """Should respect custom max_chars parameter."""
        line = "x" * 100
        result, was_truncated = truncate_line(line, max_chars=50)
        
        assert was_truncated is True
        assert result.startswith("x" * 50)


class TestFormatTruncationNotice:
    """Tests for format_truncation_notice."""
    
    def test_no_notice_when_not_truncated(self):
        """Should return empty string when not truncated."""
        result = truncate_head("short content")
        notice = format_truncation_notice(result)
        
        assert notice == ""
    
    def test_notice_includes_offset(self):
        """Should include offset instruction."""
        lines = [f"line {i}" for i in range(100)]
        content = '\n'.join(lines)
        
        result = truncate_head(content, max_lines=50)
        notice = format_truncation_notice(result, start_line=1)
        
        assert "offset=" in notice
        assert "continue" in notice.lower()
    
    def test_notice_with_temp_file(self):
        """Should include temp file path if provided."""
        lines = [f"line {i}" for i in range(100)]
        content = '\n'.join(lines)
        
        result = truncate_head(content, max_lines=50)
        notice = format_truncation_notice(result, temp_file_path="/tmp/output.log")
        
        assert "/tmp/output.log" in notice


class TestUtf8Handling:
    """Tests for proper UTF-8 handling."""
    
    def test_multibyte_characters(self):
        """Should handle multi-byte UTF-8 characters correctly."""
        # Each emoji is 4 bytes
        content = "Hello 🌍🌎🌏 World"
        
        result = truncate_head(content)
        
        assert result.truncated is False
        assert "🌍" in result.content
    
    def test_truncate_at_utf8_boundary(self):
        """Should not break multi-byte characters when truncating."""
        # Create content with emojis that might get cut
        content = "x" * 100 + "🎉" * 100
        
        result = truncate_head(content, max_bytes=150)
        
        # Should be valid UTF-8
        result.content.encode('utf-8')  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
