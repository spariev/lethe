"""Text processing utilities."""

import re


def strip_model_tags(content: str) -> str:
    """Strip reasoning and wrapper tags from model output.
    
    Removes:
    - <think>...</think> blocks (Kimi reasoning)
    - <thinking>...</thinking> blocks (Claude extended thinking)
    - <result>...</result> wrapper (keeps inner content)
    
    Args:
        content: Raw model output
        
    Returns:
        Cleaned content with tags stripped
    """
    if not content:
        return content
    
    # Strip thinking blocks entirely
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
    
    # Strip result wrapper but keep inner content
    content = re.sub(r'<result>\s*', '', content)
    content = re.sub(r'\s*</result>', '', content)
    
    return content.strip()
