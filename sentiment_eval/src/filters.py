"""
Hard Filter Module for Sentiment Classification Pipeline

Provides deterministic pre-filtering to skip posts that don't require
model inference (e.g., too short, link-only, promotional noise).
"""

import re
from typing import Tuple, Optional
from .utils import ESCALATION_KEYWORDS

# URL pattern for detecting link-only posts
URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+',
    re.IGNORECASE
)

# Promotional keywords that suggest affiliate/marketing content
PROMO_KEYWORDS = {
    "sign up",
    "join now",
    "free bet",
    "free bets",
    "bonus",
    "claim here",
    "bet £",
    "bet $",
    "new customer",
    "t&cs apply",
    "18+",
    "gamble responsibly",
    "begambleaware",
    "#ad",
    "promo code",
}

# Minimum token count for meaningful content
DEFAULT_MIN_TOKENS = 4


def count_tokens(text: str) -> int:
    """Count tokens (whitespace-separated words) in text."""
    return len(text.split())


def extract_urls(text: str) -> list[str]:
    """Extract all URLs from text."""
    return URL_PATTERN.findall(text)


def remove_urls(text: str) -> str:
    """Remove all URLs from text."""
    return URL_PATTERN.sub("", text).strip()


def is_link_only(text: str, max_non_url_tokens: int = 3) -> bool:
    """
    Check if post is primarily just links with minimal text.
    
    Args:
        text: Post content
        max_non_url_tokens: Maximum non-URL tokens to still be considered link-only
        
    Returns:
        True if post is essentially just links
    """
    text_without_urls = remove_urls(text)
    non_url_tokens = count_tokens(text_without_urls)
    return non_url_tokens <= max_non_url_tokens and len(extract_urls(text)) > 0


def is_promotional(text: str, threshold: int = 2) -> bool:
    """
    Check if post appears to be promotional/affiliate content.
    
    Args:
        text: Post content
        threshold: Minimum number of promo keywords to trigger
        
    Returns:
        True if post appears promotional
    """
    text_lower = text.lower()
    matches = sum(1 for kw in PROMO_KEYWORDS if kw in text_lower)
    return matches >= threshold


def has_escalation_trigger(text: str) -> bool:
    """
    Check if text contains keywords that should bypass filters.
    
    These are high-risk terms that should always be analyzed even
    if the post would otherwise be filtered.
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in ESCALATION_KEYWORDS)


def should_skip_post(
    text: str,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    skip_promotional: bool = True,
    skip_link_only: bool = True,
) -> Tuple[bool, str]:
    """
    Determine if a post should be skipped (not sent to model).
    
    Args:
        text: Post content
        min_tokens: Minimum token count required
        skip_promotional: Whether to skip promotional content
        skip_link_only: Whether to skip link-only posts
        
    Returns:
        Tuple of (should_skip: bool, reason: str)
        reason is empty string if should_skip is False
    """
    if not text or not text.strip():
        return True, "empty_text"
    
    text = text.strip()
    
    # Check for escalation keywords first - these override all filters
    if has_escalation_trigger(text):
        return False, ""
    
    # Check token count
    token_count = count_tokens(text)
    if token_count < min_tokens:
        return True, f"too_short_{token_count}_tokens"
    
    # Check for link-only posts
    if skip_link_only and is_link_only(text):
        return True, "link_only"
    
    # Check for promotional content
    if skip_promotional and is_promotional(text):
        return True, "promotional"
    
    return False, ""


def filter_batch(
    texts: list[str],
    min_tokens: int = DEFAULT_MIN_TOKENS,
    skip_promotional: bool = True,
    skip_link_only: bool = True,
) -> Tuple[list[int], list[int], dict[int, str]]:
    """
    Filter a batch of texts, returning indices to process and skip.
    
    Args:
        texts: List of post texts
        min_tokens: Minimum token count
        skip_promotional: Skip promotional content
        skip_link_only: Skip link-only posts
        
    Returns:
        Tuple of:
        - process_indices: List of indices to send to model
        - skip_indices: List of indices to skip
        - skip_reasons: Dict mapping skip index to reason
    """
    process_indices = []
    skip_indices = []
    skip_reasons = {}
    
    for i, text in enumerate(texts):
        should_skip, reason = should_skip_post(
            text,
            min_tokens=min_tokens,
            skip_promotional=skip_promotional,
            skip_link_only=skip_link_only,
        )
        
        if should_skip:
            skip_indices.append(i)
            skip_reasons[i] = reason
        else:
            process_indices.append(i)
    
    return process_indices, skip_indices, skip_reasons


def get_filter_stats(skip_reasons: dict[int, str]) -> dict[str, int]:
    """
    Get statistics on filter reasons.
    
    Args:
        skip_reasons: Dict from filter_batch
        
    Returns:
        Dict mapping reason category to count
    """
    stats = {}
    for reason in skip_reasons.values():
        # Normalize reason (e.g., "too_short_3_tokens" -> "too_short")
        category = reason.split("_")[0] if "_" in reason else reason
        stats[category] = stats.get(category, 0) + 1
    return stats
