"""
LLM utility functions including retry logic for API calls.

Provides robust retry mechanism with exponential backoff for all LLM calls.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar, Optional

logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 10.0  # seconds
DEFAULT_EXPONENTIAL_BASE = 2.0

# Retryable exceptions (add more as needed)
RETRYABLE_EXCEPTIONS = (
    Exception,  # Catch all for now - can be more specific
)

# Non-retryable error messages (don't retry on these)
NON_RETRYABLE_ERRORS = [
    "invalid_api_key",
    "authentication",
    "quota exceeded",
    "rate limit",  # Rate limits should use different backoff strategy
]

T = TypeVar('T')


def should_retry(exception: Exception) -> bool:
    """Determine if an exception is retryable."""
    error_msg = str(exception).lower()

    # Don't retry on authentication/quota errors
    for non_retryable in NON_RETRYABLE_ERRORS:
        if non_retryable in error_msg:
            return False

    return True


def calculate_delay(attempt: int, base_delay: float, max_delay: float, exponential_base: float) -> float:
    """Calculate delay with exponential backoff and jitter."""
    import random

    # Exponential backoff
    delay = base_delay * (exponential_base ** attempt)

    # Add jitter (±25%)
    jitter = delay * 0.25 * (2 * random.random() - 1)
    delay += jitter

    # Cap at max delay
    return min(delay, max_delay)


async def llm_call_with_retry(
    func: Callable[..., T],
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
    operation_name: str = "LLM call",
    **kwargs
) -> T:
    """
    Execute an LLM call with retry logic and exponential backoff.

    Args:
        func: The function to call (typically gemini_client.chat.completions.create)
        *args: Arguments to pass to the function
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 10.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        operation_name: Name for logging purposes
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"[LLM_RETRY] {operation_name}: Attempt {attempt + 1}/{max_retries + 1}")

            # Execute the call
            result = await asyncio.to_thread(func, *args, **kwargs)

            if attempt > 0:
                logger.info(f"[LLM_RETRY] {operation_name}: Succeeded on attempt {attempt + 1}")

            return result

        except Exception as e:
            last_exception = e

            # Check if we should retry
            if not should_retry(e):
                logger.error(f"[LLM_RETRY] {operation_name}: Non-retryable error: {e}")
                raise

            # Check if we have retries left
            if attempt >= max_retries:
                logger.error(f"[LLM_RETRY] {operation_name}: All {max_retries + 1} attempts failed. Last error: {e}")
                raise

            # Calculate delay and wait
            delay = calculate_delay(attempt, base_delay, max_delay, exponential_base)
            logger.warning(
                f"[LLM_RETRY] {operation_name}: Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:100]}. "
                f"Retrying in {delay:.2f}s..."
            )

            await asyncio.sleep(delay)

    # Should never reach here, but just in case
    raise last_exception


def llm_call_with_retry_sync(
    func: Callable[..., T],
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
    operation_name: str = "LLM call",
    **kwargs
) -> T:
    """
    Synchronous version of llm_call_with_retry.

    For use in non-async contexts.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"[LLM_RETRY] {operation_name}: Attempt {attempt + 1}/{max_retries + 1}")

            result = func(*args, **kwargs)

            if attempt > 0:
                logger.info(f"[LLM_RETRY] {operation_name}: Succeeded on attempt {attempt + 1}")

            return result

        except Exception as e:
            last_exception = e

            if not should_retry(e):
                logger.error(f"[LLM_RETRY] {operation_name}: Non-retryable error: {e}")
                raise

            if attempt >= max_retries:
                logger.error(f"[LLM_RETRY] {operation_name}: All {max_retries + 1} attempts failed. Last error: {e}")
                raise

            delay = calculate_delay(attempt, base_delay, max_delay, exponential_base)
            logger.warning(
                f"[LLM_RETRY] {operation_name}: Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:100]}. "
                f"Retrying in {delay:.2f}s..."
            )

            time.sleep(delay)

    raise last_exception


# Convenience wrapper for common Gemini call pattern
async def gemini_chat_completion_with_retry(
    gemini_client,
    model: str,
    messages: list,
    temperature: float = 0.3,
    max_tokens: int = 500,
    response_format: Optional[dict] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    operation_name: str = "Gemini chat completion",
) -> Any:
    """
    Convenience function for Gemini chat completion with retry.

    Args:
        gemini_client: OpenAI-compatible Gemini client
        model: Model name
        messages: List of message dicts
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        response_format: Optional response format (e.g., {"type": "json_object"})
        max_retries: Maximum retry attempts
        operation_name: Name for logging

    Returns:
        The API response
    """
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if response_format:
        kwargs["response_format"] = response_format

    return await llm_call_with_retry(
        gemini_client.chat.completions.create,
        max_retries=max_retries,
        operation_name=operation_name,
        **kwargs
    )
