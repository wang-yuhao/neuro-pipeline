"""
utils/api_client.py
Centralized Anthropic API client with retry logic and error handling.
"""

from __future__ import annotations

import time
import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

# Model to use for all agents
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8096


def call_claude(
    system_prompt: str,
    user_message: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.3,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    """
    Call the Anthropic Claude API with retry logic.

    Parameters
    ----------
    system_prompt : str
        The system prompt defining agent behavior.
    user_message : str
        The user message / task input.
    max_tokens : int
        Maximum tokens in response.
    temperature : float
        Sampling temperature (lower = more deterministic).
    max_retries : int
        Number of retry attempts on failure.
    retry_delay : float
        Seconds to wait between retries.

    Returns
    -------
    str
        The model's text response.

    Raises
    ------
    RuntimeError
        If all retry attempts fail.
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"API call attempt {attempt}/{max_retries}")
            response = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text
            logger.debug(f"API call succeeded on attempt {attempt}")
            return text

        except anthropic.RateLimitError as e:
            wait = retry_delay * attempt
            logger.warning(f"Rate limit hit (attempt {attempt}). Waiting {wait}s... {e}")
            time.sleep(wait)

        except anthropic.APIStatusError as e:
            logger.error(f"API status error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise RuntimeError(f"API call failed after {max_retries} attempts: {e}") from e
            time.sleep(retry_delay)

        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise RuntimeError(f"Unexpected failure after {max_retries} attempts: {e}") from e
            time.sleep(retry_delay)

    raise RuntimeError(f"All {max_retries} API attempts exhausted")
