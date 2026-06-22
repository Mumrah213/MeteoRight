"""LLM factory for the agent.

Defaults to the local OpenAI-compatible endpoint (llama-swap on bazzite) so the
agent runs without any cloud key, but every setting is env-overridable, so an
OpenAI or other compatible endpoint can be swapped in without code changes.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

DEFAULT_BASE_URL = "http://bazzite:8080/v1"
DEFAULT_MODEL = "qwen3.6-35b-q4-headroom"
DEFAULT_API_KEY = "sk-noauth"  # the local endpoint ignores auth


def make_llm(*, temperature: float = 0.0, timeout: float = 120.0) -> ChatOpenAI:
    """Build the chat model the graph drives.

    Env overrides:
        METEORIGHT_LLM_BASE_URL, METEORIGHT_LLM_MODEL, METEORIGHT_LLM_API_KEY.
    """
    return ChatOpenAI(
        model=os.environ.get("METEORIGHT_LLM_MODEL", DEFAULT_MODEL),
        base_url=os.environ.get("METEORIGHT_LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.environ.get("METEORIGHT_LLM_API_KEY", DEFAULT_API_KEY),
        temperature=temperature,
        timeout=timeout,
        max_retries=1,
    )
