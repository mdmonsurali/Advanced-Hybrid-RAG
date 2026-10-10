"""SGLang client (OpenAI-compatible API) for every LLM call in the project.

All task calls run with Qwen3.5's thinking mode switched off and, where a
pydantic model is given, with SGLang's grammar-constrained JSON output
(`response_format=json_schema`, compiled by xgrammar). A semaphore keeps the
number of in-flight requests at the server's `--max-running-requests`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Type, TypeVar

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

log = logging.getLogger("llm_service.client")

SGLANG_URL = os.getenv("SGLANG_URL", "http://localhost:30000")
MODEL_NAME = os.getenv("LLM_SERVED_NAME", "qwen")
LLM_MODEL = os.getenv("LLM_MODEL", "LostGentoo/Qwen3.5-4B-AWQ")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))
LLM_CONCURRENCY = int(os.getenv("LLM_MAX_RUNNING", "4"))
LLM_CONTEXT_LEN = int(os.getenv("LLM_CONTEXT_LEN", "8192"))

_NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

T = TypeVar("T", bound=BaseModel)

_client: AsyncOpenAI | None = None
_sem: asyncio.Semaphore | None = None
_tokenizer = None


class LLMOutputError(RuntimeError):
    """The model returned output that doesn't match the requested schema."""


def _get_client() -> AsyncOpenAI:
    global _client, _sem
    if _client is None:
        _client = AsyncOpenAI(base_url=f"{SGLANG_URL}/v1", api_key="EMPTY",
                              timeout=LLM_TIMEOUT)
        _sem = asyncio.Semaphore(max(1, LLM_CONCURRENCY))
    return _client


def _clean(text: str) -> str:
    """Drop any reasoning block and code fences the model may still emit."""
    text = _THINK_RE.sub("", text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    return text


async def chat_text(messages: list[dict], *, temperature: float = 0.0,
                    max_tokens: int = 1024) -> str:
    client = _get_client()
    assert _sem is not None
    async with _sem:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=_NO_THINKING,
        )
    return _clean(resp.choices[0].message.content or "")


async def chat_json(messages: list[dict], schema: Type[T], *,
                    temperature: float = 0.0, max_tokens: int = 1024) -> T:
    """Grammar-constrained JSON call parsed into `schema`. Retries once at a
    slightly higher temperature before giving up."""
    client = _get_client()
    assert _sem is not None
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
    }
    last_err: Exception | None = None
    for attempt, temp in enumerate((temperature, max(temperature, 0.3))):
        async with _sem:
            resp = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
                response_format=response_format,
                extra_body=_NO_THINKING,
            )
        raw = _clean(resp.choices[0].message.content or "")
        try:
            return schema.model_validate_json(raw)
        except ValidationError as e:
            last_err = e
            log.warning("[llm] %s output invalid (attempt %d): %s",
                        schema.__name__, attempt + 1, str(e)[:300])
    raise LLMOutputError(f"{schema.__name__}: {last_err}")


def count_tokens(text: str) -> int:
    """Token count with the served model's tokenizer (falls back to a
    4-chars-per-token estimate if the tokenizer can't be loaded)."""
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        except Exception as e:  # pragma: no cover - only when weights are absent
            log.warning("[llm] tokenizer unavailable (%s); estimating tokens", e)
            _tokenizer = False
    if _tokenizer is False:
        return max(1, len(text) // 4)
    return len(_tokenizer.encode(text, add_special_tokens=False))


async def health() -> None:
    async with httpx.AsyncClient(timeout=3.0) as client:
        r = await client.get(f"{SGLANG_URL}/health")
        r.raise_for_status()
