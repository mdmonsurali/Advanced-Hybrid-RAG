"""The SGLang client: output cleanup, JSON retries, token estimate."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from llm_client import client
from schemas import ExtractOutput

pytestmark = pytest.mark.anyio


def test_clean_strips_reasoning_and_code_fences():
    assert client._clean("<think>hmm\nok</think>  Answer ") == "Answer"
    assert client._clean('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert client._clean("") == ""


def test_count_tokens_estimate_without_tokenizer():
    assert client.count_tokens("x" * 40) == 10
    assert client.count_tokens("") == 1


class FakeCompletions:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.temperatures: list[float] = []

    async def create(self, **kwargs):
        self.temperatures.append(kwargs["temperature"])
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def sglang(monkeypatch):
    def install(replies: list[str]) -> FakeCompletions:
        completions = FakeCompletions(replies)
        fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(client, "_get_client", lambda: fake)
        monkeypatch.setattr(client, "_sem", asyncio.Semaphore(1))
        return completions

    return install


async def test_chat_json_parses_the_reply(sglang):
    sglang(['<think>x</think>{"relevant_text": "30 days"}'])
    out = await client.chat_json([{"role": "user", "content": "q"}], ExtractOutput)
    assert out.relevant_text == "30 days"


async def test_chat_json_retries_once_warmer_then_gives_up(sglang):
    completions = sglang(["not json", "still not json"])
    with pytest.raises(client.LLMOutputError):
        await client.chat_json([{"role": "user", "content": "q"}], ExtractOutput)
    assert completions.temperatures == [0.0, 0.3]


async def test_chat_json_recovers_on_the_second_try(sglang):
    sglang(["oops", '{"relevant_text": "ok"}'])
    out = await client.chat_json([{"role": "user", "content": "q"}], ExtractOutput)
    assert out.relevant_text == "ok"
