"""STEP 1 — figure descriptions (Qwen3.5 reads the image) for figure chunks."""

from __future__ import annotations

from llm_client.client import chat_text
from prompts.figures import figure_messages
from schemas import DescribeFigureOutput, DescribeFigureRequest


async def describe(req: DescribeFigureRequest) -> DescribeFigureOutput:
    text = await chat_text(figure_messages(req.image_b64, req.mime, req.caption, req.context),
                           max_tokens=320)
    return DescribeFigureOutput(description=text.strip().strip('"'))
