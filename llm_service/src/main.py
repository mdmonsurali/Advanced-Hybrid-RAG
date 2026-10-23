"""llm_service — task API in front of SGLang (Qwen3.5-4B-AWQ).

SGLang runs in the same container on 127.0.0.1:30000; this FastAPI app is
the only thing other services talk to. Every prompt in the project lives
here (prompts/), every LLM call goes through llm_client/, and the pipeline
steps that need the model are implemented in generation/.

Exposes:
    GET  /health                    SGLang + app
    POST /tasks/query-analysis      STEP 3+4  {question, history}        -> QueryPlan
    POST /tasks/route               STEP 5    {plan, available_routes}   -> {routes, reason}
    POST /tasks/describe-figure     STEP 1    {image_b64, caption, context} -> {description}
    POST /tasks/doc-metadata        STEP 2    {name, text}               -> {doc_type, topic, summary, language}
    POST /tasks/compress            STEP 8    {plan, evidence}           -> compressed evidence
    POST /tasks/generate            STEP 9    {plan, evidence, direct}   -> cited answer
    POST /tasks/validate            STEP 10   {plan, evidence, answer}   -> masked answer + verdict
    POST /tasks/select-tools        MCP       {question, tools}          -> tool calls for user MCP servers
    GET  /prompts                   prompt source / label / version in use (PROMPT_* in .env)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import tracing
from fastapi import FastAPI, HTTPException, Request
from generation import answer, compression, figures, query_analysis, tools, validation
from llm_client import client
from llm_client.client import LLMOutputError
from prompts import registry
from schemas import (
    CompressRequest,
    CompressResponse,
    DescribeFigureOutput,
    DescribeFigureRequest,
    DocMetadataOutput,
    DocMetadataRequest,
    GenerateRequest,
    GenerateResponse,
    QueryAnalysisRequest,
    QueryPlan,
    RouteOutput,
    RouteRequest,
    SelectToolsOutput,
    SelectToolsRequest,
    ValidateRequest,
    ValidateResponse,
)

log = logging.getLogger("llm_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create prompts missing in Langfuse from the local texts and warm the
    # prompt cache (never blocks startup for long: 3 s timeout per call).
    await asyncio.to_thread(registry.seed_missing)
    yield
    tracing.shutdown()  # flush pending Langfuse generations


app = FastAPI(title="Advanced Hybrid RAG — LLM Service", version="1.0", lifespan=lifespan)


class _SilencePathsFilter(logging.Filter):
    """Drop uvicorn access lines for /health so the actual events show."""

    def __init__(self, paths: tuple[str, ...]) -> None:
        super().__init__()
        self._paths = paths

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(f" {p} " in msg for p in self._paths)


logging.getLogger("uvicorn.access").addFilter(_SilencePathsFilter(("/health",)))


def _llm_error(exc: Exception) -> HTTPException:
    if isinstance(exc, LLMOutputError):
        return HTTPException(502, f"LLM returned invalid output: {exc}")
    return HTTPException(503, f"LLM call failed: {exc}")


@app.get("/health")
async def health():
    try:
        await client.health()
    except Exception as e:
        raise HTTPException(503, f"sglang not reachable: {e}")
    return {"status": "ok", "sglang": "ok", "model": client.LLM_MODEL}


@app.get("/prompts")
async def prompts_in_use():
    """Prompt source, label, pins and the version each prompt last resolved to."""
    return registry.status()


@app.post("/tasks/query-analysis", response_model=QueryPlan)
async def task_query_analysis(req: QueryAnalysisRequest, request: Request):
    with tracing.task("query-analysis", request.headers, input=req) as obs:
        try:
            return tracing.output(obs, await query_analysis.analyze(req.question, req.history))
        except Exception as e:
            log.exception("query analysis failed")
            raise _llm_error(e)


@app.post("/tasks/route", response_model=RouteOutput)
async def task_route(req: RouteRequest, request: Request):
    if not req.available_routes:
        raise HTTPException(400, "available_routes must not be empty")
    with tracing.task("route-llm", request.headers, input=req) as obs:
        try:
            return tracing.output(obs, await query_analysis.route(req.plan, req.available_routes))
        except Exception as e:
            log.exception("routing failed")
            raise _llm_error(e)


@app.post("/tasks/describe-figure", response_model=DescribeFigureOutput)
async def task_describe_figure(req: DescribeFigureRequest, request: Request):
    with tracing.task(
        "describe-figure", request.headers, input={"caption": req.caption, "context": req.context}
    ) as obs:
        try:
            return tracing.output(obs, await figures.describe(req))
        except Exception as e:
            log.exception("figure description failed")
            raise _llm_error(e)


@app.post("/tasks/doc-metadata", response_model=DocMetadataOutput)
async def task_doc_metadata(req: DocMetadataRequest, request: Request):
    with tracing.task("doc-metadata", request.headers, input=req) as obs:
        try:
            return tracing.output(obs, await query_analysis.doc_metadata(req.name, req.text))
        except Exception as e:
            log.exception("doc metadata failed")
            raise _llm_error(e)


@app.post("/tasks/compress", response_model=CompressResponse)
async def task_compress(req: CompressRequest, request: Request):
    with tracing.task("compress", request.headers, input=req) as obs:
        try:
            return tracing.output(obs, await compression.compress(req.plan, req.evidence))
        except Exception as e:
            log.exception("compression failed")
            raise _llm_error(e)


@app.post("/tasks/generate", response_model=GenerateResponse)
async def task_generate(req: GenerateRequest, request: Request):
    with tracing.task("generate", request.headers, input=req) as obs:
        try:
            return tracing.output(
                obs, await answer.generate(req.plan, req.evidence, req.direct, req.answer_style)
            )
        except Exception as e:
            log.exception("generation failed")
            raise _llm_error(e)


@app.post("/tasks/validate", response_model=ValidateResponse)
async def task_validate(req: ValidateRequest, request: Request):
    with tracing.task("validate", request.headers, as_type="evaluator", input=req) as obs:
        try:
            return tracing.output(
                obs, await validation.validate(req.plan, req.evidence, req.answer, req.direct)
            )
        except Exception as e:
            log.exception("validation failed")
            raise _llm_error(e)


@app.post("/tasks/select-tools", response_model=SelectToolsOutput)
async def task_select_tools(req: SelectToolsRequest, request: Request):
    if not req.tools:
        return SelectToolsOutput(calls=[], reason="no tools connected")
    with tracing.task("select-tools", request.headers, input=req) as obs:
        try:
            return tracing.output(obs, await tools.select(req.question, req.tools))
        except Exception as e:
            log.exception("tool selection failed")
            raise _llm_error(e)
