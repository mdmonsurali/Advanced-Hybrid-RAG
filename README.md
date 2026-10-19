# Advanced Hybrid RAG

A self-hosted, microservice **Advanced RAG** system that runs on a single
**6 GB GPU**. Upload PDFs, scans, images and Word files; ask questions in a
chat that answers from your documents, documents shared with you, your
Google Drive and the web, with citations, and checks every answer for
groundedness before you see it.

- **OCR / parsing:** [Docling](https://github.com/docling-project/docling) (CPU)
- **LLM:** [LostGentoo/Qwen3.5-4B-AWQ](https://huggingface.co/LostGentoo/Qwen3.5-4B-AWQ) on [SGLang](https://github.com/sgl-project/sglang) (GPU)
- **Retrieval:** Qdrant hybrid search (Qwen3-Embedding-0.6B dense + BM25 sparse, RRF fusion), then the Qwen3-Reranker-0.6B reranker
- **External sources:** Google Drive and web search as [FastMCP](https://github.com/jlowin/fastmcp) servers
- **Observability:** self-hosted [Langfuse](https://langfuse.com) — one trace per question, answer scores, versioned prompts

## Architecture

```
                         ┌──────────────┐
          Browser  ───▶  │  ui_service  │  Chainlit chat (/chat) + OCR / History / Admin pages
                         └──────┬───────┘
              ┌─────────────────┼──────────────────────┐
              ▼                 ▼                      ▼
      ┌──────────────┐  ┌───────────────┐   ┌──────────────────────┐
      │ auth_service │  │  ocr_service  │   │ orchestrator_service │  steps 3→10, retry loop, SSE
      │ users,       │  │ Docling (CPU) │   └─────┬───────────┬────┘
      │ sessions     │  │ MD/JSON/DOCX  │         │           │
      └──────────────┘  └───┬───────┬───┘         ▼           ▼
                            │ hooks │   ┌──────────────────┐ ┌──────────────────────┐
                            │       └──▶│   rag_service    │ │     llm_service      │
                            │           │ steps 1-2, 5-7   │ │ prompts, generation, │
                            │           │ embeddings (CPU) │─▶ llm_client           │
                            │           │ MCP: gdrive, web │ │ SGLang Qwen3.5 (GPU) │
                            ▼           └───┬─────────┬────┘ └──────────────────────┘
                     ┌──────────────┐       │         │
                     │    MinIO     │◀──────┘         ▼
                     │ bytes + meta │          ┌────────────┐
                     └──────────────┘          │   Qdrant   │ dense + BM25 vectors
                                               └────────────┘
```

| Service | Directory | Responsibility |
|---|---|---|
| **ui_service** | [`ui_service/`](ui_service/) | The **Chainlit chat** at `/chat` (live pipeline steps, clickable citations, file upload → ingestion, user MCP servers, chat history in MinIO) plus the Single/Batch OCR, History (per-document *Shared* toggle) and Admin pages. Gates every route behind a session and proxies to the backends with `X-User-Id` / `X-User-Role`. |
| **auth_service** | [`auth_service/`](auth_service/) | Login, sessions, password changes, user admin. Users and sessions live in a JSON file on the `auth_data` volume. |
| **ocr_service** | [`ocr_service/`](ocr_service/) | Docling conversion (layout + TableFormer + RapidOCR) on CPU. Stores source, Markdown, DoclingDocument JSON, DOCX and picture crops in MinIO. Notifies rag_service when a document is finished, shared or deleted. |
| **rag_service** | [`rag_service/`](rag_service/) | The data side: chunking, metadata, embeddings, Qdrant, query routing, hybrid retrieval, reranking, and the two MCP servers. |
| **llm_service** | [`llm_service/`](llm_service/) | SGLang serving Qwen3.5-4B-AWQ plus a task API. All prompts and LLM calls live here: query analysis, document metadata, compression, answer generation, validation. |
| **orchestrator_service** | [`orchestrator_service/`](orchestrator_service/) | Runs the query pipeline end to end, owns the validation retry loop, and streams every step to the chat over SSE. |
| **langfuse_service** | [`langfuse_service/`](langfuse_service/) | Self-hosted Langfuse (web, worker, Postgres, ClickHouse, Redis; events in MinIO). Every question is one trace across orchestrator, rag and llm services; answers get validation and 👍/👎 scores; llm_service's prompts are versioned there. |
| qdrant / minio | — | Vector store / object store. |

## The 10-step pipeline

| # | Step | Where | How |
|---|---|---|---|
| 1 | **Ingestion** | `rag_service/src/ingestion`, `chunking_techniques/` | docling-core `HybridChunker` for text: splits along headings/sections, sized in tokens with the embedder's tokenizer, merges small neighbours. Tables split across pages are stitched back into one table first. Tables become row-group chunks: each row is embedded as `Column: value` pairs (multi-level headers resolved to a path) while the LLM sees a Markdown table with the full header, the caption and the paragraph that explains it, each group citing its own page. Figures are described by the vision LLM (`/tasks/describe-figure`, cached per crop) and indexed with their caption, in-figure text and context. |
| 2 | **Metadata** | `ingestion/metadata.py` + llm `/tasks/doc-metadata` | Every chunk carries `source, page_start/end, timestamp, doc_type, topic, language, headings, chunk_type, owner_id, access_level`. Document-level type, topic and summary come from one LLM call per document. |
| 3 | **Query understanding** | `llm_service/src/generation/query_analysis.py` | Rewrites the question into a standalone question (using chat history), then classifies it: fact / comparison / troubleshooting / policy / summary / catalog / chitchat. Also extracts entities and date/type filters, and flags ambiguous questions for a clarifying question. |
| 4 | **Query transformation** | same | Keyword rewrite, up to 3 expansions, decomposition into sub-questions (comparisons), and a **HyDE** passage used for dense search. |
| 5 | **Query routing** | `rag_service/src/query_route/` | Rules first, then an LLM fallback. Routes: `docs` (vector store), `catalog` (document metadata: counts, lists), `gdrive` (Drive MCP), `web` (web-search MCP), `direct`. Routes run concurrently; a failing route is reported as `unavailable` and the others still answer. |
| 6 | **Hybrid retrieval** | `hybrid_retrieval/search.py` | One Qdrant query per sub-question. Dense prefetches cover the rewritten query (embedded with Qwen3-Embedding's query instruction) and the HyDE passage (embedded as a document); a BM25 prefetch covers query + expansions. Results are fused with RRF and filtered by access (`owner == you OR shared`) plus the step-3 filters. |
| 7 | **Reranking** | `hybrid_retrieval/rerank.py` | Qwen3-Reranker-0.6B scores pooled docs + Drive + web candidates against their (sub-)question (probability of "yes, this answers it"). Keeps the top-k above a threshold, plus the best chunk per sub-question. |
| 8 | **Context compression** | `llm_service/src/generation/compression.py` | Drops weak chunks and near-duplicates, merges adjacent chunks, and fits the token budget. Long chunks that don't fit are compressed extractively, keeping their citation number. |
| 9 | **Prompt construction** | `llm_service/src/prompts/answer.py` | Assembles the system rules, intent-specific format, question and numbered evidence `[n] source p.X (date)`. Output is grammar-constrained JSON (`answer_markdown, citations, confidence, insufficient_evidence`). |
| 10 | **Validation** | `llm_service/src/generation/validation.py` | Checks schema and citation validity, then claim-level groundedness via an LLM judge plus a safety flag. PII is masked unless it comes from your own documents. Low confidence → the orchestrator retries with broadened retrieval (filters dropped, more candidates, web added), then asks a clarifying question or answers "not enough evidence". |

## GPU budget (6 GB)

Only **llm_service** touches the GPU (unless `GPU_SWAP` is on, see below). Docling, the embedder, BM25 and the
reranker run on CPU, so plan for roughly 8–10 GB of host RAM. The Qwen3
embedder and reranker are loaded as float32 (`EMBED_DTYPE` / `RERANK_DTYPE`;
their bf16 checkpoints run about 4x slower on most CPUs) and take about
5 GB together, so rag_service's memory limit is 8G (`RAG_MEMORY_LIMIT`).

The AWQ checkpoint keeps embeddings, linear-attention layers and the vision
tower in BF16, so the weights take 3.85 GB of VRAM. On a Windows desktop
the display already uses about 1 GB, leaving roughly 5 GB for SGLang, so it
starts with:
- `--mem-fraction-static 0.92`, `--context-length 8192`, `--max-running-requests 2`
- `--max-mamba-cache-size 4` (~17k KV tokens)

These settings live in `.env` (`LLM_MEM_FRACTION`, `LLM_CONTEXT_LEN`,
`LLM_MAX_RUNNING`, `LLM_EXTRA_ARGS`). Startup takes about 5 minutes, most
of it CUDA-graph capture. If SGLang reports "Not enough memory", raise
`LLM_MEM_FRACTION` slightly, or lower `LLM_CONTEXT_LEN` to 4096. Do not add
`--language-only`: in SGLang 0.5.10 it selects encoder-disaggregated mode.

### GPU swap (`GPU_SWAP=true`)

Docling on CPU is slow (minutes per long PDF), so OCR and the LLM can take
turns on the GPU. `orchestrator_service/src/gpu_swap.py` owns the GPU and
hands out leases to the `llm` or `ocr` tenant:

- An upload stops `llm_service` (through the mounted docker socket) and
  starts ocr_service's Docling worker process on CUDA (`/gpu/enable`).
- Chat waits until the batch is done ("Waiting for GPU" step). After
  `GPU_SWAP_IDLE_SEC` (30 s) without OCR, the worker process exits, which
  frees all of its VRAM, and `llm_service` restarts (~5 min cold start).
- Documents indexed while the LLM was off get heuristic metadata and
  caption-only figures. Once the LLM is back the orchestrator calls
  `/ingest/sync?upgrade_heuristic=true`, which redoes them with the LLM.
- `GET /gpu/status` on the orchestrator shows the tenant, leases and swap
  state. If the GPU path fails, OCR falls back to the CPU converter.

The swap pays off for batches, not single files. `GPU_SWAP=false` keeps the
old layout: OCR on CPU and the LLM always up.

## Running

Requires Docker Compose and the NVIDIA container toolkit.

```bash
# 1. cp .env.example .env, then replace every CHANGEME: MASTER_PASSWORD,
#    MCP_TOKEN, CHAINLIT_AUTH_SECRET and the LANGFUSE_* secrets
#    (`openssl rand -hex 32` works for all of them; the Langfuse keys
#    must keep their pk-lf- / sk-lf- prefixes).
# 2. Optional: connect Google Drive (see below).
docker compose up -d --build
```

First start: llm_service needs a few minutes to load and capture CUDA graphs.
rag_service backfills the index for any documents already in MinIO.

| Service | Port |
|---|---|
| ui_service | `${UI_SERVICE_PORT:-8000}` → open `http://localhost:8000` (chat at `/chat/`) |
| Langfuse | `${LANGFUSE_PORT:-3000}` → `http://localhost:3000` (login: `LANGFUSE_INIT_USER_*`) |
| auth_service | `${AUTH_SERVICE_PORT:-8002}` |
| MinIO console | `${MINIO_CONSOLE_PORT:-9001}` |

All other services are internal to the `rag-network` Docker network.

### Google Drive (optional)

1. In Google Cloud Console, enable the **Drive API** and create an OAuth
   client of type **Desktop app**. Save it as `secrets/gdrive/credentials.json`.
2. On the host, run:
   ```bash
   pip install google-auth-oauthlib
   python rag_service/src/mcp_servers/gdrive_auth.py --secrets secrets/gdrive
   ```
   Approve read-only access; `secrets/gdrive/token.json` is written.
3. `docker compose restart rag_service`.

The Drive MCP connects **one** Google account, so only the roles in
`GDRIVE_ALLOWED_ROLES` (default `master,admin`) can search it. Service
accounts work too (`GDRIVE_AUTH=service_account`, with
`secrets/gdrive/service_account.json`).

### Chat UI (Chainlit)

`/chat/` is a [Chainlit](https://github.com/Chainlit/chainlit) app mounted in
ui_service ([`chat_app.py`](ui_service/src/chat_app.py), [`chat/`](ui_service/src/chat/)).
You sign in once at `/login`; the chat reuses that session.

| Feature | How |
|---|---|
| **Pipeline view** | Each answer shows a collapsible *RAG pipeline* step with the steps 3 → 10 (intent, rewrites, routes, retrieval counts, rerank scores, compression, groundedness). Retries appear as a second attempt. |
| **Citations** | `[1]`, `[2]` … in the answer are clickable: they open a side panel with the exact passage and a link to the document, Drive file or web page. |
| **Ingestion** | Drop PDF / DOCX / PNG / JPG files into the message box. They go through Docling OCR → chunking → indexing with live progress, then offer **Share with everyone** and **Summarize**. A question sent with the files is answered from them. |
| **Commands** | `/docs`, `/web`, `/drive` (Drive roles only), `/catalog` force where answers come from. |
| **Settings** | ⚙ in the composer: web search on/off, Google Drive on/off, show/hide pipeline steps, answer style (auto / concise / detailed). |
| **Your MCP servers** | Connect streamable-HTTP or SSE MCP servers from the plug icon. Only URLs starting with an entry of `MCP_USER_ALLOWED_URLS` are accepted (empty = disabled). For each question the LLM picks matching tools; calls show as tool steps and their results are reranked with the other evidence and cited as `server/tool`. |
| **History** | Threads, element contents and 👍/👎 feedback are stored in MinIO under `chat/` (sidebar history, resume a conversation with context). Deleting an account deletes its chat history. |

### MCP servers

rag_service hosts two FastMCP streamable-HTTP servers. The router uses them
server-side (the token never reaches the browser); any other MCP client can use
them with `Authorization: Bearer $MCP_TOKEN`:

| Endpoint | Tools |
|---|---|
| `http://rag_service:8003/mcp/gdrive/` | `search_files`, `read_file`, `get_file_metadata` |
| `http://rag_service:8003/mcp/websearch/` | `search`, `fetch_page` (refuses private/internal addresses) |

Web search uses `ddgs` by default (no key). Set `WEBSEARCH_PROVIDER=tavily`
or `brave` with `WEBSEARCH_API_KEY` for an API provider.

### Observability (Langfuse)

[`langfuse_service/`](langfuse_service/README.md) runs Langfuse next to the
pipeline (about 5.5 GB RAM, no GPU). With the keys in `.env` nothing has to
be set up: the project and its API keys are created on first start.

- **Traces:** one `rag-chat` trace per question with steps 3 → 10 as nested
  observations, every LLM call as a generation (messages, output, tokens,
  latency, prompt version), retrieval hits and rerank scores. Grouped by user
  and chat thread (Sessions view). Document ingestion is an `ingest-document` trace.
- **Scores:** `outcome`, `groundedness`, `citations_valid`, `answer_confidence`
  and `attempts` from validation; `user_feedback` from 👍/👎 in the chat.
- **Prompts:** the system prompts are `rag/*` prompts in Langfuse. Choose the
  version in `.env` with `PROMPT_SOURCE` (`langfuse` | `local`), `PROMPT_LABEL`
  (e.g. `production`, `staging`) and `PROMPT_VERSIONS` (pins such as
  `rag/answer:3`). If Langfuse is down the texts in the code are used.
- **Off switch:** `LANGFUSE_TRACING_ENABLED=false`.

## Key routes

Each service lists its routes in its `main.py` module docstring:
[ocr](ocr_service/src/main.py) · [rag](rag_service/src/main.py) ·
[llm](llm_service/src/main.py) · [orchestrator](orchestrator_service/src/main.py) ·
[ui](ui_service/src/main.py) · [auth](auth_service/src/main.py).

---

## UI pages

### Administration (master account)

The **master** account is the un-deletable super-admin seeded from `MASTER_USERNAME` / `MASTER_PASSWORD` on first boot. From the Administration page it can create users and admins, rename / change role / reset password / disable / delete any account (except itself), and the Profile dropdown shows the role beside the avatar. Regular admins see only `user` rows in the same table; the master and other admins are hidden from them.

### Single mode

Upload one file and see Markdown, DOCX preview, and JSON side-by-side.

### Batch mode

Drop multiple files; each row streams its own status and exposes download buttons as soon as it finishes.

### DOCX preview

Generated Word document rendered inline (DOCX → PDF via LibreOffice).

### MinIO object store

Uploads, generated artifacts (MD / JSON / DOCX), picture crops and each document's `meta.json` live under `documents/{uuid}/`; the RAG index registry lives under `rag/{uuid}/index.json`.

### Task history

Every OCR'd document is listed under `/history`, scoped to the logged-in user, with downloads (MD, DOCX, JSON, source) and a **Shared** toggle that makes the document searchable by every user in Chat.

## Notes

- Inter-service calls trust the `X-User-Id` / `X-User-Role` headers. ui_service
  strips any client-supplied copies, and backend ports are only reachable on
  the internal Docker network.
- There is no SQL database. ocr_service and rag_service each keep an
  in-memory index of their MinIO metadata and are its only writers, so run a
  single replica of each. Back up the `minio_data`, `qdrant_data` and
  `auth_data` volumes (and `langfuse_postgres_data` / `langfuse_clickhouse_data`
  to keep traces and prompt versions).
- `layout.json` is a [DoclingDocument](https://docling-project.github.io/docling/concepts/docling_document/).
  If you change `EMBED_MODEL`, use a new `QDRANT_COLLECTION` (rag_service
  refuses to start on a collection filled by another model); the next sync
  re-indexes every document. Changes to chunking bump `CHUNKER_VERSION` in
  `ingestion/pipeline.py`, which re-indexes the same way.
