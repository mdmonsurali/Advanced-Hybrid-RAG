# langfuse_service

Self-hosted [Langfuse](https://langfuse.com) v4 for the RAG pipeline: one
trace per chat question, scores on every answer, and versioned prompts. The
containers are defined in the root [`docker-compose.yml`](../docker-compose.yml)
(section *Observability*); this folder only holds their config.

## What runs

| Container | Image | Memory cap | Role |
|---|---|---|---|
| `langfuse_service` | `langfuse/langfuse:4` | 2G (`LANGFUSE_WEB_MEMORY_LIMIT`) | UI + public API, published on `${LANGFUSE_PORT:-3000}` |
| `langfuse-worker` | `langfuse/langfuse-worker:4` | 1G (`LANGFUSE_WORKER_MEMORY_LIMIT`) | ingestion queue → ClickHouse |
| `langfuse-clickhouse` | `clickhouse-server:25.12` | 2G (`LANGFUSE_CLICKHOUSE_MEMORY_LIMIT`) | traces, observations, scores |
| `langfuse-postgres` | `postgres:17` | 512M | users, projects, prompts |
| `langfuse-redis` | `redis:7` | 256M | queue + cache |

That is about 5.5 GB of RAM on top of the RAG services (CPU only, no GPU).
Langfuse's raw events go to the existing MinIO, in the bucket `langfuse`
(created by `minio-init`). Only the web UI is published; the databases are
reachable on `rag-network` alone.

## First start

`docker compose up -d` starts it with everything else. On first boot
`langfuse_service` creates, from `.env`:

- the organisation and the project `advanced-hybrid-rag`,
- the project API keys `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` that the
  services use (so nothing has to be set up in the UI),
- the login `LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD`.

Open `http://localhost:3000` and sign in. Sign-up is disabled.

## What gets recorded

```
rag-chat                       orchestrator   user, session (= chat thread), tags
├─ query-analysis              llm_service    ├─ understanding  (generation)
│                                             └─ transform      (generation)
├─ route                       rag_service    (└─ route-llm → route generation, when rules can't decide)
├─ retrieve      (retriever)   rag_service    ├─ search-docs (retriever) / gdrive / web-search (tool)
│                                             └─ rerank
├─ compress                    llm_service    └─ extract        (generation, per long chunk)
├─ generate                    llm_service    └─ answer         (generation)
└─ validate      (evaluator)   llm_service    └─ judge          (generation)
scores: outcome, groundedness, citations_valid, answer_confidence, attempts, user_feedback (👍/👎)

ingest-document                rag_service    owner as user
└─ doc-metadata                llm_service    └─ doc-metadata   (generation)
```

Generations carry the exact messages, the output, token usage, latency and
the prompt name + version they were built from. A retry shows up as a second
`retrieve … validate` round in the same trace.

Services pass the trace to each other with two internal headers,
`X-Langfuse-Trace-Id` and `X-Langfuse-Parent-Span-Id` (each service's
`src/tracing.py`). ui_service strips them from browser requests.

## Prompts

Every system prompt lives in `llm_service/src/prompts/*.py` and in Langfuse
under `rag/…` (`rag/answer`, `rag/judge`, `rag/understanding`, …). llm_service
creates missing ones at startup; after that, edit them in the Langfuse UI.

Which version llm_service uses is set in `.env` (restart `llm_service` to apply):

| Setting | Effect |
|---|---|
| `PROMPT_SOURCE=langfuse` | fetch prompts from Langfuse (default) |
| `PROMPT_SOURCE=local` | ignore Langfuse, use the texts in the code |
| `PROMPT_LABEL=production` | label fetched for every prompt; a prompt without a version under this label uses `production` |
| `PROMPT_VERSIONS=rag/answer:3,rag/judge:2` | pin single prompts to a version (overrides the label) |
| `PROMPT_CACHE_TTL=60` | seconds before a prompt is refreshed (in the background) |

`GET http://llm_service:8005/prompts` shows what each prompt resolved to. If
Langfuse is down, the local texts are used and answering does not wait for it.

To publish a prompt edited in the code as a new version:

```bash
docker compose exec llm_service python3 -m prompts.seed --update
```

## Turning it off

- `LANGFUSE_TRACING_ENABLED=false` stops all traces and scores (prompts still
  come from Langfuse unless `PROMPT_SOURCE=local`).
- To not run the containers at all, start the stack without them, e.g.
  `docker compose up -d --scale langfuse_service=0 --scale langfuse-worker=0 ...`,
  or remove the *Observability* section from `docker-compose.yml`, and set
  `PROMPT_SOURCE=local`.

## Data

Langfuse data lives in the `langfuse_postgres_data` and
`langfuse_clickhouse_data` volumes and under `langfuse/` in MinIO. Traces
contain the users' questions and the document passages sent to the LLM, so
treat them like the documents themselves.
