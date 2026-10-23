# CI/CD

CI/CD for this repo runs on **GitHub Actions**. The workflow file is
[.github/workflows/ci.yml](.github/workflows/ci.yml).

- **CI (Continuous Integration):** every pull request and every push to `main` is checked
  automatically on GitHub's machines (lint, type checks, unit and integration tests, Docker builds).
- **CD (Continuous Delivery):** after the checks pass, the built images are published so they
  can be pulled instead of rebuilt.

Everything runs in GitHub's cloud. The repo is public, so it is free and nothing runs on the local PC.
Results are on the [Actions tab](https://github.com/mdmonsurali/Advanced-Hybrid-RAG/actions).

## Current pipeline

| Job | What it checks | Services |
|---|---|---|
| `lint` | Full `ruff check` rule set (project exceptions in `ruff.toml`), and formatting (`ruff format --check`) | all |
| `test` | Per service, with its real libraries: `pyright` in basic mode (`pyrightconfig.json`), then the unit tests in `<service>/tests` | all 6 |
| `compose` | `docker-compose.yml` is valid, using `.env.example` | n/a |
| `dockerfile-check` | Dockerfile mistakes and bad practices (`docker buildx build --check`) | all 6 |
| `build` | Image builds end to end | orchestrator |
| `integration` | Builds and starts auth, ui and rag (CPU models) with Qdrant and MinIO, then runs `tests/integration` against them over HTTP | auth, ui, rag |

`ocr_service` (~16 GB) and `llm_service` (~64 GB) are too large to build on a free GitHub runner,
and need a GPU, so they get `lint`, `test` and `dockerfile-check` but no image build.

## Steps

### Phase 1: Basic pipeline

- [x] **Step 1: Create a branch.** Work on `add-ci` so `main` stays safe.
- [x] **Step 2: Lint.** `ruff` checks for syntax errors and undefined names.
- [x] **Step 3: Compose check.** `docker compose --env-file .env.example config -q`.
- [x] **Step 4: Docker checks.** Dockerfile check for all 6 services, full builds for 4.
  - [x] Fixed: `llm_service` passed `HF_TOKEN` as a build `ARG`, which stored it in the image
    history. It is now a BuildKit build secret.
- [x] **Step 5: Push and run on GitHub.** The first run failed on GitHub's newer Docker tools; both causes are fixed.
  - [x] Fixed: the nested `${...:?}` on `MASTER_PASSWORD` in `docker-compose.yml` broke newer
    Compose versions. `auth_service/src/bootstrap.py` still rejects an empty password.
  - [x] Fixed: `SecretsUsedInArgOrEnv` false positive on `AUTH_DATA_FILE` in `auth_service/Dockerfile`.
  - [x] Updated `actions/checkout` to v5 and `actions/setup-python` to v6 (Node 24).
- [x] **Step 6: Pull request and merge.** PR [#1](https://github.com/mdmonsurali/Advanced-Hybrid-RAG/pull/1)
  was merged into `main` with all checks green.
- [x] **Step 6b: Stop duplicate runs.** Checks ran twice on a PR (once for `push`, once for
  `pull_request`). CI now runs on `push` only for `main`, plus on every pull request.

### Phase 2: Code quality

- [x] **Step 7: Formatting.** Auto-formatted 76 files once with `ruff format` (line length 100,
  set in `ruff.toml`). CI now runs `ruff format --check`, with ruff pinned to `0.17.1`. The
  formatting commit is listed in `.git-blame-ignore-revs`, so `git blame` skips it.
- [x] **Step 8: Full lint.** CI now runs the full `ruff check` rule set. Issues went from 235 to 0:
  - 182 were auto-fixed with `ruff check --fix` (modern type hints, sorted imports, unused imports).
  - 3 rules are switched off in `ruff.toml`, each with its reason:
    - `BLE001`: `except Exception` is used on purpose. Revisit in Step 10.
    - `SIM103`: guard clauses are kept.
    - `B008`: FastAPI `Depends`/`Body`/`File` etc. are allowed as default arguments.
  - 2 one-line exceptions with reasons: `# noqa: SIM102` in `orchestrator_service/src/gpu_swap.py`,
    and `# noqa: S110` in `rag_service/src/ingestion/pipeline.py`.
  - About 10 were fixed by hand: merged `if`/`with`, narrowed one `except`, `Field(default_factory=set)`, and others.
- [x] **Step 9: Type checking.** `pyright` 1.1.414 in basic mode; config is in `pyrightconfig.json`.
  - **CI job `type-check`:** one job per service. Each installs that service's real libraries (CPU-only
    torch for rag/ocr; for llm, the versions from the SGLang base image) and runs pyright on its `src/`.
  - **Result:** 130 issues (first run, in pyright's stricter default mode) went to 0.
  - `reportPrivateImportUsage` is switched off. docling-core does not mark its public names as
    re-exported, although `docling_core.types.doc` is its documented import path.
  - **Real fixes:**
    - `auth_service`: updating a deleted user now returns 404 instead of a 500.
    - `rag_service` executors: a search route that fails with a non-`Exception` error
      (e.g. a cancellation) is now marked unavailable instead of crashing the query.
    - `rag_service` web fetch: stops on a redirect without a target.
    - Clear errors replace silent `None` crashes: GPU lease missing, GPU worker not initialised,
      embedding dimension missing, Qdrant collection without the named dense vector.
  - **The rest:** type annotations, `None` checks pyright could not follow, and `cast` /
    `# pyright: ignore[...]` with a reason for library type gaps (MinIO `BinaryIO`, Chainlit
    `CommandDict`, Langfuse score overloads, docling `RefItem` alias).

### Phase 3: Tests

- [x] **Step 10: Unit tests.** `pytest` 9.1.1 (`requirements-test.txt`), 184 tests in
  `<service>/tests`. Each service has a `pytest.ini`, and the CI `test` job runs the tests after pyright.
  - **No real services needed:** fakes stand in for Qdrant, MinIO, the LLM, Langfuse and the other
    services, so the whole suite runs in seconds.
  - **Covered:**
    - auth: login, sessions and the master / admin / user rules
    - orchestrator: the 10-step pipeline (retries, fallbacks, errors), request validation and GPU-swap policy
    - llm: PII masking, the validation verdict, compression and citations
    - rag: access-control filters, routing rules and the SSRF guard on web fetches
    - ocr: who may read, share or delete which documents
    - ui: the login gate, safe redirects and links, and identity headers stripped from browser requests
  - `BLE001` (`except Exception`) is still switched off. With the tests in place it can be narrowed
    service by service later.
- [x] **Step 11: Integration tests.** The CI job `integration` builds and starts auth_service,
  ui_service, rag_service (real CPU models), Qdrant and MinIO with `docker compose`.
  `tests/integration/docker-compose.ci.yml` publishes the internal ports and turns off Langfuse.
  `tests/integration` then checks, over HTTP:
  - login through ui_service into auth_service
  - a document stored in MinIO the way ocr_service writes it is chunked, embedded, written to Qdrant
    and found again by hybrid search and rerank
  - private / shared access is applied, and deleting the document removes it from search
  - The LLM, OCR and orchestrator need a GPU and are not started. rag_service's fallback without
    the LLM (heuristic metadata) is what gets tested.

### Phase 4: Security and delivery

- [ ] **Step 12: Security scans.** Vulnerable packages (`pip-audit`), leaked secrets
  (`gitleaks`), image vulnerabilities (Trivy).
- [ ] **Step 13: CD.** Publish images to GitHub Container Registry (ghcr.io) on each merge to
  `main`, so the stack can be updated with `docker compose pull && docker compose up -d`.

## Notes

- **Logs:** GitHub only shows job logs to logged-in users. Locally, use the GitHub CLI:
  `gh run view <run-id> --log-failed`.
- **Runner versions:** GitHub's runners use newer Docker and Compose versions than a typical local
  install, so a check can pass locally and fail in CI (this happened in Step 5).
- **Before you push:** run `pip install ruff==0.17.1` once. Then, on the service folders
  (`auth_service llm_service ocr_service orchestrator_service rag_service ui_service langfuse_service`), run:
  - `ruff check --fix <folders>` to auto-fix lint issues, then fix whatever it still lists
  - `ruff format <folders>` to format the code
- **Type check locally:** VS Code (Pylance) reads `pyrightconfig.json`, so the editor shows the same
  errors as CI once the service's libraries are installed in the selected interpreter. To match CI on
  the command line, run `pip install pyright==1.1.414 -r <service>/requirements.txt`, then
  `pyright --pythonpath "$(which python)" <service>/src`.
- **Run the unit tests locally:** in an environment with the service's libraries, run
  `pip install -r requirements-test.txt`, then `cd <service>` and `python -m pytest`.
- **Integration tests locally:** they start containers with the stack's fixed container names
  and volumes, so run them on a machine without your own running stack (CI does this).
- **Secrets:** `.env` is gitignored and never reaches CI. CI uses `.env.example` (dummy values).
