# CI/CD

CI/CD for this repo runs on **GitHub Actions**. The workflow file is
[.github/workflows/ci.yml](.github/workflows/ci.yml).

- **CI (Continuous Integration):** every pull request and every push to `main` is checked
  automatically on GitHub's machines (lint, config validation, Docker builds).
- **CD (Continuous Delivery):** after the checks pass, the built images are published so they
  can be pulled instead of rebuilt.

Everything runs in GitHub's cloud. The repo is public, so it is free and nothing runs on the local PC.
Results are on the [Actions tab](https://github.com/mdmonsurali/Advanced-Hybrid-RAG/actions).

## Current pipeline

| Job | What it checks | Services |
|---|---|---|
| `lint` | Python syntax errors and undefined names (`ruff`, error rules only), and formatting (`ruff format --check`) | all |
| `compose` | `docker-compose.yml` is valid, using `.env.example` | n/a |
| `dockerfile-check` | Dockerfile mistakes and bad practices (`docker buildx build --check`) | all 6 |
| `build` | Image builds end to end | ui, auth, orchestrator, rag |

`ocr_service` (~16 GB) and `llm_service` (~64 GB) are too large to build on a free GitHub runner,
so they only get `lint` and `dockerfile-check`.

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
- [ ] **Step 8: Full lint.** Turn on all configured `ruff` rules. There are 236 issues, 148 of them auto-fixable;
  fix the rest by hand.
- [ ] **Step 9: Type checking.** Add `mypy` or `pyright`, review the results, then fix the issues or relax rules.

### Phase 3: Tests

- [ ] **Step 10: Unit tests.** Add `pytest` tests per service (e.g. app starts, `/health`
  responds, models mocked) and run them in CI.
- [ ] **Step 11: Integration tests.** Start the non-GPU parts (Qdrant, MinIO, RAG API with small
  models) and test them together.

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
- **Formatting before you push:** `pip install ruff==0.17.1`, then
  `ruff format auth_service llm_service ocr_service orchestrator_service rag_service ui_service langfuse_service`.
- **Secrets:** `.env` is gitignored and never reaches CI. CI uses `.env.example` (dummy values).
