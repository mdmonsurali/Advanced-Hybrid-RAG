"""Integration tests against the running containers (see .github/workflows/ci.yml, job
`integration`): real auth_service, ui_service, rag_service (CPU models), Qdrant and MinIO.

The LLM, OCR and orchestrator need a GPU and are not started, so these tests cover the
paths that work without them (rag_service falls back to heuristic metadata).
"""

from __future__ import annotations

import os
import time

import httpx
import pytest
from minio import Minio

AUTH_URL = os.getenv("AUTH_URL", "http://localhost:8002")
UI_URL = os.getenv("UI_URL", "http://localhost:8000")
RAG_URL = os.getenv("RAG_URL", "http://localhost:8003")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "ocr")

# The values in .env.example, which CI starts the stack with.
MASTER = ("master", "CHANGEME")
MINIO_USER, MINIO_PASSWORD = "admin", "changeme"


def wait_until_up(url: str, timeout: float = 300) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=5)
            if resp.status_code == 200:
                return
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except httpx.HTTPError as e:
            last = repr(e)
        time.sleep(3)
    raise RuntimeError(f"{url} not up after {timeout}s ({last})")


@pytest.fixture(scope="session", autouse=True)
def stack_is_up():
    for url in (f"{AUTH_URL}/auth/health", f"{UI_URL}/health", f"{RAG_URL}/health"):
        wait_until_up(url)


@pytest.fixture(scope="session")
def minio() -> Minio:
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_USER, secret_key=MINIO_PASSWORD, secure=False)
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)
    return client
