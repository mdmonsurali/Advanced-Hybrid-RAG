"""Prompt registry: every system prompt llm_service sends, served from
Langfuse prompt management with the text in this package as fallback.

Which version is used is set in .env (a restart applies a change):

    PROMPT_SOURCE     langfuse | local   (local: only the texts in prompts/*.py)
    PROMPT_LABEL      label fetched for every prompt (production, staging, latest, ...)
    PROMPT_VERSIONS   pins that override the label, e.g. "rag/answer:3,rag/judge:2"
    PROMPT_CACHE_TTL  seconds a fetched prompt is cached; after that it is
                      refreshed in the background while the cached one is served

Resolving a prompt: PROMPT_SOURCE=local → local text; pinned → that version;
otherwise → PROMPT_LABEL, then production if the prompt has no version with
that label. When Langfuse is unreachable or the prompt does not exist, the
local text is used and Langfuse is not asked again for that prompt until the
TTL has passed, so answering never waits on Langfuse.

Templates use Langfuse's {{variable}} syntax, also in the local copies.
Missing prompts are created in Langfuse from the local text at startup
(`seed_missing`); `python -m prompts.seed --update` pushes changed local
texts as new versions.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any

log = logging.getLogger("llm_service.prompts")

SOURCE = os.getenv("PROMPT_SOURCE", "langfuse").strip().lower()
LABEL = os.getenv("PROMPT_LABEL", "production").strip() or "production"
CACHE_TTL = int(os.getenv("PROMPT_CACHE_TTL", "60"))
_KEYS = bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))
FETCH_TIMEOUT_SEC = 3

if SOURCE not in {"langfuse", "local"}:
    raise ValueError(f"PROMPT_SOURCE must be 'langfuse' or 'local', got {SOURCE!r}")


def _parse_versions(raw: str) -> dict[str, int]:
    pins: dict[str, int] = {}
    for item in filter(None, (p.strip() for p in raw.split(","))):
        name, sep, version = item.rpartition(":")
        if not sep or not name or not version.isdigit() or int(version) < 1:
            raise ValueError(f"PROMPT_VERSIONS entry {item!r} must look like 'rag/answer:3'")
        pins[name.strip()] = int(version)
    return pins


VERSIONS = _parse_versions(os.getenv("PROMPT_VERSIONS", ""))

# name -> local template, filled by register() as the prompt modules load.
LOCAL: dict[str, str] = {}
# name -> how it was last resolved (served on GET /prompts).
RESOLVED: dict[str, dict] = {}
_retry_after: dict[str, float] = {}     # per prompt: not found / unavailable
_down_until = 0.0                       # Langfuse unreachable: local texts until then
_lock = threading.Lock()
_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class Messages(list):
    """A chat message list that remembers the Langfuse prompt it was built
    from, so the LLM call can link its generation to that prompt version."""

    def __init__(self, messages: list[dict], prompt: Any = None, prompt_name: str | None = None):
        super().__init__(messages)
        self.prompt = prompt
        self.prompt_name = prompt_name


def register(name: str, template: str) -> str:
    LOCAL[name] = template
    return template


def use_langfuse() -> bool:
    return SOURCE == "langfuse" and _KEYS


def _compile_local(template: str, variables: dict) -> str:
    return _VAR_RE.sub(lambda m: str(variables.get(m.group(1), m.group(0))), template)


def _fetch(name: str):
    """(Langfuse prompt, label it was found under) for `name`, or (None, None)
    to use the local text."""
    global _down_until
    now = time.monotonic()
    if not use_langfuse() or now < _down_until or now < _retry_after.get(name, 0.0):
        return None, None
    from langfuse.api import NotFoundError

    from tracing import langfuse
    pinned = VERSIONS.get(name)
    # A prompt that has no version under PROMPT_LABEL (e.g. staging) uses its
    # production version, so one label can be tried on a few prompts at a time.
    labels = [None] if pinned else list(dict.fromkeys([LABEL, "production"]))
    error: Exception | None = None
    unreachable = False
    for label in labels:
        try:
            prompt = langfuse.get_prompt(
                name, version=pinned, label=label, type="text", cache_ttl_seconds=CACHE_TTL,
                max_retries=0, fetch_timeout_seconds=FETCH_TIMEOUT_SEC)
            return prompt, label
        except NotFoundError as e:
            error = e
        except Exception as e:
            error, unreachable = e, True
            break
    backoff = time.monotonic() + max(CACHE_TTL, 30)
    with _lock:
        first = name not in _retry_after
        _retry_after[name] = backoff
        if unreachable:             # one outage: stop asking for every prompt
            _down_until = backoff
    if first:
        want = f"version {pinned}" if pinned else f"label {LABEL!r}"
        log.warning("[prompts] %s (%s) unavailable in Langfuse, using the local text: %s",
                    name, want, str(error).splitlines()[0][:200])
    return None, None


def resolve(name: str, **variables: Any) -> tuple[str, Any]:
    """(compiled text, Langfuse prompt or None) for a registered prompt."""
    prompt, label = _fetch(name)
    if prompt is not None:
        _retry_after.pop(name, None)
        RESOLVED[name] = {"source": "langfuse", "version": prompt.version, "label": label,
                          "pinned": name in VERSIONS}
        return prompt.compile(**variables), prompt
    RESOLVED[name] = {"source": "local" if SOURCE == "local" or not _KEYS else "local-fallback",
                      "version": None, "label": None, "pinned": False}
    return _compile_local(LOCAL[name], variables), None


def text(name: str, **variables: Any) -> str:
    return resolve(name, **variables)[0]


def messages(name: str, rest: list[dict], **variables: Any) -> Messages:
    """System prompt `name` followed by `rest` (history, user message)."""
    system, prompt = resolve(name, **variables)
    return Messages([{"role": "system", "content": system}, *rest],
                    prompt=prompt, prompt_name=name)


def status() -> dict:
    return {"source": SOURCE if _KEYS or SOURCE == "local" else "local (no Langfuse keys)",
            "label": LABEL, "pinned_versions": VERSIONS, "cache_ttl_sec": CACHE_TTL,
            "prompts": {name: RESOLVED.get(name, {"source": "not used yet"}) for name in sorted(LOCAL)}}


def seed_missing() -> None:
    """Create prompts that don't exist in Langfuse yet from the local text
    (labelled production), then warm the cache. Existing prompts are never
    changed here, so edits made in the Langfuse UI stay."""
    if not use_langfuse():
        log.info("[prompts] source=%s: using the local prompt texts", status()["source"])
        return
    from langfuse.api import NotFoundError

    from tracing import langfuse
    for name in sorted(LOCAL):
        try:
            langfuse.get_prompt(name, type="text", cache_ttl_seconds=0, max_retries=0,
                                fetch_timeout_seconds=FETCH_TIMEOUT_SEC)
        except NotFoundError:
            try:
                langfuse.create_prompt(name=name, prompt=LOCAL[name], type="text",
                                       labels=["production"],
                                       commit_message="Seeded from llm_service/src/prompts")
                log.info("[prompts] created %s in Langfuse (label production)", name)
            except Exception as e:
                log.warning("[prompts] could not create %s: %s", name, e)
                return
        except Exception as e:
            log.warning("[prompts] Langfuse unreachable, using the local texts for now: %s",
                        str(e).splitlines()[0][:200])
            return
    for name in sorted(LOCAL):
        resolve(name)
    unknown = sorted(set(VERSIONS) - set(LOCAL))
    if unknown:
        log.warning("[prompts] PROMPT_VERSIONS names unknown prompts: %s", ", ".join(unknown))
    log.info("[prompts] %s", {n: (r["source"], r["version"]) for n, r in RESOLVED.items()})
