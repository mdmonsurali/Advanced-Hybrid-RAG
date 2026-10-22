"""Chainlit chat history stored in MinIO (no SQL database).

Layout in the shared bucket:
    chat/users/{user_id}.json                  persisted Chainlit user
    chat/index/{user_id}.json                  thread summaries for the sidebar
    chat/threads/{thread_id}.json              thread + steps + element metadata
    chat/elements/{thread_id}/{element_id}     element content (citation text, small files)
    chat/feedback/{feedback_id}.json           thumbs up/down (seed for evaluations)

Feedback is also sent to Langfuse as a `user_feedback` score on the answer's
trace (chat/feedback_scores.py).

The user's auth id is used as both the Chainlit identifier and the persisted
id, so history survives username changes. Threads are cached in memory and
written through on every change; ui_service is the only writer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from chainlit.data.base import BaseDataLayer
from chainlit.types import (
    Feedback,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User
from minio import Minio
from minio.deleteobjects import DeleteObject
from minio.error import S3Error

from chat import feedback_scores

log = logging.getLogger("ui_service.chat_history")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "changeme")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "ocr")
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Served by ui_service (main.py) with an ownership check.
ELEMENT_URL_PREFIX = "/chat-files"
MAX_ELEMENT_BYTES = int(os.getenv("CHAT_MAX_ELEMENT_BYTES", str(10 * 1024 * 1024)))
THREAD_CACHE_SIZE = 256


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Bytes:
    __slots__ = ("_b", "_pos")

    def __init__(self, b: bytes):
        self._b, self._pos = b, 0

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self._pos + n > len(self._b):
            chunk, self._pos = self._b[self._pos :], len(self._b)
            return chunk
        chunk = self._b[self._pos : self._pos + n]
        self._pos += n
        return chunk


class MinioDataLayer(BaseDataLayer):
    def __init__(self, client: Optional[Minio] = None):
        self.client = client or Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
            region=MINIO_REGION,
        )
        self._threads: "OrderedDict[str, dict]" = OrderedDict()
        self._step_thread: Dict[str, str] = {}  # step id -> thread id
        self._feedback_step: Dict[str, str] = {}  # feedback id -> step id
        self._locks: Dict[str, asyncio.Lock] = {}

    # ── MinIO helpers ───────────────────────────────────────────────────

    def _get(self, key: str) -> Optional[bytes]:
        try:
            resp = self.client.get_object(MINIO_BUCKET, key)
        except S3Error as e:
            if e.code in {"NoSuchKey", "NoSuchObject"}:
                return None
            raise
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def _put(self, key: str, body: bytes, content_type: str = "application/json") -> None:
        self.client.put_object(
            MINIO_BUCKET, key, data=_Bytes(body), length=len(body), content_type=content_type
        )

    def _remove_prefix(self, prefix: str) -> None:
        keys = [
            DeleteObject(o.object_name)
            for o in self.client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True)
        ]
        for err in self.client.remove_objects(MINIO_BUCKET, keys):
            log.warning("minio delete error: %s", err)

    async def _get_json(self, key: str):
        raw = await asyncio.to_thread(self._get, key)
        return json.loads(raw) if raw else None

    async def _put_json(self, key: str, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        await asyncio.to_thread(self._put, key, body)

    def _lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    # ── Threads (cache + write-through) ─────────────────────────────────

    def _remember(self, thread: dict) -> None:
        self._threads[thread["id"]] = thread
        self._threads.move_to_end(thread["id"])
        while len(self._threads) > THREAD_CACHE_SIZE:
            old_id, old = self._threads.popitem(last=False)
            for step in old.get("steps", []):
                self._step_thread.pop(step["id"], None)
        for step in thread.get("steps", []):
            self._step_thread[step["id"]] = thread["id"]
            fb = step.get("feedback")
            if fb and fb.get("id"):
                self._feedback_step[fb["id"]] = step["id"]

    async def _load(self, thread_id: str) -> Optional[dict]:
        if thread_id in self._threads:
            self._threads.move_to_end(thread_id)
            return self._threads[thread_id]
        thread = await self._get_json(f"chat/threads/{thread_id}.json")
        if thread:
            self._remember(thread)
        return thread

    async def _load_or_new(self, thread_id: str) -> dict:
        thread = await self._load(thread_id)
        if thread is None:
            thread = {
                "id": thread_id,
                "createdAt": _now(),
                "updatedAt": _now(),
                "name": None,
                "userId": None,
                "userIdentifier": None,
                "tags": [],
                "metadata": {},
                "steps": [],
                "elements": [],
            }
            self._remember(thread)
        return thread

    async def _save(self, thread: dict) -> None:
        # Chainlit creates a thread record for every connection; only persist
        # threads that actually have content.
        if not thread.get("name") and not thread.get("steps"):
            return
        thread["updatedAt"] = _now()
        await self._put_json(f"chat/threads/{thread['id']}.json", thread)
        if thread.get("userId"):
            await self._update_index(thread)

    async def _update_index(self, thread: dict, remove: bool = False) -> None:
        user_id = thread["userId"]
        async with self._lock(f"index:{user_id}"):
            index = await self._get_json(f"chat/index/{user_id}.json") or []
            index = [t for t in index if t["id"] != thread["id"]]
            if not remove:
                feedback = sorted(
                    {s["feedback"]["value"] for s in thread.get("steps", []) if s.get("feedback")}
                )
                index.append(
                    {
                        "id": thread["id"],
                        "name": thread.get("name"),
                        "createdAt": thread["createdAt"],
                        "updatedAt": thread["updatedAt"],
                        "tags": thread.get("tags") or [],
                        "feedback": feedback,
                    }
                )
            await self._put_json(f"chat/index/{user_id}.json", index)

    # ── Users ───────────────────────────────────────────────────────────

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        data = await self._get_json(f"chat/users/{identifier}.json")
        if not data:
            return None
        return PersistedUser(
            id=data["id"],
            createdAt=data["createdAt"],
            identifier=data["identifier"],
            display_name=data.get("display_name"),
            metadata=data.get("metadata") or {},
        )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        existing = await self._get_json(f"chat/users/{user.identifier}.json") or {}
        data = {
            "id": user.identifier,
            "identifier": user.identifier,
            "display_name": user.display_name,
            "metadata": user.metadata or {},
            "createdAt": existing.get("createdAt") or _now(),
        }
        await self._put_json(f"chat/users/{user.identifier}.json", data)
        return PersistedUser(
            **{k: data[k] for k in ("id", "createdAt", "identifier", "display_name", "metadata")}
        )

    # ── Steps ───────────────────────────────────────────────────────────

    async def _upsert_step(self, step_dict: dict) -> None:
        thread_id = step_dict.get("threadId")
        if not thread_id:
            return
        async with self._lock(thread_id):
            thread = await self._load_or_new(thread_id)
            steps = thread["steps"]
            for i, existing in enumerate(steps):
                if existing["id"] == step_dict["id"]:
                    merged = {**existing, **{k: v for k, v in step_dict.items() if v is not None}}
                    steps[i] = merged
                    break
            else:
                steps.append(dict(step_dict))
            self._step_thread[step_dict["id"]] = thread_id
            await self._save(thread)

    async def create_step(self, step_dict):
        await self._upsert_step(step_dict)

    async def update_step(self, step_dict):
        await self._upsert_step(step_dict)

    async def delete_step(self, step_id: str):
        thread_id = self._step_thread.pop(step_id, None)
        if not thread_id:
            return
        async with self._lock(thread_id):
            thread = await self._load(thread_id)
            if thread:
                thread["steps"] = [s for s in thread["steps"] if s["id"] != step_id]
                await self._save(thread)

    # ── Elements ────────────────────────────────────────────────────────

    async def create_element(self, element):
        thread_id = getattr(element, "thread_id", None)
        if not thread_id:
            return
        content = element.content
        if content is None and element.path and Path(element.path).is_file():
            if Path(element.path).stat().st_size <= MAX_ELEMENT_BYTES:
                content = await asyncio.to_thread(Path(element.path).read_bytes)
        el = element.to_dict()
        el.pop("chainlitKey", None)
        el.pop("objectKey", None)
        if content is not None:
            body = content.encode("utf-8") if isinstance(content, str) else content
            if len(body) <= MAX_ELEMENT_BYTES:
                mime = getattr(element, "mime", None) or "text/plain"
                await asyncio.to_thread(
                    self._put, f"chat/elements/{thread_id}/{element.id}", body, mime
                )
                el["url"] = f"{ELEMENT_URL_PREFIX}/{thread_id}/{element.id}"
        async with self._lock(thread_id):
            thread = await self._load_or_new(thread_id)
            thread["elements"] = [e for e in thread["elements"] if e["id"] != el["id"]] + [el]
            await self._save(thread)

    async def get_element(self, thread_id: str, element_id: str):
        thread = await self._load(thread_id)
        if not thread:
            return None
        return next((e for e in thread["elements"] if e["id"] == element_id), None)

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        if not thread_id:
            return
        async with self._lock(thread_id):
            thread = await self._load(thread_id)
            if thread:
                thread["elements"] = [e for e in thread["elements"] if e["id"] != element_id]
                await self._save(thread)
        await asyncio.to_thread(self._remove_prefix, f"chat/elements/{thread_id}/{element_id}")

    async def element_content(self, thread_id: str, element_id: str) -> Optional[tuple[bytes, str]]:
        """For the /chat-files route: (bytes, mime) of a stored element."""
        el = await self.get_element(thread_id, element_id)
        if el is None:
            return None
        body = await asyncio.to_thread(self._get, f"chat/elements/{thread_id}/{element_id}")
        return (body, el.get("mime") or "text/plain") if body is not None else None

    # ── Threads API ─────────────────────────────────────────────────────

    async def get_thread_author(self, thread_id: str) -> str:
        thread = await self._load(thread_id)
        return (thread or {}).get("userIdentifier") or ""

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        async with self._lock(thread_id):
            thread = await self._load_or_new(thread_id)
            if name is not None:
                thread["name"] = name[:200]
            if user_id is not None:
                thread["userId"] = user_id
                thread["userIdentifier"] = user_id
            if metadata is not None:
                thread["metadata"] = {**(thread.get("metadata") or {}), **metadata}
            if tags is not None:
                thread["tags"] = tags
            await self._save(thread)

    async def delete_thread(self, thread_id: str):
        async with self._lock(thread_id):
            thread = await self._load(thread_id)
            self._threads.pop(thread_id, None)
            await asyncio.to_thread(self._remove_prefix, f"chat/threads/{thread_id}.json")
            await asyncio.to_thread(self._remove_prefix, f"chat/elements/{thread_id}/")
            if thread and thread.get("userId"):
                await self._update_index(thread, remove=True)

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        if not filters.userId:
            return PaginatedResponse(
                pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None), data=[]
            )
        index = await self._get_json(f"chat/index/{filters.userId}.json") or []
        rows = [t for t in index if t.get("name")]
        if filters.search:
            needle = filters.search.lower()
            rows = [t for t in rows if needle in (t.get("name") or "").lower()]
        if filters.feedback is not None:
            rows = [t for t in rows if filters.feedback in (t.get("feedback") or [])]
        rows.sort(key=lambda t: t.get("updatedAt") or "", reverse=True)

        start = 0
        if pagination.cursor:
            ids = [t["id"] for t in rows]
            start = ids.index(pagination.cursor) + 1 if pagination.cursor in ids else 0
        page = rows[start : start + pagination.first]
        data = [
            ThreadDict(
                id=t["id"],
                createdAt=t["createdAt"],
                name=t.get("name"),
                userId=filters.userId,
                userIdentifier=filters.userId,
                tags=t.get("tags") or [],
                metadata={},
                steps=[],
                elements=[],
            )
            for t in page
        ]
        return PaginatedResponse(
            pageInfo=PageInfo(
                hasNextPage=start + pagination.first < len(rows),
                startCursor=page[0]["id"] if page else None,
                endCursor=page[-1]["id"] if page else None,
            ),
            data=data,
        )

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        thread = await self._load(thread_id)
        if not thread:
            return None
        steps = sorted(
            thread.get("steps", []), key=lambda s: s.get("createdAt") or s.get("start") or ""
        )
        return ThreadDict(
            id=thread["id"],
            createdAt=thread["createdAt"],
            name=thread.get("name"),
            userId=thread.get("userId"),
            userIdentifier=thread.get("userIdentifier"),
            tags=thread.get("tags") or [],
            metadata=thread.get("metadata") or {},
            steps=steps,
            elements=thread.get("elements") or [],
        )

    # ── Feedback ────────────────────────────────────────────────────────

    async def upsert_feedback(self, feedback: Feedback) -> str:
        feedback_id = feedback.id or str(uuid.uuid4())
        thread_id = feedback.threadId or self._step_thread.get(feedback.forId)
        trace_id = None
        if thread_id:
            async with self._lock(thread_id):
                thread = await self._load(thread_id)
                for step in (thread or {}).get("steps", []):
                    if step["id"] == feedback.forId:
                        step["feedback"] = {
                            "id": feedback_id,
                            "forId": feedback.forId,
                            "value": feedback.value,
                            "comment": feedback.comment,
                        }
                        self._feedback_step[feedback_id] = step["id"]
                        trace_id = (step.get("metadata") or {}).get("langfuse_trace_id")
                        await self._save(thread)
                        break
        await feedback_scores.send(feedback_id, trace_id, feedback.value, feedback.comment)
        await self._put_json(
            f"chat/feedback/{feedback_id}.json",
            {
                "id": feedback_id,
                "threadId": thread_id,
                "forId": feedback.forId,
                "value": feedback.value,
                "comment": feedback.comment,
                "at": _now(),
            },
        )
        return feedback_id

    async def delete_feedback(self, feedback_id: str) -> bool:
        step_id = self._feedback_step.pop(feedback_id, None)
        thread_id = self._step_thread.get(step_id) if step_id else None
        if thread_id:
            async with self._lock(thread_id):
                thread = await self._load(thread_id)
                for step in (thread or {}).get("steps", []):
                    if step["id"] == step_id:
                        step.pop("feedback", None)
                        await self._save(thread)
        await asyncio.to_thread(self._remove_prefix, f"chat/feedback/{feedback_id}.json")
        await feedback_scores.delete(feedback_id)
        return True

    async def get_favorite_steps(self, user_id: str) -> List:
        return []  # favorites are disabled in config.toml

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        pass

    # ── Account deletion ────────────────────────────────────────────────

    async def delete_user_data(self, user_id: str) -> int:
        index = await self._get_json(f"chat/index/{user_id}.json") or []
        for t in index:
            self._threads.pop(t["id"], None)
            await asyncio.to_thread(self._remove_prefix, f"chat/threads/{t['id']}.json")
            await asyncio.to_thread(self._remove_prefix, f"chat/elements/{t['id']}/")
        await asyncio.to_thread(self._remove_prefix, f"chat/index/{user_id}.json")
        await asyncio.to_thread(self._remove_prefix, f"chat/users/{user_id}.json")
        log.info("deleted chat history of %s (%d threads)", user_id, len(index))
        return len(index)
