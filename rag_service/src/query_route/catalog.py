"""Catalog route — structured questions about the document collection
("how many invoices did I upload in May?", "list my contracts"), answered
from the index registry instead of vector search.

Builds one evidence block with totals by type and month plus a table of the
matching documents the user may see (own + shared).
"""

from __future__ import annotations

import os
from collections import Counter

import storage
from identity import Identity
from schemas import Evidence, QueryPlan

CATALOG_MAX_ROWS = int(os.getenv("RAG_CATALOG_MAX_ROWS", "50"))


def _visible(entry: dict, identity: Identity) -> bool:
    return entry.get("owner_id") == str(identity.id) or entry.get("access_level") == "shared"


def _matches(entry: dict, plan: QueryPlan) -> bool:
    f = plan.filters
    created = (entry.get("created_at") or "")[:10]
    if f.doc_type and (entry.get("doc_type") or "").lower() != f.doc_type.lower():
        return False
    if f.date_from and created and created < f.date_from[:10]:
        return False
    if f.date_to and created and created > f.date_to[:10]:
        return False
    return True


def catalog_evidence(plan: QueryPlan, identity: Identity) -> list[Evidence]:
    entries = [e for e in storage.REGISTRY.values()
               if e.get("status") == "ok" and _visible(e, identity) and _matches(e, plan)]
    entries.sort(key=lambda e: e.get("created_at") or "", reverse=True)

    own = sum(1 for e in entries if e.get("owner_id") == str(identity.id))
    by_type = Counter(e.get("doc_type") or "other" for e in entries)
    by_month = Counter((e.get("created_at") or "")[:7] or "unknown" for e in entries)

    lines = [
        f"Document catalog: {len(entries)} documents match "
        f"({own} uploaded by you, {len(entries) - own} shared by others).",
    ]
    f = plan.filters
    if f.doc_type or f.date_from or f.date_to:
        lines.append(f"Filters applied: type={f.doc_type or 'any'}, "
                     f"from={f.date_from or '-'}, to={f.date_to or '-'}.")
    if by_type:
        lines.append("By type: " + ", ".join(f"{t}: {n}" for t, n in by_type.most_common()))
    if by_month:
        lines.append("By upload month: " + ", ".join(f"{m}: {n}" for m, n in sorted(by_month.items())))
    if entries:
        lines.append("")
        lines.append("| Name | Type | Topic | Pages | Uploaded | Owner |")
        lines.append("|---|---|---|---|---|---|")
        for e in entries[:CATALOG_MAX_ROWS]:
            owner = "you" if e.get("owner_id") == str(identity.id) else "shared"
            lines.append(f"| {e.get('source', '')} | {e.get('doc_type', '')} | "
                         f"{(e.get('topic') or '').replace('|', '/')} | {e.get('page_count') or ''} | "
                         f"{(e.get('created_at') or '')[:10]} | {owner} |")
        if len(entries) > CATALOG_MAX_ROWS:
            lines.append(f"(+{len(entries) - CATALOG_MAX_ROWS} more not listed)")

    return [Evidence(id="catalog:0", route="catalog", text="\n".join(lines), score=1.0,
                     source="Document catalog", owned=True)]
