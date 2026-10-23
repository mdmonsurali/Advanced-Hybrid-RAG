"""Push the local prompt texts (prompts/*.py) to Langfuse.

    python -m prompts.seed            create prompts that don't exist yet (label production)
    python -m prompts.seed --update   also add a new version, labelled production, for every
                                      prompt whose local text differs from its production version
    python -m prompts.seed --label staging --update
                                      the same against another label, e.g. to try a change first

llm_service already creates missing prompts at startup, so this is only
needed to publish edits made in the code. Inside the container:

    docker compose exec llm_service python3 -m prompts.seed --update
"""

from __future__ import annotations

import argparse
import logging

import tracing
from generation import answer, compression, figures, query_analysis, tools, validation  # noqa: F401  (register every prompt)

from prompts import registry


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().split("\n")[0])
    parser.add_argument(
        "--update", action="store_true", help="publish changed local texts as new versions"
    )
    parser.add_argument("--label", default="production", help="label to compare with and set")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("seed")

    if not registry._KEYS:
        log.error("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set")
        return 1
    from langfuse.api import NotFoundError

    lf = tracing.langfuse
    for name in sorted(registry.LOCAL):
        local = registry.LOCAL[name]
        try:
            current = lf.get_prompt(name, label=args.label, type="text", cache_ttl_seconds=0)
        except NotFoundError:
            current = None
        if current is not None and current.prompt == local:
            log.info("%-22s unchanged (v%d)", name, current.version)
            continue
        if current is not None and not args.update:
            log.info(
                "%-22s differs from %s v%d (use --update to publish)",
                name,
                args.label,
                current.version,
            )
            continue
        created = lf.create_prompt(
            name=name,
            prompt=local,
            type="text",
            labels=[args.label],
            commit_message="Published from llm_service/src/prompts",
        )
        log.info(
            "%-22s %s v%d (%s)",
            name,
            "updated to" if current else "created",
            created.version,
            args.label,
        )
    lf.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
