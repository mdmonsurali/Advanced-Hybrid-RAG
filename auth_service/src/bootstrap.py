"""Idempotent master bootstrap.

Reads MASTER_USERNAME / MASTER_PASSWORD (with ADMIN_USERNAME / ADMIN_PASSWORD
accepted as fallback so old .env files keep working) and ensures that:

1. A user row with that username exists.
2. That row has role='master'.
3. That row is not disabled.

If the row already exists with the right state, nothing changes (idempotent).
If somebody manually demoted or disabled the master, the next start
auto-heals it back to a working master — that's the lockout-prevention
contract for this account.

Pre-existing installs from earlier versions will have one of:
  - an 'admin' bootstrap row → it gets promoted to 'master' and renamed to
    MASTER_USERNAME (typically 'master');
  - no users at all → fresh insert.
"""

from __future__ import annotations

import logging
import os

import db
from security import hash_password

log = logging.getLogger("auth_service.bootstrap")


def _read_credentials() -> tuple[str, str]:
    """Prefer the MASTER_* names; fall back to ADMIN_* for older .env files."""
    username = (os.getenv("MASTER_USERNAME") or os.getenv("ADMIN_USERNAME") or "master").strip()
    password = (os.getenv("MASTER_PASSWORD") or os.getenv("ADMIN_PASSWORD") or "").strip()
    if not username or not password:
        raise RuntimeError(
            "MASTER_USERNAME and MASTER_PASSWORD must be set "
            "(ADMIN_USERNAME / ADMIN_PASSWORD are accepted as fallback)."
        )
    return username, password


async def bootstrap_admin() -> None:
    username, password = _read_credentials()

    # 1. Find an existing master, if any.
    master = await db.find_first_user("master")
    if master:
        if master["disabled"]:
            await db.update_user(master["id"], disabled=False)
            log.info("bootstrap: re-enabled disabled master '%s'", master["username"])
        else:
            log.info(
                "bootstrap: master already exists (id=%s, username=%s)",
                master["id"],
                master["username"],
            )
        return

    # 2. No master. Look for the seeded username — could be an admin from an
    #    older install, or a demoted master that we should promote back. The
    #    match is case-insensitive so the same .env value always wins.
    seeded = await db.get_user_by_username(username)
    if seeded:
        await db.update_user(seeded["id"], role="master", disabled=False)
        log.info(
            "bootstrap: promoted existing '%s' (was role=%s) to master (id=%s)",
            username,
            seeded["role"],
            seeded["id"],
        )
        return

    # 3. Fall back to any pre-existing admin so the operator's previous
    #    credentials still work, just under the new username.
    old_admin = await db.find_first_user("admin")
    if old_admin:
        await db.update_user(old_admin["id"], role="master", username=username, disabled=False)
        log.info(
            "bootstrap: promoted legacy admin '%s' to master and renamed to '%s' (id=%s)",
            old_admin["username"],
            username,
            old_admin["id"],
        )
        return

    # 4. Fresh install — create the master from env.
    created = await db.create_user(
        username, hash_password(password), "master", must_change_password=False
    )
    log.info("bootstrap: created master user '%s' (id=%s)", username, created["id"])
