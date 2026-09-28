"""SQLite-backed prompt→response cache for paid LLM APIs.

A single-file cache that stores deterministic (``temperature == 0``) responses
so that re-runs of stage 2/3/4 and evaluation can skip repeat API calls.

The cache is keyed on a SHA-256 of a canonical JSON blob of every input that
affects the response (backend, model, call type, system, prompt, temperature,
max_tokens). Non-deterministic calls (``temperature > 0``) bypass the cache.

Concurrency: a 30-second busy timeout plus a process-local lock; safe for the
≤8 concurrent workers the project uses. Inserts are ``INSERT OR IGNORE`` so
races are harmless. The default journal mode is ``TRUNCATE`` rather than
``WAL`` because the cache file typically lives on NFS/GPFS scratch, where
WAL's shared-mmap locking silently fails with ``SQLITE_PROTOCOL`` ("locking
protocol") under concurrent access and tanks the whole pipeline. Override
with ``cache.journal_mode: wal`` (or ``LLM_CACHE_JOURNAL_MODE=wal``) when
the cache is on local disk.

Resilience: every DB operation is guarded with a ``sqlite3.Error`` catch. If
the backing file goes sideways mid-run (protocol error, disk full, etc.) the
cache auto-disables itself after logging once, and the caller transparently
falls through to a live LLM call. A cache failure must never kill the run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_cache (
  key          TEXT PRIMARY KEY,
  backend      TEXT NOT NULL,
  model        TEXT NOT NULL,
  call_type    TEXT NOT NULL,
  temperature  REAL NOT NULL,
  max_tokens   INTEGER NOT NULL,
  prompt_sha   TEXT NOT NULL,
  prompt       TEXT NOT NULL,
  system       TEXT,
  response     TEXT NOT NULL,
  created_at   TEXT NOT NULL,
  hit_count    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_model ON llm_cache(model);
"""


def make_cache_key(
    *,
    backend: str,
    model: str,
    call_type: str,
    system: str | None,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Return the SHA-256 hex digest used as the primary cache key."""
    blob = json.dumps(
        {
            "backend": backend,
            "model": model,
            "call": call_type,
            "system": system or "",
            "prompt": prompt,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "cache_schema": CACHE_SCHEMA_VERSION,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


_VALID_JOURNAL_MODES = {"wal", "truncate", "delete", "memory", "persist", "off"}


def _resolve_journal_mode(db_path: Path, configured: str | None) -> str:
    """Pick a SQLite journal_mode that survives the filesystem under db_path.

    Priority: env var ``LLM_CACHE_JOURNAL_MODE`` > configured value > default.
    The default is ``TRUNCATE`` because the cache file typically sits on
    scratch/NFS/GPFS where WAL's shared-mmap locking silently breaks.
    """
    env_override = os.environ.get("LLM_CACHE_JOURNAL_MODE")
    candidate = (env_override or configured or "truncate").strip().lower()
    if candidate not in _VALID_JOURNAL_MODES:
        logger.warning(
            "Unknown journal_mode %r for LLM cache; falling back to TRUNCATE", candidate
        )
        return "truncate"
    return candidate


class LLMCache:
    """Thin SQLite wrapper. One instance per process; thread-safe for our uses.

    DB operations are best-effort: if SQLite raises, the cache auto-disables
    itself (logging once) and callers fall through to live LLM calls.
    """

    def __init__(
        self,
        db_path: Path | str,
        enabled: bool = True,
        journal_mode: str | None = None,
    ):
        self.db_path = Path(db_path)
        self.enabled = enabled and os.environ.get("LLM_CACHE_DISABLE") != "1"
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._journal_mode = _resolve_journal_mode(self.db_path, journal_mode)
        if self.enabled:
            try:
                self._open()
            except sqlite3.Error as err:
                self._disable(f"failed to open cache DB at {self.db_path}: {err}")

    def _open(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        # Journal-mode PRAGMA returns the mode actually applied as a row; if
        # the filesystem rejects the requested mode (e.g. WAL on NFS) SQLite
        # silently downgrades to the previous mode — we still log what stuck.
        try:
            applied = self._conn.execute(
                f"PRAGMA journal_mode={self._journal_mode};"
            ).fetchone()
        except sqlite3.OperationalError:
            applied = self._conn.execute("PRAGMA journal_mode=TRUNCATE;").fetchone()
        if applied:
            self._journal_mode = str(applied[0]).lower()
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.executescript(_SCHEMA_SQL)

    def _disable(self, reason: str) -> None:
        """Disable the cache after an unrecoverable DB error; log once."""
        if not self.enabled:
            return
        logger.warning(
            "LLM cache disabled for rest of this process: %s "
            "(runs continue with live LLM calls, just without caching).",
            reason,
        )
        self.enabled = False
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def get(self, key: str) -> str | None:
        if not self.enabled or self._conn is None:
            return None
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT response FROM llm_cache WHERE key = ?", (key,)
                ).fetchone()
            return row[0] if row else None
        except sqlite3.Error as err:
            self._disable(f"cache read failed ({err})")
            return None

    def put(
        self,
        key: str,
        *,
        backend: str,
        model: str,
        call_type: str,
        temperature: float,
        max_tokens: int,
        prompt: str,
        system: str | None,
        response: str,
    ) -> None:
        if not self.enabled or self._conn is None:
            return
        if not response:
            return
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO llm_cache
                      (key, backend, model, call_type, temperature, max_tokens,
                       prompt_sha, prompt, system, response, created_at, hit_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        key,
                        backend,
                        model,
                        call_type,
                        float(temperature),
                        int(max_tokens),
                        _prompt_sha(prompt),
                        prompt,
                        system,
                        response,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except sqlite3.Error as err:
            self._disable(f"cache write failed ({err})")

    def bump_hit(self, key: str) -> None:
        if not self.enabled or self._conn is None:
            return
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE llm_cache SET hit_count = hit_count + 1 WHERE key = ?",
                    (key,),
                )
        except sqlite3.Error as err:
            self._disable(f"cache hit-count update failed ({err})")

    def stats(self) -> dict:
        if not self.enabled or self._conn is None:
            return {"enabled": False}
        try:
            with self._lock:
                total = self._conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM llm_cache"
                ).fetchone()
                per_model = self._conn.execute(
                    """
                    SELECT model, COUNT(*), COALESCE(SUM(hit_count), 0)
                    FROM llm_cache GROUP BY model ORDER BY COUNT(*) DESC
                    """
                ).fetchall()
        except sqlite3.Error as err:
            self._disable(f"cache stats query failed ({err})")
            return {"enabled": False}
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "enabled": True,
            "path": str(self.db_path),
            "journal_mode": self._journal_mode,
            "rows": total[0],
            "total_hits": total[1],
            "size_bytes": size_bytes,
            "per_model": [
                {"model": m, "rows": r, "hits": h} for (m, r, h) in per_model
            ],
        }

    def close(self) -> None:
        if self._conn is not None:
            with self._lock:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None
