"""Loading the released benchmark (``data/benchmark/``).

Layout::

    bundles.jsonl            knowledge bundles of all Stage-2 APIs (retrieval pool included)
    tasks.jsonl              every Stage-3 task
    instances/<model>.txt    task ids of each backbone's instance (after C1, C2 and C3)
    filters/                 the Stage-4 records the instances are built from (construction/released.py)
    splits/rq3_train.txt     RQ3 training tasks (80/20 split by target API)
    splits/rq3_test.txt      RQ3 test tasks (before the second C3 pass; see ``load_split``)
    excluded_tasks.tsv       tasks removed from every instance, with the reason
    expected_calls.jsonl     reference call records used by the call-record check
    manifest.json            versions, counts and checksums
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from novelapibench.config import list_models
from novelapibench.io import iter_jsonl, read_lines
from novelapibench.paths import BENCHMARK_DIR, instances_dir
from novelapibench.schemas import KnowledgeBundle, Task

DOMAINS = ("agent_tool", "ai4science", "data_science", "dl", "swe")
PRIMARY_MODEL = "qwen2.5-coder-7b"


@lru_cache(maxsize=1)
def load_bundles() -> dict[str, KnowledgeBundle]:
    """All knowledge bundles, keyed by ``api_name`` (file order preserved)."""
    return {r["api_name"]: KnowledgeBundle(**r) for r in iter_jsonl(BENCHMARK_DIR / "bundles.jsonl")}


@lru_cache(maxsize=1)
def load_tasks() -> dict[str, Task]:
    """All tasks, keyed by ``task_id`` (file order preserved)."""
    return {r["task_id"]: Task(**r) for r in iter_jsonl(BENCHMARK_DIR / "tasks.jsonl")}


@lru_cache(maxsize=1)
def excluded_tasks() -> dict[str, str]:
    """``task_id -> reason`` for tasks removed from every instance."""
    with (BENCHMARK_DIR / "excluded_tasks.tsv").open() as f:
        return {r["task_id"]: r["reason"] for r in csv.DictReader(f, delimiter="\t")}


def instance_path(model: str) -> Path:
    """The released instance, else one built with ``scripts/build_instance.py``."""
    path = BENCHMARK_DIR / "instances" / f"{model}.txt"
    return path if path.exists() else instances_dir() / model / "instance.txt"


def instance_ids(model: str) -> list[str]:
    path = instance_path(model)
    if not path.exists():
        raise FileNotFoundError(f"no instance for {model!r}: build one with "
                                f"scripts/build_instance.py --model {model} (configured models: {list_models()})")
    return read_lines(path)


def load_instance(model: str, domains: list[str] | None = None, limit: int | None = None) -> list[Task]:
    """The backbone's instance, optionally restricted to some domains and to the first
    ``limit`` tasks of each domain (quick checks)."""
    tasks = load_tasks()
    out: list[Task] = []
    per_domain: dict[str, int] = {}
    for tid in instance_ids(model):
        t = tasks[tid]
        if domains and t.domain not in domains:
            continue
        if limit is not None:
            if per_domain.get(t.domain, 0) >= limit:
                continue
            per_domain[t.domain] = per_domain.get(t.domain, 0) + 1
        out.append(t)
    return out


def load_split(name: str, final: bool = True) -> list[Task]:
    """An adaptation split (``rq3_train`` or ``rq3_test``).

    The split was drawn before the second C3 pass. Training uses all of it (reference
    solutions of later-excluded tasks remain valid targets); evaluation uses only the tasks that
    are in the final primary instance (``final=True``: 348 of the 397 test tasks).
    """
    tasks = load_tasks()
    ids = read_lines(BENCHMARK_DIR / "splits" / f"{name}.txt")
    if final:
        keep = set(instance_ids(PRIMARY_MODEL))
        ids = [t for t in ids if t in keep]
    return [tasks[t] for t in ids]


def retrieval_pool(domain: str) -> list[KnowledgeBundle]:
    """Bundles indexed for retrieval in ``domain``: every Stage-2 bundle of the domain's
    libraries that contribute tasks (Appendix D.3; the eight httpx bundles are excluded)."""
    libraries_with_tasks = {t.library for t in load_tasks().values()}
    return [b for b in load_bundles().values()
            if b.domain == domain and b.library in libraries_with_tasks]


@lru_cache(maxsize=1)
def released_expected_calls() -> dict[str, dict]:
    """``task_id -> {"status", "expected"}`` reference call records of the released instances."""
    return {r["task_id"]: r for r in iter_jsonl(BENCHMARK_DIR / "expected_calls.jsonl")}


@lru_cache(maxsize=1)
def load_expected_calls() -> dict[str, dict]:
    """The released call records, plus those built for new backbones' instances."""
    records = dict(released_expected_calls())
    for path in sorted(instances_dir().glob("*/expected_calls.jsonl")):
        for r in iter_jsonl(path):
            records.setdefault(r["task_id"], r)
    return records
