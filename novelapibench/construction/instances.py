"""Benchmark instances, exclusions and the RQ3 split (Appendix B.2, Section 4.4).

A backbone's instance is every Stage-3 task that passes C1, the backbone's own C2 and C3,
minus the tasks excluded by the call-record check (``call_records.exclusions``) and the
manual exclusions of ``configs/construction.yaml``. Instances are listed domain by domain
(``benchmark.DOMAINS``); within a domain, tasks are concatenated library by library (sorted
names, Stage-3 order) and shuffled with a fixed seed.

The RQ3 split partitions the primary backbone's instance *before* the call-record exclusions
by target API: per library, the sorted (library, API) pairs are shuffled with
``random.Random(seed ^ crc32(library))`` and the first ``int(0.8 n)`` go to training. From the
released filter records this yields the same tasks as ``data/benchmark/instances/`` and
``data/benchmark/splits/rq3_{train,test}.txt`` (``scripts/build_instance.py --verify``); the
order of the released files comes from the original build and they remain the reference.
Training uses every split task; evaluation keeps the test tasks that are in the final instance.
"""

from __future__ import annotations

import random
import zlib
from collections import defaultdict

from novelapibench.benchmark import DOMAINS
from novelapibench.schemas import Task


def passing(records: list[dict], key: str = "passed") -> set[str]:
    return {r["task_id"] for r in records if r.get(key)}


def instance_order(task_ids: set[str], tasks: list[Task], seed: int) -> list[str]:
    """Domain order; within a domain, sorted libraries and Stage-3 order, then shuffled."""
    by_domain: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for t in tasks:
        if t.task_id in task_ids:
            by_domain[t.domain][t.library].append(t.task_id)
    ordered = []
    for domain in list(DOMAINS) + sorted(set(by_domain) - set(DOMAINS)):
        ids = [tid for lib in sorted(by_domain.get(domain, {})) for tid in by_domain[domain][lib]]
        random.Random(seed).shuffle(ids)
        ordered += ids
    return ordered


def rq3_split(task_ids: set[str], tasks: list[Task], seed: int = 42,
              train_fraction: float = 0.8) -> tuple[list[str], list[str]]:
    """``(train, test)`` task ids, split by target API within each library."""
    selected = [t for t in tasks if t.task_id in task_ids]
    train_apis: set[tuple[str, str]] = set()
    by_lib: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for t in selected:
        by_lib[t.library].add((t.library, t.api_name))
    for lib in sorted(by_lib):
        apis = sorted(by_lib[lib])
        random.Random(seed ^ (zlib.crc32(lib.encode("utf-8")) & 0xFFFFFFFF)).shuffle(apis)
        train_apis.update(apis[: int(len(apis) * train_fraction)])
    ordered = sorted(selected, key=lambda t: t.library)       # library order, Stage-3 order within
    train = [t.task_id for t in ordered if (t.library, t.api_name) in train_apis]
    test = [t.task_id for t in ordered if (t.library, t.api_name) not in train_apis]
    return train, test


def build_instances(tasks: list[Task], c1: set[str], c2: dict[str, set[str]], c3: set[str],
                    excluded: dict[str, str], seed: int) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """``(before_exclusion, final)`` instance id lists per backbone."""
    before, final = {}, {}
    for model, novel in c2.items():
        ids = c1 & novel & c3
        before[model] = instance_order(ids, tasks, seed)
        final[model] = [t for t in before[model] if t not in excluded]
    return before, final
