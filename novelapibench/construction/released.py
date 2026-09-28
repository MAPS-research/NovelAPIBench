"""Stage 4 on the released benchmark: the paper's filter records, and instances for new backbones.

``data/benchmark/filters/`` holds the Stage-4 records of the paper's build::

    c1.jsonl                          C1 per task
    c2/<model>.jsonl                  C2 per task and backbone (tasks passing C1)
    c2/<model>.completions.jsonl.gz   the three sampled completions behind each C2 record
    c3/<model>.jsonl                  C3 per task and backbone (tasks passing C1 and that C2)
    c3_recheck.jsonl.gz               the second C3 pass under the call-record check

C3 was run for each backbone on its own C2-novel tasks, so a task can have different C3
verdicts for different backbones. The 32B backbone has no torch records: its torch tasks were
excluded before C2 (their distributed-execution workloads exceeded the cluster's memory limit).

``rebuild_released`` recomputes the six instances and the RQ3 split from these records. It
compares them as sets: the order of the released files comes from the original build, where
it followed the completion order of parallel C3 checks, and those files remain the reference.
``build_instance`` gives a new backbone an instance under the same protocol: C2 on every
released task that passes C1, C3 on the backbone's novel tasks, and, for novel tasks that are in
no released instance, the reference call records and the second C3 pass. Everything it writes
goes to ``outputs/instances/<model>/``; ``benchmark.load_instance`` finds the instance there.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from omegaconf import DictConfig

from novelapibench.benchmark import (BENCHMARK_DIR, load_bundles, load_expected_calls, load_tasks,
                                     released_expected_calls)
from novelapibench.construction.instances import build_instances, instance_order, rq3_split
from novelapibench.io import iter_jsonl, read_jsonl, read_lines, write_jsonl
from novelapibench.log import logger
from novelapibench.paths import instances_dir
from novelapibench.schemas import Task

FILTERS_DIR = BENCHMARK_DIR / "filters"
RELEASED_MODELS = ["qwen2.5-coder-7b", "qwen2.5-coder-14b", "qwen2.5-coder-32b",
                   "opencoder-8b-instruct", "seed-coder-8b-instruct", "r1-distill-qwen-7b"]


def _records(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    return read_jsonl(path)


def _passing(records: list[dict], key: str) -> set[str]:
    return {r["task_id"] for r in records if r[key]}


def released_c1() -> set[str]:
    return _passing(_records(FILTERS_DIR / "c1.jsonl"), "passed")


def released_c2(model: str) -> set[str]:
    return _passing(_records(FILTERS_DIR / "c2" / f"{model}.jsonl"), "novel")


def released_c3(model: str) -> set[str]:
    return _passing(_records(FILTERS_DIR / "c3" / f"{model}.jsonl"), "passed")


def released_exclusions() -> dict[str, str]:
    with (BENCHMARK_DIR / "excluded_tasks.tsv").open() as f:
        return {r["task_id"]: r["reason"] for r in csv.DictReader(f, delimiter="\t")}


def _same(ids: list[str], path: Path) -> bool:
    released = read_lines(path)
    return len(ids) == len(released) and set(ids) == set(released)


def rebuild_released(cfg: DictConfig) -> dict[str, bool]:
    """Recompute the released instances and the RQ3 split; ``name -> same tasks as data/``."""
    tasks = list(load_tasks().values())
    c1 = released_c1()
    seed = int(cfg.construction.instances.shuffle_seed)
    excluded = released_exclusions()
    same = {}
    before = {}
    for m in RELEASED_MODELS:
        b, final = build_instances(tasks, c1, {m: released_c2(m)}, released_c3(m), excluded, seed)
        before[m] = b[m]
        same[f"instances/{m}"] = _same(final[m], BENCHMARK_DIR / "instances" / f"{m}.txt")
    rs = cfg.construction.rq3_split
    train, test = rq3_split(set(before[rs.model]), tasks, int(rs.seed), float(rs.train_fraction))
    same["splits/rq3_train"] = _same(train, BENCHMARK_DIR / "splits" / "rq3_train.txt")
    same["splits/rq3_test"] = _same(test, BENCHMARK_DIR / "splits" / "rq3_test.txt")
    recheck = {r["task_id"]: r for r in _records(FILTERS_DIR / "c3_recheck.jsonl.gz")}
    from novelapibench.construction.call_records import exclusions
    derived = exclusions(released_expected_calls(), recheck, int(cfg.construction.c3_recheck.num_samples))
    manual = dict(cfg.construction.instances.manual_exclusions or {})
    same["excluded_tasks"] = {**derived, **manual} == excluded
    return same


# ---------------------------------------------------------------------------
# A new backbone
# ---------------------------------------------------------------------------


class NewInstance:
    """Stage 4 for one backbone on the released tasks; a rerun skips tasks already recorded."""

    def __init__(self, model: str, cfg: DictConfig, libraries: list[str] | None = None,
                 limit: int | None = None):
        self.model, self.cfg = model, cfg
        self.dir = instances_dir() / model
        c1 = released_c1()
        tasks = [t for t in load_tasks().values() if t.task_id in c1]
        if libraries:
            tasks = [t for t in tasks if t.library in libraries]
        if limit:
            per_lib: dict[str, int] = {}
            kept = []
            for t in tasks:
                if per_lib.get(t.library, 0) < limit:
                    per_lib[t.library] = per_lib.get(t.library, 0) + 1
                    kept.append(t)
            tasks = kept
        self.tasks: list[Task] = tasks

    def path(self, name: str) -> Path:
        return self.dir / name

    def _done(self, name: str) -> dict[str, dict]:
        p = self.path(name)
        return {r["task_id"]: r for r in iter_jsonl(p)} if p.exists() else {}

    def generate(self) -> None:
        """C2, GPU part: three completions per task without knowledge."""
        from novelapibench.construction.stage4 import c2
        done = self._done("c2.completions.jsonl")
        todo = [t for t in self.tasks if t.task_id not in done]
        if todo:
            comps = c2.generate(todo, self.model, self.cfg)
            write_jsonl(self.path("c2.completions.jsonl"),
                        [{"task_id": k, "completions": v} for k, v in comps.items()], append=True)
        logger.info(f"C2 {self.model}: completions for {len(done) + len(todo)} tasks")

    def score(self) -> None:
        """C2, CPU part: a task is novel for the backbone when all three completions fail."""
        from novelapibench.construction.stage4 import c2
        comps = {k: v["completions"] for k, v in self._done("c2.completions.jsonl").items()}
        done = self._done("c2.jsonl")
        todo = [t for t in self.tasks if t.task_id in comps and t.task_id not in done]
        if todo:
            write_jsonl(self.path("c2.jsonl"), c2.score(todo, comps, self.model, self.cfg), append=True)

    def c3(self) -> None:
        """C3 (GPT-5-mini) on the backbone's novel tasks."""
        from novelapibench.construction.stage4 import c3
        from novelapibench.llm.strong import StrongLLM
        novel = _passing(list(self._done("c2.jsonl").values()), "novel")
        done = self._done("c3.jsonl")
        todo = [t for t in self.tasks if t.task_id in novel and t.task_id not in done]
        if todo:
            write_jsonl(self.path("c3.jsonl"),
                        c3.run_c3(todo, load_bundles(), StrongLLM(self.cfg), self.cfg), append=True)

    def candidates(self) -> list[Task]:
        keep = (_passing(list(self._done("c2.jsonl").values()), "novel")
                & _passing(list(self._done("c3.jsonl").values()), "passed"))
        return [t for t in self.tasks if t.task_id in keep]

    def call_records(self) -> None:
        """Reference call records and the second C3 pass for tasks without released records."""
        from novelapibench.construction.call_records import build_expected_records, run_recheck
        from novelapibench.llm.strong import StrongLLM
        released = released_expected_calls()
        skip = set(released_exclusions()) | set(self.cfg.construction.instances.manual_exclusions or {})
        new = [t for t in self.candidates() if t.task_id not in released and t.task_id not in skip]
        done = self._done("expected_calls.jsonl")
        todo = [t for t in new if t.task_id not in done]
        if todo:
            write_jsonl(self.path("expected_calls.jsonl"), build_expected_records(todo, self.cfg),
                        append=True)
        expected = self._done("expected_calls.jsonl")
        done = self._done("c3_recheck.jsonl")
        todo = [t for t in new if t.task_id in expected and t.task_id not in done]
        if todo:
            write_jsonl(self.path("c3_recheck.jsonl"),
                        run_recheck(todo, load_bundles(), expected, StrongLLM(self.cfg), self.cfg),
                        append=True)
        logger.info(f"{self.model}: {len(new)} candidate tasks outside the released call records")

    def exclusions(self) -> dict[str, str]:
        from novelapibench.construction.call_records import exclusions
        out = released_exclusions()
        out.update(exclusions(self._done("expected_calls.jsonl"), self._done("c3_recheck.jsonl"),
                              int(self.cfg.construction.c3_recheck.num_samples)))
        out.update(dict(self.cfg.construction.instances.manual_exclusions or {}))
        return out

    def instance(self) -> list[str]:
        """Write ``outputs/instances/<model>/instance.txt``: C1, C2 and C3 minus the exclusions."""
        released = set(released_expected_calls()) | set(self._done("expected_calls.jsonl"))
        excluded = self.exclusions()
        keep = {t.task_id for t in self.candidates()}
        missing = keep - released - set(excluded)
        if missing:
            raise SystemExit(f"{len(missing)} candidate tasks have no call records yet; run --phase call-records")
        ids = [t for t in instance_order(keep, list(load_tasks().values()),
                                         int(self.cfg.construction.instances.shuffle_seed))
               if t not in excluded]
        self.path("instance.txt").write_text("".join(f"{t}\n" for t in ids))
        logger.info(f"{self.model}: {len(keep)} tasks pass C1, C2 and C3; {len(ids)} in the instance "
                    f"-> {self.path('instance.txt')}")
        return ids
