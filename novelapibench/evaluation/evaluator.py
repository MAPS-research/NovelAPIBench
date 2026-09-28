"""Scoring completions (paper Section 4.1, Appendix B.3).

A completion passes when, run in the task's program context and the library's execution
environment, it (1) raises no error, (2) calls the target API (target-call monitor), and
(3) reproduces the reference solution's calls to it (call-record check); the task's scenario
assertions then run as well. Failures are labelled by ``failure_taxonomy``.

Per cell, ``evaluate_cell`` reads ``predictions.jsonl`` and writes ``results.jsonl`` (one record
per task) and ``summary.json``. It resumes from a partial ``results.jsonl`` written by the same
evaluator version.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from omegaconf import DictConfig
from tqdm import tqdm

from novelapibench.benchmark import load_expected_calls, load_tasks
from novelapibench.config import load_library_config
from novelapibench.evaluation import call_check
from novelapibench.evaluation.extraction import extract_code
from novelapibench.evaluation.failure_taxonomy import PASS, classify_failure, extract_exception_type
from novelapibench.io import iter_jsonl, read_json, write_json
from novelapibench.log import logger
from novelapibench.paths import BENCHMARK_DIR
from novelapibench.runtime.envs import env_python
from novelapibench.runtime.sandbox import ExecutionResult, execute_test_harness, infer_cli_argv
from novelapibench.schemas import Task

RESULTS_FILE = "results.jsonl"
SUMMARY_FILE = "summary.json"
META_FILE = "eval_meta.json"


def evaluator_version() -> str:
    """Hash of the evaluation code and the reference call records; stamped on every cell."""
    h = hashlib.sha256()
    here = Path(__file__).parent
    for name in ("evaluator.py", "extraction.py", "harness.py", "call_check.py", "failure_taxonomy.py"):
        h.update((here / name).read_bytes())
    h.update((here.parent / "runtime" / "sandbox.py").read_bytes())
    h.update((BENCHMARK_DIR / "expected_calls.jsonl").read_bytes())
    return h.hexdigest()[:16]


def run_completion(task: Task, code: str, expected: list[dict], timeout: int, python: str,
                   max_memory_mb: int = 16384) -> ExecutionResult:
    """Execute ``code`` as the task's completion under the call-record harness."""
    h = task.test_harness
    setup = call_check.build_setup(task.api_name, h.setup_code, task.context_code,
                                   bind_target_leaf=True)
    test = call_check.build_test(h.execution_test, expected)
    return execute_test_harness(setup_code=setup, solution_code=code, test_code=test,
                                timeout=timeout, env_python=python, max_memory_mb=max_memory_mb,
                                argv=infer_cli_argv(h.execution_test))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    import math
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


@dataclass
class TaskResult:
    task_id: str
    api_name: str
    library: str
    domain: str
    difficulty: str
    passed: bool
    label: str               # "Pass" or one of failure_taxonomy.LABELS
    rationale: str = ""
    exception_type: str = ""
    error: str = ""          # last 500 characters of stderr
    pass_at_5: float | None = None


class Evaluator:
    def __init__(self, cfg: DictConfig, classify_failures: bool = True):
        self.cfg = cfg
        self.classify_failures = classify_failures
        self._strong = None
        self._serial = threading.Lock()   # timeouts are re-run one at a time
        self._expected = load_expected_calls()

    @property
    def strong(self):
        if self._strong is None:
            from novelapibench.llm.strong import StrongLLM
            self._strong = StrongLLM(self.cfg)
        return self._strong

    def _timeout(self, library: str) -> int:
        return int(load_library_config(library).get("eval_timeout_seconds",
                                                    self.cfg.evaluation.timeout_seconds))

    def _expected_for(self, task: Task) -> list[dict]:
        rec = self._expected.get(task.task_id)
        if rec is None or rec["status"] != "ok":
            raise ValueError(f"{task.task_id} has no usable reference call record "
                             f"({rec['status'] if rec else 'missing'}); it is not part of any instance")
        return rec["expected"]

    def run(self, task: Task, response: str) -> tuple[str, ExecutionResult]:
        """Extract and execute one response; a timeout is retried once, serially, so that
        load on the machine cannot turn a slow import into a failure."""
        code = extract_code(response)
        python = env_python(task.library)
        expected = self._expected_for(task)
        timeout = self._timeout(task.library)
        mem = int(self.cfg.evaluation.max_memory_mb)
        res = run_completion(task, code, expected, timeout, python, mem)
        if res.timed_out:
            with self._serial:
                res = run_completion(task, code, expected, timeout, python, mem)
        return code, res

    def evaluate_task(self, task: Task, samples: list[str], k5_samples: list[str] | None = None) -> TaskResult:
        code, res = self.run(task, samples[0] if samples else "")
        label, rationale, exc = PASS, "", ""
        if not res.passed:
            exc = "Timeout" if res.timed_out else extract_exception_type(res.stderr)
            if self.classify_failures:
                label, rationale = classify_failure(task, code, res, self.strong)
            else:
                label = ""
        pass5 = None
        if k5_samples:
            passes = [self.run(task, s)[1].passed for s in k5_samples]
            pass5 = pass_at_k(len(passes), sum(passes), 5) if len(passes) >= 5 else None
        return TaskResult(task_id=task.task_id, api_name=task.api_name, library=task.library,
                          domain=task.domain, difficulty=task.difficulty, passed=res.passed,
                          label=label, rationale=rationale, exception_type=exc,
                          error=(res.stderr or "").strip()[-500:] if not res.passed else "",
                          pass_at_5=pass5)

    def evaluate_cell(self, cell_dir: str | Path, workers: int | None = None,
                      pass_at_5: bool = False) -> dict:
        """Score ``cell_dir/predictions.jsonl`` into ``results.jsonl`` and ``summary.json``."""
        cell_dir = Path(cell_dir)
        tasks = load_tasks()
        preds = {r["task_id"]: r for r in iter_jsonl(cell_dir / "predictions.jsonl")}
        k5 = {}
        if pass_at_5:
            k5_path = cell_dir / "predictions_k5.jsonl"
            if not k5_path.exists():
                raise FileNotFoundError(f"{k5_path} (run inference with --pass-at-5)")
            k5 = {r["task_id"]: r["samples"] for r in iter_jsonl(k5_path)}

        meta = {"evaluator_version": evaluator_version(), "classify_failures": self.classify_failures,
                "pass_at_5": pass_at_5}
        out = cell_dir / RESULTS_FILE
        done: dict[str, dict] = {}
        meta_path = cell_dir / META_FILE
        if out.exists():
            old = read_json(meta_path) if meta_path.exists() else {}
            if old == meta:
                done = {r["task_id"]: r for r in iter_jsonl(out)}
            else:
                logger.warning(f"{cell_dir}: results from a different evaluator; re-scoring")
                out.unlink()
        write_json(meta_path, meta)

        todo = [tid for tid in preds if tid not in done]
        logger.info(f"{cell_dir}: scoring {len(todo)} tasks ({len(done)} already scored)")
        lock = threading.Lock()
        workers = workers or int(self.cfg.evaluation.task_workers)
        with out.open("a") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(self.evaluate_task, tasks[t], preds[t]["samples"], k5.get(t)): t
                    for t in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=cell_dir.name):
                rec = asdict(fut.result())
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                done[rec["task_id"]] = rec

        records = [done[t] for t in preds]  # prediction order
        summary = summarize(records)
        write_json(cell_dir / SUMMARY_FILE, summary)
        logger.info(f"{cell_dir.name}: pass@1 = {100 * summary['pass_at_1']:.1f}% (n={summary['n_tasks']})")
        return summary


def summarize(records: list[dict]) -> dict:
    n = len(records)
    labels: dict[str, int] = {}
    for r in records:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    p5 = [r["pass_at_5"] for r in records if r.get("pass_at_5") is not None]
    return {"n_tasks": n,
            "pass_at_1": sum(r["passed"] for r in records) / n if n else 0.0,
            "pass_at_5": sum(p5) / len(p5) if p5 else None,
            "labels": dict(sorted(labels.items()))}
