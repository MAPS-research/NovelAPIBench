"""Reference call records and the second C3 pass (Appendix B.3, "Call-record comparison").

Expected records. For every task in at least one instance, the reference completion is the
masked region, or the full reference solution when the masked region only runs inside it
(both must pass the task's harness first). It is run four times under the recording monitor
(``evaluation.call_check``), with global seeds 11 and 22 as written and seeds 33 and 44 with
its constant seeds randomised; the fields that agree across runs are the expected records.
Status: ``ok``, ``unstable`` (the runs disagree on the number of calls), ``no_calls``, or
``reference_failed`` (fewer than two runs produced records).

Second C3 pass. C3 is repeated with the same prompt, sampling and extraction, now scoring each
sample with the evaluation harness including the call-record check. A task whose three
samples all fail is excluded from every instance, as are tasks whose reference records are
not ``ok``.
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from omegaconf import DictConfig
from tqdm import tqdm

from novelapibench.config import load_library_config
from novelapibench.construction.envs import stage_python
from novelapibench.construction.stage4.c3 import c3_prompt, extract_first_block
from novelapibench.evaluation import call_check as cc
from novelapibench.evaluation.evaluator import run_completion
from novelapibench.evaluation.extraction import extract_code
from novelapibench.evaluation.harness import build_setup
from novelapibench.llm.strong import StrongLLM
from novelapibench.runtime.sandbox import execute_test_harness, infer_cli_argv
from novelapibench.schemas import KnowledgeBundle, Task

#: Exclusion reasons written to ``excluded_tasks.tsv``.
REASONS = {"c3": "c3_unsolved_under_call_record_check",
           "reference_failed": "reference_call_record_failed",
           "unstable": "reference_call_record_unstable",
           "no_calls": "reference_call_record_no_calls"}


def eval_timeout(library: str, cfg: DictConfig) -> int:
    return int(load_library_config(library).get("eval_timeout_seconds",
                                                cfg.evaluation.timeout_seconds))


def passes_harness(task: Task, code: str, python: str, timeout: int) -> bool:
    """The completion passes the task's harness (monitor + scenario assertions, no call-record
    check), with the evaluation-time extraction and leaf binding."""
    th = task.test_harness
    r = execute_test_harness(
        setup_code=build_setup(task.api_name, th.setup_code, task.context_code, bind_target_leaf=True),
        solution_code=extract_code(f"```python\n{code}\n```"), test_code=th.execution_test,
        timeout=timeout, env_python=python)
    return r.passed


def expected_record(task: Task, runs: list[list], python: str, timeout: int) -> dict:
    """Expected call records of one task (see module docstring)."""
    th = task.test_harness
    base, source = task.masked_region or "", "masked_region"
    if not passes_harness(task, base, python, timeout):
        base, source = task.reference_solution or "", "reference_solution"
        if not passes_harness(task, base, python, timeout):
            source = "none"
    argv = infer_cli_argv(th.execution_test)
    unseeded = cc.unseed(base)
    uses_torch = "torch" in (task.context_code or "") + base
    records, notes = [], []
    for seed, randomise in (runs if source != "none" else ()):
        code = unseeded if randomise and unseeded is not None else base
        setup = cc.build_setup(task.api_name, th.setup_code, task.context_code, True,
                               cc.reseed_prelude(int(seed), uses_torch))
        fd, path = tempfile.mkstemp(suffix=".json", prefix="call_records_")
        os.close(fd)
        try:
            r = execute_test_harness(setup, code, cc.dump_records(path), timeout=timeout,
                                     env_python=python, argv=argv)
            recs = cc.read_records(path) if r.passed else None
        finally:
            os.unlink(path)
        if recs is None:
            notes.append(f"seed{seed}: " + ("timeout" if r.timed_out else
                                            (r.stderr or r.error_msg or "")[-300:] if not r.passed
                                            else "records file unreadable"))
        else:
            records.append(recs)
    if len(records) < 2:
        expected, status = [], "reference_failed"
    else:
        expected, status = cc.expected_from_runs(records)
    return {"task_id": task.task_id, "api_name": task.api_name, "status": status,
            "expected": expected, "reference_source": source, "n_runs": len(records),
            "notes": notes}


def _expected_worker(task_json: str, runs: list[list], python: str, timeout: int) -> dict:
    return expected_record(Task.model_validate_json(task_json), runs, python, timeout)


def build_expected_records(tasks: list[Task], cfg: DictConfig) -> list[dict]:
    """Expected records of execute-then-assert tasks (in input order)."""
    runs = [list(r) for r in cfg.construction.call_records.runs]
    todo = [t for t in tasks if t.test_harness.generation_method == "execute_then_assert"]
    out: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=int(cfg.construction.call_records.workers)) as pool:
        futures = [pool.submit(_expected_worker, t.model_dump_json(), runs,
                               stage_python(t.library), eval_timeout(t.library, cfg)) for t in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="call records"):
            rec = fut.result()
            out[rec["task_id"]] = rec
    return [out[t.task_id] for t in todo]


def recheck_task(task: Task, bundle: KnowledgeBundle, expected: list[dict], llm: StrongLLM,
                 cfg: DictConfig) -> dict:
    """C3 samples scored by the evaluation harness with the call-record check."""
    rc = cfg.construction.c3_recheck
    python, timeout = stage_python(task.library), eval_timeout(task.library, cfg)
    prompt = c3_prompt(task, bundle)
    samples = []
    for _ in range(int(rc.num_samples)):
        try:
            raw = llm.generate(prompt, temperature=float(rc.temperature))
        except Exception as exc:  # noqa: BLE001
            samples.append({"code": "", "passed": None, "error": f"llm: {str(exc)[:200]}"})
            continue
        code = extract_first_block(raw)
        r = run_completion(task, extract_code(f"```python\n{code}\n```"), expected, timeout, python)
        rec = {"code": code, "passed": bool(r.passed)}
        if not r.passed:
            rec["error"] = (r.stderr or r.error_msg or "")[-300:]
        samples.append(rec)
    return {"task_id": task.task_id, "api_name": task.api_name, "library": task.library,
            "passed": any(s.get("passed") for s in samples),
            "n_llm_errors": sum(s.get("passed") is None for s in samples), "samples": samples}


def run_recheck(tasks: list[Task], bundles: dict[str, KnowledgeBundle],
                expected: dict[str, dict], llm: StrongLLM, cfg: DictConfig) -> list[dict]:
    """Second C3 pass over the tasks whose reference records are ``ok``."""
    todo = [t for t in tasks if expected.get(t.task_id, {}).get("status") == "ok"]
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=int(cfg.construction.c3_recheck.workers)) as pool:
        futures = [pool.submit(recheck_task, t, bundles[t.api_name],
                               expected[t.task_id]["expected"], llm, cfg) for t in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="C3 recheck"):
            rec = fut.result()
            out[rec["task_id"]] = rec
    return [out[t.task_id] for t in todo]


def exclusions(expected: dict[str, dict], recheck: dict[str, dict], num_samples: int) -> dict[str, str]:
    """``task_id -> reason`` for tasks removed by the call-record check."""
    out = {}
    for tid, e in expected.items():
        if e["status"] != "ok":
            out[tid] = REASONS[e["status"]]
    for tid, r in recheck.items():
        if (not r["passed"] and r["n_llm_errors"] < num_samples
                and expected.get(tid, {}).get("status") == "ok"):
            out[tid] = REASONS["c3"]
    return out
