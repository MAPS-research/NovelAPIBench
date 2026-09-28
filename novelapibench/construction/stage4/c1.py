"""C1: reference validity (Appendix B.2, "Stage 4").

The reference solution must pass its own harness: the target-call monitor must install, and
``import <module>`` + monitor + context + reference + ``execution_test`` (monitor assertion
and scenario assertions) must run without error. A task without any assertion layer fails.
The same check validates tasks in Stage 3. The result is shared by all backbones.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from omegaconf import DictConfig
from tqdm import tqdm

from novelapibench.construction.envs import stage_python
from novelapibench.construction.stage3.tasks import stage3_settings
from novelapibench.evaluation.harness import build_setup
from novelapibench.runtime.sandbox import execute_code, execute_test_harness
from novelapibench.schemas import Task


def validate_reference(task: Task, timeout: int = 30, max_memory_mb: int = 16384,
                       env_python: str | None = None,
                       pid_namespace: bool = False) -> tuple[bool, str | None]:
    """``(passed, error)`` of the reference solution under the task's harness."""
    th = task.test_harness
    if not any((s or "").strip() for s in (th.execution_test, th.shape_type_test, th.mock_test)):
        return False, "empty_test_harness"
    use_mock = env_python is None
    if (th.setup_code or "").strip():
        probe = execute_code(th.setup_code + "\nassert _target_api_spy_installed, 'spy_install_failed'\n",
                             timeout=timeout, max_memory_mb=max_memory_mb, env_python=env_python,
                             use_mock_imports=use_mock, pid_namespace=pid_namespace)
        if not probe.passed:
            return False, "spy_install_failed"
    result = execute_test_harness(
        setup_code=build_setup(task.api_name, th.setup_code, task.context_code),
        solution_code=task.reference_solution, test_code=th.execution_test, timeout=timeout,
        max_memory_mb=max_memory_mb, env_python=env_python, use_mock_imports=use_mock,
        pid_namespace=pid_namespace)
    if result.passed:
        return True, None
    if result.timed_out:
        return False, f"timed_out after {timeout}s"
    detail = result.stderr[-500:] if result.stderr else result.stdout[-500:]
    if result.returncode < 0:
        return False, f"process_killed_by_signal_{-result.returncode}" + (f": {detail}" if detail else "")
    return False, detail or "execution_failed (no stderr)"


def run_c1(tasks: list[Task], cfg: DictConfig) -> list[dict]:
    """One record per task: ``{task_id, api_name, passed, error}`` (input order)."""
    pid_ns = bool(cfg.construction.sandbox.pid_namespace)

    def check(task: Task) -> dict:
        limits = stage3_settings(task.library, cfg)
        ok, err = validate_reference(task, timeout=limits["validation_timeout"],
                                     max_memory_mb=limits["memory_mb"],
                                     env_python=stage_python(task.library), pid_namespace=pid_ns)
        return {"task_id": task.task_id, "api_name": task.api_name, "passed": ok, "error": err}

    records: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=int(cfg.construction.stage4.c1.workers)) as pool:
        futures = [pool.submit(check, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="C1"):
            rec = fut.result()
            records[rec["task_id"]] = rec
    return [records[t.task_id] for t in tasks]
