"""Stage 3 driver: generate, validate and write the tasks of one library.

Enum-like classes that introspection sees with a generic ``(*args, **kwargs)`` constructor and
that have neither parameters nor examples are skipped (a constructor call is meaningless for
them). Each generated task must pass C1 (``stage4.c1.validate_reference``); on failure its
harness is regenerated once and re-validated. Tasks are deduplicated by id.

Output: ``outputs/construction/stage3/<library>.jsonl`` (``Task`` records, appended per bundle;
with ``resume`` APIs already written are skipped, otherwise the file is rewritten).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from omegaconf import DictConfig
from tqdm import tqdm

from novelapibench.construction.envs import stage_python
from novelapibench.construction.schemas import APIEntry
from novelapibench.construction.stage3.harness import generate_test_harness
from novelapibench.construction.stage3.tasks import generate_tasks, stage3_settings
from novelapibench.construction.stage4.c1 import validate_reference
from novelapibench.io import iter_jsonl
from novelapibench.llm.strong import StrongLLM
from novelapibench.log import logger
from novelapibench.schemas import KnowledgeBundle, Task

_GENERIC_CTORS = {"(self, *args, **kwds)", "(self, *args, **kwargs)", "(self, /, *args, **kwargs)"}


def is_supported(bundle: KnowledgeBundle, entry: APIEntry | None) -> bool:
    if entry is None or entry.kind != "class":
        return True
    leaf = bundle.api_name.split(".")[-1]
    docstring = (entry.docstring or "").strip()
    return not ((entry.signature or "").strip() in _GENERIC_CTORS
                and not bundle.s_param and not bundle.examples
                and (leaf.endswith(("Name", "Enum", "Type", "Mode"))
                     or docstring.startswith(("str(", "int(", "float(", "bytes(", "tuple(",
                                              "list(", "dict(", "set("))))


def validate_task(task: Task, cfg: DictConfig, llm: StrongLLM, env_python: str) -> Task | None:
    """The task if its reference passes its harness (regenerating the harness once)."""
    if not task.test_harness.execution_test.strip():
        return None
    limits = stage3_settings(task.library, cfg)
    kw = dict(timeout=limits["validation_timeout"], max_memory_mb=limits["memory_mb"],
              env_python=env_python, pid_namespace=bool(cfg.construction.sandbox.pid_namespace))
    ok, err = validate_reference(task, **kw)
    if ok:
        return task
    logger.debug(f"{task.api_name} [{task.difficulty}]: harness validation failed ({err}); regenerating")
    try:
        harness = generate_test_harness(task.api_name, task.reference_solution, task.description,
                                        llm, cfg, task.context_code or "", env_python, None,
                                        task.library)
        retried = task.model_copy(update={"test_harness": harness})
        if validate_reference(retried, **kw)[0]:
            return retried
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"{task.api_name}: harness regeneration error ({exc})")
    return None


def generate_library(library: str, bundles: list[KnowledgeBundle], entries: dict[str, APIEntry],
                     cfg: DictConfig, llm: StrongLLM, out_path: Path,
                     resume: bool = False) -> list[Task]:
    env_python = stage_python(library)
    supported = [b for b in bundles if is_supported(b, entries.get(b.api_name))]
    if len(supported) < len(bundles):
        logger.info(f"{library}: skipping {len(bundles) - len(supported)} enum-like classes")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if resume and out_path.exists():
        existing = list(iter_jsonl(out_path))
        seen = {t["task_id"] for t in existing}
        done_apis = {t["api_name"] for t in existing}
        supported = [b for b in supported if b.api_name not in done_apis]
    else:
        out_path.write_text("")
    logger.info(f"Stage 3 {library}: {len(supported)} bundles, execution env {env_python}")

    lock = threading.Lock()
    n_generated = n_written = 0

    def work(bundle: KnowledgeBundle) -> list[Task]:
        try:
            tasks = generate_tasks(bundle, cfg, llm, env_python)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{bundle.api_name}: task generation failed ({exc})")
            return []
        return tasks

    with ThreadPoolExecutor(max_workers=int(cfg.construction.stage3.workers)) as pool, \
            out_path.open("a") as fh:
        futures = {pool.submit(work, b): b for b in supported}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Stage 3 [{library}]"):
            tasks = fut.result()
            valid = [v for v in (validate_task(t, cfg, llm, env_python) for t in tasks) if v]
            with lock:
                n_generated += len(tasks)
                for t in valid:
                    if t.task_id in seen:
                        continue
                    seen.add(t.task_id)
                    fh.write(t.model_dump_json() + "\n")
                    n_written += 1
                fh.flush()
    tasks = [Task(**r) for r in iter_jsonl(out_path)]
    logger.info(f"Stage 3 {library}: {n_written} of {n_generated} generated tasks kept; "
                f"{len(tasks)} in {out_path.name}")
    return tasks
