"""C2: empirical novelty, per backbone (Appendix B.2, "Stage 4", Listing "C2 prompt").

The backbone receives only the context code and the task description (no knowledge), through
its chat template, and three completions are sampled at T = 0.8 (repetition penalty 1.1, the
setting construction ran with; evaluation uses 1.0). The first fenced code block of each
completion (else the whole response) runs after the monitor setup and the context code, with
the task's ``execution_test``. A task is kept for the backbone only if all three fail.

Two phases, so that the GPU is not held while completions execute: ``generate`` (vLLM) writes
the completions, ``score`` (CPU) runs them.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from novelapibench.config import load_model_config
from novelapibench.construction.envs import stage_python
from novelapibench.construction.stage3.tasks import stage3_settings
from novelapibench.log import logger
from novelapibench.runtime.sandbox import execute_test_harness
from novelapibench.schemas import Task

C2_PROMPT = """\
{context_code}

# Task:
{description}

# Complete the following code (output only the missing lines, no explanation):
"""

_FIRST_BLOCK_RE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)


def build_prompt(task: Task) -> str:
    return C2_PROMPT.format(context_code=task.context_code or "", description=task.description)


def first_code_block(text: str) -> str:
    m = _FIRST_BLOCK_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def generate(tasks: list[Task], model: str, cfg: DictConfig) -> dict[str, list[str]]:
    """``task_id -> completions`` of the backbone (reasoning traces stripped)."""
    from novelapibench.llm.local import LocalLLM

    c2 = cfg.construction.stage4.c2
    run_cfg = OmegaConf.merge(cfg, {"inference": {"repetition_penalty": float(c2.repetition_penalty)}})
    llm = LocalLLM(run_cfg, load_model_config(model))
    try:
        outputs = llm.generate([build_prompt(t) for t in tasks], temperature=float(c2.temperature),
                               n=int(c2.num_samples))
    finally:
        llm.close()
    return {t.task_id: [g.text for g in gens] for t, gens in zip(tasks, outputs)}


def sample_fails(task: Task, completion: str, cfg: DictConfig) -> bool:
    limits = stage3_settings(task.library, cfg)
    th = task.test_harness
    python = stage_python(task.library)
    result = execute_test_harness(
        setup_code="\n\n".join(filter(None, [th.setup_code, task.context_code or ""])),
        solution_code=first_code_block(completion), test_code=th.execution_test or th.mock_test,
        timeout=limits["validation_timeout"], max_memory_mb=limits["memory_mb"],
        env_python=python, use_mock_imports=python is None,
        pid_namespace=bool(cfg.construction.sandbox.pid_namespace))
    return not result.passed


def score(tasks: list[Task], completions: dict[str, list[str]], model: str,
          cfg: DictConfig) -> list[dict]:
    """One record per task with completions: ``{task_id, api_name, novel, fail_count, sample_fails}``."""
    c2 = cfg.construction.stage4.c2
    threshold = int(c2.fail_threshold)

    def check(task: Task) -> dict:
        fails = [sample_fails(task, c, cfg) for c in completions[task.task_id]]
        if len(fails) < threshold:
            raise ValueError(f"{task.task_id}: {len(fails)} samples cannot reach fail_threshold={threshold}")
        return {"task_id": task.task_id, "api_name": task.api_name, "model": model,
                "novel": sum(fails) >= threshold, "fail_count": sum(fails), "sample_fails": fails}

    todo = [t for t in tasks if t.task_id in completions]
    records: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=int(c2.workers)) as pool:
        futures = [pool.submit(check, t) for t in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"C2 [{model}]"):
            rec = fut.result()
            records[rec["task_id"]] = rec
    kept = sum(r["novel"] for r in records.values())
    logger.info(f"C2 {model}: {kept}/{len(records)} tasks novel (all {threshold} samples fail)")
    return [records[t.task_id] for t in todo]
