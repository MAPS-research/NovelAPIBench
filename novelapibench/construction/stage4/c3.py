"""C3: solvability with knowledge (Appendix B.2, "Stage 4").

GPT-5-mini receives the task with the Full bundle (S, E, M, C) as reference documentation,
preceded by an explicit ``from <module> import <leaf>`` line, under the evaluation prompt
(``prompts.build_prompt``). Three samples are drawn (nominal T = 0.2; GPT-5 models sample at
their default temperature); the first fenced code block of each runs after the monitor setup
and the context code with the task's ``execution_test``. A task is dropped only if all three
samples fail. The call-record check later re-runs this gate under the full harness
(``construction.call_records``).
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from omegaconf import DictConfig
from tqdm import tqdm

from novelapibench.construction.envs import stage_python
from novelapibench.construction.stage3.tasks import stage3_settings
from novelapibench.llm.strong import StrongLLM
from novelapibench.prompts import build_prompt
from novelapibench.runtime.sandbox import execute_test_harness
from novelapibench.schemas import KnowledgeBundle, Task

_FIRST_BLOCK_RE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)


def knowledge_text(bundle: KnowledgeBundle) -> str:
    """The Full bundle as C3 renders it (import line, S, E, M, C). The section headers are
    part of the prompt that built the benchmark and differ slightly from ``render_knowledge``."""
    import_line = ""
    if "." in bundle.s_name:
        parent, _, leaf = bundle.s_name.rpartition(".")
        import_line = f"Import:\n    from {parent} import {leaf}\n\n"
    surface = [f"API: {bundle.s_name}"]
    if bundle.s_param:
        surface.append("Parameters:\n" + "\n".join(p.render() for p in bundle.s_param))
    if bundle.examples:
        surface.append("\n\n".join(f"Example {i + 1}:\n```python\n{ex.code}\n```"
                                   for i, ex in enumerate(bundle.examples)))
    parts = ["\n\n".join(surface)]
    mechanism = bundle.mechanism.render()
    if mechanism:
        parts.append(f"Conceptual background:\n{mechanism}")
    if bundle.implementation:
        parts.append(f"Implementation source (M_code):\n```python\n{bundle.implementation}\n```")
    return import_line + "\n\n".join(parts)


def c3_prompt(task: Task, bundle: KnowledgeBundle) -> str:
    return build_prompt(task, knowledge_text(bundle))


def extract_first_block(text: str) -> str:
    """First fenced block (else the whole response), without a ``def solution():`` header."""
    m = _FIRST_BLOCK_RE.search(text)
    code = m.group(1).strip() if m else text.strip()
    if code.startswith("def solution():"):
        return "\n".join(code.split("\n")[1:])
    if "def solution():" in code:
        return code[code.index("def solution():") + len("def solution():"):]
    return code


def check_task(task: Task, bundle: KnowledgeBundle, llm: StrongLLM, cfg: DictConfig) -> dict:
    c3 = cfg.construction.stage4.c3
    limits = stage3_settings(task.library, cfg)
    python = stage_python(task.library)
    th = task.test_harness
    prompt = c3_prompt(task, bundle)
    passes, last_error = 0, None
    for _ in range(int(c3.num_samples)):
        try:
            code = extract_first_block(llm.generate(prompt, temperature=float(c3.temperature)))
            r = execute_test_harness(
                setup_code="\n\n".join(filter(None, [th.setup_code, task.context_code or ""])),
                solution_code=code, test_code=th.execution_test or th.mock_test,
                timeout=limits["validation_timeout"], max_memory_mb=limits["memory_mb"],
                env_python=python, use_mock_imports=python is None,
                pid_namespace=bool(cfg.construction.sandbox.pid_namespace))
            if r.passed:
                passes += 1
            else:
                last_error = r.stderr[:200]
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)[:200]
    solved = passes >= int(c3.pass_threshold)
    return {"task_id": task.task_id, "api_name": task.api_name, "passed": solved,
            "n_passed": passes, "error": None if solved else last_error}


def run_c3(tasks: list[Task], bundles: dict[str, KnowledgeBundle], llm: StrongLLM,
           cfg: DictConfig) -> list[dict]:
    """One record per task (input order); a task without a bundle passes through."""
    def check(task: Task) -> dict:
        bundle = bundles.get(task.api_name)
        if bundle is None:
            return {"task_id": task.task_id, "api_name": task.api_name, "passed": True,
                    "n_passed": None, "error": "no_bundle"}
        return check_task(task, bundle, llm, cfg)

    records: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=int(cfg.construction.stage4.c3.workers)) as pool:
        futures = [pool.submit(check, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="C3"):
            rec = fut.result()
            records[rec["task_id"]] = rec
    return [records[t.task_id] for t in tasks]
