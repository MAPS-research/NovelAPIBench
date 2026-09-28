"""Training examples of the adaptation methods (Appendix D.4, "Training examples").

Every method learns the reference solution of each RQ3 training task (the context code plus the
missing lines) verbatim, in the split's order; only the input differs:

    SFT, AlphaEdit-LoRA  the no-knowledge task prompt, as a user/assistant chat pair
    RAFT                 the task prompt with the target API's Full bundle and the Full bundles of
                         two other training APIs in random order; with probability 0.2 the target
                         bundle is left out
    GRACE                the no-knowledge task prompt
    MEMIT                the S_name task prompt, whose ``API: <name>`` line carries the subject

All prompts come from :func:`novelapibench.prompts.build_prompt`, so training and evaluation
prompts are identical for the same knowledge.
"""

from __future__ import annotations

import random
from typing import Any

from novelapibench.knowledge import Condition, render_knowledge
from novelapibench.prompts import build_prompt
from novelapibench.schemas import KnowledgeBundle, Task

#: The line of the S_name prompt that holds MEMIT's subject; ``{}`` is the subject placeholder.
MEMIT_SUBJECT_LINE = "API: {}"


def chat_example(task: Task, prompt: str) -> dict[str, Any]:
    return {"task_id": task.task_id,
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": task.reference_solution}]}


def sft_examples(tasks: list[Task]) -> list[dict[str, Any]]:
    """SFT and AlphaEdit-LoRA: no-knowledge prompt -> reference solution."""
    return [chat_example(t, build_prompt(t)) for t in tasks]


def raft_examples(tasks: list[Task], bundles: dict[str, KnowledgeBundle], num_distractors: int,
                  drop_target_prob: float, seed: int) -> list[dict[str, Any]]:
    """RAFT (Zhang et al., 2024): the target's Full bundle among distractor bundles.

    Distractors are drawn from the training APIs only. Each task has its own random stream
    (seeded with ``"<seed>::<task_id>"``), so an example does not depend on the other tasks.
    """
    by_key = {(b.library, b.api_name): b for b in bundles.values()}
    pool = sorted({(t.library, t.api_name) for t in tasks})
    examples = []
    for task in tasks:
        target_key = (task.library, task.api_name)
        target = by_key[target_key]
        rng = random.Random(f"{seed}::{task.task_id}")
        candidates = [k for k in pool if k != target_key and k in by_key]
        shown = [by_key[k] for k in rng.sample(candidates, num_distractors)]
        drop_target = rng.random() < drop_target_prob
        if not drop_target:
            shown = [target, *shown]
            rng.shuffle(shown)
        knowledge = "\n\n".join(t for t in (render_knowledge(b, Condition.FULL) for b in shown)
                                if t.strip())
        example = chat_example(task, build_prompt(task, knowledge))
        example["target_shown"] = not drop_target
        examples.append(example)
    return examples


def grace_examples(tasks: list[Task]) -> list[dict[str, Any]]:
    """GRACE: one edit per task, no-knowledge prompt -> reference solution."""
    return [{"task_id": t.task_id, "prompt": build_prompt(t), "target": t.reference_solution}
            for t in tasks]


def memit_requests(tasks: list[Task], bundles: dict[str, KnowledgeBundle]) -> list[dict[str, Any]]:
    """MEMIT edit requests in the upstream format.

    The edit prompt is the task's S_name prompt with the API name in its first
    ``API: <name>`` line replaced by the ``{}`` placeholder; the subject is the bundle's S_name.
    """
    requests = []
    for task in tasks:
        bundle = bundles[task.api_name]
        prompt = build_prompt(task, render_knowledge(bundle, Condition.S_NAME))
        line = MEMIT_SUBJECT_LINE.replace("{}", bundle.s_name)
        if line not in prompt or "{}" in prompt.split(line, 1)[0]:
            raise ValueError(f"{task.task_id}: the S_name line is not the first placeholder site")
        requests.append({"case_id": task.task_id,
                         "prompt": prompt.replace(line, MEMIT_SUBJECT_LINE, 1),
                         "subject": bundle.s_name,
                         "target_new": {"str": task.reference_solution}})
    return requests


def build_examples(method: str, cfg: Any, tasks: list[Task],
                   bundles: dict[str, KnowledgeBundle]) -> list[dict[str, Any]]:
    """Training examples of ``method`` (``cfg`` is the ``adaptation`` config section)."""
    if method in ("sft", "alphaedit_lora"):
        return sft_examples(tasks)
    if method == "raft":
        r = cfg.raft
        return raft_examples(tasks, bundles, int(r.num_distractors), float(r.drop_target_prob),
                             int(r.sampling_seed))
    if method == "grace":
        return grace_examples(tasks)
    if method == "memit":
        return memit_requests(tasks, bundles)
    raise ValueError(f"unknown adaptation method {method!r}")


def chat_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """A user/assistant pair rendered with the backbone's chat template (the SFT training text)."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
