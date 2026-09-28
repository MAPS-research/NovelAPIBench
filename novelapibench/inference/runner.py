"""Inference for the three research questions.

Each *cell* (one knowledge condition, or one adaptation method under one condition) writes
``outputs/runs/<run>/<cell>/predictions.jsonl`` with one record per task::

    {"task_id", "api_name", "cell", "samples": [response, ...]}

(``raw_samples`` / ``finish_reasons`` are added for reasoning models). Existing predictions are
kept and only missing tasks are generated.

Runs and cells:

* ``rq1-<model>``  cells = knowledge conditions with oracle knowledge (``none``, ``S``, ``S+E`` ...)
* ``rq2-<model>``  the same conditions with retrieved knowledge (top-5 chunks)
* ``rq3-<model>``  cells ``<method>__<condition>`` for method in base, sft, raft, grace, memit,
  alphaedit_lora and condition in ``none`` / ``Full`` (``S_name`` is also supported)
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from novelapibench.benchmark import load_bundles
from novelapibench.io import iter_jsonl, write_json, write_jsonl
from novelapibench.knowledge import Condition, render_knowledge
from novelapibench.llm.local import LocalLLM
from novelapibench.log import logger
from novelapibench.prompts import build_prompt
from novelapibench.schemas import Task


def oracle_prompts(tasks: list[Task], condition: Condition) -> list[str]:
    bundles = load_bundles()
    prompts = []
    for t in tasks:
        text = ""
        if condition != Condition.NONE:
            text = render_knowledge(bundles[t.api_name], condition)
            if not text:
                raise RuntimeError(f"{t.task_id}: condition {condition.value} renders no knowledge")
        prompts.append(build_prompt(t, text or None))
    return prompts


def retrieval_prompts(cfg: DictConfig, tasks: list[Task], condition: Condition) -> list[str]:
    from novelapibench.inference.retrieval import Retriever

    if condition == Condition.NONE:
        return [build_prompt(t) for t in tasks]
    retrievers: dict[str, Retriever] = {}
    encoder = None
    prompts = []
    for t in tasks:
        if t.domain not in retrievers:
            retrievers[t.domain] = Retriever(cfg, t.domain, condition, encoder=encoder)
            encoder = retrievers[t.domain].encoder
        prompts.append(build_prompt(t, retrievers[t.domain].knowledge_text(t) or None))
    return prompts


def _missing(tasks: list[Task], path: Path) -> list[Task]:
    if not path.exists():
        return list(tasks)
    have = {r["task_id"] for r in iter_jsonl(path)}
    return [t for t in tasks if t.task_id not in have]


def run_cell(cfg: DictConfig, llm: LocalLLM, tasks: list[Task], prompts: list[str], cell_dir: Path,
             cell: str, pass_at_5: bool = False) -> None:
    """Generate greedy predictions (and, with ``pass_at_5``, 20 samples at T=0.8)."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    by_id = dict(zip((t.task_id for t in tasks), prompts))
    jobs = [("predictions.jsonl", float(cfg.inference.temperature), 1)]
    if pass_at_5:
        jobs.append(("predictions_k5.jsonl", float(cfg.inference.temperature_pass5),
                     int(cfg.inference.num_samples_pass5)))
    for fname, temperature, n in jobs:
        todo = _missing(tasks, cell_dir / fname)
        if not todo:
            logger.info(f"{cell_dir}/{fname}: complete")
            continue
        outs = llm.generate([by_id[t.task_id] for t in todo], temperature=temperature, n=n)
        recs = []
        for t, gens in zip(todo, outs):
            rec = {"task_id": t.task_id, "api_name": t.api_name, "cell": cell,
                   "samples": [g.text for g in gens]}
            if llm.thinking_mode:
                rec["raw_samples"] = [g.raw_text for g in gens]
                rec["finish_reasons"] = [g.finish_reason for g in gens]
            recs.append(rec)
        write_jsonl(cell_dir / fname, recs, append=(cell_dir / fname).exists())
        logger.info(f"{cell_dir}/{fname}: wrote {len(recs)} predictions")
    write_json(cell_dir / "inference_meta.json", {
        "cell": cell, "model": llm.model_cfg.name, "revision": llm.model_cfg.get("revision"),
        "adapter": llm.adapter_path, "edited_model": llm.edited_model_path, "grace": llm.grace_path,
        "repetition_penalty": float(cfg.inference.repetition_penalty),
        "max_new_tokens": int(llm.model_cfg.max_new_tokens), "chat_template": True,
    })
