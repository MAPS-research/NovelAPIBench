"""Generate completions for RQ1 (oracle knowledge), RQ2 (retrieved knowledge) or RQ3 (adaptation).

Examples:
    # RQ1: all twelve knowledge conditions on the primary backbone
    python scripts/run_inference.py --experiment rq1 --model qwen2.5-coder-7b
    # RQ1 on another backbone (the nine conditions of the cross-backbone study)
    python scripts/run_inference.py --experiment rq1 --model opencoder-8b-instruct --conditions cross-backbone
    # RQ2: retrieval (top-5 BGE-small chunks) instead of the oracle bundle
    python scripts/run_inference.py --experiment rq2 --model qwen2.5-coder-7b
    # RQ3: base model and adapted models, without knowledge and with the Full bundle
    python scripts/run_inference.py --experiment rq3 --methods base sft raft grace memit alphaedit_lora
    # Quick check of the setup: two tasks per domain, written under outputs/debug/
    python scripts/run_inference.py --experiment rq1 --conditions none S+E --debug

Predictions go to outputs/runs/<experiment>-<model>/<cell>/predictions.jsonl.
Score them with scripts/evaluate.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novelapibench.benchmark import DOMAINS, PRIMARY_MODEL, load_instance, load_split  # noqa: E402
from novelapibench.config import load_config, load_model_config  # noqa: E402
from novelapibench.inference.runner import oracle_prompts, retrieval_prompts, run_cell  # noqa: E402
from novelapibench.knowledge import (ALL_CONDITIONS, CROSS_BACKBONE_CONDITIONS,  # noqa: E402
                                     Condition, parse_condition)
from novelapibench.llm.local import LocalLLM  # noqa: E402
from novelapibench.log import logger, setup_logging  # noqa: E402
from novelapibench.paths import run_dir, use_debug_outputs  # noqa: E402

RQ3_METHODS = ["base", "sft", "raft", "grace", "memit", "alphaedit_lora"]
DEBUG_LIMIT = 2


def conditions_from(args) -> list[Condition]:
    if args.conditions in (None, ["all"]):
        return ALL_CONDITIONS
    if args.conditions == ["cross-backbone"]:
        return CROSS_BACKBONE_CONDITIONS
    return [parse_condition(c) for c in args.conditions]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=["rq1", "rq2", "rq3"])
    ap.add_argument("--model", default=PRIMARY_MODEL, help="backbone (configs/models/<model>.yaml)")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="knowledge conditions (e.g. none S E S+E Full), 'all' or 'cross-backbone'; "
                         "RQ3 default: none Full")
    ap.add_argument("--methods", nargs="+", default=RQ3_METHODS, choices=RQ3_METHODS, help="RQ3 only")
    ap.add_argument("--domains", nargs="+", choices=DOMAINS, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N tasks of each domain")
    ap.add_argument("--debug", action="store_true",
                    help=f"a few tasks ({DEBUG_LIMIT} per domain unless --limit), outputs under outputs/debug/")
    ap.add_argument("--pass-at-5", action="store_true", help="also draw 20 samples at T=0.8")
    ap.add_argument("--run-name", default=None, help="default: <experiment>-<model>")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. inference.gpu_memory_utilization=0.8")
    args = ap.parse_args()
    setup_logging()
    if args.debug:
        use_debug_outputs()
        if args.limit is None:
            args.limit = DEBUG_LIMIT

    cfg = load_config(args.overrides)
    model_cfg = load_model_config(args.model)
    run = args.run_name or f"{args.experiment}-{args.model}"

    if args.experiment in ("rq1", "rq2"):
        tasks = load_instance(args.model, args.domains, args.limit)
        logger.info(f"{run}: {len(tasks)} tasks")
        llm = LocalLLM(cfg, model_cfg)
        for cond in conditions_from(args):
            prompts = (oracle_prompts(tasks, cond) if args.experiment == "rq1"
                       else retrieval_prompts(cfg, tasks, cond))
            run_cell(cfg, llm, tasks, prompts, run_dir(run, cond.value), cond.value, args.pass_at_5)
        llm.close()
        return

    # RQ3: adaptation methods on the held-out test split of the primary backbone.
    from novelapibench.adaptation import llm_kwargs

    if args.model != PRIMARY_MODEL:
        raise SystemExit("RQ3 adapts the primary backbone only")
    tasks = load_split("rq3_test")
    if args.domains:
        tasks = [t for t in tasks if t.domain in args.domains]
    if args.limit:
        tasks = tasks[: args.limit]
    conds = [parse_condition(c) for c in (args.conditions or ["none", "Full"])]
    logger.info(f"{run}: {len(tasks)} test tasks, methods {args.methods}")
    for method in args.methods:
        llm = LocalLLM(cfg, model_cfg, **llm_kwargs(method))
        for cond in conds:
            cell = f"{method}__{cond.value}"
            run_cell(cfg, llm, tasks, oracle_prompts(tasks, cond), run_dir(run, cell), cell, args.pass_at_5)
        llm.close()


if __name__ == "__main__":
    main()
