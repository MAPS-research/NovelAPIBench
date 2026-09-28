"""Build a backbone's instance from the released benchmark (paper Section 3.2, Stage 4).

Benchmark instances are model-conditioned: a backbone's instance holds the tasks it fails
without knowledge (C2) that are valid (C1) and solvable with the Full bundle (C3). This script
applies Stage 4 to a new backbone on the released tasks, with the paper's settings, and
checks that the released records reproduce the six released instances.

    --verify                 rebuild the released instances, exclusion list and RQ3 split from
                             data/benchmark/filters/ and check that they hold the same tasks as
                             data/benchmark/ (CPU, seconds)
    --model <name>           build configs/models/<name>.yaml's instance, phase by phase:
        --phase generate     C2: three completions per task at T = 0.8 (GPU)
        --phase score        C2: run them; a task is novel when all three fail (CPU)
        --phase c3           C3 on the novel tasks (GPT-5-mini)
        --phase call-records reference call records and the second C3 pass for novel tasks that
                             are in no released instance (CPU and GPT-5-mini)
        --phase instance     write outputs/instances/<name>/instance.txt
        --phase all          everything, in order (default)

Examples:
    python scripts/build_instance.py --verify
    python scripts/build_instance.py --model my-coder --phase generate        # on a GPU node
    python scripts/build_instance.py --model my-coder --phase score c3 call-records instance
    python scripts/build_instance.py --model my-coder --debug                 # two tasks per library

Rerunning skips the work already recorded. Once the instance exists, the other scripts take the model
like a released one:  python scripts/run_inference.py --experiment rq1 --model my-coder
C3 and the second C3 pass need OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novelapibench.config import list_libraries, load_config, load_model_config  # noqa: E402
from novelapibench.log import logger, setup_logging  # noqa: E402
from novelapibench.paths import use_debug_outputs  # noqa: E402

PHASES = ["generate", "score", "c3", "call-records", "instance"]
DEBUG_LIMIT = 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="check the released instances against their records")
    ap.add_argument("--model", default=None, help="backbone (configs/models/<model>.yaml)")
    ap.add_argument("--phase", nargs="+", default=["all"], metavar="PHASE",
                    help=f"one or more of {', '.join(PHASES + ['all'])} (default: all)")
    ap.add_argument("--libraries", nargs="+", choices=list_libraries(), default=None,
                    help="only these libraries (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks per library")
    ap.add_argument("--debug", action="store_true",
                    help=f"{DEBUG_LIMIT} tasks per library unless --limit, outputs under outputs/debug/")
    ap.add_argument("overrides", nargs="*", help="config overrides KEY=VALUE")
    args = ap.parse_args()
    # KEY=VALUE overrides written after --phase end up in its list
    args.overrides += [p for p in args.phase if "=" in p]
    args.phase = [p for p in args.phase if "=" not in p]
    bad = set(args.phase) - set(PHASES + ["all"])
    if bad:
        ap.error(f"unknown phase(s) {sorted(bad)}; choose from {PHASES + ['all']}")
    if args.verify == bool(args.model):
        ap.error("pass either --verify or --model")
    setup_logging()
    if args.debug:
        use_debug_outputs()
        if args.limit is None:
            args.limit = DEBUG_LIMIT
    cfg = load_config(args.overrides, "construction")

    from novelapibench.construction.released import NewInstance, rebuild_released

    if args.verify:
        same = rebuild_released(cfg)
        for name, ok in same.items():
            logger.info(f"  {name:36s} {'same tasks' if ok else 'DIFFERENT'}")
        if not all(same.values()):
            raise SystemExit("the released records do not reproduce data/benchmark/")
        logger.info("the released records reproduce every instance, the exclusions and the RQ3 split")
        return

    phases = PHASES if "all" in args.phase else [p for p in PHASES if p in args.phase]
    if "generate" in phases:
        load_model_config(args.model)   # fail early on a missing config
    job = NewInstance(args.model, cfg, args.libraries, args.limit)
    logger.info(f"{args.model}: {len(job.tasks)} released tasks pass C1 -> {job.dir}")
    steps = {"generate": job.generate, "score": job.score, "c3": job.c3,
             "call-records": job.call_records, "instance": job.instance}
    for phase in phases:
        logger.info(f"=== {phase} ===")
        steps[phase]()


if __name__ == "__main__":
    main()
