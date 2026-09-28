"""Score predictions: execute each completion in its library's environment under the
call-record harness, and label failures (GPT-5-mini; needs OPENAI_API_KEY unless
--no-failure-labels).

Examples:
    python scripts/evaluate.py --run rq1-qwen2.5-coder-7b              # every cell of a run
    python scripts/evaluate.py --run rq3-qwen2.5-coder-7b --cells sft__Full base__Full
    python scripts/evaluate.py --run rq1-qwen2.5-coder-7b --no-failure-labels --workers 8
    python scripts/evaluate.py --run rq1-qwen2.5-coder-7b --debug     # a run made with --debug

Writes <cell>/results.jsonl (one record per task: passed, label, ...) and <cell>/summary.json.
Evaluation is CPU-only; keep --workers at or below the number of cores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novelapibench.config import load_config  # noqa: E402
from novelapibench.evaluation.evaluator import Evaluator  # noqa: E402
from novelapibench.log import logger, setup_logging  # noqa: E402
from novelapibench.paths import run_dir, use_debug_outputs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="directory name under outputs/runs/")
    ap.add_argument("--cells", nargs="+", default=None, help="default: every cell with predictions")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-failure-labels", action="store_true",
                    help="skip the GPT-5-mini failure classifier (pass/fail only)")
    ap.add_argument("--pass-at-5", action="store_true", help="also score predictions_k5.jsonl")
    ap.add_argument("--debug", action="store_true", help="score a run under outputs/debug/")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    setup_logging()
    if args.debug:
        use_debug_outputs()

    cfg = load_config(args.overrides)
    root = run_dir(args.run)
    cells = args.cells or sorted(p.name for p in root.iterdir() if (p / "predictions.jsonl").exists())
    if not cells:
        raise SystemExit(f"no predictions under {root}")
    classify = cfg.evaluation.classify_failures and not args.no_failure_labels
    ev = Evaluator(cfg, classify_failures=classify)
    rows = []
    for cell in cells:
        s = ev.evaluate_cell(root / cell, workers=args.workers, pass_at_5=args.pass_at_5)
        rows.append((cell, s))
    logger.info("pass@1 by cell:")
    for cell, s in rows:
        logger.info(f"  {cell:28s} {100 * s['pass_at_1']:5.1f}%  (n={s['n_tasks']})")


if __name__ == "__main__":
    main()
