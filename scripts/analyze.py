"""Reproduce the tables and figures of the paper's main body (Section 4) and print the numbers
quoted in the text.

Examples:
    # from the released per-task verdicts behind the paper (data/paper_results/)
    python scripts/analyze.py
    # from your own runs (outputs/runs/<run>/<cell>/results.jsonl, see scripts/evaluate.py)
    python scripts/analyze.py --source runs
    # ... and rank the target API under retrieval for Figure 5b (needs faiss and
    # sentence-transformers; uses the retrieval indexes built by the RQ2 inference run)
    python scripts/analyze.py --source runs --retrieval-hits
    # ... or from runs made with --debug (outputs/debug/runs/)
    python scripts/analyze.py --source runs --debug

Writes CSV tables to <out>/tables/ and PDF/PNG figures to <out>/figures/
(default <out>: outputs/analysis/<source>). Percentages throughout; intervals are 95%
API-cluster bootstrap intervals (Appendix D.5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from novelapibench.analysis import collect, figures, rq1, rq2, rq3  # noqa: E402
from novelapibench.benchmark import PRIMARY_MODEL, load_bundles, load_tasks  # noqa: E402
from novelapibench.config import load_config  # noqa: E402
from novelapibench.log import logger, setup_logging  # noqa: E402
from novelapibench.paths import analysis_dir, use_debug_outputs  # noqa: E402


def write(table: pd.DataFrame, out: Path, name: str) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{name}.csv", index=False)
    logger.info(f"wrote {out / name}.csv ({len(table)} rows)")
    return table


def section(title: str, lines: list[str]) -> None:
    print(f"\n== {title}")
    for line in lines:
        print(f"  {line}")


def run_rq1(data: collect.AnalysisData, tables: Path, figs: Path) -> None:
    primary = data.oracle()
    if primary.empty:
        logger.warning(f"RQ1: no oracle results for {PRIMARY_MODEL}; skipped")
        return
    is_modified = {a: b.is_modified for a, b in load_bundles().items()}
    conds = set(primary["condition"])
    cells = write(rq1.condition_table(primary), tables, "rq1_conditions")
    failures = write(rq1.failure_table(primary), tables, "rq1_failure_labels")
    figures.rq1_components(cells, failures, figs)
    novelty = interaction = xb = None
    if {"S", "E", "S+E"} <= conds:
        novelty = write(rq1.novelty_table(primary, is_modified), tables, "rq1_novelty_type")
        interaction = write(rq1.novelty_interaction(data.rq12, is_modified), tables,
                            "rq1_novelty_interaction")
    cross = rq1.cross_backbone(data.rq12)
    if not cross.empty:
        xb = write(cross, tables, "rq1_cross_backbone")
    if novelty is not None and xb is not None:
        figures.rq1_novelty_backbones(novelty, xb, figs)
    else:
        logger.warning("RQ1: Figure 4 needs S, E and S+E on the primary backbone and the same "
                       "conditions on the other backbones; skipped")
    section(f"RQ1 ({PRIMARY_MODEL}, {cells['n_tasks'].iloc[0]:,} tasks)",
            rq1.headline(cells, failures, novelty, interaction, xb))


def run_rq2(data: collect.AnalysisData, tables: Path, figs: Path, compute_hits: bool,
            overrides: list[str], runs_root: Path | None) -> None:
    oracle, retrieval = data.oracle(), data.retrieval()
    if oracle.empty or retrieval.empty:
        logger.warning(f"RQ2: need oracle and retrieval results for {PRIMARY_MODEL}; skipped")
        return
    overall = write(rq2.oracle_vs_retrieval(oracle, retrieval), tables, "rq2_oracle_vs_retrieval")
    hits = data.retrieval_hits
    if hits is None and compute_hits:
        tasks = load_tasks()
        conds = [c for c in collect.CONDITIONS if c != "none" and c in set(retrieval["condition"])]
        ids = list(dict.fromkeys(retrieval["task_id"]))
        hits = rq2.compute_retrieval_hits(load_config(overrides), [tasks[t] for t in ids], conds)
        logger.info(f"wrote {collect.write_hits(hits, runs_root)}")
    shared = None
    if hits is None:
        logger.warning("RQ2: no retrieval hits (run with --retrieval-hits); Figure 5b skipped")
    elif set(rq2.SHARED_HIT_CONDITIONS) <= set(hits["condition"]):
        shared = write(rq2.shared_hits(oracle, retrieval, hits), tables, "rq2_shared_hits")
    figures.rq2_oracle_vs_real(overall, shared, figs)
    section(f"RQ2 ({PRIMARY_MODEL})", rq2.headline(overall, shared))


def run_rq3(data: collect.AnalysisData, tables: Path, figs: Path) -> None:
    if data.rq3.empty:
        logger.warning("RQ3: no results; skipped")
        return
    table = write(rq3.headline(data.rq3), tables, "rq3_headline")
    byp = write(rq3.bypass(data.rq3, data.rq3_responses), tables, "rq3_bypass")
    have = set(zip(data.rq3["method"], data.rq3["condition"]))
    fx = None
    if {(m, "Full") for m in [rq3.BASE, *rq3.FIX_METHODS]} <= have:
        fx = write(rq3.fixes(data.rq3), tables, "rq3_fixes")
        figures.rq3_bypass_fixes(byp, fx, figs)
    else:
        logger.warning("RQ3: Figure 6 needs base, sft and raft with Full; skipped")
    section(f"RQ3 ({table['n_tasks'].iloc[0]} held-out tasks)", rq3.headline_lines(table, byp, fx))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["paper", "runs"], default="paper",
                    help="paper: data/paper_results/; runs: outputs/runs/ (default: paper)")
    ap.add_argument("--runs-dir", type=Path, default=None, help="default: outputs/runs")
    ap.add_argument("--out", type=Path, default=None, help="default: outputs/analysis/<source>")
    ap.add_argument("--retrieval-hits", action="store_true",
                    help="runs only: rank the target API in the retrieval indexes (Figure 5b)")
    ap.add_argument("--debug", action="store_true",
                    help="read runs from and write to outputs/debug/")
    ap.add_argument("overrides", nargs="*", help="config overrides for --retrieval-hits")
    args = ap.parse_args()
    setup_logging()
    if args.debug:
        use_debug_outputs()

    out = args.out or analysis_dir() / args.source
    tables, figs = out / "tables", out / "figures"
    data = collect.load(args.source, args.runs_dir)
    run_rq1(data, tables, figs)
    run_rq2(data, tables, figs, args.source == "runs" and args.retrieval_hits, args.overrides,
            args.runs_dir)
    run_rq3(data, tables, figs)
    print(f"\ntables: {tables}\nfigures: {figs}")


if __name__ == "__main__":
    main()
