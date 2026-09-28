"""Rebuild the benchmark from scratch (paper Section 3.2, Appendix B.2).

Subcommands, in pipeline order (each reads the previous one's output under
outputs/construction/, override the location with NOVELAPIBENCH_OUTPUTS):

    pool          Stage 1: construction environments, API maps, frozen candidate pool
    extract       Stage 2: knowledge bundles                         (GPT-5-mini, web search)
    generate      Stage 3: tasks with execute-then-assert harnesses  (GPT-5-mini)
    filter c1     Stage 4, C1: reference validity
    filter c2     Stage 4, C2: empirical novelty of one backbone     (GPU; --phase generate|score)
    filter c3     Stage 4, C3: solvability with the Full bundle      (GPT-5-mini)
    call-records  reference call records of the call-record check
    c3-recheck    C3 under the call-record check, and the resulting exclusions (GPT-5-mini)
    instances     per-backbone instances, exclusion list, RQ3 split  (outputs/construction/benchmark/)

Examples:
    python scripts/build_benchmark.py pool --libraries flask --boundaries opencoder-8b-instruct
    python scripts/build_benchmark.py extract --libraries flask --from-pool --max-apis 2
    python scripts/build_benchmark.py generate --libraries flask --max-apis 1
    python scripts/build_benchmark.py filter c1 --libraries flask
    python scripts/build_benchmark.py filter c2 --model qwen2.5-coder-7b --libraries flask
    python scripts/build_benchmark.py filter c3 --libraries flask
    python scripts/build_benchmark.py call-records --libraries flask
    python scripts/build_benchmark.py c3-recheck --libraries flask
    python scripts/build_benchmark.py instances

Every subcommand takes --debug: outputs go to outputs/debug/construction/, and Stages 2-3 take one
API per library unless --max-apis is given.

LLM stages need OPENAI_API_KEY; deterministic calls are cached (configs/default.yaml `cache`).
KEY=VALUE arguments right after the subcommand override the configuration, e.g.
    python scripts/build_benchmark.py extract construction.stage2.workers=4 --libraries torch
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import DictConfig  # noqa: E402

from novelapibench.config import list_libraries, load_config, load_library_config  # noqa: E402
from novelapibench.construction.schemas import APIEntry  # noqa: E402
from novelapibench.io import iter_jsonl, read_jsonl, write_json, write_jsonl  # noqa: E402
from novelapibench.log import logger, setup_logging  # noqa: E402
from novelapibench.paths import POOL_DIR, construction_dir, use_debug_outputs  # noqa: E402
from novelapibench.schemas import KnowledgeBundle, Task  # noqa: E402

ROOT = construction_dir()
POOL_OUT = ROOT / "pool"
CALLS = ROOT / "call_records"
BENCH = ROOT / "benchmark"


def stage2_path(lib: str) -> Path:
    return ROOT / "stage2" / f"{lib}.jsonl"


def stage3_path(lib: str) -> Path:
    return ROOT / "stage3" / f"{lib}.jsonl"


def check_path(check: str, lib: str, model: str | None = None) -> Path:
    d = ROOT / "stage4" / check
    return (d / model if model else d) / f"{lib}.jsonl"


def strong_llm(cfg: DictConfig):
    from novelapibench.llm.strong import StrongLLM
    return StrongLLM(cfg)


def load_records(path: Path) -> list[dict]:
    return read_jsonl(path) if path.exists() else []


def load_tasks(libs: list[str]) -> list[Task]:
    return [Task(**r) for lib in libs for r in load_records(stage3_path(lib))]


def load_bundles(libs: list[str]) -> dict[str, KnowledgeBundle]:
    return {r["api_name"]: KnowledgeBundle(**r) for lib in libs for r in load_records(stage2_path(lib))}


def c2_novel(libs: list[str], models: list[str]) -> dict[str, set[str]]:
    """``model -> novel task ids`` for models with C2 records in any of ``libs``."""
    out = {}
    for m in models:
        paths = [check_path("c2", lib, m) for lib in libs if check_path("c2", lib, m).exists()]
        if paths:
            out[m] = {r["task_id"] for p in paths for r in read_jsonl(p) if r["novel"]}
    return out


def passing(check: str, libs: list[str]) -> set[str]:
    return {r["task_id"] for lib in libs for r in load_records(check_path(check, lib)) if r["passed"]}


def in_some_instance(libs: list[str], cfg: DictConfig) -> list[Task]:
    """Tasks passing C1, C3 and at least one backbone's C2 (C2 is skipped, with a warning,
    when no C2 results exist), minus the manual exclusions."""
    tasks = load_tasks(libs)
    keep = passing("c1", libs) & passing("c3", libs)
    novel = c2_novel(libs, list(cfg.construction.instances.models))
    if novel:
        keep &= set().union(*novel.values())
    else:
        logger.warning("no C2 results found: C2 is not applied to this selection")
    manual = set(cfg.construction.instances.manual_exclusions or {})
    return [t for t in tasks if t.task_id in keep and t.task_id not in manual]


def limit(items: list, n: int | None) -> list:
    return items[:n] if n else items


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_pool(args, cfg: DictConfig) -> None:
    from novelapibench.construction.stage1 import api_maps, pool

    if args.resolve_boundaries:
        for lib in args.libraries:
            versions = pool.resolve_old_versions(str(load_library_config(lib).package),
                                                 pool.boundaries(cfg))
            print(f"    {lib}: " + "{" + ", ".join(f"{m}: {json.dumps(v)}" for m, v in versions.items()) + "}")
        return
    if args.boundaries:
        cfg.construction.stage1.boundaries = {m: cfg.construction.stage1.boundaries[m]
                                              for m in args.boundaries}
    maps_dir = Path(args.api_maps_dir) if args.api_maps_dir else api_maps.default_maps_dir()
    if not args.no_build:
        for lib in args.libraries:
            for version in pool.required_versions(lib, cfg):
                api_maps.build_version_map(lib, version, cfg, maps_dir, force=args.force,
                                           repair=not args.no_repair)
    out = Path(args.out) if args.out else POOL_OUT
    pool.freeze_pool(args.libraries, cfg, maps_dir, out)
    ok = pool.verify_pool(out, cfg, maps_dir)
    logger.info(f"pool checks {'passed' if ok else 'FAILED'}")


def cmd_extract(args, cfg: DictConfig) -> None:
    from novelapibench.construction.stage2.extract import extract_library

    src = Path(args.pool) if args.pool else (POOL_DIR if args.from_pool else POOL_OUT) / "candidates.jsonl"
    entries = [APIEntry(**r) for r in iter_jsonl(src)]
    llm = strong_llm(cfg)
    for lib in args.libraries:
        mine = [e for e in entries if e.library == lib and (not args.apis or e.api_name in args.apis)]
        extract_library(lib, limit(mine, args.max_apis), cfg, llm, stage2_path(lib))


def cmd_generate(args, cfg: DictConfig) -> None:
    from novelapibench.construction.stage3.generate import generate_library

    src = Path(args.pool) if args.pool else (POOL_DIR if args.from_pool else POOL_OUT) / "candidates.jsonl"
    entries = {r["api_name"]: APIEntry(**r) for r in iter_jsonl(src)} if src.exists() else {}
    llm = strong_llm(cfg)
    for lib in args.libraries:
        bundles = [KnowledgeBundle(**r) for r in load_records(stage2_path(lib))]
        bundles = [b for b in bundles if not args.apis or b.api_name in args.apis]
        generate_library(lib, limit(bundles, args.max_apis), entries, cfg, llm, stage3_path(lib),
                         resume=args.resume)


def cmd_filter(args, cfg: DictConfig) -> None:
    from novelapibench.construction.stage4 import c1, c2, c3

    for lib in args.libraries:
        tasks = limit(load_tasks([lib]), args.limit)
        if not tasks:
            logger.info(f"{lib}: no Stage-3 tasks")
            continue
        if args.check == "c1":
            write_jsonl(check_path("c1", lib), c1.run_c1(tasks, cfg))
            continue
        c1_ok = passing("c1", [lib])
        tasks = [t for t in tasks if t.task_id in c1_ok]
        if args.check == "c2":
            responses = check_path("c2", lib, args.model).with_suffix(".responses.jsonl")
            if args.phase in ("generate", "all"):
                comps = c2.generate(tasks, args.model, cfg)
                write_jsonl(responses, [{"task_id": k, "completions": v} for k, v in comps.items()])
            if args.phase in ("score", "all"):
                comps = {r["task_id"]: r["completions"] for r in read_jsonl(responses)}
                write_jsonl(check_path("c2", lib, args.model), c2.score(tasks, comps, args.model, cfg))
        else:
            novel = c2_novel([lib], list(cfg.construction.instances.models))
            if novel:
                union = set().union(*novel.values())
                tasks = [t for t in tasks if t.task_id in union]
            records = c3.run_c3(tasks, load_bundles([lib]), strong_llm(cfg), cfg)
            write_jsonl(check_path("c3", lib), records)
            logger.info(f"C3 {lib}: {sum(r['passed'] for r in records)}/{len(records)} solvable")


def cmd_call_records(args, cfg: DictConfig) -> None:
    from novelapibench.construction.call_records import build_expected_records

    tasks = limit(in_some_instance(args.libraries, cfg), args.limit)
    path = CALLS / "expected_calls.jsonl"
    merged = {r["task_id"]: r for r in load_records(path)}
    for rec in build_expected_records(tasks, cfg):
        merged[rec["task_id"]] = rec
    write_jsonl(path, merged.values())
    statuses = [merged[t.task_id]["status"] for t in tasks if t.task_id in merged]
    logger.info(f"call records: {len(statuses)} tasks, {statuses.count('ok')} ok -> {path}")


def cmd_c3_recheck(args, cfg: DictConfig) -> None:
    from novelapibench.construction.call_records import exclusions, run_recheck

    expected = {r["task_id"]: r for r in load_records(CALLS / "expected_calls.jsonl")}
    tasks = [t for t in limit(in_some_instance(args.libraries, cfg), args.limit)
             if t.task_id in expected]
    path = CALLS / "c3_recheck.jsonl"
    merged = {r["task_id"]: r for r in load_records(path)}
    for rec in run_recheck(tasks, load_bundles(args.libraries), expected, strong_llm(cfg), cfg):
        merged[rec["task_id"]] = rec
    write_jsonl(path, merged.values())
    excluded = exclusions(expected, merged, int(cfg.construction.c3_recheck.num_samples))
    with (CALLS / "excluded_tasks.tsv").open("w") as f:
        f.write("task_id\treason\n")
        for tid in sorted(excluded):
            f.write(f"{tid}\t{excluded[tid]}\n")
    logger.info(f"C3 recheck: {sum(not merged[t.task_id]['passed'] for t in tasks if t.task_id in merged)}"
                f"/{len(tasks)} unsolved; {len(excluded)} exclusions in total")


def cmd_instances(args, cfg: DictConfig) -> None:
    from novelapibench.construction.instances import build_instances, rq3_split

    libs = args.libraries
    tasks = load_tasks(libs)
    BENCH.mkdir(parents=True, exist_ok=True)
    excluded = {}
    tsv = CALLS / "excluded_tasks.tsv"
    if tsv.exists():
        with tsv.open() as f:
            excluded = {r["task_id"]: r["reason"] for r in csv.DictReader(f, delimiter="\t")}
    else:
        logger.warning(f"{tsv} missing: no call-record exclusions applied")
    excluded.update(dict(cfg.construction.instances.manual_exclusions or {}))
    novel = c2_novel(libs, args.models)
    for m in set(args.models) - set(novel):
        logger.warning(f"{m}: no C2 results, no instance written")
    ic = cfg.construction.instances
    before, final = build_instances(tasks, passing("c1", libs), novel, passing("c3", libs),
                                    excluded, int(ic.shuffle_seed))
    for model, ids in final.items():
        (BENCH / "instances").mkdir(parents=True, exist_ok=True)
        (BENCH / "instances" / f"{model}.txt").write_text("".join(f"{t}\n" for t in ids))
        logger.info(f"{model}: {len(before[model])} tasks before exclusion, {len(ids)} final")
    rs = cfg.construction.rq3_split
    if rs.model in before:
        train, test = rq3_split(set(before[rs.model]), tasks, int(rs.seed), float(rs.train_fraction))
        (BENCH / "splits").mkdir(parents=True, exist_ok=True)
        (BENCH / "splits" / "rq3_train.txt").write_text("".join(f"{t}\n" for t in train))
        (BENCH / "splits" / "rq3_test.txt").write_text("".join(f"{t}\n" for t in test))
        logger.info(f"RQ3 split: {len(train)} train / {len(test)} test tasks")
    union = set().union(*before.values()) if before else set()
    with (BENCH / "excluded_tasks.tsv").open("w") as f:
        f.write("task_id\treason\n")
        for tid in sorted(t for t in excluded if t in union):
            f.write(f"{tid}\t{excluded[tid]}\n")
    write_jsonl(BENCH / "tasks.jsonl", (t.model_dump() for t in tasks))
    write_jsonl(BENCH / "bundles.jsonl", (b.model_dump() for b in load_bundles(libs).values()))
    expected = load_records(CALLS / "expected_calls.jsonl")
    write_jsonl(BENCH / "expected_calls.jsonl",
                ({k: r[k] for k in ("task_id", "api_name", "status", "expected", "reference_source",
                                    "n_runs")} for r in expected))
    write_json(BENCH / "manifest.json", {
        "stage3_tasks": len(tasks), "tasks_in_some_instance": len(union),
        "instances": {m: {"before_exclusion": len(before[m]), "final": len(final[m])} for m in final},
        "excluded": sum(t in union for t in excluded)})


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, llm_limit: str | None = None):
        p.add_argument("--libraries", nargs="+", default=list_libraries(),
                       help="library names (configs/libraries/<name>.yaml); default: all")
        if llm_limit == "apis":
            p.add_argument("--max-apis", type=int, default=None, help="first N APIs per library")
            p.add_argument("--apis", nargs="+", default=None, help="only these APIs")
            p.add_argument("--from-pool", action="store_true",
                           help="read the released frozen pool (data/pool/) instead of "
                                "outputs/construction/pool/")
            p.add_argument("--pool", default=None, help="explicit candidates.jsonl")
        elif llm_limit == "tasks":
            p.add_argument("--limit", type=int, default=None, help="first N tasks (per library)")
        p.add_argument("--debug", action="store_true",
                       help="outputs under outputs/debug/construction/"
                            + ("; one API per library unless --max-apis" if llm_limit == "apis" else ""))
        p.add_argument("overrides", nargs="*", help="config overrides KEY=VALUE")

    p = sub.add_parser("pool", help="Stage 1")
    common(p)
    p.add_argument("--boundaries", nargs="+", default=None,
                   help="only these backbones' boundaries (default: all six)")
    p.add_argument("--api-maps-dir", default=None,
                   help="read/write API maps here (default: outputs/construction/api_maps)")
    p.add_argument("--no-build", action="store_true", help="only use cached API maps")
    p.add_argument("--no-repair", action="store_true", help="skip environment repair")
    p.add_argument("--force", action="store_true", help="rebuild cached API maps")
    p.add_argument("--out", default=None, help="pool directory (default: outputs/construction/pool)")
    p.add_argument("--resolve-boundaries", action="store_true",
                   help="print each library's last release before every boundary (PyPI) and exit")
    p.set_defaults(func=cmd_pool)

    p = sub.add_parser("extract", help="Stage 2")
    common(p, "apis")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("generate", help="Stage 3")
    common(p, "apis")
    p.add_argument("--resume", action="store_true", help="keep tasks already written")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("filter", help="Stage 4")
    p.add_argument("check", choices=["c1", "c2", "c3"])
    common(p, "tasks")
    p.add_argument("--model", default=None, help="C2: backbone (configs/models/<model>.yaml)")
    p.add_argument("--phase", choices=["generate", "score", "all"], default="all",
                   help="C2: generate completions (GPU), score them (CPU), or both")
    p.set_defaults(func=cmd_filter)

    p = sub.add_parser("call-records", help="expected call records")
    common(p, "tasks")
    p.set_defaults(func=cmd_call_records)

    p = sub.add_parser("c3-recheck", help="C3 under the call-record check")
    common(p, "tasks")
    p.set_defaults(func=cmd_c3_recheck)

    p = sub.add_parser("instances", help="instances, exclusions, RQ3 split")
    common(p)
    p.add_argument("--models", nargs="+", default=None, help="default: construction.instances.models")
    p.set_defaults(func=cmd_instances)

    args = ap.parse_args()
    setup_logging(os.environ.get("NOVELAPIBENCH_LOG_LEVEL", "INFO"))
    if args.debug:
        global ROOT, POOL_OUT, CALLS, BENCH
        use_debug_outputs()
        ROOT = construction_dir()
        POOL_OUT, CALLS, BENCH = ROOT / "pool", ROOT / "call_records", ROOT / "benchmark"
        if getattr(args, "max_apis", 0) is None:
            args.max_apis = 1
    cfg = load_config(args.overrides, "construction")
    if getattr(args, "check", None) == "c2" and not args.model:
        ap.error("filter c2 needs --model")
    if args.command == "instances" and not args.models:
        args.models = list(cfg.construction.instances.models)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
