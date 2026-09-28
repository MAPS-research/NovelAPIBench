"""Per-task frames that every table and figure of Section 4 aggregates.

Two sources give frames with the same columns:

* ``paper``  the released verdicts behind the paper (``data/paper_results/``);
* ``runs``   fresh ``outputs/runs/<run>/<cell>/results.jsonl`` files written by
  ``scripts/evaluate.py`` (runs ``rq1-<model>``, ``rq2-<model>``, ``rq3-<model>``).

Frames:

    rq12            model, setting (oracle | retrieval), condition, task_id, api_name, library,
                    domain, difficulty, passed (0/1), label (Pass or a failure label), pass_at_5
    rq3             method, condition (none | Full), the same task columns, passed, label,
                    call_check_failed (the completion called the target API but failed only the
                    call-record check), pass_at_5
    rq3_responses   method, condition, task_id, response (greedy)
    retrieval_hits  task_id, condition, target_rank, hit5: rank of the first chunk of the
                    target API under retrieval on the primary backbone (Section 4.3)

Row order is preserved from the source (paper: the order the paper's numbers were computed in;
runs: prediction order), because bootstrap interval endpoints depend on it (see ``stats``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from novelapibench.benchmark import PRIMARY_MODEL
from novelapibench.evaluation.failure_taxonomy import LABELS, PASS
from novelapibench.io import iter_jsonl
from novelapibench.knowledge import Condition
from novelapibench.log import logger
from novelapibench.paths import PAPER_RESULTS_DIR, runs_dir

#: Backbones in the order of the paper's figures, with display names.
MODELS: dict[str, str] = {
    "qwen2.5-coder-7b": "Qwen2.5-Coder-7B",
    "seed-coder-8b-instruct": "Seed-Coder-8B",
    "opencoder-8b-instruct": "OpenCoder-8B",
    "r1-distill-qwen-7b": "R1-Distill-Qwen-7B",
    "qwen2.5-coder-14b": "Qwen2.5-Coder-14B",
    "qwen2.5-coder-32b": "Qwen2.5-Coder-32B",
}
#: RQ3 methods with display names (Table 1 order is Base, SFT, RAFT, GRACE, MEMIT, AlphaEdit-LoRA).
METHODS: dict[str, str] = {
    "base": "Base", "sft": "SFT", "raft": "RAFT", "grace": "GRACE", "memit": "MEMIT",
    "alphaedit_lora": "AlphaEdit-LoRA",
}
CONDITIONS: list[str] = [c.value for c in Condition]
#: Pass followed by the six failure labels (Section 3.3).
LABEL_ORDER: list[str] = [PASS, *LABELS]
#: Label of a failure scored without the failure classifier (``--no-failure-labels``).
UNLABELLED = "Unlabelled"
#: Assertion message of the call-record check (``evaluation.call_check``).
CALL_CHECK_MESSAGE = "no call to the target API reproduces reference call"

RQ12_COLUMNS = ["model", "setting", "condition", "task_id", "api_name", "library", "domain",
                "difficulty", "passed", "label", "pass_at_5"]
RQ3_COLUMNS = ["method", "condition", "task_id", "api_name", "library", "domain", "difficulty",
               "passed", "label", "call_check_failed", "pass_at_5"]
HITS_COLUMNS = ["task_id", "condition", "target_rank", "hit5"]
HITS_FILE = "retrieval_hits.csv"


@dataclass
class AnalysisData:
    rq12: pd.DataFrame
    rq3: pd.DataFrame
    rq3_responses: pd.DataFrame
    retrieval_hits: pd.DataFrame | None

    def oracle(self, model: str = PRIMARY_MODEL) -> pd.DataFrame:
        f = self.rq12
        return f[(f["model"] == model) & (f["setting"] == "oracle")]

    def retrieval(self, model: str = PRIMARY_MODEL) -> pd.DataFrame:
        f = self.rq12
        return f[(f["model"] == model) & (f["setting"] == "retrieval")]


def _string_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c not in ("passed", "pass_at_5", "call_check_failed", "target_rank", "hit5"):
            df[c] = df[c].astype(str)
    return df


def read_hits(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values={"target_rank": [""]},
                       dtype={"task_id": str, "condition": str})


# ---------------------------------------------------------------------------------------------
# data/paper_results
# ---------------------------------------------------------------------------------------------

def from_paper_results(root: Path = PAPER_RESULTS_DIR) -> AnalysisData:
    """Frames of the released per-task verdicts behind the paper."""
    rq12 = _string_columns(pd.read_csv(root / "rq12_per_task.csv.gz", keep_default_na=False,
                                       na_values={"pass_at_5": [""]}))
    rq3 = _string_columns(pd.read_csv(root / "rq3_per_task.csv.gz", keep_default_na=False,
                                      na_values={"pass_at_5": [""]}))
    responses = pd.read_json(root / "rq3_responses.jsonl.gz", lines=True, dtype=False,
                             convert_dates=False)
    hits = read_hits(root / "rq2_retrieval_hits.csv.gz")
    return AnalysisData(rq12, rq3, responses, hits)


# ---------------------------------------------------------------------------------------------
# outputs/runs
# ---------------------------------------------------------------------------------------------

def _cell_records(cell_dir: Path) -> list[dict]:
    """``results.jsonl`` records of one cell in prediction order."""
    results = {r["task_id"]: r for r in iter_jsonl(cell_dir / "results.jsonl")}
    pred_path = cell_dir / "predictions.jsonl"
    order = [r["task_id"] for r in iter_jsonl(pred_path)] if pred_path.exists() else list(results)
    missing = [t for t in order if t not in results]
    if missing:
        logger.warning(f"{cell_dir}: {len(missing)} predictions not scored yet; skipped")
    return [results[t] for t in order if t in results]


def _row(rec: dict) -> dict:
    label = rec.get("label") or (PASS if rec["passed"] else UNLABELLED)
    return {"task_id": rec["task_id"], "api_name": rec["api_name"], "library": rec["library"],
            "domain": rec["domain"], "difficulty": rec["difficulty"],
            "passed": int(bool(rec["passed"])), "label": label,
            "call_check_failed": int(not rec["passed"] and CALL_CHECK_MESSAGE in (rec.get("error") or "")),
            "pass_at_5": rec.get("pass_at_5")}


def _condition_order(name: str) -> int:
    return CONDITIONS.index(name) if name in CONDITIONS else len(CONDITIONS)


def from_runs(root: Path | None = None) -> AnalysisData:
    """Frames of every scored cell under ``outputs/runs`` (cells without ``results.jsonl`` are
    ignored)."""
    root = root or runs_dir()
    rq12, rq3, responses = [], [], []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        exp, _, model = run.name.partition("-")
        if exp not in ("rq1", "rq2", "rq3") or not model:
            continue
        cells = [c for c in run.iterdir() if (c / "results.jsonl").exists()]
        if exp == "rq3":
            for cell in sorted(cells):
                method, _, cond = cell.name.partition("__")
                preds = {r["task_id"]: (r.get("samples") or [""])[0]
                         for r in iter_jsonl(cell / "predictions.jsonl")}
                for rec in _cell_records(cell):
                    rq3.append({"method": method, "condition": cond, **_row(rec)})
                    responses.append({"method": method, "condition": cond,
                                      "task_id": rec["task_id"], "response": preds.get(rec["task_id"], "")})
            continue
        setting = "oracle" if exp == "rq1" else "retrieval"
        for cell in sorted(cells, key=lambda c: _condition_order(c.name)):
            if cell.name not in CONDITIONS:
                logger.warning(f"{cell}: not a knowledge condition; skipped")
                continue
            for rec in _cell_records(cell):
                row = _row(rec)
                row.pop("call_check_failed")
                rq12.append({"model": model, "setting": setting, "condition": cell.name, **row})
    if not rq12 and not rq3:
        raise FileNotFoundError(f"no scored cells (results.jsonl) under {root}")
    hits_path = root / f"rq2-{PRIMARY_MODEL}" / HITS_FILE
    hits = read_hits(hits_path) if hits_path.exists() else None
    return AnalysisData(
        rq12=_string_columns(pd.DataFrame(rq12, columns=RQ12_COLUMNS)),
        rq3=_string_columns(pd.DataFrame(rq3, columns=RQ3_COLUMNS)),
        rq3_responses=pd.DataFrame(responses, columns=["method", "condition", "task_id", "response"]),
        retrieval_hits=hits,
    )


def load(source: str, runs_root: Path | None = None) -> AnalysisData:
    if source == "paper":
        return from_paper_results()
    if source == "runs":
        return from_runs(runs_root)
    raise ValueError(f"unknown source {source!r} (paper | runs)")


def write_hits(hits: pd.DataFrame, runs_root: Path | None = None) -> Path:
    path = (runs_root or runs_dir()) / f"rq2-{PRIMARY_MODEL}" / HITS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    hits[HITS_COLUMNS].to_csv(path, index=False)
    return path
