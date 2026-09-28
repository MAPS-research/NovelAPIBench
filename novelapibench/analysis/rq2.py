"""RQ2: what limits the effectiveness of retrieved knowledge? (Section 4.3, Figure 5)

* ``oracle_vs_retrieval``  per condition, pass@1 with the oracle bundle and with the top-5
                           retrieved chunks on the same tasks: marginal intervals, the paired
                           difference (retrieved minus oracle), exact McNemar, Holm over the
                           conditions (Figure 5a);
* ``shared_hits``          the same on the tasks where the target API's first chunk is ranked in
                           the top five under every one of S, S+M, S+E and Full (Figure 5b);
* ``compute_retrieval_hits`` the per-task rank of the target API's first chunk, replaying the
                           inference-time query against the retrieval indexes (Appendix D.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from novelapibench.analysis.collect import CONDITIONS
from novelapibench.analysis.stats import holm, mcnemar_exact, mean_ci, paired_diff_ci
from novelapibench.schemas import Task

#: Conditions whose top-5 hits define the shared-hit subset (Figure 5b).
SHARED_HIT_CONDITIONS = ["S", "S+M", "S+E", "Full"]
TOP_K = 5


def _ci_cols(prefix: str, ci) -> dict:
    return {prefix: 100 * ci.point, f"{prefix}_lo": 100 * ci.lo, f"{prefix}_hi": 100 * ci.hi}


def oracle_vs_retrieval(oracle: pd.DataFrame, retrieval: pd.DataFrame) -> pd.DataFrame:
    rows, pvals = [], {}
    for cond in CONDITIONS:
        a = oracle[oracle["condition"] == cond].set_index("task_id")
        b = retrieval[retrieval["condition"] == cond].set_index("task_id")
        shared = a.index.intersection(b.index)
        if shared.empty:
            continue
        a, b = a.loc[shared], b.loc[shared]
        cl = a["api_name"].tolist()
        ci_o, ci_r = mean_ci(a["passed"], cl), mean_ci(b["passed"], cl)
        d = paired_diff_ci(b["passed"], a["passed"], cl)
        n01, n10, p = mcnemar_exact(b["passed"], a["passed"])
        pvals[cond] = p
        rows.append({"condition": cond, "n_tasks": len(shared), "n_apis": ci_o.n_clusters,
                     **_ci_cols("oracle", ci_o), **_ci_cols("retrieval", ci_r), **_ci_cols("delta", d),
                     "n_retrieval_only": n10, "n_oracle_only": n01, "mcnemar_p": p})
    adj = holm(pvals)
    for r in rows:
        r["holm_p"], r["significant"] = adj[r["condition"]]
    return pd.DataFrame(rows)


def shared_hits(oracle: pd.DataFrame, retrieval: pd.DataFrame, hits: pd.DataFrame,
                conditions: list[str] = SHARED_HIT_CONDITIONS) -> pd.DataFrame:
    """Oracle vs retrieval on the tasks whose target is in the top five under all ``conditions``."""
    o = oracle.set_index(["condition", "task_id"])["passed"]
    r = retrieval.set_index(["condition", "task_id"])["passed"]
    api = oracle.drop_duplicates("task_id").set_index("task_id")["api_name"]
    sub = hits[hits["condition"].isin(conditions)
               & hits["task_id"].isin(set(oracle["task_id"]) & set(retrieval["task_id"]))]
    piv = sub.pivot(index="task_id", columns="condition", values="hit5")
    shared = set(piv.index[(piv.reindex(columns=conditions) == 1).all(axis=1)])
    rows = []
    for cond in conditions:
        g = sub[(sub["condition"] == cond) & sub["task_id"].isin(shared)]
        keys = list(zip(g["condition"], g["task_id"]))
        po, pr = o.reindex(keys).to_numpy(float), r.reindex(keys).to_numpy(float)
        cl = api.reindex(g["task_id"]).tolist()
        ci_o, ci_r, d = mean_ci(po, cl), mean_ci(pr, cl), paired_diff_ci(pr, po, cl)
        rows.append({"condition": cond, "n_tasks": ci_o.n, "n_apis": ci_o.n_clusters,
                     **_ci_cols("oracle", ci_o), **_ci_cols("retrieval", ci_r), **_ci_cols("delta", d)})
    return pd.DataFrame(rows)


def compute_retrieval_hits(cfg: DictConfig, tasks: list[Task], conditions: list[str]) -> pd.DataFrame:
    """Rank of the first chunk of each task's target API in the full retrieval ranking.

    Uses the same indexes and query as inference (``inference.retrieval``); an index that does
    not exist yet is built. Needs ``faiss`` and ``sentence-transformers``.
    """
    from novelapibench.inference.retrieval import Retriever
    from novelapibench.knowledge import parse_condition

    rows, encoder = [], None
    by_domain: dict[str, list[Task]] = {}
    for t in tasks:
        by_domain.setdefault(t.domain, []).append(t)
    for domain, dtasks in by_domain.items():
        queries = None
        for name in conditions:
            ret = Retriever(cfg, domain, parse_condition(name), encoder=encoder)
            encoder = ret.encoder
            if queries is None:
                queries = np.asarray(encoder.encode([Retriever.query(t) for t in dtasks],
                                                    normalize_embeddings=True, batch_size=64),
                                     dtype="float32")
            _, order = ret.index.search(queries, len(ret.texts))
            chunk_api = np.array([m["api_name"] for m in ret.metadata])
            for t, ranked in zip(dtasks, order):
                is_target = chunk_api[ranked] == t.api_name
                rank = int(np.flatnonzero(is_target)[0]) + 1 if is_target.any() else None
                rows.append({"task_id": t.task_id, "condition": name, "target_rank": rank,
                             "hit5": int(rank is not None and rank <= TOP_K)})
    return pd.DataFrame(rows).astype({"target_rank": "Int64"})


def headline(overall: pd.DataFrame, shared: pd.DataFrame | None) -> list[str]:
    """The RQ2 numbers quoted in Section 4.3."""
    o = overall.set_index("condition")
    fig = [c for c in ["S", "C", "E", "S+E", "Full"] if c in o.index]
    lines = []
    if fig:
        d = o.loc[fig, "delta"]
        lines.append(f"retrieval - oracle over {', '.join(fig)}: {d.max():.1f} to {d.min():.1f} pp "
                     f"(n={int(o['n_tasks'].max())})")
    if "Full" in o.index:
        lines.append(f"Full: oracle {o.loc['Full', 'oracle']:.1f} -> retrieval {o.loc['Full', 'retrieval']:.1f}")
    if {"S+E", "S"} <= set(o.index):
        lines.append(f"retrieval: S+E - S = {o.loc['S+E', 'retrieval'] - o.loc['S', 'retrieval']:.1f} pp")
    if shared is not None and not shared.empty:
        s = shared.set_index("condition")
        lines.append(f"shared-hit tasks: n={int(s['n_tasks'].iloc[0])}; retrieval - oracle: "
                     + ", ".join(f"{c} {s.loc[c, 'delta']:+.1f}" for c in s.index))
    return lines
