"""RQ3: what does parametric adaptation contribute? (Section 4.4, Table 1, Figure 6)

Every method is scored on the same 348 held-out tasks, without knowledge (``none``) and with
the Full bundle (``Full``):

* ``headline``     pass@1 with and without the bundle (API-cluster BCa intervals), and each
                   adapted model's Full cell against Base with Full: paired bootstrap interval,
                   exact McNemar, Holm over the five comparisons (Table 1);
* ``bypass``       failures in which the target API was never called, split by what the code
                   did instead (Figure 6a; Appendix G.2);
* ``fixes``        Base+Full failures by label, and how many of them SFT and RAFT (with Full)
                   solve (Figure 6b).

Bypass modes of a failed greedy completion labelled ``WrongAPISelection`` (failures that called
the target and only failed the call-record check are not considered), from its extracted code:

    empty                   no code (at most five characters)
    same_name_reimpl        defines its own ``def`` / ``class`` named like the target's leaf name
    reimplemented           the leaf name never appears
    referenced_not_invoked  the name appears, but the target never runs

A *bypass* is ``reimplemented`` or ``same_name_reimpl``: the model wrote its own version of the
functionality instead of calling the documented API.
"""

from __future__ import annotations

import re

import pandas as pd

from novelapibench.analysis.collect import METHODS
from novelapibench.analysis.stats import holm, mcnemar_exact, mean_ci, paired_diff_ci
from novelapibench.evaluation.extraction import extract_code
from novelapibench.evaluation.failure_taxonomy import LABELS

BASE = "base"
BYPASS_MODES = ["reimplemented", "same_name_reimpl", "referenced_not_invoked", "empty"]
#: Figure 6b pools the small failure categories.
FIX_GROUPS = {"API selection": ["WrongAPISelection"], "Program logic": ["WrongLogic"],
              "Other errors": ["WrongImport", "WrongSyntax", "WrongParam", "WrongShapeDtype"]}
FIX_METHODS = ["sft", "raft"]


def cell(method: str, condition: str) -> str:
    return f"{method}__{condition}"


def _wide(rq3: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    f = rq3.assign(cell=rq3["method"] + "__" + rq3["condition"])
    w = f.pivot_table(index="task_id", columns="cell", values="passed", observed=True)
    return w, f.drop_duplicates("task_id").set_index("task_id")["api_name"]


def _ci_cols(prefix: str, ci) -> dict:
    return {prefix: 100 * ci.point, f"{prefix}_lo": 100 * ci.lo, f"{prefix}_hi": 100 * ci.hi}


def headline(rq3: pd.DataFrame) -> pd.DataFrame:
    """Table 1: pass@1 without / with the Full bundle, and the change from Base with Full."""
    w, api = _wide(rq3)
    ref = cell(BASE, "Full")
    rows, pvals = [], {}
    for m in METHODS:
        wo, wi = cell(m, "none"), cell(m, "Full")
        cols = [c for c in (wo, wi) if c in w.columns]
        if not cols:
            continue
        sub = w[cols].dropna()
        cl = api.reindex(sub.index).tolist()
        row = {"method": m, "n_tasks": len(sub), "n_apis": len(set(cl))}
        for name, c in (("without", wo), ("full", wi)):
            if c in sub.columns:
                row.update(_ci_cols(name, mean_ci(sub[c], cl)))
        if m != BASE and ref in w.columns and wi in w.columns:
            pair = w[[wi, ref]].dropna()
            cl = api.reindex(pair.index).tolist()
            d = paired_diff_ci(pair[wi], pair[ref], cl)
            n01, n10, p = mcnemar_exact(pair[wi], pair[ref])
            pvals[m] = p
            row.update({**_ci_cols("delta_vs_base_full", d), "n_gained": n10, "n_lost": n01,
                        "mcnemar_p": p})
        rows.append(row)
    adj = holm(pvals)
    for r in rows:
        if r["method"] in adj:
            r["holm_p"], r["significant"] = adj[r["method"]]
    return pd.DataFrame(rows)


def bypass_mode(response: str, api_name: str) -> str:
    """Bypass mode of a completion that never called its target API (see module docstring)."""
    code = extract_code(response).strip()
    if len(code) <= 5:
        return "empty"
    leaf = re.escape(api_name.split(".")[-1])
    if re.search(rf"^\s*(?:async\s+def|def|class)\s+{leaf}\b", code, re.M):
        return "same_name_reimpl"
    if not re.search(rf"\b{leaf}\b", code):
        return "reimplemented"
    return "referenced_not_invoked"


def _cells(rq3: pd.DataFrame, responses: pd.DataFrame) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Per-cell frames indexed by task id, on the tasks every cell scored (sorted), with the
    bypass mode of each task."""
    resp = responses.set_index(["method", "condition", "task_id"])["response"]
    groups = {cell(m, c): g.set_index("task_id") for (m, c), g in rq3.groupby(["method", "condition"])}
    ids = sorted(set.intersection(*(set(g.index) for g in groups.values())))
    out = {}
    for name, g in groups.items():
        g = g.loc[ids].copy()
        m, c = name.split("__")
        candidate = (g["passed"] == 0) & (g["label"] == "WrongAPISelection") & (g["call_check_failed"] == 0)
        g["mode"] = [bypass_mode(resp.get((m, c, t), ""), g.at[t, "api_name"]) if candidate[t] else None
                     for t in ids]
        g["bypass"] = g["mode"].isin(["reimplemented", "same_name_reimpl"]).astype(float)
        out[name] = g
    return ids, out


def bypass(rq3: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    """Figure 6a: bypass modes per method and condition; bypass share (%) of all tasks with its
    interval, and for adapted models with Full the paired difference to Base with Full."""
    ids, cells = _cells(rq3, responses)
    ref = cells.get(cell(BASE, "Full"))
    cl = next(iter(cells.values()))["api_name"].tolist()
    rows = []
    for m in METHODS:
        for c in ("none", "Full"):
            g = cells.get(cell(m, c))
            if g is None:
                continue
            row = {"method": m, "condition": c, "n_tasks": len(g),
                   "n_selection_failures": int(g["mode"].notna().sum()),
                   **{k: int((g["mode"] == k).sum()) for k in BYPASS_MODES},
                   **_ci_cols("bypass_pct", mean_ci(g["bypass"], cl))}
            if c == "Full" and m != BASE and ref is not None:
                d = paired_diff_ci(g["bypass"], ref["bypass"], cl)
                row.update({**_ci_cols("bypass_vs_base_full", d),
                            "mcnemar_p": mcnemar_exact(g["bypass"], ref["bypass"])[2]})
            rows.append(row)
    return pd.DataFrame(rows)


def fixes(rq3: pd.DataFrame) -> pd.DataFrame:
    """Figure 6b: Base+Full failures by label and how many SFT / RAFT with Full solve."""
    w = {m: rq3[(rq3["method"] == m) & (rq3["condition"] == "Full")].set_index("task_id")
         for m in [BASE, *FIX_METHODS]}
    ids = sorted(set.intersection(*(set(g.index) for g in w.values())))
    base = w[BASE].loc[ids]
    fails = base[base["passed"] == 0]
    rows = []
    for lab in [*LABELS, "TOTAL"]:
        sub = fails.index if lab == "TOTAL" else fails.index[fails["label"] == lab]
        row = {"base_label": lab, "n": len(sub)}
        for m in FIX_METHODS:
            k = int(w[m].loc[sub, "passed"].sum())
            row[f"{m}_fixed"] = k
            row[f"{m}_fixed_pct"] = 100 * k / len(sub) if len(sub) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def fix_groups(fx: pd.DataFrame) -> pd.DataFrame:
    """``fixes`` pooled into the three groups of Figure 6b."""
    f = fx.set_index("base_label")
    rows = []
    for name, labels in FIX_GROUPS.items():
        row = {"group": name, "n": int(f.loc[labels, "n"].sum())}
        for m in FIX_METHODS:
            row[f"{m}_fixed"] = int(f.loc[labels, f"{m}_fixed"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def headline_lines(table: pd.DataFrame, byp: pd.DataFrame, fx: pd.DataFrame | None) -> list[str]:
    """The RQ3 numbers quoted in Section 4.4."""
    lines = [f"{'method':<16}{'none':>7}{'Full':>7}   delta vs Base+Full [95% CI]"]
    for r in table.to_dict("records"):
        d = (f"{r['delta_vs_base_full']:+.1f} [{r['delta_vs_base_full_lo']:+.1f}, "
             f"{r['delta_vs_base_full_hi']:+.1f}]" if pd.notna(r.get("delta_vs_base_full")) else "---")
        lines.append(f"{METHODS[r['method']]:<16}{r.get('without', float('nan')):7.1f}"
                     f"{r.get('full', float('nan')):7.1f}   {d}")
    b = byp[byp["condition"] == "Full"].set_index("method")["bypass_pct"]
    if not b.empty:
        lines.append("bypass with Full (%): " + ", ".join(f"{METHODS[m]} {v:.1f}" for m, v in b.items()))
    if fx is None:
        return lines
    f = fx.set_index("base_label")
    for lab in ("WrongAPISelection", "WrongLogic"):
        lines.append(f"Base+Full {lab} failures fixed: "
                     + ", ".join(f"{METHODS[m]} {int(f.loc[lab, f'{m}_fixed'])}" for m in FIX_METHODS)
                     + f" of {int(f.loc[lab, 'n'])}")
    lines.append(f"Base+Full failures: {int(f.loc['TOTAL', 'n'])}")
    return lines
