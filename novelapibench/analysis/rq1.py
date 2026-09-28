"""RQ1: what knowledge enables effective API use? (Section 4.2)

Tables behind Figures 3 and 4:

* ``condition_table``     pass@1 per knowledge condition with API-cluster BCa intervals, and the
                          paired contrast against ``none`` (bootstrap interval, exact McNemar,
                          Holm over the conditions);
* ``failure_table``       share of tasks per outcome label (Pass and the six failure labels)
                          per condition, each with its own marginal interval (Figure 3);
* ``novelty_table``       pass@1 of S, E and S+E on newly introduced vs signature-modified APIs
                          (Figure 4a; the novelty type is the bundle's ``is_modified``);
* ``novelty_interaction`` per backbone, the gain from adding S to E on modified APIs minus the
                          same gain on new APIs, with the stratified bootstrap interval and the
                          API-level permutation p, Holm over the six backbones;
* ``cross_backbone``      ``condition_table`` of each backbone on the tasks shared by all six
                          instances (Figure 4b).
"""

from __future__ import annotations

import pandas as pd

from novelapibench.analysis.collect import CONDITIONS, LABEL_ORDER, MODELS, UNLABELLED
from novelapibench.analysis.stats import holm, interaction_test, mcnemar_exact, mean_ci, paired_diff_ci

NOVELTY_CONDITIONS = ["S", "E", "S+E"]


def wide(frame: pd.DataFrame, value: str = "passed") -> pd.DataFrame:
    """One row per task (sorted by task id), one column per condition; tasks missing from any
    condition are dropped, so every column scores the same tasks."""
    w = frame.pivot_table(index="task_id", columns="condition", values=value, observed=True)
    return w.dropna(axis=0, how="any")


def api_of(frame: pd.DataFrame) -> pd.Series:
    return frame.drop_duplicates("task_id").set_index("task_id")["api_name"]


def condition_table(frame: pd.DataFrame, reference: str = "none") -> pd.DataFrame:
    """Pass@1 (%) per condition and the paired contrast of each condition against ``reference``."""
    w = wide(frame)
    clusters = api_of(frame).reindex(w.index).tolist()
    conds = [c for c in CONDITIONS if c in w.columns]
    has_ref = reference in w.columns
    adj = holm({c: mcnemar_exact(w[c], w[reference])[2] for c in conds if c != reference}) if has_ref else {}
    rows = []
    for c in conds:
        ci = mean_ci(w[c], clusters)
        row = {"condition": c, "n_tasks": ci.n, "n_apis": ci.n_clusters,
               "pass1": 100 * ci.point, "ci_lo": 100 * ci.lo, "ci_hi": 100 * ci.hi}
        if has_ref and c != reference:
            d = paired_diff_ci(w[c], w[reference], clusters)
            n01, n10, p = mcnemar_exact(w[c], w[reference])
            row.update({f"delta_vs_{reference}": 100 * d.point, "delta_lo": 100 * d.lo,
                        "delta_hi": 100 * d.hi, "n_gained": n10, "n_lost": n01,
                        "mcnemar_p": p, "holm_p": adj[c][0], "significant": adj[c][1]})
        rows.append(row)
    return pd.DataFrame(rows)


def failure_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Share (%) of tasks with each outcome label per condition, with marginal intervals
    (they do not sum to 100)."""
    labels = LABEL_ORDER + ([UNLABELLED] if (frame["label"] == UNLABELLED).any() else [])
    rows = []
    for cond in [c for c in CONDITIONS if c in set(frame["condition"])]:
        g = frame[frame["condition"] == cond]
        for lab in labels:
            ci = mean_ci((g["label"] == lab).astype(float), g["api_name"].tolist())
            rows.append({"condition": cond, "label": lab, "share": 100 * ci.point,
                         "ci_lo": 100 * ci.lo, "ci_hi": 100 * ci.hi, "n": ci.n})
    return pd.DataFrame(rows)


def novelty_table(frame: pd.DataFrame, is_modified: dict[str, bool]) -> pd.DataFrame:
    """Pass@1 (%) of S, E and S+E on new vs signature-modified APIs (one backbone)."""
    f = frame[frame["condition"].isin(NOVELTY_CONDITIONS)]
    novelty = f["api_name"].map(lambda a: "modified" if is_modified[a] else "new")
    rows = []
    for nov in ("new", "modified"):
        for cond in NOVELTY_CONDITIONS:
            g = f[(novelty == nov) & (f["condition"] == cond)]
            ci = mean_ci(g["passed"], g["api_name"].tolist())
            rows.append({"novelty": nov, "condition": cond, "n_tasks": ci.n, "n_apis": ci.n_clusters,
                         "pass1": 100 * ci.point, "ci_lo": 100 * ci.lo, "ci_hi": 100 * ci.hi})
    return pd.DataFrame(rows)


def novelty_interaction(rq12: pd.DataFrame, is_modified: dict[str, bool]) -> pd.DataFrame:
    """(S+E - E) on modified APIs minus (S+E - E) on new APIs, per backbone (own instance)."""
    rows = []
    for model in MODELS:
        f = rq12[(rq12["model"] == model) & (rq12["setting"] == "oracle")]
        if f.empty:
            continue
        w = wide(f)
        if not {"S+E", "E"} <= set(w.columns):
            continue
        api = api_of(f).reindex(w.index)
        mod = api.map(is_modified).astype(bool)
        if mod.all() or not mod.any():
            continue
        d = w["S+E"] - w["E"]
        r = interaction_test(d[mod], api[mod], d[~mod], api[~mod])
        rows.append({"model": model, "n_new": int((~mod).sum()), "n_modified": int(mod.sum()),
                     "delta": 100 * r["delta"], "ci_lo": 100 * r["ci_lo"], "ci_hi": 100 * r["ci_hi"],
                     "perm_p": r["perm_p"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        adj = holm(dict(zip(out.index, out["perm_p"])))
        out["holm_p"] = [adj[i][0] for i in out.index]
    return out


def cross_backbone(rq12: pd.DataFrame) -> pd.DataFrame:
    """``condition_table`` of each backbone on the tasks shared by every backbone's instance."""
    f = rq12[rq12["setting"] == "oracle"]
    models = [m for m in MODELS if m in set(f["model"])]
    shared = set.intersection(*(set(f.loc[f["model"] == m, "task_id"]) for m in models))
    f = f[f["task_id"].isin(shared)]
    parts = []
    for m in models:
        t = condition_table(f[f["model"] == m])
        t.insert(0, "model", m)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def headline(cells: pd.DataFrame, failures: pd.DataFrame, novelty: pd.DataFrame | None,
             interaction: pd.DataFrame | None, xb: pd.DataFrame | None) -> list[str]:
    """The RQ1 numbers quoted in Section 4.2 (those whose inputs are available)."""
    p = cells.set_index("condition")["pass1"]
    share = failures.set_index(["condition", "label"])["share"]
    lines = [
        "pass@1: " + ", ".join(f"{c} {p[c]:.1f}" for c in ["E", "M", "C", "S", "S+E", "Full"] if c in p),
        f"WrongAPISelection: none {share['none', 'WrongAPISelection']:.1f} -> "
        f"S_name {share['S_name', 'WrongAPISelection']:.1f}"
        if {("none", "WrongAPISelection"), ("S_name", "WrongAPISelection")} <= set(share.index) else "",
        f"WrongParam: S_name {share['S_name', 'WrongParam']:.1f} -> S {share['S', 'WrongParam']:.1f}"
        if {("S_name", "WrongParam"), ("S", "WrongParam")} <= set(share.index) else "",
    ]
    if novelty is not None:
        nv = novelty.set_index(["novelty", "condition"])["pass1"]
        for nov in ("new", "modified"):
            if (nov, "E") in nv and (nov, "S+E") in nv:
                lines.append(f"{nov} APIs: E {nv[nov, 'E']:.1f} -> S+E {nv[nov, 'S+E']:.1f} "
                             f"(adding S: {nv[nov, 'S+E'] - nv[nov, 'E']:+.1f} pp)")
    if interaction is not None and not interaction.empty:
        lines.append(f"(S+E - E) modified minus new: Holm-adjusted p <= {interaction['holm_p'].max():.4f} "
                     f"over {len(interaction)} backbones")
    if xb is not None and not xb.empty:
        x = xb.set_index(["model", "condition"])["pass1"]
        models = list(dict.fromkeys(xb["model"]))
        if all((m, c) in x for m in models for c in ("S", "E", "S+E", "Full")):
            es = [x[m, "E"] - x[m, "S"] for m in models]
            fs = [x[m, "Full"] - x[m, "S+E"] for m in models]
            n = xb.drop_duplicates("model")
            lines.append(f"shared tasks: n={int(n['n_tasks'].iloc[0])} ({int(n['n_apis'].iloc[0])} APIs); "
                         f"E - S {min(es):.1f} to {max(es):.1f} pp; "
                         f"Full - (S+E) {min(fs):+.1f} to {max(fs):+.1f} pp")
    return [s for s in lines if s]
