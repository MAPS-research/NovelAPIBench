"""The main-body figures (Figures 3-6), drawn from the tables of ``rq1``, ``rq2`` and ``rq3``.

    rq1_components          Figure 3: outcome composition and pass@1 per knowledge condition
    rq1_novelty_backbones   Figure 4: (a) S / E / S+E on new vs modified APIs, (b) six backbones
                            on their shared tasks
    rq2_oracle_vs_real      Figure 5: oracle vs retrieved knowledge, all tasks and shared-hit tasks
    rq3_bypass_fixes        Figure 6: (a) bypassing the target API, (b) Base+Full failures fixed

Each figure is written as PDF and PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from novelapibench.analysis.collect import LABEL_ORDER, METHODS, MODELS, UNLABELLED  # noqa: E402
from novelapibench.analysis.rq2 import SHARED_HIT_CONDITIONS  # noqa: E402
from novelapibench.analysis.rq3 import fix_groups  # noqa: E402

# Plain DejaVu Sans; canvases are wider than the text width and scaled down by LaTeX.
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12, "legend.fontsize": 9.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .8,
    "xtick.major.width": .8, "ytick.major.width": .8, "xtick.major.size": 3,
    "ytick.major.size": 3, "pdf.fonttype": 42, "ps.fonttype": 42,
}
# Desaturated seaborn "deep" hues.
PAL = {"blue": "#7A9CC6", "orange": "#E3A76F", "green": "#78AE7E", "red": "#CC8B86",
       "purple": "#A695C0", "tan": "#D8C89A", "gray": "#AFAFAF", "cyan": "#8FBFD4"}

TEX = {"none": "No knowledge", "S_name": r"$S_{\mathrm{name}}$",
       "S": r"$S$  ($S_{\mathrm{name}}{+}S_{\mathrm{param}}$)", "M": "$M$", "C": "$C$",
       "S+M": "$S{+}M$", "S+C": "$S{+}C$", "S+M+C": "$S{+}M{+}C$", "E": "$E$", "S+E": "$S{+}E$",
       "S+E+C": "$S{+}E{+}C$", "Full": "Full"}
SHORT = {"S": "$S$", "C": "$C$", "E": "$E$", "S+M": "$S{+}M$", "S+E": "$S{+}E$", "Full": "Full"}


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=.02)
        paths.append(p)
    plt.close(fig)
    return paths


def _fmt_n(k: int) -> str:
    return f"{int(k):,}".replace(",", "{,}")


# ---------------------------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------------------------

def rq1_components(cells: pd.DataFrame, failures: pd.DataFrame, out_dir: Path) -> list[Path]:
    groups = [["none", "S_name", "S"], ["M", "C", "S+M", "S+C", "S+M+C"], ["E", "S+E", "S+E+C", "Full"]]
    have = set(cells["condition"])
    groups = [[c for c in g if c in have] for g in groups]
    conds = [c for g in groups for c in g]
    seps = np.cumsum([len(g) for g in groups if g])[:-1] - .5
    bands = LABEL_ORDER + ([UNLABELLED] if (failures["label"] == UNLABELLED).any() else [])
    names = ["OK (pass@1)", *LABEL_ORDER[1:], UNLABELLED][:len(bands)]
    colors = [PAL["green"], PAL["red"], PAL["orange"], PAL["tan"], PAL["blue"], PAL["purple"],
              PAL["gray"], "#e6e6e6"][:len(bands)]
    b = failures.pivot(index="condition", columns="label", values="share").loc[conds]
    c = cells.set_index("condition").loc[conds]
    assert np.allclose(b[bands].sum(axis=1), 100) and np.allclose(b["Pass"], c["pass1"])

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 2.95))
        fig.subplots_adjust(left=.22, right=.90, bottom=.30, top=.90)
        y = np.arange(len(conds))
        left = np.zeros(len(conds))
        for band, color in zip(bands, colors):
            ax.barh(y, b[band], left=left, height=.74, color=color, edgecolor="white", linewidth=.5)
            left += b[band].to_numpy()
        ax.errorbar(c["pass1"], y, xerr=np.array([c["pass1"] - c["ci_lo"], c["ci_hi"] - c["pass1"]]),
                    fmt="none", ecolor="#2b2b2b", elinewidth=.9, capsize=2)
        for yy, val in zip(y, c["pass1"]):
            ax.text(102.5, yy, f"{val:.1f}", va="center", ha="left", fontsize=9.5, clip_on=False)
        ax.text(102.5, -1.05, "pass@1", fontsize=9.5, ha="left", va="center", clip_on=False)
        for sep in seps:
            ax.axhline(sep, color="#cfcfcf", lw=.7)
        ax.set(yticks=y, yticklabels=[TEX[k] for k in conds], xlim=(0, 100),
               ylim=(len(conds) - .4, -1.4), xticks=np.arange(0, 101, 20))
        ax.set_xlabel("Share of test samples (%)", fontsize=10, labelpad=2)
        ax.set_title("Failure composition under each knowledge condition", fontweight="bold",
                     fontsize=11, pad=6)
        ax.tick_params(axis="y", length=0, labelsize=9.5)
        ax.spines["left"].set_visible(False)
        handles = [Patch(facecolor=col, label=n) for col, n in zip(colors, names)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .165), ncol=4,
                   frameon=False, columnspacing=1.6, handlelength=1.5, handletextpad=.6,
                   labelspacing=.35, fontsize=9.5)
        return _save(fig, out_dir, "rq1_components")


# ---------------------------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------------------------

def rq1_novelty_backbones(novelty: pd.DataFrame, xb: pd.DataFrame, out_dir: Path) -> list[Path]:
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.1))
        fig.subplots_adjust(left=.075, right=.985, bottom=.30, top=.88, wspace=.26)

        n = novelty.set_index(["novelty", "condition"])
        for j, (cond, color) in enumerate(zip(["S", "E", "S+E"], [PAL["blue"], PAL["orange"], PAL["green"]])):
            g = n.loc[[(t, cond) for t in ["new", "modified"]]]
            x = np.arange(2) + (j - 1) * .26
            ax.bar(x, g["pass1"], .24, color=color, label=cond,
                   yerr=np.array([g["pass1"] - g["ci_lo"], g["ci_hi"] - g["pass1"]]),
                   error_kw=dict(elinewidth=.9, capsize=2, ecolor="#2b2b2b"))
            for xx, v in zip(x, g["pass1"]):
                ax.text(xx, v + 6.5, f"{v:.1f}", ha="center", fontsize=9.5)
        ax.set(xticks=[i + d for i in range(2) for d in [-.26, 0, .26]],
               xticklabels=["$S$", "$E$", "$S{+}E$"] * 2, xlim=(-.55, 1.55), ylim=(0, 100),
               ylabel="Pass@1 (%)", yticks=[0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="x", length=0, pad=4)
        counts = [int(n.loc[(t, "S"), "n_tasks"]) for t in ["new", "modified"]]
        for i, label in enumerate([f"New APIs\n($n={_fmt_n(counts[0])}$)",
                                   f"Modified APIs\n($n={_fmt_n(counts[1])}$)"]):
            ax.text(i, -.15, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10)
        ax.set_title("(a) Component effects by API type", fontweight="bold", fontsize=11, pad=8)
        for i, t in enumerate(["new", "modified"]):
            delta = n.loc[(t, "S+E"), "pass1"] - n.loc[(t, "E"), "pass1"]
            ax.text(i, 93, f"adding $S$: {delta:+.1f} pp", ha="center", fontsize=9.5, color="#333333")

        colors = [PAL["blue"], PAL["red"], PAL["green"], PAL["purple"], PAL["orange"], PAL["gray"]]
        markers = ["o", "s", "^", "D", "v", "P"]
        c = xb.set_index(["model", "condition"])
        assert xb["n_tasks"].nunique() == 1
        for m, col, mark in zip(MODELS, colors, markers):
            if m not in set(xb["model"]):
                continue
            g = c.loc[[(m, k) for k in ["S", "E", "S+E", "Full"]]]
            bx.errorbar(np.arange(4), g["pass1"], yerr=np.array([g["pass1"] - g["ci_lo"], g["ci_hi"] - g["pass1"]]),
                        color=col, marker=mark, markersize=5, linewidth=1.4, label=MODELS[m],
                        elinewidth=.8, capsize=1.8,
                        linestyle="--" if m in ("qwen2.5-coder-14b", "qwen2.5-coder-32b") else "-")
        bx.set(xticks=range(4), xticklabels=["$S$", "$E$", "$S{+}E$", "Full"], xlim=(-.25, 3.25),
               ylim=(0, 100), yticks=[0, 20, 40, 60, 80, 100], ylabel="Pass@1 (%)")
        bx.set_title("(b) Component effects across models", fontweight="bold", fontsize=11, pad=8)
        bx.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2, frameon=False,
                  columnspacing=1.2, handlelength=1.8, handletextpad=.5, labelspacing=.4, fontsize=9.5)
        for a in (ax, bx):
            a.set_axisbelow(True)
            a.grid(axis="y", alpha=.25, lw=.6, color="#b0b0b0")
        return _save(fig, out_dir, "rq1_novelty_backbones")


# ---------------------------------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------------------------------

def _dumbbell(ax, frame: pd.DataFrame, labels: list[str], title: str, subtitle: str | None = None) -> None:
    """Paired oracle / retrieved pass@1, one row per knowledge condition."""
    y = np.arange(len(frame))[::-1]
    colors = {"oracle": "#8C8C8C", "retrieval": "#5B84B1"}
    # Rows whose two points nearly coincide are split vertically instead of labelled left/right.
    split = (frame["oracle"] - frame["retrieval"]).abs() < 2.5
    dy = {"oracle": np.where(split, .17, 0.), "retrieval": np.where(split, -.17, 0.)}
    ax.hlines(y[~split], frame["oracle"][~split], frame["retrieval"][~split], color="#c4c4c4",
              lw=1.6, zorder=1)
    for mode, label in [("oracle", "Oracle knowledge"), ("retrieval", "Real retrieval (top-5)")]:
        ax.errorbar(frame[mode], y + dy[mode],
                    xerr=np.array([frame[mode] - frame[mode + "_lo"], frame[mode + "_hi"] - frame[mode]]),
                    fmt="o", markersize=5.5, color=colors[mode], label=label, zorder=3,
                    elinewidth=.9, capsize=1.8)
    for i, (yy, (_, r)) in enumerate(zip(y, frame.iterrows())):
        if split.iloc[i]:
            for mode, va, off in [("oracle", "bottom", .30), ("retrieval", "top", -.30)]:
                ax.text(r[mode], yy + off, f"{r[mode]:.1f}", ha="center", va=va, fontsize=8.5,
                        color=colors[mode])
        else:
            # Labels sit outside the pair, so they never stack on a neighbouring row.
            lo_mode = "retrieval" if r["retrieval"] <= r["oracle"] else "oracle"
            hi_mode = "oracle" if lo_mode == "retrieval" else "retrieval"
            pad = 1.9 + max(r[lo_mode] - r[lo_mode + "_lo"], 0)
            ax.text(r[lo_mode] - pad, yy, f"{r[lo_mode]:.1f}", ha="right", va="center", fontsize=8.5,
                    color=colors[lo_mode])
            pad = 1.9 + max(r[hi_mode + "_hi"] - r[hi_mode], 0)
            ax.text(r[hi_mode] + pad, yy, f"{r[hi_mode]:.1f}", ha="left", va="center", fontsize=8.5,
                    color=colors[hi_mode])
        ax.text(102, yy, f"${r['delta']:+.1f}$", ha="left", va="center", fontsize=9.5, clip_on=False)
    ax.text(102, len(frame) - .42, r"$\Delta$ (pp)", ha="left", va="center", fontsize=8.5, clip_on=False)
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 100), xticks=np.arange(0, 101, 20),
           ylim=(-.6, len(frame) - .4))
    ax.set_xlabel("Pass@1 (%)", fontsize=10, labelpad=2)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, fontweight="bold", fontsize=10.5, pad=21 if subtitle else 6)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color="#333333")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.25, lw=.6, color="#b0b0b0")


def rq2_oracle_vs_real(overall: pd.DataFrame, shared: pd.DataFrame | None, out_dir: Path) -> list[Path]:
    o = overall.set_index("condition")
    conds = [c for c in ["S", "C", "E", "S+E", "Full"] if c in o.index]
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.5))
        fig.subplots_adjust(left=.085, right=.955, bottom=.32, top=.775, wspace=.42)
        _dumbbell(ax, o.loc[conds], [SHORT[c] for c in conds], "(a) Overall performance",
                  f"All tasks ($n={_fmt_n(o['n_tasks'].max())}$)")
        if shared is not None and not shared.empty:
            s = shared.set_index("condition").loc[SHARED_HIT_CONDITIONS]
            assert s["n_tasks"].nunique() == 1
            _dumbbell(bx, s, [SHORT[c] for c in SHARED_HIT_CONDITIONS],
                      "(b) Performance on shared-hit tasks",
                      f"Target hit in all four conditions ($n={_fmt_n(s['n_tasks'].iloc[0])}$)")
        else:
            bx.set_axis_off()
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .155), ncol=2,
                   frameon=False, fontsize=9.5, columnspacing=1.8, handlelength=1.3,
                   handletextpad=.5, numpoints=1)
        return _save(fig, out_dir, "rq2_oracle_vs_real")


# ---------------------------------------------------------------------------------------------
# Figure 6
# ---------------------------------------------------------------------------------------------

def rq3_bypass_fixes(byp: pd.DataFrame, fx: pd.DataFrame, out_dir: Path) -> list[Path]:
    b = byp[byp["condition"] == "Full"].set_index("method")
    methods = [m for m in METHODS if m in b.index]
    b = b.loc[methods]
    n_tasks = int(b["n_tasks"].iloc[0])
    assert (b["n_tasks"] == n_tasks).all()
    fixes = fix_groups(fx).set_index("group")
    n_fail = int(fx.set_index("base_label").loc["TOTAL", "n"])
    assert fixes["n"].sum() == n_fail

    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1, 1.05]))
        fig.subplots_adjust(left=.075, right=.975, bottom=.30, top=.775, wspace=.40)

        y = np.arange(len(methods))[::-1]
        ax.barh(y, b["bypass_pct"], .6, color=PAL["orange"],
                xerr=np.array([b["bypass_pct"] - b["bypass_pct_lo"], b["bypass_pct_hi"] - b["bypass_pct"]]),
                error_kw=dict(elinewidth=.9, capsize=2, ecolor="#2b2b2b"))
        for yy, r in zip(y, b.itertuples()):
            ax.text(r.bypass_pct_hi + .7, yy, f"{r.bypass_pct:.1f}%", ha="left", va="center", fontsize=9)
        ax.set(yticks=y, yticklabels=[METHODS[m] for m in methods], xlim=(0, 21),
               xticks=np.arange(0, 16, 5), ylim=(-.6, len(methods) - .4))
        ax.spines["bottom"].set_bounds(0, 15)
        ax.set_xlabel("Share of tasks (%)", fontsize=10, labelpad=2)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title("(a) Bypassing the target API", fontweight="bold", fontsize=10.5, pad=21)
        ax.text(0, 1.015, f"Full bundle in the prompt ($n={n_tasks}$)", transform=ax.transAxes,
                fontsize=8.5, color="#333333")

        yb = np.arange(len(fixes))[::-1]
        for m, color, dy in [("sft", "#5B84B1", .17), ("raft", "#78AE7E", -.17)]:
            share = 100 * fixes[f"{m}_fixed"] / fixes["n"]
            bx.barh(yb + dy, share, .32, color=color, label=METHODS[m])
            for yy, v, k, n in zip(yb, share, fixes[f"{m}_fixed"], fixes["n"]):
                bx.text(v + 1.8, yy + dy, f"{int(k)}/{int(n)}", ha="left", va="center", fontsize=8.5,
                        color=color)
        bx.set(yticks=yb, yticklabels=list(fixes.index), xlim=(0, 108), xticks=np.arange(0, 101, 25),
               ylim=(-.6, len(fixes) - .4))
        bx.spines["bottom"].set_bounds(0, 100)
        bx.set_xlabel("Base failures corrected (%)", fontsize=10, labelpad=2)
        bx.tick_params(axis="y", length=0)
        bx.spines["left"].set_visible(False)
        bx.set_title("(b) Correcting the base model's failures", fontweight="bold", fontsize=10.5, pad=21)
        bx.text(0, 1.015, f"Base+bundle failures ($n={n_fail}$)", transform=bx.transAxes,
                fontsize=8.5, color="#333333")
        bx.legend(loc="upper center", bbox_to_anchor=(.5, -.24), ncol=2, frameon=False, fontsize=9.5,
                  handlelength=1.3, handletextpad=.5, columnspacing=1.8)
        for a in (ax, bx):
            a.set_axisbelow(True)
            a.grid(axis="x", alpha=.25, lw=.6, color="#b0b0b0")
        return _save(fig, out_dir, "rq3_bypass_fixes")
