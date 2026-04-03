#!/usr/bin/env python3
"""
generate_v3_figures.py  (v3-rev2)
=================================
Generates 8 figures (fig_v3_1 … fig_v3_8) for the v3 comprehensive mapping
report.  Old Finding 6 (productive illusion) removed; 7->6, 8->7, 9->8.
All line plots use smooth cubic splines.
"""
from __future__ import annotations
import argparse, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import make_interp_spline

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})
COLOURS = {
    64: "#9467bd", 256: "#2ca02c", 1024: "#1f77b4",
    4096: "#d62728", 6000: "#ff7f0e", 8000: "#8c564b", 10000: "#e377c2",
}
Z_MARKERS = {0: "x", 1: "^", 5: "s", 10: "o", 20: "D", 50: "v", 100: "P"}

rff  = pd.read_csv(DATA / "all_rff_results.csv")
comb = pd.read_csv(DATA / "combined_longer_backtest_results.csv")
if "T" in comb.columns:
    comb24 = comb[comb["T"] == 24].copy()
else:
    comb24 = comb.copy()

all_data = pd.concat([
    rff.assign(source="rff"),
    comb24.rename(columns={"run": "run_label"}).assign(source="comb"),
], ignore_index=True)
all_data = all_data.sort_values("source", ascending=True)
all_data = all_data.drop_duplicates(subset=["k", "P", "z"], keep="first")
all_data = all_data.reset_index(drop=True)
print(f"Loaded {len(rff)} rff + {len(comb24)} comb -> {len(all_data)} unique")


def smooth(x, y, num=200, log_x=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    if len(x) < 2:
        return x, y
    xw = np.log10(x) if log_x else x
    k = min(3, len(x) - 1)
    try:
        spl = make_interp_spline(xw, y, k=k)
        xs = np.linspace(xw.min(), xw.max(), num)
        ys = spl(xs)
        return (10**xs if log_x else xs), ys
    except Exception:
        return x, y


def _plot_smooth(ax, x, y, color, marker, label, log_x=False, ms=6, marker_only=False):
    xs, ys = smooth(x, y, log_x=log_x)
    if not marker_only:
        ax.plot(xs, ys, alpha=0.55, linewidth=1.8, color=color)
    ax.plot(x, y, marker=marker, ms=ms, label=label,
            alpha=0.85, linestyle="none", color=color)


# ═══════════════════════════════════════════════════════════════
# FIG 1: Sharpe vs P — k=20 (left), k=24 (right)
# ═══════════════════════════════════════════════════════════════
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, kval in zip(axes, [20, 24]):
        sub = all_data[all_data["k"] == kval].copy()
        z_list = sorted(sub["z"].unique())
        cmap = plt.cm.tab10
        for i, z_val in enumerate(z_list):
            sz = sub[sub["z"] == z_val].sort_values("P")
            if len(sz) < 2:
                continue
            marker = Z_MARKERS.get(int(z_val), "o")
            _plot_smooth(ax, sz["P"].values, sz["avg_sharpe"].values,
                         color=cmap(i % 10), marker=marker,
                         label=f"z = {int(z_val)}", log_x=True)
        ax.axhline(y=0.12, color="grey", ls="--", lw=1, alpha=0.6,
                    label="Plain IPCA")
        ax.set_xlabel("RFF dimension $P$")
        ax.set_title(f"$k = {kval}$")
        ax.set_xscale("log")
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("Annualised Sharpe Ratio")
    fig.suptitle("Finding 1: Virtue of Complexity — Sharpe vs $P$",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_v3_1.pdf", bbox_inches="tight")
    print("  ok fig_v3_1.pdf")
    return fig


# ═══════════════════════════════════════════════════════════════
# FIG 2: R2 vs P — k=20 (left), k=24 (right)
# ═══════════════════════════════════════════════════════════════
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, kval in zip(axes, [20, 24]):
        sub = all_data[all_data["k"] == kval].copy()
        z_list = sorted(sub["z"].unique())
        cmap = plt.cm.tab10
        for i, z_val in enumerate(z_list):
            sz = sub[sub["z"] == z_val].sort_values("P")
            if len(sz) < 2:
                continue
            marker = Z_MARKERS.get(int(z_val), "o")
            _plot_smooth(ax, sz["P"].values, sz["avg_r2_oos"].values,
                         color=cmap(i % 10), marker=marker,
                         label=f"z = {int(z_val)}", log_x=True)
        ax.axhline(y=-0.03, color="grey", ls="--", lw=1, alpha=0.6,
                    label="Plain IPCA")
        ax.set_xlabel("RFF dimension $P$")
        ax.set_title(f"$k = {kval}$")
        ax.set_xscale("log")
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("OOS $R^2$")
    fig.suptitle("Finding 2: OOS $R^2$ Improves with $P$ (Benign Overfitting)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_v3_2.pdf", bbox_inches="tight")
    print("  ok fig_v3_2.pdf")
    return fig


# ═══════════════════════════════════════════════════════════════
# FIG 3: Spectral gap — highlight k=20-24 collapse boundary
# ═══════════════════════════════════════════════════════════════
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    p_val = 1024
    sub = all_data[all_data["P"] == p_val].copy()

    ax = axes[0]
    cmap = plt.cm.tab10
    z_list = sorted(sub["z"].unique())
    for i, z_val in enumerate(z_list):
        sz = sub[sub["z"] == z_val].sort_values("k")
        if len(sz) < 2:
            continue
        marker = Z_MARKERS.get(int(z_val), "o")
        _plot_smooth(ax, sz["k"].values, sz["avg_spectral_gap"].values,
                     color=cmap(i % 10), marker=marker,
                     label=f"z = {int(z_val)}")
    ax.set_xlabel("Number of factors $k$")
    ax.set_ylabel("Spectral gap")
    ax.set_title(f"Spectral gap vs $k$ at $P = {p_val}$")
    ax.legend(fontsize=7, ncol=2)
    ax.axvspan(20, 24, alpha=0.10, color="green")
    ax.annotate("Peak Sharpe zone\n(sg collapses to 0)", xy=(22, 0.08),
                fontsize=8, color="green", ha="center", alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green",
                          alpha=0.7))

    ax = axes[1]
    for kval in [4, 8, 12, 20, 24, 32]:
        sk = sub[sub["k"] == kval].sort_values("z")
        if len(sk) < 2:
            continue
        _plot_smooth(ax, sk["z"].values, sk["avg_spectral_gap"].values,
                     color=None, marker="o", label=f"k = {kval}")
    ax.set_xlabel("Ridge parameter $z$")
    ax.set_ylabel("Spectral gap")
    ax.set_title(f"Spectral gap vs $z$ at $P = {p_val}$")
    ax.legend(fontsize=8, ncol=2)

    fig.suptitle("Finding 3: Spectral Gap as Regime Separator — "
                 "Optimal $k$ at Collapse Boundary ($k = 20$–$24$)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_v3_3.pdf", bbox_inches="tight")
    print("  ok fig_v3_3.pdf")
    return fig


# ═══════════════════════════════════════════════════════════════
# FIG 4: rho_theta scatter only — correlation with Sharpe
# ═══════════════════════════════════════════════════════════════
def fig4():
    fig, ax = plt.subplots(figsize=(8, 6))
    df = all_data.copy()
    df["pa_ratio"] = np.where(
        df["avg_max_principal_angle"] > 1e-8,
        df["avg_mean_principal_angle"] / df["avg_max_principal_angle"],
        np.nan)
    valid = df.dropna(subset=["pa_ratio", "avg_sharpe"]).copy()

    sc = ax.scatter(valid["pa_ratio"], valid["avg_sharpe"],
                    c=np.log10(valid["P"]), cmap="viridis",
                    s=valid["z"].clip(1, 100) * 1.5 + 15,
                    alpha=0.65, edgecolors="k", linewidth=0.3)
    plt.colorbar(sc, ax=ax, label=r"$\log_{10}(P)$")
    ax.set_xlabel(r"$\rho_\theta = \bar{\theta}\,/\,\theta_{\max}$")
    ax.set_ylabel("Annualised Sharpe")

    mask = valid["pa_ratio"].notna() & np.isfinite(valid["pa_ratio"])
    if mask.sum() > 5:
        from scipy.stats import spearmanr
        rp = np.corrcoef(valid.loc[mask, "pa_ratio"],
                         valid.loc[mask, "avg_sharpe"])[0, 1]
        rs, _ = spearmanr(valid.loc[mask, "pa_ratio"],
                          valid.loc[mask, "avg_sharpe"])
        ax.annotate(f"Pearson $r = {rp:.2f}$\nSpearman $\\rho_s = {rs:.2f}$",
                    xy=(0.58, 0.92), xycoords="axes fraction", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                              alpha=0.8))

    ax.annotate("Signal-concentrated\n(best Sharpe)", xy=(0.06, 0.38),
                fontsize=9, color="green", alpha=0.8)
    ax.annotate("Diffuse rotation", xy=(0.40, 0.05),
                fontsize=9, color="orange", alpha=0.8)

    ax.set_title(r"Finding 4: Sharpe vs $\rho_\theta$ — "
                 "Low Concentration Ratio Predicts Performance", fontsize=12)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_v3_4.pdf", bbox_inches="tight")
    print("  ok fig_v3_4.pdf")
    return fig


# ═══════════════════════════════════════════════════════════════
# FIG 5: Erank — include k=20, highlight 20-24 sweet spot
# ═══════════════════════════════════════════════════════════════
def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    p_val = 1024
    sub = all_data[all_data["P"] == p_val].copy()
    sub["erank_over_k"] = sub["avg_erank"] / sub["k"]

    ax = axes[0]
    for kval in sorted(sub["k"].unique()):
        sk = sub[sub["k"] == kval].sort_values("z")
        if len(sk) < 2:
            continue
        _plot_smooth(ax, sk["z"].values, sk["erank_over_k"].values,
                     color=None, marker="o", label=f"k = {int(kval)}")
    ax.set_xlabel("Ridge parameter $z$")
    ax.set_ylabel("erank / $k$")
    ax.set_title(f"erank/$k$ vs $z$ at $P = {p_val}$")
    ax.axhline(y=0.8, color="green", ls=":", lw=1, alpha=0.5,
               label="Healthy threshold")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    pivot = sub.pivot_table(index="k", columns="z",
                            values="avg_erank_collapse", aggfunc="mean")
    pivot = pivot.sort_index(ascending=True)
    if not pivot.empty:
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                       vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([int(c) for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([int(i) for i in pivot.index], fontsize=9)
        ax.set_xlabel("Ridge parameter $z$")
        ax.set_ylabel("Number of factors $k$")
        ax.set_title(f"Erank collapse at $P = {p_val}$")
        plt.colorbar(im, ax=ax, label="Frac. windows erank < $k/2$")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    c = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=8, color=c)

    fig.suptitle("Finding 5: Erank as Regime Separator — "
                 "Behaviour Changes Past $k = 20$–$24$",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_v3_5.pdf", bbox_inches="tight")
    print("  ok fig_v3_5.pdf")
    return fig


# ═══════════════════════════════════════════════════════════════
# FIG 6 (was 7): RFF premium — single panel z=10, no P=4096
# ═══════════════════════════════════════════════════════════════
def fig6():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    z_val = 10
    sub = all_data[all_data["z"] == z_val].copy()
    skip = {4096}
    for p_val in sorted(sub["P"].unique()):
        if int(p_val) in skip:
            continue
        sp = sub[sub["P"] == p_val].sort_values("k")
        if len(sp) < 2:
            continue
        color = COLOURS.get(int(p_val), "grey")
        _plot_smooth(ax, sp["k"].values, sp["avg_sharpe"].values,
                     color=color, marker="o", label=f"P = {int(p_val)}")
    ax.axhline(y=0.22, color="black", ls="--", lw=1.5, alpha=0.7,
               label="Plain IPCA peak")
    ax.axvspan(20, 24, alpha=0.08, color="green")
    ax.annotate("Peak Sharpe\nzone", xy=(22, 0.05), fontsize=8,
                color="green", ha="center")
    ax.set_xlabel("Number of factors $k$")
    ax.set_ylabel("Annualised Sharpe Ratio")
    ax.set_title(f"Finding 6: RFF Complexity Premium ($z = {z_val}$)",
                 fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xticks([4, 8, 12, 16, 20, 24, 28, 32])
    fig.tight_layout()
    fig.savefig(DOCS / "fig_v3_6.pdf", bbox_inches="tight")
    print("  ok fig_v3_6.pdf")
    return fig


# ═══════════════════════════════════════════════════════════════
# FIG 7 (was 8): z=0 vs z>=10, remove P=256,10000
# ═══════════════════════════════════════════════════════════════
def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    skip = {256, 10000}

    ax = axes[0]
    sub = all_data[all_data["z"] == 0].copy()
    for p_val in sorted(sub["P"].unique()):
        if int(p_val) in skip:
            continue
        sp = sub[sub["P"] == p_val].sort_values("k")
        if len(sp) < 2:
            continue
        color = COLOURS.get(int(p_val), "grey")
        xs, ys = smooth(sp["k"].values, sp["avg_sharpe"].values)
        ax.plot(xs, ys, alpha=0.4, linewidth=1.5, color=color, ls="--")
        ax.plot(sp["k"], sp["avg_sharpe"], marker="x", ms=7, color=color,
                label=f"P = {int(p_val)}", alpha=0.85, linestyle="none")
    ax.set_xlabel("Number of factors $k$")
    ax.set_ylabel("Annualised Sharpe Ratio")
    ax.set_title("$z = 0$ (no ridge) — erratic")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xticks([4, 8, 12, 16, 20, 24, 28, 32])
    ax.axhline(y=0, color="grey", ls="-", lw=0.5, alpha=0.3)

    ax = axes[1]
    sub = all_data[all_data["z"] >= 10].copy()
    for p_val in sorted(sub["P"].unique()):
        if int(p_val) in skip:
            continue
        sp_all = sub[sub["P"] == p_val]
        avg_k = sp_all.groupby("k")["avg_sharpe"].mean().reset_index()
        avg_k = avg_k.sort_values("k")
        if len(avg_k) < 2:
            continue
        color = COLOURS.get(int(p_val), "grey")
        _plot_smooth(ax, avg_k["k"].values, avg_k["avg_sharpe"].values,
                     color=color, marker="o", label=f"P = {int(p_val)}")
    ax.axvspan(20, 24, alpha=0.08, color="green")
    ax.set_xlabel("Number of factors $k$")
    ax.set_title("$z \\geq 10$ — clean peaks at $k = 20$–$24$")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xticks([4, 8, 12, 16, 20, 24, 28, 32])
    ax.axhline(y=0, color="grey", ls="-", lw=0.5, alpha=0.3)

    fig.suptitle("Finding 7: $z = 0$ Never Finds the Optimal $k$ Range",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_v3_7.pdf", bbox_inches="tight")
    print("  ok fig_v3_7.pdf")
    return fig


# ═══════════════════════════════════════════════════════════════
# FIG 8 (was 9): Sharpe vs d_proj — Goldilocks zone
# ═══════════════════════════════════════════════════════════════
def fig8():
    fig, ax = plt.subplots(figsize=(8, 6))
    df = all_data.dropna(subset=["avg_subspace_stability", "avg_sharpe"]).copy()
    norm = Normalize(vmin=np.log10(64), vmax=np.log10(10000))
    sc = ax.scatter(
        df["avg_subspace_stability"], df["avg_sharpe"],
        c=np.log10(df["P"]), cmap="viridis", norm=norm,
        s=df["z"].clip(1, 100) * 1.5 + 20,
        alpha=0.6, edgecolors="k", linewidth=0.3)
    plt.colorbar(sc, ax=ax, label="$\\log_{10}(P)$")

    ax.axvspan(0.3, 1.0, alpha=0.08, color="green", label="Goldilocks zone")
    ax.axvline(x=0.3, color="green", ls=":", lw=1, alpha=0.4)
    ax.axvline(x=1.0, color="green", ls=":", lw=1, alpha=0.4)
    ax.set_xlabel("$d_{\\mathrm{proj}}$ (subspace stability)")
    ax.set_ylabel("Annualised Sharpe Ratio")
    ax.set_title("Finding 8: Sharpe vs $d_{\\mathrm{proj}}$ — Goldilocks Zone")
    ax.legend(fontsize=9, loc="upper left")
    ax.annotate("Frozen", xy=(0.08, -0.05), fontsize=8, color="red", alpha=0.7)
    ax.annotate("Wandering", xy=(1.6, -0.05), fontsize=8, color="red", alpha=0.7)

    fig.tight_layout()
    fig.savefig(DOCS / "fig_v3_8.pdf", bbox_inches="tight")
    print("  ok fig_v3_8.pdf")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    print("Generating v3-rev2 figures ...")
    figs = [fig1(), fig2(), fig3(), fig4(), fig5(), fig6(), fig7(), fig8()]
    print(f"\nDone - {len(figs)} figures saved to {DOCS}/")
    if args.show:
        plt.show()
    else:
        plt.close("all")

if __name__ == "__main__":
    main()
