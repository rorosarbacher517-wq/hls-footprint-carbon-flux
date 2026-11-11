#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4.3 main figure (Fig. 6): what is, and what is not, related to the FAT gain.  Annual scale (site-years, >= 6 HLS days).

a  Spearman correlation of every factor with (i) the normalised RMSE of FAT, (ii) the normalised RMSE of the uniform-weighted
   Transformer and (iii) the relative gain (share of the uniform model's error removed by FAT), per flux.
   normalised RMSE = RMSE / SD of the observations in the site-year;  relative gain = (RMSE_uniform - RMSE_FAT) / RMSE_uniform.
   * = FDR q < 0.05 (site-resampling p-values, Benjamini-Hochberg over the 20 factors within each column).
b  Positive control: correlation with the error of FAT (x) against correlation with the error of the uniform model (y), 20 factors x 3 fluxes.
   Points on the 1:1 line = the factors change the error of both models alike, so they cancel in the gain.
c  Mean annual gain (delta RMSE, flux units) in the lowest / middle / highest third of NDVI heterogeneity (site-resampling 95% intervals).

Supplement: the same analysis at all five time scales (Fig_S_*), absolute-RMSE version, precision of the null result.
Outputs: ../outputs/Fig6_main.{png,pdf}, fig6_correlations_annual.csv, fig6_correlations_all_scales_rho.csv,
         fig6_null_precision.csv, Fig_S_scatter_all_scales.*, Fig_S_heatmap_<scale>.*
Run    : python fig_6_main.py
"""
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g = load("gain_correlates")
OUT = g.OUT
OUTCOMES = [("nrmse_d", "FAT\nerror"), ("nrmse_s", "Uni.\nerror"), ("rel", "Rel.\ngain")]
VMAX = 0.6


# ------------------------------------------------------------------ computation
def annual_correlations(table):
    parts = []
    for oc, _ in OUTCOMES + [("rmse_d", ""), ("rmse_s", ""), ("delta", "")]:
        c = g.correlations(table, "year", y_col=oc)
        c["q"] = np.nan
        for _, idx in c.groupby(["flux"]).groups.items():
            c.loc[idx, "q"] = g.bh_q(c.loc[idx, "p"].to_numpy())
        c["sig"] = c["q"] < 0.05
        parts.append(c)
    return pd.concat(parts, ignore_index=True)


def rho_all_scales(d):
    """Point estimates only (no bootstrap) for every scale, outcome and flux."""
    rows = []
    for scale in g.SCALES:
        t = g.aggregate(d, scale)
        for oc, _ in OUTCOMES:
            for flux in g.FLUXES:
                x = t[t["flux"] == flux].dropna(subset=[oc])
                rho = g.spearman_matrix(x[g.FCOLS].to_numpy(float), x[oc].to_numpy(float))
                for j, c in enumerate(g.FCOLS):
                    rows.append({"scale": scale, "outcome": oc, "flux": flux, "factor": g.LABEL[c], "group": g.GROUP[c], "rho": rho[j], "n": len(x)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ drawing
def heat(ax, cor, with_stars=True):
    cols = [(fl, oc) for fl in g.FLUXES for oc, _ in OUTCOMES]
    mat = np.full((len(g.FCOLS), len(cols)), np.nan)
    star = np.zeros_like(mat, dtype=bool)
    for jc, (fl, oc) in enumerate(cols):
        sub = cor[(cor["outcome"] == oc) & (cor["flux"] == fl)].set_index("factor")
        for i, c in enumerate(g.FCOLS):
            mat[i, jc] = sub.loc[c, "rho"]
            star[i, jc] = bool(sub.loc[c, "sig"]) if with_stars else False
    im = ax.imshow(mat, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-VMAX, vcenter=0, vmax=VMAX), aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if star[i, j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=9, fontweight="bold", color="white" if abs(mat[i, j]) > 0.35 else "#222222")
    ax.set_yticks(range(len(g.FCOLS)))
    ax.set_yticklabels([g.LABEL[c] for c in g.FCOLS], fontsize=7.5)
    for lab, c in zip(ax.get_yticklabels(), g.FCOLS):
        lab.set_color(g.GROUP_COLORS[g.GROUP[c]])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([lab for _ in g.FLUXES for _, lab in OUTCOMES], fontsize=6.5)
    ax.xaxis.tick_top()
    for k, fl in enumerate(g.FLUXES):
        ax.text(3 * k + 1, -1.9, fl, ha="center", va="bottom", fontsize=9, fontweight="bold", color=g.FLUX_COLORS[fl])
        if k:
            ax.axvline(3 * k - 0.5, color="white", lw=3)
    prev = None
    for i, c in enumerate(g.FCOLS):
        if prev is not None and g.GROUP[c] != prev:
            ax.axhline(i - 0.5, color="white", lw=2.5)
        prev = g.GROUP[c]
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def scatter(ax, rho, annotate=True, size=14):
    """rho: long table with outcomes nrmse_d / nrmse_s (columns factor, group, flux, rho)."""
    a = rho[rho["outcome"] == "nrmse_d"].set_index(["flux", "factor"])["rho"]
    b = rho[rho["outcome"] == "nrmse_s"].set_index(["flux", "factor"])["rho"].reindex(a.index)
    grp = rho[rho["outcome"] == "nrmse_d"].set_index(["flux", "factor"])["group"].reindex(a.index)
    for gname, col in g.GROUP_COLORS.items():
        m = grp == gname
        ax.scatter(a[m], b[m], s=size, color=col, alpha=0.85, edgecolor="none", label=gname)
    lim = max(np.nanmax(np.abs(a)), np.nanmax(np.abs(b))) * 1.08
    ax.plot([-lim, lim], [-lim, lim], color="#444444", lw=0.8, ls="--")
    ax.axhline(0, color="#999999", lw=0.5)
    ax.axvline(0, color="#999999", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    r = np.corrcoef(a.values, b.values)[0, 1]
    if annotate:
        ax.text(0.04, 0.95, f"r = {r:.3f}\nmean |Δρ| = {np.nanmean(np.abs(a.values - b.values)):.3f}", transform=ax.transAxes, va="top", fontsize=7)
    ax.set_xlabel("ρ with FAT error", fontsize=8)
    ax.set_ylabel("ρ with Uniform error", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return r


def bars(ax, ter):
    shades = {"NEE": ["#C9D8EC", "#8FAED6", "#4F81BD"], "GPP": ["#CDE5D6", "#8CC5A2", "#3C9A61"], "RECO": ["#F1D9BF", "#E0A96B", "#C87521"]}
    width = 0.26
    for k, name in enumerate(["Low", "Middle", "High"]):
        for i, flux in enumerate(g.FLUXES):
            r = ter[(ter["flux"] == flux) & (ter["group"] == name)].iloc[0]
            ax.bar(i + (k - 1) * width, r["mean_gain"], width=width * 0.94, color=shades[flux][k], edgecolor="#444444", linewidth=0.5)
            ax.errorbar(i + (k - 1) * width, r["mean_gain"], yerr=[[r["mean_gain"] - r["ci_low"]], [r["ci_high"] - r["mean_gain"]]],
                        fmt="none", ecolor="#222222", elinewidth=0.8, capsize=1.8)
    ymax = ter["ci_high"].max()
    for i, flux in enumerate(g.FLUXES):
        r = ter[(ter["flux"] == flux) & (ter["group"] == "High")].iloc[0]
        ax.text(i, ymax * 1.06, f"{r['high_minus_low']:+.2f}\n[{r['diff_ci_low']:+.2f}, {r['diff_ci_high']:+.2f}]", ha="center", va="bottom", fontsize=6.3, linespacing=1.1)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(g.FLUXES, fontsize=8)
    ax.set_ylim(min(0, ter["ci_low"].min()) * 1.1, ymax * 1.75)
    ax.set_ylabel(r"Mean annual gain, $\Delta$RMSE", fontsize=8)
    ax.tick_params(labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#444444", lw=0.5) for c in ["#D9D9D9", "#A6A6A6", "#595959"]]
    ax.legend(handles, ["Low", "Middle", "High"], title="NDVI heterogeneity (thirds)", frameon=False, fontsize=6.5, title_fontsize=6.5, loc="upper left", ncol=3,
              columnspacing=0.8, handlelength=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main_figure(cor, ter):
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0], wspace=0.5, hspace=0.45)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    im = heat(ax_a, cor)
    cb = fig.colorbar(im, ax=ax_a, orientation="horizontal", fraction=0.035, pad=0.03)
    cb.set_label("Spearman ρ  (* FDR q < 0.05)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    r = scatter(ax_b, cor)
    ax_b.legend(frameon=False, fontsize=5.8, loc="lower right", handletextpad=0.2, borderpad=0.1)
    bars(ax_c, ter)
    for ax, lab, dx in [(ax_a, "a", -0.62), (ax_b, "b", -0.3), (ax_c, "c", -0.3)]:
        ax.text(dx, 1.04 if ax is not ax_a else 1.06, lab, transform=ax.transAxes, fontsize=12, fontweight="bold")
    g.save_manuscript_png(fig, OUT / "Fig6_main.png", width_in=7.2, render_dpi=600)
    fig.savefig(OUT / "Fig6_main.pdf", bbox_inches="tight")
    plt.close(fig)
    return r


def supplement(rho_all, cor):
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.9), sharex=False)
    for ax, s in zip(axes, g.SCALES):
        r = scatter(ax, rho_all[rho_all["scale"] == s], annotate=False, size=6)
        ax.set_title(f"{g.SCALE_LABEL[s]}  (r = {r:.3f})", fontsize=7)
        ax.set_xlabel("ρ with FAT error", fontsize=6.5)
        ax.set_ylabel("ρ with Uniform error" if s == "daily" else "", fontsize=6.5)
        ax.tick_params(labelsize=5.5)
    fig.subplots_adjust(wspace=0.45)
    g.save_manuscript_png(fig, OUT / "Fig_S_scatter_all_scales.png", width_in=7.2, render_dpi=500)
    fig.savefig(OUT / "Fig_S_scatter_all_scales.pdf", bbox_inches="tight")
    plt.close(fig)
    for s in g.SCALES:
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        sub = rho_all[rho_all["scale"] == s].copy()
        sub["factor_col"] = sub["factor"].map({g.LABEL[c]: c for c in g.FCOLS})
        cor_s = sub.rename(columns={"factor": "label"}).assign(factor=sub["factor_col"], sig=False)
        im = heat(ax, cor_s, with_stars=False)
        ax.set_title(f"{g.SCALE_LABEL[s]} scale (point estimates)", fontsize=8, pad=24)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03).set_label("Spearman ρ", fontsize=7)
        g.save_manuscript_png(fig, OUT / f"Fig_S_heatmap_{s}.png", width_in=4.6, render_dpi=400)
        fig.savefig(OUT / f"Fig_S_heatmap_{s}.pdf", bbox_inches="tight")
        plt.close(fig)


def main():
    import sys
    redraw = "--redraw" in sys.argv                      # re-use the saved tables (only the drawing is repeated)
    f_cor, f_ter, f_all = OUT / "fig6_correlations_annual.csv", OUT / "heterogeneity_terciles_annual.csv", OUT / "fig6_correlations_all_scales_rho.csv"
    if redraw and f_cor.exists() and f_ter.exists() and f_all.exists():
        cor, ter, rho_all = pd.read_csv(f_cor), pd.read_csv(f_ter), pd.read_csv(f_all)
    else:
        d = g.daily_gain_table()
        year = g.aggregate(d, "year")
        cor = annual_correlations(year)
        cor.to_csv(f_cor, index=False)
        ter = load("heterogeneity_effect").terciles(year)
        ter.to_csv(f_ter, index=False)
        rho_all = rho_all_scales(d)
        rho_all.to_csv(f_all, index=False)
    r = main_figure(cor[cor["outcome"].isin([o for o, _ in OUTCOMES])], ter)
    supplement(rho_all, cor)

    # precision of the null result for the relative gain (annual)
    gain = cor[cor["outcome"] == "rel"]
    prec = gain.assign(half_width=(gain["ci_high"] - gain["ci_low"]) / 2).groupby("flux").agg(
        max_abs_rho=("rho", lambda v: v.abs().max()), median_half_width=("half_width", "median"), max_half_width=("half_width", "max"),
        n_sig_fdr=("sig", "sum")).reset_index()
    prec.to_csv(OUT / "fig6_null_precision.csv", index=False)
    summary = rho_all.groupby(["scale", "outcome"])["rho"].agg(lambda v: v.abs().mean()).unstack().round(3)
    print("scatter r (FAT vs Uniform error correlations), annual: %.4f" % r)
    print("mean |rho| over 20 factors x 3 fluxes:\n", summary.to_string())
    print(prec.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
