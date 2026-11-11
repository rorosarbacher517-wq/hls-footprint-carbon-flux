#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4.3 main figure: the FAT gain and surface heterogeneity (descriptive).

a  Mean annual gain (delta RMSE = RMSE_uniform - RMSE_FAT, site-years) in the lowest, middle and highest third of the
   NDVI-heterogeneity distribution, per flux; 95% intervals by resampling whole sites; the high-minus-low difference is
   printed with its interval.
b  Within vegetation types (IGBP): Spearman correlation between NDVI heterogeneity and the annual gain, with 95% site-resampling
   intervals.  Only types with >= 60 site-years and >= 12 sites.  Removes the "forest vs cropland" background difference.

Inputs/factors as in gain_correlates.py (annual scale: site-years with >= 6 HLS days).
Outputs: ../outputs/Fig_heterogeneity_effect.{png,pdf}, heterogeneity_terciles_annual.csv, heterogeneity_within_igbp_annual.csv
Run    : python heterogeneity_effect.py
"""
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g", HERE / "gain_correlates.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

OUT = g.OUT
COL = "NDVI_STD_DAY"
N_BOOT = 2000
MIN_SITEYEARS, MIN_SITES = 60, 12
IGBP_ORDER = ["ENF", "DBF", "OSH", "GRA", "CRO", "WET"]
SHADES = {"NEE": ["#C9D8EC", "#8FAED6", "#4F81BD"], "GPP": ["#CDE5D6", "#8CC5A2", "#3C9A61"], "RECO": ["#F1D9BF", "#E0A96B", "#C87521"]}
rng = np.random.default_rng(11)


def terciles(table):
    rows = []
    for flux in g.FLUXES:
        x = table[table["flux"].eq(flux)].dropna(subset=["delta", COL])
        q1, q2 = x[COL].quantile([1 / 3, 2 / 3])
        grp = np.where(x[COL] <= q1, 0, np.where(x[COL] >= q2, 2, 1))
        dv, gv = x["delta"].to_numpy(), grp
        blocks = g.site_blocks(x["SITE_ID"].to_numpy())
        means = np.array([dv[gv == k].mean() for k in range(3)])
        boots = []
        for i in g.boot_indices(blocks, N_BOOT, rng):
            d_, g_ = dv[i], gv[i]
            boots.append([d_[g_ == k].mean() if (g_ == k).any() else np.nan for k in range(3)])
        boots = np.array(boots)
        diff = boots[:, 2] - boots[:, 0]
        for k, name in enumerate(["Low", "Middle", "High"]):
            lo, hi = np.nanpercentile(boots[:, k], [2.5, 97.5])
            rows.append({"flux": flux, "group": name, "mean_gain": means[k], "ci_low": lo, "ci_high": hi, "n_site_years": int((gv == k).sum()),
                         "n_sites": x.loc[gv == k, "SITE_ID"].nunique(), "ndvi_std_lower": [x[COL].min(), q1, q2][k], "ndvi_std_upper": [q1, q2, x[COL].max()][k],
                         "high_minus_low": means[2] - means[0], "diff_ci_low": np.nanpercentile(diff, 2.5), "diff_ci_high": np.nanpercentile(diff, 97.5)})
    return pd.DataFrame(rows)


def within_igbp(table):
    rows = []
    for flux in g.FLUXES:
        x = table[table["flux"].eq(flux)].dropna(subset=["delta", COL])
        for b in IGBP_ORDER:
            s = x[x["IGBP"] == b]
            if len(s) < MIN_SITEYEARS or s["SITE_ID"].nunique() < MIN_SITES:
                continue
            xv, yv = s[COL].to_numpy(), s["delta"].to_numpy()
            rho = spearmanr(xv, yv)[0]
            bs = [spearmanr(xv[i], yv[i])[0] for i in g.boot_indices(g.site_blocks(s["SITE_ID"].to_numpy()), 1000, rng)]
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            rows.append({"flux": flux, "IGBP": b, "rho": rho, "ci_low": lo, "ci_high": hi, "n_site_years": len(s), "n_sites": s["SITE_ID"].nunique()})
    return pd.DataFrame(rows)


def figure(ter, wi):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.62})
    # a: terciles
    width, xs = 0.26, np.arange(3)
    for k, name in enumerate(["Low", "Middle", "High"]):
        for i, flux in enumerate(g.FLUXES):
            r = ter[(ter["flux"] == flux) & (ter["group"] == name)].iloc[0]
            ax_a.bar(i + (k - 1) * width, r["mean_gain"], width=width * 0.94, color=SHADES[flux][k], edgecolor="#444444", linewidth=0.5,
                     label=name if i == 0 else None)
            ax_a.errorbar(i + (k - 1) * width, r["mean_gain"], yerr=[[r["mean_gain"] - r["ci_low"]], [r["ci_high"] - r["mean_gain"]]],
                          fmt="none", ecolor="#222222", elinewidth=0.9, capsize=2)
    ymax = ter["ci_high"].max()
    for i, flux in enumerate(g.FLUXES):
        r = ter[(ter["flux"] == flux) & (ter["group"] == "High")].iloc[0]
        ax_a.text(i, ymax * 1.08, f"high − low\n{r['high_minus_low']:+.2f}\n[{r['diff_ci_low']:+.2f}, {r['diff_ci_high']:+.2f}]",
                  ha="center", va="bottom", fontsize=6.5, linespacing=1.15)
    ax_a.axhline(0, color="#333333", lw=0.8)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(g.FLUXES)
    ax_a.set_ylim(min(0, ter["ci_low"].min()) * 1.1, ymax * 2.35)
    ax_a.set_ylabel(r"Mean annual gain, $\Delta$RMSE")
    ax_a.set_xlabel("")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#444444", lw=0.5) for c in ["#D9D9D9", "#A6A6A6", "#595959"]]
    ax_a.legend(handles, ["Low", "Middle", "High"], title="NDVI heterogeneity\n(thirds of site-years)", frameon=False, fontsize=7, title_fontsize=7, loc="upper left")
    ax_a.text(-0.12, 1.06, "a", transform=ax_a.transAxes, fontsize=12, fontweight="bold")
    # b: within IGBP
    types = [b for b in IGBP_ORDER if (wi["IGBP"] == b).any()]
    ypos = {b: len(types) - 1 - j for j, b in enumerate(types)}
    for i, flux in enumerate(g.FLUXES):
        for b in types:
            r = wi[(wi["flux"] == flux) & (wi["IGBP"] == b)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = ypos[b] + (1 - i) * 0.24
            ax_b.plot([r["ci_low"], r["ci_high"]], [y, y], color=SHADES[flux][2], lw=1.6)
            ax_b.plot(r["rho"], y, "o", color=SHADES[flux][2], ms=4.5, label=flux if b == types[0] else None)
    ax_b.axvline(0, color="#333333", lw=0.8, ls="--")
    nlab = {b: int(wi[wi["IGBP"] == b]["n_sites"].iloc[0]) for b in types}
    ax_b.set_yticks([ypos[b] for b in types])
    ax_b.set_yticklabels([f"{b} ({nlab[b]})" for b in types])
    ax_b.set_xlabel("Spearman ρ, heterogeneity vs gain\nwithin vegetation type (n sites)", fontsize=8.5)
    ax_b.legend(frameon=False, fontsize=7, loc="lower right")
    ax_b.text(-0.42, 1.06, "b", transform=ax_b.transAxes, fontsize=12, fontweight="bold")
    for ax in (ax_a, ax_b):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y" if ax is ax_a else "x", linewidth=0.3, alpha=0.4)
        ax.set_axisbelow(True)
    g.save_manuscript_png(fig, OUT / "Fig_heterogeneity_effect.png", width_in=7.2, render_dpi=600)
    fig.savefig(OUT / "Fig_heterogeneity_effect.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    d = g.daily_gain_table()
    table = g.aggregate(d, "year")
    ter = terciles(table)
    wi = within_igbp(table)
    ter.to_csv(OUT / "heterogeneity_terciles_annual.csv", index=False)
    wi.to_csv(OUT / "heterogeneity_within_igbp_annual.csv", index=False)
    figure(ter, wi)
    print(ter[["flux", "group", "mean_gain", "ci_low", "ci_high", "n_site_years", "n_sites"]].round(3).to_string(index=False))
    print(ter[ter["group"] == "High"][["flux", "high_minus_low", "diff_ci_low", "diff_ci_high"]].round(3).to_string(index=False))
    print(wi.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
