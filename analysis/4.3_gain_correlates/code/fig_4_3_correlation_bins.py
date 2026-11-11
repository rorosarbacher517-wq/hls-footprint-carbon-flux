#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4.3 figure (correlation + bins): which variables are associated with the FAT gain (annual scale, site-years with >= 6 HLS days).

Left   Spearman correlation between each of the 20 variables and the annual gain (delta RMSE = RMSE_uniform - RMSE_FAT) for NEE, GPP, RECO;
       values are printed in the cells, * marks Benjamini-Hochberg q < 0.05 (p values from resampling whole sites).
Right  Gain in six equal-count bins of NDVI heterogeneity (the pre-specified primary variable) for NEE, GPP and RECO, stacked vertically;
       points = mean of the bin, bars = 95% interval from resampling whole sites.

Inputs : ../outputs/correlation_all_scales.csv, ../outputs/binned_response_all_factors_annual.csv  (written by gain_correlates.py)
Outputs: ../outputs/Fig_correlation_bins.{png,pdf}
Run    : python fig_4_3_correlation_bins.py
"""
import importlib.util
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g", HERE / "gain_correlates.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
OUT = g.OUT
FLUXES = g.FLUXES
NDVI = "NDVI_STD_DAY"
VMAX = 0.25


def save(fig, path, **kw):
    for _ in range(8):
        try:
            return g.save_manuscript_png(fig, path, **kw)
        except OSError:
            time.sleep(2)
    raise OSError(path)


def main():
    cor = pd.read_csv(OUT / "correlation_all_scales.csv")
    cor = cor[cor["scale"] == "year"]
    binned = pd.read_csv(OUT / "binned_response_all_factors_annual.csv")
    order = [c for c, _, _, _ in g.FACTORS]
    rho = cor.pivot(index="factor", columns="flux", values="rho").reindex(order)[FLUXES]
    q = cor.pivot(index="factor", columns="flux", values="q").reindex(order)[FLUXES]
    n_sy, n_sites = int(cor["n"].iloc[0]), int(cor["n_sites"].iloc[0])

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.9], left=0.285, right=0.985, bottom=0.135, top=0.955, wspace=0.34, hspace=0.42)
    axh = fig.add_subplot(gs[:, 0])
    norm = TwoSlopeNorm(vmin=-VMAX, vcenter=0, vmax=VMAX)
    im = axh.imshow(rho.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            v = rho.iloc[i, j]
            star = "*" if q.iloc[i, j] < 0.05 else ""
            axh.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=6.2, color="white" if abs(v) > 0.17 else "#222222")
    labels = [g.LABEL[c] for c in order]
    axh.set_yticks(range(len(order)))
    axh.set_yticklabels(labels, fontsize=6.4)
    axh.tick_params(axis="y", pad=7)
    axh.set_xticks(range(3))
    axh.set_xticklabels(FLUXES, fontsize=7.5, fontweight="bold")
    axh.xaxis.tick_top()
    axh.tick_params(length=0)
    for s in axh.spines.values():
        s.set_visible(False)
    # block separators and block names
    groups = [g.GROUP[c] for c in order]
    start = 0
    for k in range(1, len(order) + 1):
        if k == len(order) or groups[k] != groups[start]:
            if k < len(order):
                axh.axhline(k - 0.5, color="white", lw=2.2)
            axh.add_patch(plt.Rectangle((-0.62, start - 0.42), 0.07, k - start - 0.16, color=g.GROUP_COLORS[groups[start]], clip_on=False, lw=0))
            start = k
    axh.text(-0.28, 1.045, "a", transform=axh.transAxes, fontsize=10, fontweight="bold", va="bottom")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in g.GROUP_COLORS.values()]
    fig.legend(handles, list(g.GROUP_COLORS.keys()), loc="lower left", ncol=4, frameon=False, fontsize=6.2, bbox_to_anchor=(0.285, 0.0), handlelength=1.0, columnspacing=1.0)
    cax = fig.add_axes([0.285, 0.098, 0.28, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.2, -0.1, 0, 0.1, 0.2])
    cb.set_label(r"Spearman $\rho$ with the annual gain ($\Delta$RMSE)", fontsize=6.6, labelpad=1.5)
    cb.ax.tick_params(labelsize=6, length=2)

    # right: binned response to NDVI heterogeneity, stacked
    cols = g.FLUX_COLORS
    for r, flux in enumerate(FLUXES):
        ax = fig.add_subplot(gs[r, 1])
        b = binned[(binned["flux"] == flux) & (binned["factor"] == NDVI)].sort_values("bin")
        ax.errorbar(b["x_median"], b["mean_delta"], yerr=[b["mean_delta"] - b["ci_low"], b["ci_high"] - b["mean_delta"]], fmt="o", color=cols[flux],
                    ecolor="#444444", elinewidth=0.9, capsize=2, ms=4.5)
        ax.plot(b["x_median"], b["mean_delta"], color=cols[flux], lw=1.0, alpha=0.8)
        ax.axhline(0, color="#333333", lw=0.7, ls="--")
        ax.set_ylabel(r"Mean $\Delta$RMSE" + "\n" + r"($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)", fontsize=6.6)
        ax.set_xlim(left=0)
        ax.grid(axis="y", lw=0.3, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.22 * (hi_ - lo_))                       # head room for the flux label
        rr, qq = rho.loc[NDVI, flux], q.loc[NDVI, flux]
        ax.text(0.03, 0.97, flux, transform=ax.transAxes, fontsize=9, va="top", ha="left", fontweight="bold", color=cols[flux])
        ax.text(-0.3, 1.0, "bcd"[r], transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
        if r < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("NDVI heterogeneity", fontsize=7)
    # sample size (n site-years, sites) is given in the caption
    save(fig, OUT / "Fig_correlation_bins.png", width_in=7.2, render_dpi=600)
    fig.savefig(OUT / "Fig_correlation_bins.pdf", bbox_inches="tight")
    plt.close(fig)
    print(rho.round(2).to_string())


if __name__ == "__main__":
    main()
