#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4.3 supplement: the annual gain in bins of every variable other than NDVI heterogeneity (annual scale, site-years with >= 6 HLS days).

One figure per variable class, one panel per variable, NEE / GPP / RECO as coloured lines in the same panel (same unit, shared y axis in all panels
of all supplementary figures so that flat panels can be compared with the NDVI-heterogeneity panels of the main figure):
  Fig_S_bins_model_inputs        12 variables the networks were trained with (air/dewpoint temperature, VPD, shortwave radiation, soil temperature and moisture, 4 layers)
  Fig_S_bins_footprint_wind      5 footprint-calculation parameters + 2 wind variables
Points = mean gain of six equal-count bins, shaded band = 95% interval from resampling whole sites.  The Spearman rho (NEE, GPP, RECO) is printed in each panel.

Inputs : ../outputs/binned_response_all_factors_annual.csv, correlation_all_scales.csv  (written by gain_correlates.py)
Outputs: ../outputs/Fig_S_bins_*.{png,pdf}
Run    : python fig_4_3_supp_bins.py
"""
import importlib.util
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g", HERE / "gain_correlates.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
OUT = g.OUT
FLUXES = g.FLUXES
COLS = g.FLUX_COLORS
UNIT = r"Mean $\Delta$RMSE ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)"


def save(fig, path, **kw):
    for _ in range(8):
        try:
            return g.save_manuscript_png(fig, path, **kw)
        except OSError:
            time.sleep(2)
    raise OSError(path)


def classes():
    allf = [(c, l, grp) for c, l, grp, _ in g.FACTORS if c != "NDVI_STD_DAY"]
    return {
        "model_inputs": ("Model inputs", [x for x in allf if x[2] == "Model inputs"], 4, 3),
        "footprint_wind": ("Footprint calculation and wind", [x for x in allf if x[2] in ("Footprint calculation", "Wind")], 3, 3),
    }


def main():
    binned = pd.read_csv(OUT / "binned_response_all_factors_annual.csv")
    cor = pd.read_csv(OUT / "correlation_all_scales.csv")
    cor = cor[cor["scale"] == "year"]
    rho = cor.pivot(index="factor", columns="flux", values="rho")
    qv = cor.pivot(index="factor", columns="flux", values="q")
    others = binned[binned["factor"] != "NDVI_STD_DAY"]
    ylo, yhi = others["ci_low"].min(), others["ci_high"].max()
    pad = 0.06 * (yhi - ylo)
    ylim = (ylo - pad, yhi + pad)
    for key, (title, facs, nr, nc) in classes().items():
        fig, axes = plt.subplots(nr, nc, figsize=(7.2, 1.85 * nr + 0.3), sharey=True, squeeze=False,
                                 gridspec_kw={"wspace": 0.1, "hspace": 0.62})
        fig.subplots_adjust(left=0.085, right=0.99, top=0.975, bottom=0.085 if nr == 4 else 0.095)
        for k, ax in enumerate(axes.ravel()):
            if k >= len(facs):
                ax.axis("off")
                continue
            col, label, _ = facs[k]
            for fl in FLUXES:
                b = binned[(binned["factor"] == col) & (binned["flux"] == fl)].sort_values("bin")
                ax.fill_between(b["x_median"], b["ci_low"], b["ci_high"], color=COLS[fl], alpha=0.14, lw=0)
                ax.plot(b["x_median"], b["mean_delta"], "o-", color=COLS[fl], lw=1.0, ms=3.2, label=fl)
            ax.axhline(0, color="#333333", lw=0.6, ls="--")
            ax.set_ylim(*ylim)
            ax.set_title(label, fontsize=7, pad=2.5)
            r = rho.loc[col]
            sig = [("*" if qv.loc[col, f] < 0.05 else "") for f in FLUXES]
            ax.text(0.97, 0.04, r"$\rho$ = " + " / ".join(f"{r[f]:+.2f}{s_}" for f, s_ in zip(FLUXES, sig)), transform=ax.transAxes, fontsize=5.4, ha="right", va="bottom",
                    color="#333333", bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", lw=0.3, alpha=0.4)
            ax.tick_params(labelsize=6, length=2.5)
            if k % nc == 0:
                ax.set_ylabel("")
            ax.text(-0.02 if k % nc else -0.2, 1.2, "abcdefghijkl"[k], transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")
        fig.supylabel(UNIT, x=0.004, fontsize=7)
        h = [plt.Line2D([0], [0], color=COLS[f], marker="o", lw=1.0, ms=3.2) for f in FLUXES]
        fig.legend(h, FLUXES, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 0.0), columnspacing=2.0)
        save(fig, OUT / f"Fig_S_bins_{key}.png", width_in=7.2, render_dpi=600)
        fig.savefig(OUT / f"Fig_S_bins_{key}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(key, len(facs), "panels")


if __name__ == "__main__":
    main()
