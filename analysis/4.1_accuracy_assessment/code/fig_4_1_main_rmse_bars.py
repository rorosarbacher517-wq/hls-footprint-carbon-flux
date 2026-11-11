#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4.1 main figure (updated for the four-model comparison).

a  Bar charts of pooled RMSE for the four models (FAT, Transformer with uniform-weighted aggregation,
   Transformer with dynamic-footprint pre-aggregation, random forest with pre-aggregated input).
b  Density scatter of the FAT estimates only; the annotation shows R2, Bias and n (R2 replaces RMSE).

Style (fonts, sizes, colour map, layout, jitter, DPI) is identical to the previous version in
_previous_version/multi_model_flux_visualization.py.  Only the content of panel a (R2 -> RMSE, 3 -> 4 bars,
unit on the y axis) and the first annotation line of panel b (RMSE -> R2) changed.

Inputs : ../../data/four_model_predictions/<Model>/fold{0-4}/{true,pred}.npy   (pooled over the five folds)
Outputs: ../outputs/  nature_flux_validation_ocean_sunset.{png,pdf}, flux_validation_metrics_pooled.csv
Run    : python fig_4_1_main_rmse_bars.py
Metrics: Bias = estimated - observed.  Metrics in the figure are pooled over all held-out observations
         (as in the previous version); the fold-wise mean +/- SD table is made by table_S4_fourmodel.py.
"""
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_squared_error, r2_score

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 23,
    "axes.titlesize": 27,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.linewidth": 1.25,
    "xtick.major.width": 1.25,
    "ytick.major.width": 1.25,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "axes.spines.top": True,
    "axes.spines.right": True,
})

_CMAP_NODES = [
    (0.18, 0.33, 0.59), (0.35, 0.63, 0.79), (0.67, 0.85, 0.74), (0.99, 0.88, 0.56),
    (0.96, 0.53, 0.35), (0.84, 0.19, 0.15), (0.50, 0.00, 0.15),
]

VARIABLES = ("NEE", "GPP", "RECO")
UNIT_LABEL = r"$\mu$mol m$^{-2}$ s$^{-1}$"
DEFAULT_RANGES = {"NEE": (-15, 15), "GPP": (0, 25), "RECO": (0, 15)}
RMSE_YLIM = (1.0, 3.5)   # bar-panel y range, identical for NEE / GPP / RECO
INVALID_VALUE = -9999.0

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE.parents[1] / "common"))
from figure_io import save_manuscript_png   # PNG at print size: 183 mm, 600 dpi
DATA_ROOT = Path(os.environ.get("PAPER_DATA_ROOT", HERE.parent.parent / "data" / "four_model_predictions"))
DEFAULT_OUTDIR = HERE.parent / "outputs"

# key -> folder name, bar label, colour.  Bar order = order of this dict.
EXPERIMENTS = {
    "FAT": {"folder": "FAT", "label": "FAT", "short_label": "FAT", "color": "#D63026"},
    "UNIFORM": {"folder": "Uniform_Transformer", "label": "Transformer",
                "short_label": "Uniform-\nweighted\nTransformer", "color": "#F0A050"},
    "PREAGG": {"folder": "Preaggregation_Transformer", "label": "Pre-aggregation Transformer",
               "short_label": "Pre-agg.\nTransformer", "color": "#7A4FA3"},
    "RF": {"folder": "RF", "label": "RF", "short_label": "Pre-agg.\nRF", "color": "#4A84B0"},
}
DEFAULT_SHOW_KEYS = ["FAT", "UNIFORM", "PREAGG", "RF"]


def build_academic_cmap():
    cmap = LinearSegmentedColormap.from_list("ocean_sunset", _CMAP_NODES, N=256)
    cmap.set_under("white")
    cmap.set_bad("white")
    return cmap


def load_pooled(key: str):
    """Concatenate the five held-out folds (fold order 0-4) of one model."""
    base = DATA_ROOT / EXPERIMENTS[key]["folder"]
    ys, ps = [], []
    for k in range(5):
        ys.append(np.load(base / f"fold{k}" / "true.npy", allow_pickle=True))
        ps.append(np.load(base / f"fold{k}" / "pred.npy", allow_pickle=True))
    return np.concatenate(ys), np.concatenate(ps)


def flatten_valid_pair(y_true, y_pred, flux_idx: int):
    x = y_true[:, :, flux_idx].reshape(-1).astype(float)
    y = y_pred[:, :, flux_idx].reshape(-1).astype(float)
    mask = (x != INVALID_VALUE) & (y != INVALID_VALUE) & np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def compute_metrics(obs, est):
    if obs.size == 0:
        return {"R2": np.nan, "RMSE": np.nan, "Bias": np.nan, "N": 0}
    return {
        "R2": float(r2_score(obs, est)),
        "RMSE": float(np.sqrt(mean_squared_error(obs, est))),
        "Bias": float(np.mean(est - obs)),
        "N": int(obs.size),
    }


def compute_all_metrics(keys: List[str]) -> Tuple[pd.DataFrame, Dict]:
    rows, data = [], {}
    for key in keys:
        y_t, y_p = load_pooled(key)
        data[key] = (y_t, y_p)
        for j, var in enumerate(VARIABLES):
            obs, est = flatten_valid_pair(y_t, y_p, j)
            m = compute_metrics(obs, est)
            m.update({"Experiment": key, "Label": EXPERIMENTS[key]["label"],
                      "ShortLabel": EXPERIMENTS[key]["short_label"], "Variable": var})
            rows.append(m)
    return pd.DataFrame(rows), data


def plot_density_scatter(ax, obs, est, var_name, xy_range, model_label, panel_label=None, gridsize=180,
                         use_plot_jitter=False, jitter_sd=0.015, show_ylabel=True, show_colorbar_label=True):
    """Smoothed density scatter.  Metrics use the original obs/est; jitter / masking are visual only."""
    vmin, vmax = xy_range
    plot_mask = (np.isfinite(obs) & np.isfinite(est) & (obs >= vmin) & (obs <= vmax)
                 & (est >= vmin) & (est <= vmax))
    obs_plot, est_plot = obs[plot_mask], est[plot_mask]
    if use_plot_jitter and est_plot.size > 0:
        rng = np.random.default_rng(42)
        est_plot = est_plot + rng.normal(0.0, jitter_sd, size=est_plot.shape)

    counts, _, _ = np.histogram2d(obs_plot, est_plot, bins=gridsize, range=[[vmin, vmax], [vmin, vmax]])
    density = gaussian_filter(counts, sigma=0.45) if gaussian_filter is not None else counts
    positive = density[density > 0]
    if positive.size:
        low_cut = max(1.0, float(np.percentile(positive, 2)))
        high_cut = max(low_cut * 1.2, float(np.percentile(positive, 99.85)))
    else:
        low_cut, high_cut = 1.0, 10.0
    density_masked = np.ma.masked_where(density < low_cut, density)

    img = ax.imshow(density_masked.T, origin="lower", extent=[vmin, vmax, vmin, vmax], aspect="equal",
                    cmap=build_academic_cmap(), norm=LogNorm(vmin=low_cut, vmax=high_cut),
                    interpolation="nearest", rasterized=True)
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1.35, color="#333333", zorder=5)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(var_name, fontweight="bold", pad=6)
    ax.set_xlabel(f"Observed ({UNIT_LABEL})", labelpad=3)
    ax.set_ylabel(f"{model_label} estimated ({UNIT_LABEL})" if show_ylabel else "", labelpad=3)

    m = compute_metrics(obs, est)
    stats = f"$R^{{2}}$ = {m['R2']:.2f}\nBias = {m['Bias']:.2f}\n$n$ = {m['N']:,}"   # R2 replaces RMSE
    ax.text(0.04, 0.96, stats, transform=ax.transAxes, ha="left", va="top", fontsize=18, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="none", alpha=0.88))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(img, cax=cax)
    cb.set_label("Count" if show_colorbar_label else "", rotation=90, labelpad=6, fontsize=15)
    cb.ax.tick_params(labelsize=13.5)
    if panel_label:
        ax.text(-0.15, 1.08, panel_label, transform=ax.transAxes, fontsize=26, fontweight="bold",
                va="bottom", ha="right")


def plot_comparison_bars(ax, metrics_df, var_name, order, panel_label=None, show_ylabel=True):
    sub = metrics_df[metrics_df["Variable"] == var_name].copy()
    sub["Experiment"] = pd.Categorical(sub["Experiment"], categories=order, ordered=True)
    sub = sub.sort_values("Experiment")

    x = np.arange(len(sub))
    labels = list(sub["ShortLabel"])
    colors = [EXPERIMENTS[e]["color"] for e in sub["Experiment"]]
    bars = ax.bar(x, sub["RMSE"], width=0.65, color=colors, edgecolor="#333333", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=14.5)   # 16 -> 14.5: four bars per panel
    ax.set_title(var_name, fontweight="bold", pad=6)
    ax.set_ylabel(f"RMSE ({UNIT_LABEL})" if show_ylabel else "", labelpad=3)

    ax.set_ylim(RMSE_YLIM)                      # same range in all three panels
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.grid(True, linewidth=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    for rect, val in zip(bars, sub["RMSE"]):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.02 * (RMSE_YLIM[1] - RMSE_YLIM[0]), f"{val:.2f}",
                ha="center", va="bottom", fontsize=18)
    if panel_label:
        ax.text(-0.15, 1.08, panel_label, transform=ax.transAxes, fontsize=28, fontweight="bold",
                va="bottom", ha="right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--show-keys", default=",".join(DEFAULT_SHOW_KEYS))
    parser.add_argument("--scatter-key", default="FAT")
    parser.add_argument("--fig-prefix", default="nature_flux_validation_ocean_sunset")
    parser.add_argument("--gridsize", type=int, default=220)
    parser.add_argument("--use-plot-jitter", action="store_true", default=True)
    parser.add_argument("--jitter-sd", type=float, default=0.035)
    parser.add_argument("--png-dpi", type=int, default=900)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    show_keys = [k.strip() for k in args.show_keys.split(",") if k.strip()]
    for k in show_keys + [args.scatter_key]:
        if k not in EXPERIMENTS:
            raise ValueError(f"Unknown key {k}; available: {list(EXPERIMENTS)}")
    load_keys = list(dict.fromkeys(show_keys + [args.scatter_key]))

    print(f"[CONFIG] data={DATA_ROOT}\n         out={args.outdir}\n         bars={show_keys} scatter={args.scatter_key}")
    metrics_df, data = compute_all_metrics(load_keys)
    metrics_df.to_csv(os.path.join(args.outdir, "flux_validation_metrics_pooled.csv"), index=False)

    metrics_show = metrics_df[metrics_df["Experiment"].isin(show_keys)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(18.4, 11.2),
                             gridspec_kw={"hspace": 0.30, "wspace": 0.24, "height_ratios": [1.0, 1.25]})
    fig.subplots_adjust(left=0.048, right=0.992, top=0.95, bottom=0.062)

    for j, var in enumerate(VARIABLES):
        plot_comparison_bars(axes[0, j], metrics_show, var, show_keys,
                             panel_label=("a" if j == 0 else None), show_ylabel=(j == 0))

    y_t_s, y_p_s = data[args.scatter_key]
    for j, var in enumerate(VARIABLES):
        obs, est = flatten_valid_pair(y_t_s, y_p_s, j)
        plot_density_scatter(axes[1, j], obs, est, var, DEFAULT_RANGES[var], EXPERIMENTS[args.scatter_key]["label"],
                             panel_label=("b" if j == 0 else None), gridsize=args.gridsize,
                             use_plot_jitter=args.use_plot_jitter, jitter_sd=args.jitter_sd,
                             show_ylabel=(j == 0), show_colorbar_label=(j == len(VARIABLES) - 1))

    pdf = os.path.join(args.outdir, f"{args.fig_prefix}.pdf")
    png = os.path.join(args.outdir, f"{args.fig_prefix}.png")
    plt.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.015)
    save_manuscript_png(fig, png, render_dpi=args.png_dpi)
    plt.close(fig)
    print(f"[DONE] {png}\n       {pdf}")


if __name__ == "__main__":
    main()
