#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4.2, temporal part (Fig. 5): temporal emergence of the FAT gain over the uniform-weighted Transformer.

delta RMSE = RMSE(uniform-weighted Transformer) - RMSE(FAT); positive = lower RMSE for FAT.

Method (identical to the original three-step cluster pipeline, now in one script):
  1. pair FAT and uniform-weighted predictions half-hour by half-hour (same held-out records, five folds concatenated);
  2. phenological stage from day of year (Dormant: DoY <= 90 or >= 335; Green-up: 91-160; Peak growing: 161-250;
     Senescence: 251-334) and local hour bin = floor(Hour);
  3. site-first aggregation: delta RMSE is computed per site and group (>= 24 valid half-hours), then the MEDIAN over
     sites is reported; uncertainty = 95% bootstrap interval of the median over sites (500 resamples, seed 42 + row index).

Inputs : ../../data/four_model_predictions/{FAT,Uniform_Transformer}/fold{0-4}/{true,pred,site}.npy
Outputs: ../outputs/temporal/  summary_temporal_delta_rmse.csv, unit_level_temporal_delta_rmse.csv,
         temporal_stage_summary.csv, temporal_hourly_summary.csv, Fig5_temporal_emergence.{png,pdf}
Run    : python fig_4_2_temporal.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "common"))
from figure_io import save_manuscript_png  # noqa: E402

DATA = PROJECT / "data" / "four_model_predictions"
OUT = HERE.parent / "outputs" / "temporal"

VARS = ["NEE", "GPP", "RECO"]
FILL = {-9999, -9999.0, -999, -999.0}
MIN_GROUP_SAMPLES = 24       # per site and group (original: max(3, min(30, 24)))
N_BOOT, SEED = 500, 42
SEASONS = ["Dormant", "Green-up", "Peak growing", "Senescence"]
COLORS = {"NEE": "#4F81BD", "GPP": "#3C9A61", "RECO": "#C87521"}
GROUP_LABELS = {"Green_up": "Green-up", "Peak_growing": "Peak growing"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------- data
def load_model(name):
    base = DATA / name
    ys, ps, ss = [], [], []
    for k in range(5):
        ys.append(np.load(base / f"fold{k}" / "true.npy", allow_pickle=True))
        ps.append(np.load(base / f"fold{k}" / "pred.npy", allow_pickle=True))
        ss.append(np.load(base / f"fold{k}" / "site.npy", allow_pickle=True))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ss)


def num(a):
    return pd.to_numeric(pd.Series(np.asarray(a).reshape(-1)), errors="coerce").values


def build_table(y_true, y_dyn, y_stat, site):
    """Half-hour table with errors of both models (dynamic = FAT, static = uniform-weighted)."""
    if not (y_true.shape == y_dyn.shape == y_stat.shape) or site.shape[:2] != y_true.shape[:2]:
        raise ValueError("Prediction / site shapes are not aligned")
    n, t, _ = y_true.shape
    hour = num(site[:, :, 7]) if site.shape[-1] > 7 else np.tile(np.arange(t) / 2.0, n)
    df = pd.DataFrame({
        "SITE_ID": np.asarray(site[:, :, 0]).reshape(-1).astype(str),
        "DoY": num(site[:, :, 4]),
        "Hour": hour,
    })
    if np.nanmax(df["Hour"]) > 24:        # half-hour index 0..47 -> hours
        df["Hour"] = df["Hour"] * 0.5
    df["hour_bin"] = np.floor(df["Hour"]).astype("Int64")   # numeric order matters for the bootstrap seeds
    doy = df["DoY"]
    df["season"] = np.select(
        [(doy <= 90) | (doy >= 335), (doy > 90) & (doy <= 160), (doy > 160) & (doy <= 250), (doy > 250) & (doy < 335)],
        SEASONS, default="Unknown")
    df["daynight"] = np.where((df["Hour"] >= 6) & (df["Hour"] < 18), "Day", "Night")
    for i, v in enumerate(VARS):
        obs = y_true[:, :, i].reshape(-1).astype(float)
        obs[np.isin(obs, list(FILL))] = np.nan
        ed = y_dyn[:, :, i].reshape(-1).astype(float) - obs
        es = y_stat[:, :, i].reshape(-1).astype(float) - obs
        df[f"{v}_ed"], df[f"{v}_es"] = ed, es
    return df[df["SITE_ID"] != "-9999"].copy()


# ---------------------------------------------------------------- statistics
def rmse(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan


def unit_delta(df, group_col):
    rows = []
    for v in VARS:
        for (site, g), sub in df.groupby(["SITE_ID", group_col], dropna=False):
            ed, es = sub[f"{v}_ed"].values, sub[f"{v}_es"].values
            ok = np.isfinite(ed) & np.isfinite(es)
            if ok.sum() < MIN_GROUP_SAMPLES:
                continue
            rows.append({"SITE_ID": site, group_col: g, "var": v, "delta_rmse_unit": rmse(es[ok]) - rmse(ed[ok]),
                         "n_samples_unit": int(ok.sum()), "win_rate_unit": float((np.abs(ed[ok]) < np.abs(es[ok])).mean())})
    return pd.DataFrame(rows)


def bootstrap_ci(x, seed):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    med = [np.median(rng.choice(x, size=x.size, replace=True)) for _ in range(N_BOOT)]
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def summarize(df, group_col):
    units = unit_delta(df, group_col)
    rows = []
    for (v, g), sub in units.groupby(["var", group_col], dropna=False):
        x = sub["delta_rmse_unit"].astype(float).values
        rows.append({"aggregation": "unit_first", "group_col": group_col, "group": str(g), "var": v,
                     "delta_rmse": float(np.median(x)), "delta_rmse_mean": float(np.mean(x)),
                     "delta_rmse_p25": float(np.percentile(x, 25)), "delta_rmse_p75": float(np.percentile(x, 75)),
                     "n_samples": int(sub["n_samples_unit"].sum()), "n_units": int(len(sub)),
                     "win_rate": float(np.mean(sub["win_rate_unit"]))})
    summ = pd.DataFrame(rows)
    ci = [bootstrap_ci(units[(units["var"] == r["var"]) & (units[group_col].astype(str) == r["group"])]["delta_rmse_unit"].values,
                       SEED + i) for i, r in summ.iterrows()]
    summ["ci_low"], summ["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    units["group_col_for_unit"] = group_col
    return summ, units.rename(columns={group_col: "group"})


def compute(y_true, y_dyn, y_stat, site):
    df = build_table(y_true, y_dyn, y_stat, site)
    s, u = [], []
    for col in ["season", "daynight", "hour_bin"]:
        a, b = summarize(df, col)
        s.append(a)
        u.append(b)
    return pd.concat(s, ignore_index=True), pd.concat(u, ignore_index=True)


# ---------------------------------------------------------------- plotting
def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.35, alpha=0.35)
    ax.set_axisbelow(True)


def panel_label(ax, label):
    ax.text(-0.11, 1.09, label, transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")


def plot_stage_bars(ax, summary):
    sub = summary[summary["group_col"].eq("season")]
    width, x, ymax = 0.23, np.arange(len(SEASONS)), 0.0
    for i, flux in enumerate(VARS):
        vals, lows, highs = [], [], []
        for g in SEASONS:
            r = sub[sub["group"].eq(g) & sub["var"].eq(flux)].iloc[0]
            y, lo, hi = float(r["delta_rmse"]), float(r["ci_low"]), float(r["ci_high"])
            vals.append(y)
            lows.append(max(0.0, y - lo))
            highs.append(max(0.0, hi - y))
            ymax = max(ymax, y, hi)
        xpos = x + (i - 1) * width
        ax.bar(xpos, vals, width=width, color=COLORS[flux], edgecolor="none", alpha=0.92, label=flux)
        ax.errorbar(xpos, vals, yerr=np.vstack([lows, highs]), fmt="none", ecolor="#333333", elinewidth=0.8,
                    capsize=2.2, capthick=0.8, alpha=0.85)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(SEASONS)
    ax.set_ylabel(r"$\Delta$RMSE")
    ax.set_ylim(top=ymax * 1.18)
    clean_axis(ax)


def plot_hourly(ax, summary):
    sub = summary[summary["group_col"].eq("hour_bin")].copy()
    sub["hour"] = pd.to_numeric(sub["group"], errors="coerce")
    ymax = 0.0
    for flux in VARS:
        sv = sub[sub["var"].eq(flux)].dropna(subset=["hour"]).sort_values("hour")
        x, y = sv["hour"].to_numpy(float), sv["delta_rmse"].to_numpy(float)
        lo, hi = sv["ci_low"].to_numpy(float), sv["ci_high"].to_numpy(float)
        ymax = max(ymax, float(np.nanmax(y)), float(np.nanmax(hi)))
        ax.plot(x, y, color=COLORS[flux], marker="o", markersize=2.5, linewidth=1.5, label=flux)
        ax.fill_between(x, lo, hi, color=COLORS[flux], alpha=0.14, linewidth=0)
    ax.axvspan(10, 16, color="#eeeeee", alpha=0.75, zorder=0)
    ax.set_ylim(top=ymax * 1.08)
    ax.text(13, ax.get_ylim()[1] * 0.92, "midday", ha="center", va="top", fontsize=7, color="#555555")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("Local hour")
    ax.set_ylabel(r"$\Delta$RMSE")
    clean_axis(ax)


def make_figure(summary, out_png, out_pdf):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.0, 3.05), gridspec_kw={"wspace": 0.16})
    plot_stage_bars(ax_a, summary)
    panel_label(ax_a, "a")
    plot_hourly(ax_b, summary)
    panel_label(ax_b, "b")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[f]) for f in VARS]
    fig.legend(handles, VARS, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.035))
    fig.subplots_adjust(bottom=0.22, left=0.075, right=0.985, top=0.93)
    save_manuscript_png(fig, out_png, render_dpi=600, pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y_t, y_f, s_f = load_model("FAT")
    y_t2, y_u, s_u = load_model("Uniform_Transformer")
    if not (np.array_equal(np.where(np.isfinite(y_t), y_t, 0), np.where(np.isfinite(y_t2), y_t2, 0)) and np.array_equal(s_f, s_u)):
        raise RuntimeError("FAT and uniform-weighted predictions are not paired on the same held-out records")
    summary, units = compute(y_t, y_f, y_u, s_f)
    summary.to_csv(OUT / "summary_temporal_delta_rmse.csv", index=False)
    units.to_csv(OUT / "unit_level_temporal_delta_rmse.csv", index=False)
    summary[summary["group_col"].eq("season")].to_csv(OUT / "temporal_stage_summary.csv", index=False)
    summary[summary["group_col"].eq("hour_bin")].to_csv(OUT / "temporal_hourly_summary.csv", index=False)
    make_figure(summary, OUT / "Fig5_temporal_emergence.png", OUT / "Fig5_temporal_emergence.pdf")
    st = summary[summary["group_col"].eq("season")].pivot(index="group", columns="var", values="delta_rmse").reindex(SEASONS)
    print(st.round(3).to_string())
    print(json.dumps({"figure": str(OUT / "Fig5_temporal_emergence.png")}, indent=1))


if __name__ == "__main__":
    main()
