#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4.3: which factors are associated with the FAT gain?  (descriptive; correlation, not causation)

Factors (20, in four conceptual blocks):
  1 Surface heterogeneity     NDVI standard deviation of the HLS patch (the only landscape factor)
  2 Model inputs              the 12 meteorological / soil covariates the networks were trained with:
                              air temperature, dewpoint, VPD, shortwave radiation, soil temperature (4 layers),
                              soil moisture (4 layers)
  3 Footprint calculation     the parameters that drive the footprint model: friction velocity, stability (z/L),
                              boundary-layer height, sigma_v, roughness length (measurement height is a site constant: excluded)
  4 Wind                      wind speed, within-day wind-direction variability
Step 1  Gain per HLS-matched day and flux: delta RMSE = RMSE(uniform-weighted Transformer) - RMSE(FAT) (positive = FAT better).
Step 2  Aggregate to a time scale: daily, 8-day, monthly, seasonal, annual (weights = valid half-hours; minimum days per window).
Step 3  Marginal Spearman rho between each factor and the gain, 95% interval by resampling WHOLE SITES.
        Adjusted (partial) Spearman rho controlling for site type (IGBP) and, except at the annual scale, the seasonal
        cycle (sin/cos of day of year): factors that only reflect "forest vs cropland" or "summer vs winter" lose their
        association.  Intervals use the same site resampling on the residuals (approximate).
Step 4  Significance: two-sided bootstrap p-values (site resampling), Benjamini-Hochberg FDR across the 20 factors within every
        scale x flux family; * = q < 0.05.
Step 5  Pre-stated rule for the binned figure (independent of effect size): the pre-specified primary factor (NDVI
        heterogeneity) plus every factor whose ADJUSTED association is significant (q < 0.05) at >= 3 of the 5 time scales
        for the same flux (consistency across time scales); one layer per layered family.  Binned responses of ALL factors
        are in the supplement.

Inputs : ../../data/four_model_predictions/{FAT,Uniform_Transformer}/fold{0-4}/{true,pred,site}.npy
         ../../data/site_day_drivers.csv  (daily factors; independent of the models)
         ../../4.2_estimation_robustness/outputs/spatial/figures/Fig4_v25_site_metadata_with_climate.csv (IGBP)
Outputs: ../outputs/  correlation_all_scales.csv, soil_layer_correlations.csv, binned_response_*.csv,
         Fig_correlation_heatmap_all_scales.*, Fig_correlation_heatmap_adjusted_all_scales.*,
         Fig_binned_response_selected.*, Fig_binned_response_all_factors.*
Run    : python gain_correlates.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "common"))
from figure_io import save_manuscript_png  # noqa: E402

DATA = PROJECT / "data" / "four_model_predictions"
DRIVERS = PROJECT / "data" / "site_day_drivers.csv"
SITE_META = PROJECT / "4.2_estimation_robustness" / "outputs" / "spatial" / "figures" / "Fig4_v25_site_metadata_with_climate.csv"
OUT = HERE.parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

FLUXES = ["NEE", "GPP", "RECO"]
FLUX_COLORS = {"NEE": "#4F81BD", "GPP": "#3C9A61", "RECO": "#C87521"}
KEYS = ["SITE_ID", "Year", "DoY", "HLS_Hour"]
MAIN_SCALE = "year"
SCALES = ["daily", "8day", "monthly", "season", "year"]
SCALE_LABEL = {"daily": "Daily", "8day": "8-day", "monthly": "Monthly", "season": "Seasonal", "year": "Annual"}
MIN_DAYS = {"daily": 1, "8day": 1, "monthly": 2, "season": 3, "year": 6}
N_BOOT = {"daily": 1000, "8day": 1000, "monthly": 1000, "season": 1500, "year": 2000}
SEED = 2024

# column, label, block, family (layered variables share a family)
FACTORS = [
    ("NDVI_STD_DAY", "NDVI heterogeneity", "Surface heterogeneity", "ndvi"),
    ("t2m_C_day_median", "Air temperature", "Model inputs", "air"),
    ("d2m_C_day_median", "Dewpoint temperature", "Model inputs", "dew"),
    ("VPD_kPa_day_median", "VPD", "Model inputs", "vpd"),
    ("SSRD_Wm2_day_median", "Shortwave radiation", "Model inputs", "ssrd"),
    ("stl1_C_day_median", "Soil temperature, 0-7 cm", "Model inputs", "soilT"),
    ("stl2_C_day_median", "Soil temperature, 7-28 cm", "Model inputs", "soilT"),
    ("stl3_C_day_median", "Soil temperature, 28-100 cm", "Model inputs", "soilT"),
    ("stl4_C_day_median", "Soil temperature, 100-289 cm", "Model inputs", "soilT"),
    ("swvl1_pct_day_median", "Soil moisture, 0-7 cm", "Model inputs", "soilM"),
    ("swvl2_pct_day_median", "Soil moisture, 7-28 cm", "Model inputs", "soilM"),
    ("swvl3_pct_day_median", "Soil moisture, 28-100 cm", "Model inputs", "soilM"),
    ("swvl4_pct_day_median", "Soil moisture, 100-289 cm", "Model inputs", "soilM"),
    ("USTAR_day_median", "Friction velocity", "Footprint calculation", "ustar"),
    ("z_over_L_day_median", "Stability z/L", "Footprint calculation", "zL"),
    ("blh_day_median", "Boundary-layer height", "Footprint calculation", "blh"),
    ("sigma_v_rf_day_median", "Lateral turbulence σv", "Footprint calculation", "sigv"),
    ("z0_day_median", "Roughness length", "Footprint calculation", "z0"),
    ("WS_ms_day_median", "Wind speed", "Wind", "ws"),
    ("WD_deg_day_circular_sd", "Wind-direction variability", "Wind", "wd"),
]
LABEL = {c: l for c, l, _, _ in FACTORS}
GROUP = {c: g for c, _, g, _ in FACTORS}
FAMILY = {c: f for c, _, _, f in FACTORS}
FCOLS = [c for c, _, _, _ in FACTORS]
GROUP_COLORS = {"Surface heterogeneity": "#8E5EA2", "Model inputs": "#4393C3",
                "Footprint calculation": "#D6604D", "Wind": "#4D9221"}
SEASON_COLS = ["sinD", "cosD"]

plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


# ------------------------------------------------------------------ data
def load_model(name):
    base = DATA / name
    ys, ps, ss = [], [], []
    for k in range(5):
        ys.append(np.load(base / f"fold{k}" / "true.npy", allow_pickle=True))
        ps.append(np.load(base / f"fold{k}" / "pred.npy", allow_pickle=True))
        ss.append(np.load(base / f"fold{k}" / "site.npy", allow_pickle=True))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ss)


def daily_gain_table():
    """One row per HLS-matched day (record) and flux with RMSE of both models over its valid half-hours."""
    y, pf, site = load_model("FAT")
    y2, pu, site2 = load_model("Uniform_Transformer")
    if not np.array_equal(site, site2):
        raise RuntimeError("FAT / Uniform records are not aligned")
    s0 = site[:, 0, :]
    rec = pd.DataFrame({"SITE_ID": s0[:, 0].astype(str), "Year": pd.to_numeric(s0[:, 3]).astype(int),
                        "DoY": pd.to_numeric(s0[:, 4]).astype(int), "HLS_Hour": pd.to_numeric(s0[:, 5]).astype(int)})
    rows = []
    for i, flux in enumerate(FLUXES):
        obs = y[:, :, i].astype(float)
        ef = pf[:, :, i].astype(float) - obs
        eu = pu[:, :, i].astype(float) - obs
        valid = np.isfinite(obs) & (obs > -9000) & np.isfinite(ef) & np.isfinite(eu)
        n = valid.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rf = np.sqrt(np.where(valid, ef ** 2, 0).sum(1) / n)
            ru = np.sqrt(np.where(valid, eu ** 2, 0).sum(1) / n)
        t = rec.copy()
        ov = np.where(valid, obs, 0.0)
        t["flux"], t["n_valid"], t["rmse_dynamic"], t["rmse_static"] = flux, n, rf, ru
        t["obs_sum"], t["obs_sumsq"] = ov.sum(1), (ov ** 2).sum(1)      # for the spread of the observations in a window
        rows.append(t[n > 0])
    d = pd.concat(rows, ignore_index=True)
    d = d.merge(pd.read_csv(DRIVERS), on=KEYS, how="inner")          # keep days with factor information
    d["date"] = pd.to_datetime(d["Year"].astype(str), format="%Y") + pd.to_timedelta(d["DoY"] - 1, unit="D")
    d["month"] = d["date"].dt.month
    d["season"] = d["month"].map(lambda m: "DJF" if m in (12, 1, 2) else "MAM" if m in (3, 4, 5) else "JJA" if m in (6, 7, 8) else "SON")
    d["season_year"] = np.where(d["month"].eq(12), d["Year"] + 1, d["Year"])
    d["eight"] = ((d["DoY"] - 1) // 8 + 1).clip(upper=46)
    d["sinD"], d["cosD"] = np.sin(2 * np.pi * d["DoY"] / 365.25), np.cos(2 * np.pi * d["DoY"] / 365.25)
    return d


def aggregate(d, scale):
    cols = FCOLS + SEASON_COLS
    if scale == "daily":
        out = d.copy()
        out["n_days"] = 1
        out["rmse_d"], out["rmse_s"] = out["rmse_dynamic"], out["rmse_static"]
        m = out["obs_sum"] / out["n_valid"]
        out["sd_obs"] = np.sqrt(np.maximum(out["obs_sumsq"] / out["n_valid"] - m ** 2, 0))
    else:
        keys = {"8day": ["SITE_ID", "Year", "eight"], "monthly": ["SITE_ID", "Year", "month"],
                "season": ["SITE_ID", "season_year", "season"], "year": ["SITE_ID", "Year"]}[scale]
        g = d.assign(w=d["n_valid"].astype(float), md=lambda x: x["rmse_dynamic"] ** 2 * x["n_valid"],
                     ms=lambda x: x["rmse_static"] ** 2 * x["n_valid"], sx=lambda x: x["obs_sum"],
                     sxx=lambda x: x["obs_sumsq"]).groupby(keys + ["flux"], sort=False)
        base = g.agg(w=("w", "sum"), md=("md", "sum"), ms=("ms", "sum"), sx=("sx", "sum"), sxx=("sxx", "sum"),
                     n_days=("DoY", "size")).reset_index()
        base["sd_obs"] = np.sqrt(np.maximum(base["sxx"] / base["w"] - (base["sx"] / base["w"]) ** 2, 0))
        base["rmse_d"], base["rmse_s"] = np.sqrt(base["md"] / base["w"]), np.sqrt(base["ms"] / base["w"])
        for c in cols:                                  # weighted mean of each factor over the days of the window
            x = d[c]
            tmp = d[keys + ["flux"]].assign(num=(x * d["n_valid"]).fillna(0), den=d["n_valid"].where(x.notna(), 0))
            s = tmp.groupby(keys + ["flux"], sort=False)[["num", "den"]].sum().reset_index()
            base = base.merge(s.assign(**{c: s["num"] / s["den"].replace(0, np.nan)})[keys + ["flux", c]], on=keys + ["flux"])
        out = base
    out = out[out["n_days"] >= MIN_DAYS[scale]].copy()
    out["delta"] = out["rmse_s"] - out["rmse_d"]
    out["rel"] = out["delta"] / out["rmse_s"]                       # share of the uniform model's error removed by FAT
    out["nrmse_d"], out["nrmse_s"] = out["rmse_d"] / out["sd_obs"], out["rmse_s"] / out["sd_obs"]   # RMSE / SD of observations
    meta = pd.read_csv(SITE_META)[["SITE_ID", "IGBP"]].drop_duplicates("SITE_ID")
    out = out.merge(meta, on="SITE_ID", how="left")
    out["IGBP"] = out["IGBP"].fillna("NA")
    return out.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------ statistics
def spearman_matrix(x, y):
    """Spearman rho of each column of x (n,p) with y (n,), pairwise complete."""
    out = np.full(x.shape[1], np.nan)
    for j in range(x.shape[1]):
        ok = np.isfinite(x[:, j]) & np.isfinite(y)
        if ok.sum() < 10:
            continue
        a, b = rankdata(x[ok, j]), rankdata(y[ok])
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        out[j] = (a * b).sum() / den if den > 0 else np.nan
    return out


def pearson_matrix(x, y):
    xc, yc = x - x.mean(0), y - y.mean()
    return (xc * yc[:, None]).sum(0) / np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum())


def boot_p(boots):
    """Two-sided bootstrap p-value of 'rho = 0' from site-resampled estimates (resolution 1/(B+1))."""
    b = np.asarray(boots)
    pos, neg = (np.nansum(b > 0, 0) + 1) / (b.shape[0] + 1), (np.nansum(b < 0, 0) + 1) / (b.shape[0] + 1)
    return np.minimum(1.0, 2 * np.minimum(pos, neg))


def bh_q(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def add_fdr(cor):
    """FDR across the 20 factors within every (scale, flux) family."""
    cor = cor.copy()
    cor["q"], cor["adj_q"] = np.nan, np.nan
    for _, idx in cor.groupby(["scale", "flux"]).groups.items():
        cor.loc[idx, "q"] = bh_q(cor.loc[idx, "p"].to_numpy())
        cor.loc[idx, "adj_q"] = bh_q(cor.loc[idx, "adj_p"].to_numpy())
    cor["sig"], cor["adj_sig"] = cor["q"] < 0.05, cor["adj_q"] < 0.05
    return cor


def site_blocks(sites):
    codes, uniq = pd.factorize(sites)
    return [np.flatnonzero(codes == k) for k in range(len(uniq))]


def boot_indices(blocks, n_boot, rng):
    for _ in range(n_boot):
        yield np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])


def residualize(v, C):
    beta, *_ = np.linalg.lstsq(C, v, rcond=None)
    return v - C @ beta


def correlations(table, scale, y_col="delta"):
    rows = []
    rng = np.random.default_rng(SEED)
    for flux in FLUXES:
        d = table[table["flux"].eq(flux)].dropna(subset=[y_col])
        x, y = d[FCOLS].to_numpy(float), d[y_col].to_numpy(float)
        blocks = site_blocks(d["SITE_ID"].to_numpy())
        rho = spearman_matrix(x, y)
        idxs = list(boot_indices(blocks, N_BOOT[scale], rng))
        boots = np.array([spearman_matrix(x[i], y[i]) for i in idxs])
        lo, hi = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)
        pv = boot_p(boots)
        # adjusted: partial Spearman given site type (+ seasonal cycle below the annual scale)
        ctrl = pd.get_dummies(d["IGBP"], drop_first=True).astype(float)
        if scale != "year":
            ctrl[SEASON_COLS] = d[SEASON_COLS].to_numpy()
        keep = np.isfinite(x).all(1) & np.isfinite(ctrl.to_numpy()).all(1)
        Cm = np.column_stack([np.ones(keep.sum()), ctrl.to_numpy()[keep]])
        ry = residualize(rankdata(y[keep]), Cm)
        rx = np.column_stack([residualize(rankdata(x[keep][:, j]), Cm) for j in range(x.shape[1])])
        arho = pearson_matrix(rx, ry)
        blocks_k = site_blocks(d["SITE_ID"].to_numpy()[keep])
        aboots = np.array([pearson_matrix(rx[i], ry[i]) for i in boot_indices(blocks_k, N_BOOT[scale], rng)])
        alo, ahi = np.percentile(aboots, 2.5, axis=0), np.percentile(aboots, 97.5, axis=0)
        apv = boot_p(aboots)
        for j, c in enumerate(FCOLS):
            rows.append({"scale": scale, "outcome": y_col, "flux": flux, "factor": c, "label": LABEL[c], "group": GROUP[c], "family": FAMILY[c],
                         "rho": rho[j], "ci_low": lo[j], "ci_high": hi[j], "ci_excludes_0": bool(lo[j] > 0 or hi[j] < 0), "p": pv[j],
                         "rho_adj": arho[j], "adj_ci_low": alo[j], "adj_ci_high": ahi[j],
                         "adj_ci_excludes_0": bool(alo[j] > 0 or ahi[j] < 0), "adj_p": apv[j], "n": len(d), "n_sites": d["SITE_ID"].nunique()})
    return pd.DataFrame(rows)


PRIMARY = "NDVI_STD_DAY"          # pre-specified hypothesis factor: always shown


def select_factors(cor):
    """Pre-stated rule (independent of effect size and of the number of factors):
    show (a) the pre-specified primary factor and (b) every factor whose ADJUSTED association (site type + season removed)
    is significant after FDR (q < 0.05) at >= 3 of the 5 time scales for the same flux.  Layered variables: all layers are
    judged separately but only the layer with the most significant scales is drawn per family."""
    cnt = cor.groupby(["factor", "flux"])["adj_sig"].sum().reset_index(name="n_scales")
    keep = cnt[cnt["n_scales"] >= 3]
    best = keep.sort_values("n_scales", ascending=False).drop_duplicates("factor")
    rows = [{"factor": f, "family": FAMILY[f], "n_scales_max": int(n)} for f, n in zip(best["factor"], best["n_scales"])]
    sel = pd.DataFrame(rows)
    if len(sel):
        sel = sel.sort_values("n_scales_max", ascending=False).drop_duplicates("family")
    out = [PRIMARY] + [f for f in (sel["factor"] if len(sel) else []) if f != PRIMARY]
    return out


def binned_response(table, factors, n_bins=6):
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for flux in FLUXES:
        d = table[table["flux"].eq(flux)].dropna(subset=["delta"])
        for c in factors:
            dd = d.dropna(subset=[c])
            edges = np.unique(np.quantile(dd[c], np.linspace(0, 1, n_bins + 1)))
            b = np.clip(np.searchsorted(edges, dd[c].to_numpy(), side="right") - 1, 0, len(edges) - 2)
            dd = dd.assign(bin=b)
            blocks = site_blocks(dd["SITE_ID"].to_numpy())
            mean, xm, n = dd.groupby("bin")["delta"].mean(), dd.groupby("bin")[c].median(), dd.groupby("bin").size()
            delta, binv = dd["delta"].to_numpy(), dd["bin"].to_numpy()
            boots = np.array([pd.Series(delta[i]).groupby(binv[i]).mean().reindex(mean.index).to_numpy()
                              for i in boot_indices(blocks, 500, rng)])
            for k, bi in enumerate(mean.index):
                rows.append({"flux": flux, "factor": c, "label": LABEL[c], "bin": int(bi), "x_median": xm[bi], "n": int(n[bi]),
                             "mean_delta": mean[bi], "ci_low": np.nanpercentile(boots[:, k], 2.5), "ci_high": np.nanpercentile(boots[:, k], 97.5)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def heatmap(ax, cor, cols, value, flag, vmax=0.25):
    mat = np.full((len(FCOLS), len(cols)), np.nan)
    star = np.zeros_like(mat, dtype=bool)
    for jc, (sc, fl) in enumerate(cols):
        sub = cor[(cor["scale"] == sc) & (cor["flux"] == fl)].set_index("factor")
        for i, c in enumerate(FCOLS):
            mat[i, jc], star[i, jc] = sub.loc[c, value], bool(sub.loc[c, flag])
    im = ax.imshow(mat, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
    ax.set_yticks(range(len(FCOLS)))
    ax.set_yticklabels([LABEL[c] for c in FCOLS])
    for lab, c in zip(ax.get_yticklabels(), FCOLS):
        lab.set_color(GROUP_COLORS[GROUP[c]])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, ("*" if star[i, j] else ""), ha="center", va="center", fontsize=10, fontweight="bold",
                        color="white" if abs(mat[i, j]) > 0.3 else "#222222")
    prev = None
    for i, c in enumerate(FCOLS):
        if prev is not None and GROUP[c] != prev:
            ax.axhline(i - 0.5, color="white", lw=2.5)
        prev = GROUP[c]
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f for _, f in cols], fontsize=7.5)
    ax.xaxis.tick_top()
    for k, s in enumerate(SCALES):
        ax.text(3 * k + 1, -1.7, SCALE_LABEL[s], ha="center", va="bottom", fontsize=9, fontweight="bold")
        if k:
            ax.axvline(3 * k - 0.5, color="white", lw=2.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig_heatmap(cor, value, flag, name, label):
    cols = [(s, f) for s in SCALES for f in FLUXES]
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    im = heatmap(ax, cor, cols, value, flag)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(label)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in GROUP_COLORS]
    fig.legend(handles, list(GROUP_COLORS), loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.45, -0.02))
    fig.text(0.02, -0.05, "* FDR q < 0.05 (site-resampling p-values, Benjamini-Hochberg over the 20 factors)", fontsize=7.5)
    save_manuscript_png(fig, OUT / f"{name}.png", width_in=7.2, render_dpi=500)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_binned(binned, cor, factors, name, vertical=False):
    nf = len(factors)
    if not vertical:
        fig, axes = plt.subplots(3, nf, figsize=(2.15 * nf + 0.4, 6.0), sharey="row", squeeze=False)
        grid = [(i, j, FLUXES[i], factors[j]) for i in range(3) for j in range(nf)]
    else:
        fig, axes = plt.subplots(nf, 3, figsize=(7.2, 1.65 * nf + 0.5), sharey="col", squeeze=False)
        grid = [(j, i, FLUXES[i], factors[j]) for i in range(3) for j in range(nf)]
    for r, c_, flux, c in grid:
        ax = axes[r, c_]
        b = binned[(binned["flux"] == flux) & (binned["factor"] == c)].sort_values("bin")
        col = FLUX_COLORS[flux]
        ax.fill_between(b["x_median"], b["ci_low"], b["ci_high"], color=col, alpha=0.2, linewidth=0)
        ax.plot(b["x_median"], b["mean_delta"], "-o", color=col, ms=3.5, lw=1.5)
        ax.axhline(0, color="#333333", lw=0.7)
        s = cor[(cor["scale"] == MAIN_SCALE) & (cor["flux"] == flux) & (cor["factor"] == c)].iloc[0]
        ax.text(0.04, 0.95, f"ρ = {s['rho']:+.2f}" + ("*" if s["sig"] else ""), transform=ax.transAxes, va="top", fontsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)
        ax.tick_params(labelsize=7)
        if (not vertical and r == 0) or (vertical and c_ == 0):
            pass
        if not vertical:
            if r == 0:
                ax.set_title(LABEL[c], fontsize=8.5, color=GROUP_COLORS[GROUP[c]], fontweight="bold")
            if c_ == 0:
                ax.set_ylabel(f"{flux}\nmean ΔRMSE")
            if r == 2:
                ax.set_xlabel("factor value (bin median)")
        else:
            if r == 0:
                ax.set_title(flux, fontsize=9, color=FLUX_COLORS[flux], fontweight="bold")
            if c_ == 0:
                ax.set_ylabel(LABEL[c], fontsize=7.5, color=GROUP_COLORS[GROUP[c]], rotation=0, ha="right", va="center", labelpad=6)
    fig.subplots_adjust(wspace=0.14, hspace=0.34 if not vertical else 0.55)
    save_manuscript_png(fig, OUT / f"{name}.png", width_in=7.2, render_dpi=500)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    d = daily_gain_table()
    print("daily gain table:", len(d), "rows; sites", d["SITE_ID"].nunique())
    cors, tables = [], {}
    for s in SCALES:
        tables[s] = aggregate(d, s)
        cors.append(correlations(tables[s], s))
        print(s, "windows", len(tables[s]) // 3, "done", flush=True)
    cor = add_fdr(pd.concat(cors, ignore_index=True))
    cor.to_csv(OUT / "correlation_all_scales.csv", index=False)
    tables[MAIN_SCALE].to_csv(OUT / "annual_gain_with_factors.csv", index=False)
    # layer redundancy (annual NEE table): shows why the layers are treated as one family
    layers = [c for c in FCOLS if FAMILY[c] in ("soilT", "soilM")]
    pd.DataFrame(tables[MAIN_SCALE].query("flux == 'NEE'")[layers].corr(method="spearman")).round(3).to_csv(OUT / "soil_layer_correlations.csv")

    factors = select_factors(cor)
    print("shown in the binned figure:", [LABEL[f] for f in factors])
    binned_top = binned_response(tables[MAIN_SCALE], factors)
    binned_top.to_csv(OUT / "binned_response_selected_annual.csv", index=False)
    binned_all = binned_response(tables[MAIN_SCALE], FCOLS)
    binned_all.to_csv(OUT / "binned_response_all_factors_annual.csv", index=False)

    fig_heatmap(cor, "rho", "sig", "Fig_correlation_heatmap_all_scales", "Spearman ρ with FAT gain (ΔRMSE)")
    fig_heatmap(cor, "rho_adj", "adj_sig", "Fig_correlation_heatmap_adjusted_all_scales",
                "partial Spearman ρ (adjusted for site type and season)")
    fig_binned(binned_top, cor, factors, "Fig_binned_response_selected")
    fig_binned(binned_all, cor, FCOLS, "Fig_binned_response_all_factors", vertical=True)

    a = cor[cor["scale"] == MAIN_SCALE]
    print(a.pivot(index="label", columns="flux", values="rho").reindex([LABEL[c] for c in FCOLS]).round(2).to_string())
    print("adjusted (annual):")
    print(a.pivot(index="label", columns="flux", values="rho_adj").reindex([LABEL[c] for c in FCOLS]).round(2).to_string())


if __name__ == "__main__":
    main()
