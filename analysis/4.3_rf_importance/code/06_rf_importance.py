"""
Multi-scale relative variable importance for dynamic-FFP gains.

Scales:
  daily, 8-day, monthly, season, year

For each scale, this script computes:
  - scale-specific RMSE_static and RMSE_dynamic
  - ΔRMSE = RMSE_static - RMSE_dynamic
  - VIF13 predictor summaries on the same scale, with wind-direction sin/cos removed
  - SITE_ID-blocked Random Forest permutation importance
  - relative variable importance (%), normalized from positive scores
  - grouped importance as supplementary output

Outputs are written into separate folders for each scale.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


SITE_DAY = Path(os.environ.get("RSE_SITE_DAY", './data/external/site_day_delta_rmse_with_drivers.csv'))
if not SITE_DAY.exists():
    SITE_DAY = Path('./data/external/site_day_delta_rmse_with_drivers.csv')
VIF_TABLE = Path(os.environ.get("RSE_VIF_TABLE", './data/external/vif_after_representative_filter.csv'))
if not VIF_TABLE.exists():
    VIF_TABLE = Path('./data/external/vif_after_representative_filter.csv')
VIF_FALLBACK_TABLE = Path(os.environ.get("RSE_VIF_FALLBACK_TABLE", './data/external/vif_all_variables.csv'))
OUT = Path(
    os.environ.get(
        "RSE_MULTISCALE_OUT",
        str(
            Path(
                os.environ.get(
                    "HLS_FINAL_RESULT_ROOT",
                    './data/external/HLS_METE_CARBON_FFP'
    r"\work\final_4_1_to_4_5\result",
                )
            )
            / "4.3_multiscale_relative_importance"
            / "outputs_vif13_no_wd_sincos"
        ),
    )
)

# --- project paths (this folder): new paired results, annual scale only (site-year gains)
_HERE = Path(__file__).resolve().parent
SITE_DAY = _HERE.parent / "data" / "site_day_delta_rmse_with_drivers_V6.csv"
OUT = _HERE.parent / "outputs"

FLUXES = ["GPP", "NEE", "RECO"]
SCALES = [
    s.strip()
    for s in os.environ.get("RSE_MULTISCALE_SCALES", "year").split(",")
    if s.strip()
]
EXCLUDE_ANALYSIS_VARIABLES = {
    s.strip()
    for s in os.environ.get(
        "RSE_EXCLUDE_ANALYSIS_VARIABLES",
        "WD_cos_day_derived,WD_sin_day_derived",
    ).split(",")
    if s.strip()
}

VIF15_MAPPING = [
    ("WS_ms_median", "WS_ms_day_median", "Footprint/turbulence dynamics", "Wind speed", "direct"),
    ("VPD_kPa_median", "VPD_kPa_day_median", "Meteorological forcing", "VPD", "direct"),
    ("swvl1_pct_median", "swvl1_pct_day_median", "Soil hydrothermal state", "Surface soil moisture", "direct"),
    ("USTAR_median", "USTAR_day_median", "Footprint/turbulence dynamics", "Friction velocity", "direct"),
    ("SSRD_Wm2_median", "SSRD_Wm2_day_median", "Meteorological forcing", "Solar radiation", "direct"),
    ("WD_deg_circular_sd", "WD_deg_day_circular_sd", "Footprint/turbulence dynamics", "Wind direction variability", "direct"),
    ("NDVI_std", "NDVI_STD_DAY", "Surface heterogeneity", "NDVI heterogeneity", "direct"),
    ("edge_index", "dw_edge_density_median", "Surface heterogeneity", "Land-cover edge density", "site-day metric"),
    ("d2m_C_median", "d2m_C_day_median", "Meteorological forcing", "Dewpoint temperature", "direct"),
    ("swvl4_pct_median", "swvl4_pct_day_median", "Soil hydrothermal state", "Deep soil moisture", "direct"),
    ("z0_median", "z0_day_median", "Canopy/surface structure", "Roughness length", "direct"),
    ("WD_cos_median", "WD_cos_day_derived", "Footprint/turbulence dynamics", "Wind direction cos", "derived"),
    ("WD_sin_median", "WD_sin_day_derived", "Footprint/turbulence dynamics", "Wind direction sin", "derived"),
    ("z_over_L_median", "z_over_L_day_median", "Footprint/turbulence dynamics", "Stability z/L", "direct"),
    ("P_mm_median", "P_mm_day_median", "Meteorological forcing", "Precipitation", "direct"),
]


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def build_mapping() -> pd.DataFrame:
    mapping = pd.DataFrame(
        VIF15_MAPPING,
        columns=["original_variable", "analysis_variable", "group", "label", "source"],
    )
    if EXCLUDE_ANALYSIS_VARIABLES:
        mapping = mapping.loc[~mapping["analysis_variable"].isin(EXCLUDE_ANALYSIS_VARIABLES)].copy()
    vif = pd.read_csv(VIF_TABLE)
    merged = mapping.merge(vif, left_on="original_variable", right_on="variable", how="left").drop(columns=["variable"])
    missing = merged["VIF"].isna() & VIF_FALLBACK_TABLE.exists()
    if missing.any():
        fallback = pd.read_csv(VIF_FALLBACK_TABLE).set_index("variable")
        for col in ["VIF", "R2_against_others"]:
            merged.loc[missing, col] = merged.loc[missing, "original_variable"].map(fallback[col])
    return merged


def load_site_day(mapping: pd.DataFrame) -> pd.DataFrame:
    required = {
        "record_id",
        "SITE_ID",
        "Year",
        "DoY",
        "flux",
        "n_valid_halfhours",
        "rmse_dynamic_day",
        "rmse_static_day",
        "delta_RMSE_day",
        "relative_delta_RMSE_day",
        "WD_deg_day_median",
    }
    for var in mapping["analysis_variable"]:
        if not var.endswith("_derived"):
            required.add(var)
    df = pd.read_csv(SITE_DAY, usecols=sorted(required))
    df["WD_sin_day_derived"] = np.sin(np.deg2rad(df["WD_deg_day_median"]))
    df["WD_cos_day_derived"] = np.cos(np.deg2rad(df["WD_deg_day_median"]))
    base = pd.to_datetime(df["Year"].astype(str), format="%Y")
    df["date"] = base + pd.to_timedelta(df["DoY"].astype(int) - 1, unit="D")
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(season_from_month)
    df["season_year"] = df["Year"].astype(int)
    df.loc[df["month"].eq(12), "season_year"] = df.loc[df["month"].eq(12), "Year"].astype(int) + 1
    df["eight_day_period"] = ((df["DoY"].astype(int) - 1) // 8 + 1).clip(upper=46)
    df["daily_id"] = df["Year"].astype(str) + "_" + df["DoY"].astype(str).str.zfill(3)
    df["eight_day_id"] = df["Year"].astype(str) + "_" + df["eight_day_period"].astype(str).str.zfill(2)
    df["month_id"] = df["Year"].astype(str) + "_" + df["month"].astype(str).str.zfill(2)
    df["season_id"] = df["season_year"].astype(str) + "_" + df["season"]
    df["year_id"] = df["Year"].astype(str)
    return df


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def aggregate_scale(df: pd.DataFrame, scale: str, features: list[str]) -> pd.DataFrame:
    if scale == "daily":
        out = df.copy()
        out["window_id"] = out["daily_id"]
        out["n_hls_days"] = 1
        out["rmse_dynamic"] = out["rmse_dynamic_day"]
        out["rmse_static"] = out["rmse_static_day"]
        out["delta_RMSE"] = out["delta_RMSE_day"]
        out["relative_improvement"] = out["relative_delta_RMSE_day"]
        cols = ["SITE_ID", "Year", "window_id", "flux", "n_hls_days", "n_valid_halfhours", "rmse_dynamic", "rmse_static", "delta_RMSE", "relative_improvement"] + features
        return out[cols].copy()

    if scale == "8day":
        group_cols = ["SITE_ID", "Year", "eight_day_id", "flux"]
        time_col = "Year"
        window_col = "eight_day_id"
    elif scale == "monthly":
        group_cols = ["SITE_ID", "Year", "month_id", "flux"]
        time_col = "Year"
        window_col = "month_id"
    elif scale == "season":
        group_cols = ["SITE_ID", "season_year", "season_id", "flux"]
        time_col = "season_year"
        window_col = "season_id"
    elif scale == "year":
        group_cols = ["SITE_ID", "Year", "year_id", "flux"]
        time_col = "Year"
        window_col = "year_id"
    else:
        raise ValueError(scale)

    rows = []
    for keys, g in df.groupby(group_cols, sort=False):
        w = g["n_valid_halfhours"].fillna(0).astype(float)
        row = dict(zip(group_cols, keys))
        row["window_id"] = row[window_col]
        row["time_year"] = int(row[time_col])
        row["n_hls_days"] = int(g["record_id"].nunique())
        row["n_valid_halfhours"] = float(w.sum())
        if row["n_valid_halfhours"] > 0:
            dyn_mse = np.average(g["rmse_dynamic_day"] ** 2, weights=w)
            sta_mse = np.average(g["rmse_static_day"] ** 2, weights=w)
            row["rmse_dynamic"] = math.sqrt(max(dyn_mse, 0.0))
            row["rmse_static"] = math.sqrt(max(sta_mse, 0.0))
            row["delta_RMSE"] = row["rmse_static"] - row["rmse_dynamic"]
            row["relative_improvement"] = row["delta_RMSE"] / row["rmse_static"] if row["rmse_static"] > 0 else np.nan
        else:
            row["rmse_dynamic"] = np.nan
            row["rmse_static"] = np.nan
            row["delta_RMSE"] = np.nan
            row["relative_improvement"] = np.nan
        for feat in features:
            row[feat] = weighted_mean(g[feat], w)
        rows.append(row)
    return pd.DataFrame(rows)


def min_days_for_scale(scale: str) -> int:
    return {
        "daily": 1,
        "8day": 1,
        "monthly": 2,
        "season": 3,
        "year": 6,
    }[scale]


def model_input(table: pd.DataFrame, features: list[str], scale: str) -> pd.DataFrame:
    min_days = min_days_for_scale(scale)
    out = (
        table.loc[table["n_hls_days"].ge(min_days)]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["delta_RMSE"] + features)
        .copy()
    )
    return out


def run_importance(model_df: pd.DataFrame, mapping: pd.DataFrame, scale: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = mapping["analysis_variable"].tolist()
    label_map = mapping.set_index("analysis_variable")["label"].to_dict()
    group_map = mapping.set_index("analysis_variable")["group"].to_dict()
    original_map = mapping.set_index("analysis_variable")["original_variable"].to_dict()
    group_features = mapping.groupby("group")["analysis_variable"].apply(list).to_dict()
    perf_rows, var_rows, group_rows = [], [], []

    for flux in FLUXES:
        d = model_df[model_df["flux"].eq(flux)].copy()
        x = d[features]
        y = d["delta_RMSE"].to_numpy()
        groups = d["SITE_ID"].to_numpy()
        if len(d) < 50 or pd.Series(groups).nunique() < 5:
            continue
        cv = GroupKFold(n_splits=min(5, pd.Series(groups).nunique()))
        yhat = np.full(len(d), np.nan)

        for fold, (tr, te) in enumerate(cv.split(x, y, groups=groups), 1):
            rf = RandomForestRegressor(
                n_estimators=140,
                min_samples_leaf=6,
                max_features=0.7,
                random_state=9000 + fold,
                n_jobs=-1,
            )
            rf.fit(x.iloc[tr], y[tr])
            pred = rf.predict(x.iloc[te])
            yhat[te] = pred
            base_r2 = r2_score(y[te], pred)
            rng = np.random.default_rng(10000 + fold)
            repeats = 10

            for feat in features:
                for rep in range(repeats):
                    xp = x.iloc[te].copy()
                    xp[feat] = rng.permutation(xp[feat].to_numpy())
                    drop = base_r2 - r2_score(y[te], rf.predict(xp))
                    var_rows.append(
                        {
                            "scale": scale,
                            "flux": flux,
                            "fold": fold,
                            "analysis_variable": feat,
                            "original_variable": original_map[feat],
                            "label": label_map[feat],
                            "group": group_map[feat],
                            "importance_drop_cv_r2": drop,
                            "baseline_fold_r2": base_r2,
                        }
                    )

            for group, feats in group_features.items():
                for rep in range(repeats):
                    xp = x.iloc[te].copy()
                    for feat in feats:
                        xp[feat] = rng.permutation(xp[feat].to_numpy())
                    drop = base_r2 - r2_score(y[te], rf.predict(xp))
                    group_rows.append(
                        {
                            "scale": scale,
                            "flux": flux,
                            "fold": fold,
                            "group": group,
                            "n_variables": len(feats),
                            "variables": ";".join(feats),
                            "importance_drop_cv_r2": drop,
                            "baseline_fold_r2": base_r2,
                        }
                    )

        perf_rows.append(
            {
                "scale": scale,
                "flux": flux,
                "n": len(d),
                "n_sites": d["SITE_ID"].nunique(),
                "cv_r2": r2_score(y, yhat),
                "cv_rmse": mean_squared_error(y, yhat, squared=False),
                "cv_mae": mean_absolute_error(y, yhat),
                "min_hls_days": min_days_for_scale(scale),
            }
        )

    return pd.DataFrame(perf_rows), pd.DataFrame(var_rows), pd.DataFrame(group_rows)


def summarize(raw: pd.DataFrame, keys: list[str], grouped: bool = False) -> pd.DataFrame:
    out = (
        raw.groupby(keys, as_index=False)
        .agg(
            importance_mean=("importance_drop_cv_r2", "mean"),
            importance_sd=("importance_drop_cv_r2", "std"),
            importance_q025=("importance_drop_cv_r2", lambda s: s.quantile(0.025)),
            importance_q975=("importance_drop_cv_r2", lambda s: s.quantile(0.975)),
            positive_frequency=("importance_drop_cv_r2", lambda s: float((s > 0).mean())),
            mean_baseline_fold_r2=("baseline_fold_r2", "mean"),
        )
    )
    out["positive_importance"] = out["importance_mean"].clip(lower=0)
    if grouped:
        out["importance_per_variable"] = out["importance_mean"] / out["n_variables"]
    out["relative_importance_percent"] = 0.0
    for (scale, flux), idx in out.groupby(["scale", "flux"]).groups.items():
        denom = out.loc[idx, "positive_importance"].sum()
        if denom > 0:
            out.loc[idx, "relative_importance_percent"] = 100 * out.loc[idx, "positive_importance"] / denom
    return out.sort_values(["scale", "flux", "relative_importance_percent"], ascending=[True, True, False])


def plot_scale(var_summary: pd.DataFrame, group_summary: pd.DataFrame, perf: pd.DataFrame, scale_dir: Path, scale: str) -> None:
    fig_dir = scale_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    palette = {
        "Surface heterogeneity": "#2A9D8F",
        "Footprint/turbulence dynamics": "#1D3557",
        "Canopy/surface structure": "#8D6E63",
        "Meteorological forcing": "#E76F51",
        "Soil hydrothermal state": "#6A994E",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2), sharex=True)
    for ax, flux in zip(axes, FLUXES):
        d = var_summary[(var_summary["scale"].eq(scale)) & (var_summary["flux"].eq(flux))].copy()
        d = d.sort_values("relative_importance_percent", ascending=True)
        ax.barh(d["label"], d["relative_importance_percent"], color=[palette.get(g, "0.5") for g in d["group"]])
        p = perf[(perf["scale"].eq(scale)) & (perf["flux"].eq(flux))]
        if len(p):
            pp = p.iloc[0]
            ax.text(
                0.98,
                0.05,
                f"CV R²={pp.cv_r2:.2f}\nn={int(pp.n):,}; sites={int(pp.n_sites)}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.8", "alpha": 0.9},
            )
        ax.set_title(flux)
        ax.set_xlabel("Relative variable importance (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=g) for g, c in palette.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{scale}: relative variable importance of VIF-screened predictors", y=1.02)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(fig_dir / f"Figure_A_relative_variable_importance_{scale}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True)
    color_map = dict(zip(sorted(group_summary["group"].unique()), sns.color_palette("Set2", n_colors=group_summary["group"].nunique())))
    for ax, flux in zip(axes, FLUXES):
        d = group_summary[(group_summary["scale"].eq(scale)) & (group_summary["flux"].eq(flux))].copy()
        d = d.sort_values("relative_importance_percent", ascending=True)
        ax.barh(d["group"], d["relative_importance_percent"], color=[color_map[g] for g in d["group"]])
        ax.set_title(flux)
        ax.set_xlabel("Relative grouped importance (%)")
    fig.suptitle(f"{scale}: supplementary grouped importance", y=1.02)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(fig_dir / f"FigS_relative_grouped_importance_{scale}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = build_mapping()
    mapping.to_csv(OUT / "vif15_variable_mapping_to_site_day.csv", index=False)
    day = load_site_day(mapping)
    features = mapping["analysis_variable"].tolist()

    all_perf, all_var_raw, all_group_raw = [], [], []
    scale_summaries = []
    for scale in SCALES:
        print(f"Running scale: {scale}")
        scale_dir = OUT / scale
        tab_dir = scale_dir / "tables"
        tab_dir.mkdir(parents=True, exist_ok=True)
        table = aggregate_scale(day, scale, features)
        table.to_csv(tab_dir / f"{scale}_delta_rmse_with_vif15_drivers.csv", index=False)
        model_df = model_input(table, features, scale)
        model_df.to_csv(tab_dir / f"{scale}_model_input_vif15.csv", index=False)
        sample = (
            table.groupby("flux", as_index=False)
            .agg(
                n_windows=("delta_RMSE", "size"),
                n_model_windows=("n_hls_days", lambda s: int((s >= min_days_for_scale(scale)).sum())),
                n_sites=("SITE_ID", "nunique"),
                median_hls_days=("n_hls_days", "median"),
                median_halfhours=("n_valid_halfhours", "median"),
                mean_delta_RMSE=("delta_RMSE", "mean"),
                median_delta_RMSE=("delta_RMSE", "median"),
                positive_fraction=("delta_RMSE", lambda s: float((s > 0).mean())),
            )
        )
        sample["scale"] = scale
        sample["min_hls_days"] = min_days_for_scale(scale)
        sample.to_csv(tab_dir / f"{scale}_sample_summary.csv", index=False)
        scale_summaries.append(sample)

        perf, var_raw, group_raw = run_importance(model_df, mapping, scale)
        var_summary = summarize(
            var_raw,
            ["scale", "flux", "analysis_variable", "original_variable", "label", "group"],
            grouped=False,
        )
        group_summary = summarize(
            group_raw,
            ["scale", "flux", "group", "n_variables", "variables"],
            grouped=True,
        )
        perf.to_csv(tab_dir / f"{scale}_rf_cross_site_performance.csv", index=False)
        var_raw.to_csv(tab_dir / f"{scale}_raw_variable_permutation_importance.csv", index=False)
        group_raw.to_csv(tab_dir / f"{scale}_raw_grouped_permutation_importance.csv", index=False)
        var_summary.to_csv(tab_dir / f"{scale}_relative_variable_importance.csv", index=False)
        group_summary.to_csv(tab_dir / f"{scale}_relative_grouped_importance_supplement.csv", index=False)
        plot_scale(var_summary, group_summary, perf, scale_dir, scale)

        all_perf.append(perf)
        all_var_raw.append(var_raw)
        all_group_raw.append(group_raw)

    perf_all = pd.concat(all_perf, ignore_index=True)
    var_all = summarize(
        pd.concat(all_var_raw, ignore_index=True),
        ["scale", "flux", "analysis_variable", "original_variable", "label", "group"],
        grouped=False,
    )
    group_all = summarize(
        pd.concat(all_group_raw, ignore_index=True),
        ["scale", "flux", "group", "n_variables", "variables"],
        grouped=True,
    )
    pd.concat(scale_summaries, ignore_index=True).to_csv(OUT / "all_scales_sample_summary.csv", index=False)
    perf_all.to_csv(OUT / "all_scales_rf_cross_site_performance.csv", index=False)
    var_all.to_csv(OUT / "all_scales_relative_variable_importance.csv", index=False)
    group_all.to_csv(OUT / "all_scales_relative_grouped_importance_supplement.csv", index=False)

    manifest = {
        "input_site_day": str(SITE_DAY),
        "input_vif_table": str(VIF_TABLE),
        "output": str(OUT),
        "scales": SCALES,
        "excluded_analysis_variables": sorted(EXCLUDE_ANALYSIS_VARIABLES),
        "min_hls_days": {s: min_days_for_scale(s) for s in SCALES},
        "note": "Each scale has separate tables and figures. Relative importance is normalized from positive permutation scores within each flux and scale.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

