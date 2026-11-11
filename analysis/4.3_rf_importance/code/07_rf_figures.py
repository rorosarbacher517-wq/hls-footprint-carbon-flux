"""
Publication-decision Figure A/B/C suite for multi-scale VIF13 no-wind-direction-sincos analyses.

This script consumes the scale-specific model-input and importance tables
created by 06_multiscale_relative_importance_vif13_no_wd_sincos.py and writes a comparable
Figure A/B/C set for each temporal scale:
  - Figure A: top-10 relative variable importance (%)
  - Figure B: one-dimensional RF partial-response curves for core variables
  - Figure C: NDVI heterogeneity x every other VIF13 variable response surface

The outputs are intended for manuscript-scale selection: use one scale in the
main text and keep the remaining scales as supplementary sensitivity analyses.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


ROOT = Path(
    os.environ.get(
        "RSE_MULTISCALE_ROOT",
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
ROOT = Path(__file__).resolve().parent.parent / "outputs"          # project folder (annual scale only)
SCALES = [
    s.strip()
    for s in os.environ.get("RSE_MULTISCALE_SCALES", "year").split(",")
    if s.strip()
]
FLUXES = ["NEE", "GPP", "RECO"]
FEATURE_NDVI = "NDVI_STD_DAY"
CORE_B_FEATURES = [
    "NDVI_STD_DAY",
    "dw_edge_density_median",
    "z0_day_median",
    "SSRD_Wm2_day_median",
]

GROUP_PALETTE = {
    "Surface heterogeneity": "#1B9E77",
    "Footprint/turbulence dynamics": "#377EB8",
    "Meteorological forcing": "#E6AB02",
    "Soil hydrothermal state": "#7570B3",
    "Canopy/surface structure": "#A6761D",
}

NATURE_RC = {
    "font.family": "Arial",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
}


def apply_nature_style() -> None:
    sns.set_theme(style="white", font_scale=1.0)
    plt.rcParams.update(NATURE_RC)


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.18, y: float = 1.08, fontsize: float = 9) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def flux_label(ax: plt.Axes, flux: str) -> None:
    ax.text(
        0.02,
        0.98,
        flux,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.2},
    )


def clean_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="0.88", linewidth=0.45)
    else:
        ax.grid(False)


def feature_meta(scale_dir: Path) -> pd.DataFrame:
    table = pd.read_csv(scale_dir / "tables" / f"{scale_dir.name}_relative_variable_importance.csv")
    return table[["analysis_variable", "label", "group", "original_variable"]].drop_duplicates()


def load_inputs(scale: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scale_dir = ROOT / scale
    model_df = pd.read_csv(scale_dir / "tables" / f"{scale}_model_input_vif15.csv")
    imp = pd.read_csv(scale_dir / "tables" / f"{scale}_relative_variable_importance.csv")
    perf = pd.read_csv(scale_dir / "tables" / f"{scale}_rf_cross_site_performance.csv")
    return model_df, imp, perf


def features_from_importance(imp: pd.DataFrame) -> list[str]:
    return imp["analysis_variable"].drop_duplicates().tolist()


def fit_rf(df: pd.DataFrame, features: list[str], seed: int) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        max_features=0.6,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(df[features], df["delta_RMSE"])
    return model


def observed_cv_r2(df: pd.DataFrame, features: list[str], seed: int) -> float:
    sites = df["SITE_ID"].astype(str)
    n_splits = min(5, sites.nunique())
    if n_splits < 2:
        return np.nan
    preds = np.full(len(df), np.nan)
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=n_splits).split(df, groups=sites)):
        model = fit_rf(df.iloc[train_idx], features, seed + fold)
        preds[test_idx] = model.predict(df.iloc[test_idx][features])
    mask = np.isfinite(preds)
    return float(r2_score(df.loc[mask, "delta_RMSE"], preds[mask])) if mask.any() else np.nan


def plot_figure_a_top10(scale: str, imp: pd.DataFrame, perf: pd.DataFrame) -> None:
    fig_dir = ROOT / scale / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    label_map = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")["label"].to_dict()
    group_map = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")["group"].to_dict()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.25), sharex=False)

    for i, (ax, flux) in enumerate(zip(axes, FLUXES)):
        d = (
            imp.loc[imp["flux"].eq(flux)]
            .sort_values("relative_importance_percent", ascending=False)
            .head(10)
            .iloc[::-1]
            .copy()
        )
        colors = [GROUP_PALETTE.get(group_map.get(v, ""), "#777777") for v in d["analysis_variable"]]
        ax.barh(d["label"], d["relative_importance_percent"], color=colors, edgecolor="0.25", linewidth=0.35)
        panel_letter(ax, chr(ord("a") + i), x=-0.24, y=1.06)
        flux_label(ax, flux)
        ax.set_xlabel("Relative variable importance (%)")
        ax.set_xlim(left=0)
        clean_axis(ax, grid_axis="x")

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=8, label=g)
        for g, c in GROUP_PALETTE.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"Figure A ({scale}): top-10 drivers of ΔRMSE gains", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / f"Figure_A_top10_relative_variable_importance_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_A_top10_relative_variable_importance_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)


def partial_curve(model: RandomForestRegressor, df: pd.DataFrame, features: list[str], feature: str) -> pd.DataFrame:
    lo, hi = df[feature].quantile([0.05, 0.95])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return pd.DataFrame()
    grid = np.linspace(lo, hi, 60)
    base = df[features].median(numeric_only=True).to_frame().T
    rows = []
    for x in grid:
        xp = pd.concat([base] * 200, ignore_index=True)
        xp[feature] = x
        rows.append({"x": x, "y": float(model.predict(xp[features]).mean())})
    return pd.DataFrame(rows)


def observed_bins(df: pd.DataFrame, feature: str, n_bins: int = 10) -> pd.DataFrame:
    q = df[feature].rank(method="first")
    try:
        bins = pd.qcut(q, q=min(n_bins, df[feature].nunique()), duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    out = (
        df.assign(_bin=bins)
        .groupby("_bin", observed=True)
        .agg(x=(feature, "median"), y=("delta_RMSE", "mean"), n=("delta_RMSE", "size"))
        .reset_index(drop=True)
    )
    return out


def plot_figure_b(scale: str, model_df: pd.DataFrame, imp: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    fig_dir = ROOT / scale / "figures"
    tab_dir = ROOT / scale / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")
    features = features_from_importance(imp)
    use_features = [f for f in CORE_B_FEATURES if f in features]
    records = []
    fig, axes = plt.subplots(len(FLUXES), len(use_features), figsize=(4.0 * len(use_features), 8.8), squeeze=False)

    for r, flux in enumerate(FLUXES):
        d = model_df.loc[model_df["flux"].eq(flux)].dropna(subset=["delta_RMSE"] + features).copy()
        if d.empty:
            continue
        cv_r2 = observed_cv_r2(d, features, seed=7100 + r)
        model = fit_rf(d, features, seed=7200 + r)
        for c, feat in enumerate(use_features):
            ax = axes[r, c]
            curve = partial_curve(model, d, features, feat)
            bins = observed_bins(d, feat)
            label = meta.loc[feat, "label"] if feat in meta.index else feat
            if not curve.empty:
                ax.plot(curve["x"], curve["y"], color="#222222", lw=2.0, label="RF partial response")
                tmp = curve.assign(scale=scale, flux=flux, feature=feat, label=label, cv_r2=cv_r2, n=len(d), n_sites=d["SITE_ID"].nunique())
                records.extend(tmp.to_dict(orient="records"))
            if not bins.empty:
                ax.scatter(bins["x"], bins["y"], s=np.sqrt(bins["n"]) * 8, color="#D95F02", alpha=0.75, label="Observed bins")
            ax.axhline(0, color="0.6", lw=0.8, ls="--")
            ax.set_title(f"{flux}: {label}\nn={len(d)}, sites={d['SITE_ID'].nunique()}, CV R2={cv_r2:.3f}", fontsize=9)
            ax.set_xlabel(label)
            ax.set_ylabel("Predicted ΔRMSE")
            ax.grid(alpha=0.22)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=8)

    fig.suptitle(f"Figure B ({scale}): marginal response of ΔRMSE gains", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / f"Figure_B_core_partial_response_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_B_core_partial_response_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)
    out = pd.DataFrame(records)
    out.to_csv(tab_dir / f"{scale}_Figure_B_core_partial_response_curves.csv", index=False)
    return out


def surface_grid(model: RandomForestRegressor, df: pd.DataFrame, features: list[str], xfeat: str, yfeat: str) -> pd.DataFrame:
    xlo, xhi = df[xfeat].quantile([0.05, 0.95])
    ylo, yhi = df[yfeat].quantile([0.05, 0.95])
    if not all(np.isfinite([xlo, xhi, ylo, yhi])) or xlo == xhi or ylo == yhi:
        return pd.DataFrame()
    xgrid = np.linspace(xlo, xhi, 35)
    ygrid = np.linspace(ylo, yhi, 35)
    base = df[features].median(numeric_only=True).to_frame().T
    rows = []
    for xv in xgrid:
        xp = pd.concat([base] * len(ygrid), ignore_index=True)
        xp[xfeat] = xv
        xp[yfeat] = ygrid
        pred = model.predict(xp[features])
        for yv, zv in zip(ygrid, pred):
            rows.append({"x": xv, "y": yv, "pred_delta_RMSE": float(zv)})
    return pd.DataFrame(rows)


def write_surface_figure(path: Path, grid: pd.DataFrame, flux: str, xlab: str, ylab: str, scale: str, cv_r2: float) -> None:
    pivot = grid.pivot(index="y", columns="x", values="pred_delta_RMSE")
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    mesh = ax.contourf(pivot.columns.values, pivot.index.values, pivot.values, levels=16, cmap="viridis")
    ax.contour(pivot.columns.values, pivot.index.values, pivot.values, colors="white", linewidths=0.35, alpha=0.65)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("Predicted ΔRMSE")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f"{flux}, {scale}: NDVI x {ylab}\nCV R2={cv_r2:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure_c(scale: str, model_df: pd.DataFrame, imp: pd.DataFrame) -> pd.DataFrame:
    fig_dir = ROOT / scale / "figures"
    tab_dir = ROOT / scale / "tables"
    surf_dir = fig_dir / "Figure_C_NDVI_x_all_partners"
    surf_tab = tab_dir / "Figure_C_NDVI_x_all_partners"
    surf_dir.mkdir(parents=True, exist_ok=True)
    surf_tab.mkdir(parents=True, exist_ok=True)

    meta = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")
    features = features_from_importance(imp)
    partners = [f for f in features if f != FEATURE_NDVI]
    records = []

    for fi, flux in enumerate(FLUXES):
        d = model_df.loc[model_df["flux"].eq(flux)].dropna(subset=["delta_RMSE"] + features).copy()
        if d.empty:
            continue
        cv_r2 = observed_cv_r2(d, features, seed=8100 + fi)
        model = fit_rf(d, features, seed=8200 + fi)
        xlab = meta.loc[FEATURE_NDVI, "label"] if FEATURE_NDVI in meta.index else FEATURE_NDVI
        for partner in partners:
            ylab = meta.loc[partner, "label"] if partner in meta.index else partner
            grid = surface_grid(model, d, features, FEATURE_NDVI, partner)
            if grid.empty:
                continue
            grid = grid.assign(scale=scale, flux=flux, x_feature=FEATURE_NDVI, y_feature=partner, x_label=xlab, y_label=ylab, cv_r2=cv_r2)
            safe = partner.replace("/", "_").replace("\\", "_")
            grid.to_csv(surf_tab / f"{scale}_{flux}_NDVI_x_{safe}_surface.csv", index=False)
            write_surface_figure(surf_dir / f"Figure_C_{scale}_{flux}_NDVI_x_{safe}.png", grid, flux, xlab, ylab, scale, cv_r2)
            z = grid["pred_delta_RMSE"]
            low_low = grid.loc[(grid["x"].eq(grid["x"].min())) & (grid["y"].eq(grid["y"].min())), "pred_delta_RMSE"].mean()
            high_high = grid.loc[(grid["x"].eq(grid["x"].max())) & (grid["y"].eq(grid["y"].max())), "pred_delta_RMSE"].mean()
            records.append(
                {
                    "scale": scale,
                    "flux": flux,
                    "partner": partner,
                    "partner_label": ylab,
                    "cv_r2": cv_r2,
                    "surface_range": float(z.max() - z.min()),
                    "high_high_minus_low_low": float(high_high - low_low),
                    "pred_min": float(z.min()),
                    "pred_max": float(z.max()),
                }
            )

    summary = pd.DataFrame(records)
    summary.to_csv(tab_dir / f"{scale}_Figure_C_NDVI_x_all_partner_surface_summary.csv", index=False)
    if not summary.empty:
        plot_figure_c_top_summary(scale, summary, surf_tab, fig_dir)
    return summary


def plot_figure_c_top_summary(scale: str, summary: pd.DataFrame, surf_tab: Path, fig_dir: Path) -> None:
    chosen = summary.sort_values(["flux", "surface_range"], ascending=[True, False]).groupby("flux", as_index=False).head(2)
    if chosen.empty:
        return
    fig, axes = plt.subplots(len(FLUXES), 2, figsize=(10.5, 10.0), squeeze=False)
    for r, flux in enumerate(FLUXES):
        rows = chosen.loc[chosen["flux"].eq(flux)].reset_index(drop=True)
        for c in range(2):
            ax = axes[r, c]
            if c >= len(rows):
                ax.axis("off")
                continue
            row = rows.iloc[c]
            safe = row["partner"].replace("/", "_").replace("\\", "_")
            grid = pd.read_csv(surf_tab / f"{scale}_{flux}_NDVI_x_{safe}_surface.csv")
            pivot = grid.pivot(index="y", columns="x", values="pred_delta_RMSE")
            mesh = ax.contourf(pivot.columns.values, pivot.index.values, pivot.values, levels=14, cmap="viridis")
            ax.set_xlabel("NDVI heterogeneity")
            ax.set_ylabel(row["partner_label"])
            ax.set_title(
                f"{flux}: {row['partner_label']}\nrange={row['surface_range']:.3f}, high-high - low-low={row['high_high_minus_low_low']:.3f}",
                fontsize=10,
            )
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Predicted ΔRMSE", shrink=0.75)
    fig.suptitle(f"Figure C ({scale}): strongest NDVI interaction surfaces", y=0.98, fontsize=13)
    fig.savefig(fig_dir / f"Figure_C_top_NDVI_interaction_surfaces_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_C_top_NDVI_interaction_surfaces_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_figure_a_top10(scale: str, imp: pd.DataFrame, perf: pd.DataFrame) -> None:
    fig_dir = ROOT / scale / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    group_map = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")["group"].to_dict()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.25), sharex=False)

    for i, (ax, flux) in enumerate(zip(axes, FLUXES)):
        d = (
            imp.loc[imp["flux"].eq(flux)]
            .sort_values("relative_importance_percent", ascending=False)
            .head(10)
            .iloc[::-1]
            .copy()
        )
        colors = [GROUP_PALETTE.get(group_map.get(v, ""), "#777777") for v in d["analysis_variable"]]
        ax.barh(d["label"], d["relative_importance_percent"], color=colors, edgecolor="0.25", linewidth=0.35)
        panel_letter(ax, chr(ord("a") + i), x=-0.24, y=1.06)
        flux_label(ax, flux)
        ax.set_xlabel("Relative variable importance (%)")
        ax.set_xlim(left=0)
        clean_axis(ax, grid_axis="x")

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=6, label=g)
        for g, c in GROUP_PALETTE.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(fig_dir / f"Figure_A_top10_relative_variable_importance_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_A_top10_relative_variable_importance_{scale}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_A_top10_relative_variable_importance_{scale}.png", dpi=600, bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_A_top10_relative_variable_importance_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_figure_b(scale: str, model_df: pd.DataFrame, imp: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    fig_dir = ROOT / scale / "figures"
    tab_dir = ROOT / scale / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")
    features = features_from_importance(imp)
    use_features = [f for f in CORE_B_FEATURES if f in features]
    records = []
    fig, axes = plt.subplots(len(FLUXES), len(use_features), figsize=(7.25, 5.7), squeeze=False)
    letter_idx = 0

    for r, flux in enumerate(FLUXES):
        d = model_df.loc[model_df["flux"].eq(flux)].dropna(subset=["delta_RMSE"] + features).copy()
        if d.empty:
            continue
        cv_r2 = observed_cv_r2(d, features, seed=7100 + r)
        model = fit_rf(d, features, seed=7200 + r)
        for c, feat in enumerate(use_features):
            ax = axes[r, c]
            curve = partial_curve(model, d, features, feat)
            bins = observed_bins(d, feat)
            label = meta.loc[feat, "label"] if feat in meta.index else feat
            if not curve.empty:
                ax.plot(curve["x"], curve["y"], color="#222222", lw=1.2, label="RF partial response")
                tmp = curve.assign(scale=scale, flux=flux, feature=feat, label=label, cv_r2=cv_r2, n=len(d), n_sites=d["SITE_ID"].nunique())
                records.extend(tmp.to_dict(orient="records"))
            if not bins.empty:
                ax.scatter(bins["x"], bins["y"], s=np.sqrt(bins["n"]) * 4.2, color="#D95F02", alpha=0.72, linewidth=0, label="Observed bins")
            ax.axhline(0, color="0.55", lw=0.55, ls="--")
            panel_letter(ax, chr(ord("a") + letter_idx), x=-0.2, y=1.08)
            letter_idx += 1
            flux_label(ax, flux)
            ax.set_xlabel(label)
            ax.set_ylabel("Predicted ΔRMSE")
            clean_axis(ax, grid_axis="y")
            if r == 0 and c == 0:
                ax.legend(frameon=False, loc="best", handlelength=1.6)

    fig.tight_layout()
    fig.savefig(fig_dir / f"Figure_B_core_partial_response_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_B_core_partial_response_{scale}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_B_core_partial_response_{scale}.png", dpi=600, bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_B_core_partial_response_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)
    out = pd.DataFrame(records)
    out.to_csv(tab_dir / f"{scale}_Figure_B_core_partial_response_curves.csv", index=False)
    return out


def write_surface_figure(path: Path, grid: pd.DataFrame, flux: str, xlab: str, ylab: str, scale: str, cv_r2: float) -> None:
    pivot = grid.pivot(index="y", columns="x", values="pred_delta_RMSE")
    fig, ax = plt.subplots(figsize=(3.35, 2.7))
    mesh = ax.contourf(pivot.columns.values, pivot.index.values, pivot.values, levels=16, cmap="viridis")
    ax.contour(pivot.columns.values, pivot.index.values, pivot.values, colors="white", linewidths=0.25, alpha=0.55)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("Predicted ΔRMSE")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    flux_label(ax, flux)
    clean_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure_c_top_summary(scale: str, summary: pd.DataFrame, surf_tab: Path, fig_dir: Path) -> None:
    chosen = summary.sort_values(["flux", "surface_range"], ascending=[True, False]).groupby("flux", as_index=False).head(2)
    if chosen.empty:
        return
    fig, axes = plt.subplots(len(FLUXES), 2, figsize=(7.2, 6.2), squeeze=False)
    letter_idx = 0
    mesh = None
    for r, flux in enumerate(FLUXES):
        rows = chosen.loc[chosen["flux"].eq(flux)].reset_index(drop=True)
        for c in range(2):
            ax = axes[r, c]
            if c >= len(rows):
                ax.axis("off")
                continue
            row = rows.iloc[c]
            safe = row["partner"].replace("/", "_").replace("\\", "_")
            grid = pd.read_csv(surf_tab / f"{scale}_{flux}_NDVI_x_{safe}_surface.csv")
            pivot = grid.pivot(index="y", columns="x", values="pred_delta_RMSE")
            mesh = ax.contourf(pivot.columns.values, pivot.index.values, pivot.values, levels=14, cmap="viridis")
            ax.contour(pivot.columns.values, pivot.index.values, pivot.values, colors="white", linewidths=0.22, alpha=0.45)
            panel_letter(ax, chr(ord("a") + letter_idx), x=-0.16, y=1.06)
            letter_idx += 1
            flux_label(ax, flux)
            ax.set_xlabel("NDVI heterogeneity")
            ax.set_ylabel(row["partner_label"])
            clean_axis(ax)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Predicted ΔRMSE", shrink=0.72, pad=0.02)
    fig.savefig(fig_dir / f"Figure_C_top_NDVI_interaction_surfaces_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_C_top_NDVI_interaction_surfaces_{scale}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_C_top_NDVI_interaction_surfaces_{scale}.png", dpi=600, bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_C_top_NDVI_interaction_surfaces_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_figure_b(scale: str, model_df: pd.DataFrame, imp: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    fig_dir = ROOT / scale / "figures"
    tab_dir = ROOT / scale / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")
    features = features_from_importance(imp)
    use_features = [f for f in CORE_B_FEATURES if f in features]
    records = []
    fig, axes = plt.subplots(len(FLUXES), len(use_features), figsize=(7.25, 5.85), squeeze=False)
    letter_idx = 0
    line_handle = None
    point_handle = None

    for r, flux in enumerate(FLUXES):
        d = model_df.loc[model_df["flux"].eq(flux)].dropna(subset=["delta_RMSE"] + features).copy()
        if d.empty:
            continue
        cv_r2 = observed_cv_r2(d, features, seed=7100 + r)
        model = fit_rf(d, features, seed=7200 + r)
        for c, feat in enumerate(use_features):
            ax = axes[r, c]
            curve = partial_curve(model, d, features, feat)
            bins = observed_bins(d, feat)
            label = meta.loc[feat, "label"] if feat in meta.index else feat
            if not curve.empty:
                (line_handle,) = ax.plot(curve["x"], curve["y"], color="#222222", lw=1.2, label="RF partial response")
                tmp = curve.assign(scale=scale, flux=flux, feature=feat, label=label, cv_r2=cv_r2, n=len(d), n_sites=d["SITE_ID"].nunique())
                records.extend(tmp.to_dict(orient="records"))
            if not bins.empty:
                point_handle = ax.scatter(bins["x"], bins["y"], s=np.sqrt(bins["n"]) * 4.2, color="#D95F02", alpha=0.72, linewidth=0, label="Observed bins")
            ax.axhline(0, color="0.55", lw=0.55, ls="--")
            panel_letter(ax, chr(ord("a") + letter_idx), x=-0.2, y=1.08)
            letter_idx += 1
            flux_label(ax, flux)
            ax.set_xlabel(label)
            ax.set_ylabel("Predicted ΔRMSE")
            clean_axis(ax, grid_axis="y")

    if line_handle is not None and point_handle is not None:
        fig.legend([line_handle, point_handle], ["RF partial response", "Observed bins"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(fig_dir / f"Figure_B_core_partial_response_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_B_core_partial_response_{scale}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_B_core_partial_response_{scale}.png", dpi=600, bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_B_core_partial_response_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)
    out = pd.DataFrame(records)
    out.to_csv(tab_dir / f"{scale}_Figure_B_core_partial_response_curves.csv", index=False)
    return out


def plot_figure_c_top_summary(scale: str, summary: pd.DataFrame, surf_tab: Path, fig_dir: Path) -> None:
    chosen = summary.sort_values(["flux", "surface_range"], ascending=[True, False]).groupby("flux", as_index=False).head(2)
    if chosen.empty:
        return
    fig, axes = plt.subplots(len(FLUXES), 2, figsize=(7.2, 6.35), squeeze=False)
    letter_idx = 0
    mesh = None
    for r, flux in enumerate(FLUXES):
        rows = chosen.loc[chosen["flux"].eq(flux)].reset_index(drop=True)
        for c in range(2):
            ax = axes[r, c]
            if c >= len(rows):
                ax.axis("off")
                continue
            row = rows.iloc[c]
            safe = row["partner"].replace("/", "_").replace("\\", "_")
            grid = pd.read_csv(surf_tab / f"{scale}_{flux}_NDVI_x_{safe}_surface.csv")
            pivot = grid.pivot(index="y", columns="x", values="pred_delta_RMSE")
            mesh = ax.contourf(pivot.columns.values, pivot.index.values, pivot.values, levels=14, cmap="viridis")
            ax.contour(pivot.columns.values, pivot.index.values, pivot.values, colors="white", linewidths=0.22, alpha=0.45)
            panel_letter(ax, chr(ord("a") + letter_idx), x=-0.16, y=1.06)
            letter_idx += 1
            flux_label(ax, flux)
            ax.set_xlabel("NDVI heterogeneity" if r == len(FLUXES) - 1 else "")
            ax.set_ylabel(row["partner_label"])
            clean_axis(ax)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Predicted ΔRMSE", shrink=0.72, pad=0.02)
    fig.savefig(fig_dir / f"Figure_C_top_NDVI_interaction_surfaces_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_C_top_NDVI_interaction_surfaces_{scale}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_C_top_NDVI_interaction_surfaces_{scale}.png", dpi=600, bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_C_top_NDVI_interaction_surfaces_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)


def flux_label(ax: plt.Axes, flux: str) -> None:
    ax.text(
        0.02,
        1.015,
        flux,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        clip_on=False,
    )


def top_features_for_flux(imp: pd.DataFrame, flux: str, n: int = 5) -> list[str]:
    d = (
        imp.loc[imp["flux"].eq(flux)]
        .sort_values(["relative_importance_percent", "positive_frequency"], ascending=False)
        .head(n)
    )
    return d["analysis_variable"].tolist()


def representative_ndvi_partners(imp: pd.DataFrame, flux: str, n: int = 4) -> list[str]:
    """Return the top non-NDVI predictors for NDVI interaction surfaces."""
    d = imp.loc[imp["flux"].eq(flux) & ~imp["analysis_variable"].eq(FEATURE_NDVI)].copy()
    d = d.sort_values(
        ["relative_importance_percent", "positive_frequency"],
        ascending=False,
    )
    return d.head(n)["analysis_variable"].tolist()


def plot_figure_a_top10(scale: str, imp: pd.DataFrame, perf: pd.DataFrame) -> None:
    fig_dir = ROOT / scale / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    group_map = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")["group"].to_dict()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.15), sharex=False)

    for i, (ax, flux) in enumerate(zip(axes, FLUXES)):
        d = (
            imp.loc[imp["flux"].eq(flux)]
            .sort_values("relative_importance_percent", ascending=False)
            .loc[lambda x: x["relative_importance_percent"] > 0]      # variables without importance are not shown
            .head(10)
            .iloc[::-1]
            .copy()
        )
        colors = [GROUP_PALETTE.get(group_map.get(v, ""), "#777777") for v in d["analysis_variable"]]
        ax.barh(d["label"], d["relative_importance_percent"], color=colors, edgecolor="0.25", linewidth=0.35)
        panel_letter(ax, chr(ord("a") + i), x=-0.24, y=1.08)
        flux_label(ax, flux)
        ax.set_xlabel("Relative variable importance (%)")
        ax.set_xlim(left=0)
        ax.margins(y=0.12)
        clean_axis(ax, grid_axis="x")

    shown = set(imp.loc[imp["relative_importance_percent"] > 0, "group"])
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=5.5, label=g)
        for g, c in GROUP_PALETTE.items() if g in shown
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=0.9,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(fig_dir / f"Figure_A_top10_relative_variable_importance_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_A_top10_relative_variable_importance_{scale}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_A_top10_relative_variable_importance_{scale}.png", dpi=600, bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_A_top10_relative_variable_importance_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_figure_b(scale: str, model_df: pd.DataFrame, imp: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    fig_dir = ROOT / scale / "figures"
    tab_dir = ROOT / scale / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")
    features = features_from_importance(imp)
    flux_features = {flux: [f for f in top_features_for_flux(imp, flux, 5) if f in features] for flux in FLUXES}
    max_cols = max(len(v) for v in flux_features.values())
    records = []
    fig, axes = plt.subplots(len(FLUXES), max_cols, figsize=(8.8, 5.85), squeeze=False)
    letter_idx = 0
    line_handle = None
    point_handle = None

    for r, flux in enumerate(FLUXES):
        d = model_df.loc[model_df["flux"].eq(flux)].dropna(subset=["delta_RMSE"] + features).copy()
        if d.empty:
            continue
        cv_r2 = observed_cv_r2(d, features, seed=7100 + r)
        model = fit_rf(d, features, seed=7200 + r)
        for c in range(max_cols):
            ax = axes[r, c]
            if c >= len(flux_features[flux]):
                ax.axis("off")
                continue
            feat = flux_features[flux][c]
            curve = partial_curve(model, d, features, feat)
            bins = observed_bins(d, feat)
            label = meta.loc[feat, "label"] if feat in meta.index else feat
            if not curve.empty:
                (line_handle,) = ax.plot(curve["x"], curve["y"], color="#222222", lw=1.2, label="RF partial response")
                tmp = curve.assign(scale=scale, flux=flux, feature=feat, label=label, cv_r2=cv_r2, n=len(d), n_sites=d["SITE_ID"].nunique())
                records.extend(tmp.to_dict(orient="records"))
            if not bins.empty:
                point_handle = ax.scatter(bins["x"], bins["y"], s=np.sqrt(bins["n"]) * 4.2, color="#D95F02", alpha=0.72, linewidth=0, label="Observed bins")
            ax.axhline(0, color="0.55", lw=0.55, ls="--")
            panel_letter(ax, chr(ord("a") + letter_idx), x=-0.2, y=1.08)
            letter_idx += 1
            flux_label(ax, flux)
            ax.set_xlabel(label)
            ax.set_ylabel("Predicted ΔRMSE")
            clean_axis(ax, grid_axis="y")

    if line_handle is not None and point_handle is not None:
        fig.legend([line_handle, point_handle], ["RF partial response", "Observed bins"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(fig_dir / f"Figure_B_core_partial_response_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_B_core_partial_response_{scale}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_B_core_partial_response_{scale}.png", dpi=600, bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_B_core_partial_response_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)
    out = pd.DataFrame(records)
    out.to_csv(tab_dir / f"{scale}_Figure_B_top5_partial_response_curves.csv", index=False)
    out.to_csv(tab_dir / f"{scale}_Figure_B_core_partial_response_curves.csv", index=False)
    return out


def plot_figure_c_top_summary(scale: str, summary: pd.DataFrame, surf_tab: Path, fig_dir: Path) -> None:
    imp_path = fig_dir.parent / "tables" / f"{scale}_relative_variable_importance.csv"
    imp = pd.read_csv(imp_path)
    label_map = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")["label"].to_dict()
    chosen_rows = []
    for flux in FLUXES:
        for partner in representative_ndvi_partners(imp, flux):
            row = summary.loc[summary["flux"].eq(flux) & summary["partner"].eq(partner)]
            if not row.empty:
                chosen_rows.append(row.iloc[0].to_dict())
    chosen = pd.DataFrame(chosen_rows)
    if chosen.empty:
        return

    max_cols = max(len(chosen.loc[chosen["flux"].eq(flux)]) for flux in FLUXES)
    fig, axes = plt.subplots(len(FLUXES), max_cols, figsize=(11.2, 6.35), squeeze=False)
    letter_idx = 0
    mesh = None
    for r, flux in enumerate(FLUXES):
        rows = chosen.loc[chosen["flux"].eq(flux)].reset_index(drop=True)
        for c in range(max_cols):
            ax = axes[r, c]
            if c >= len(rows):
                ax.axis("off")
                continue
            row = rows.iloc[c]
            safe = row["partner"].replace("/", "_").replace("\\", "_")
            grid = pd.read_csv(surf_tab / f"{scale}_{flux}_NDVI_x_{safe}_surface.csv")
            pivot = grid.pivot(index="y", columns="x", values="pred_delta_RMSE")
            mesh = ax.contourf(pivot.columns.values, pivot.index.values, pivot.values, levels=14, cmap="viridis")
            ax.contour(pivot.columns.values, pivot.index.values, pivot.values, colors="white", linewidths=0.22, alpha=0.45)
            panel_letter(ax, chr(ord("a") + letter_idx), x=-0.16, y=1.06)
            letter_idx += 1
            flux_label(ax, flux)
            ax.set_xlabel("NDVI heterogeneity" if r == len(FLUXES) - 1 else "")
            ax.set_ylabel(label_map.get(row["partner"], row["partner_label"]), fontsize=6, labelpad=1.5)
            clean_axis(ax)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Predicted ΔRMSE", shrink=0.72, pad=0.02)
    chosen.to_csv(fig_dir.parent / "tables" / f"{scale}_Figure_C_representative_group_interactions.csv", index=False)
    fig.savefig(fig_dir / f"Figure_C_top_NDVI_interaction_surfaces_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_C_top_NDVI_interaction_surfaces_{scale}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_C_top_NDVI_interaction_surfaces_{scale}.png", dpi=600, bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_C_top_NDVI_interaction_surfaces_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)


def row_flux_label(ax: plt.Axes, flux: str, x: float = -0.38) -> None:
    ax.text(
        x,
        0.5,
        flux,
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )


def draw_b_axis(
    ax: plt.Axes,
    model: RandomForestRegressor,
    d: pd.DataFrame,
    features: list[str],
    feat: str,
    label: str,
    letter: str,
    records: list[dict],
    scale: str,
    flux: str,
    cv_r2: float,
) -> tuple[object | None, object | None]:
    curve = partial_curve(model, d, features, feat)
    bins = observed_bins(d, feat)
    line_handle = None
    point_handle = None
    if not curve.empty:
        (line_handle,) = ax.plot(curve["x"], curve["y"], color="#222222", lw=1.2, label="RF partial response")
        tmp = curve.assign(scale=scale, flux=flux, feature=feat, label=label, cv_r2=cv_r2, n=len(d), n_sites=d["SITE_ID"].nunique())
        records.extend(tmp.to_dict(orient="records"))
    if not bins.empty:
        point_handle = ax.scatter(bins["x"], bins["y"], s=np.sqrt(bins["n"]) * 4.2, color="#D95F02", alpha=0.72, linewidth=0, label="Observed bins")
    ax.axhline(0, color="0.55", lw=0.55, ls="--")
    panel_letter(ax, letter, x=-0.2, y=1.08)
    ax.set_xlabel(label)
    ax.set_ylabel("Predicted ΔRMSE")
    clean_axis(ax, grid_axis="y")
    return line_handle, point_handle


def plot_figure_b(scale: str, model_df: pd.DataFrame, imp: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    fig_dir = ROOT / scale / "figures"
    tab_dir = ROOT / scale / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")
    features = features_from_importance(imp)
    flux_features = {flux: [f for f in top_features_for_flux(imp, flux, 5) if f in features] for flux in FLUXES}
    max_cols = max(len(v) for v in flux_features.values())
    records = []
    fig, axes = plt.subplots(len(FLUXES), max_cols, figsize=(8.8, 5.85), squeeze=False)
    letter_idx = 0
    line_handle = None
    point_handle = None

    for r, flux in enumerate(FLUXES):
        d = model_df.loc[model_df["flux"].eq(flux)].dropna(subset=["delta_RMSE"] + features).copy()
        if d.empty:
            continue
        cv_r2 = observed_cv_r2(d, features, seed=7100 + r)
        model = fit_rf(d, features, seed=7200 + r)
        row_flux_label(axes[r, 0], flux, x=-0.46)
        for c in range(max_cols):
            ax = axes[r, c]
            if c >= len(flux_features[flux]):
                ax.axis("off")
                continue
            feat = flux_features[flux][c]
            label = meta.loc[feat, "label"] if feat in meta.index else feat
            lh, ph = draw_b_axis(
                ax, model, d, features, feat, label, chr(ord("a") + letter_idx), records, scale, flux, cv_r2
            )
            letter_idx += 1
            line_handle = lh or line_handle
            point_handle = ph or point_handle

    if line_handle is not None and point_handle is not None:
        fig.legend([line_handle, point_handle], ["RF partial response", "Observed bins"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.035, 0.045, 1, 1))
    fig.savefig(fig_dir / f"Figure_B_core_partial_response_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_B_core_partial_response_{scale}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_B_core_partial_response_{scale}.png", dpi=600, bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_B_core_partial_response_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)

    for r, flux in enumerate(FLUXES):
        d = model_df.loc[model_df["flux"].eq(flux)].dropna(subset=["delta_RMSE"] + features).copy()
        if d.empty:
            continue
        cv_r2 = observed_cv_r2(d, features, seed=9100 + r)
        model = fit_rf(d, features, seed=9200 + r)
        fig_one, axes_one = plt.subplots(1, len(flux_features[flux]), figsize=(8.8, 2.25), squeeze=False)
        row_flux_label(axes_one[0, 0], flux, x=-0.50)
        one_line = None
        one_point = None
        for c, feat in enumerate(flux_features[flux]):
            ax = axes_one[0, c]
            label = meta.loc[feat, "label"] if feat in meta.index else feat
            lh, ph = draw_b_axis(ax, model, d, features, feat, label, chr(ord("a") + c), records, scale, flux, cv_r2)
            one_line = lh or one_line
            one_point = ph or one_point
        if one_line is not None and one_point is not None:
            fig_one.legend([one_line, one_point], ["RF partial response", "Observed bins"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
        fig_one.tight_layout(rect=(0.035, 0.1, 1, 1))
        fig_one.savefig(fig_dir / f"Nature_Figure_B_top5_partial_response_{scale}_{flux}.png", dpi=600, bbox_inches="tight")
        fig_one.savefig(fig_dir / f"Nature_Figure_B_top5_partial_response_{scale}_{flux}.pdf", bbox_inches="tight")
        plt.close(fig_one)

    out = pd.DataFrame(records).drop_duplicates(subset=["scale", "flux", "feature", "x", "y"], keep="first")
    out.to_csv(tab_dir / f"{scale}_Figure_B_top5_partial_response_curves.csv", index=False)
    out.to_csv(tab_dir / f"{scale}_Figure_B_core_partial_response_curves.csv", index=False)
    return out


def draw_c_axis(
    ax: plt.Axes,
    grid: pd.DataFrame,
    y_label: str,
    letter: str,
    show_xlabel: bool,
) -> object:
    pivot = grid.pivot(index="y", columns="x", values="pred_delta_RMSE")
    mesh = ax.contourf(pivot.columns.values, pivot.index.values, pivot.values, levels=14, cmap="viridis")
    ax.contour(pivot.columns.values, pivot.index.values, pivot.values, colors="white", linewidths=0.22, alpha=0.45)
    panel_letter(ax, letter, x=-0.16, y=1.06)
    ax.set_xlabel("NDVI heterogeneity" if show_xlabel else "")
    ax.set_ylabel(y_label, fontsize=6, labelpad=4.0)
    clean_axis(ax)
    return mesh


def plot_figure_c_top_summary(scale: str, summary: pd.DataFrame, surf_tab: Path, fig_dir: Path) -> None:
    imp_path = fig_dir.parent / "tables" / f"{scale}_relative_variable_importance.csv"
    imp = pd.read_csv(imp_path)
    label_map = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")["label"].to_dict()
    chosen_rows = []
    for flux in FLUXES:
        for partner in representative_ndvi_partners(imp, flux):
            row = summary.loc[summary["flux"].eq(flux) & summary["partner"].eq(partner)]
            if not row.empty:
                chosen_rows.append(row.iloc[0].to_dict())
    chosen = pd.DataFrame(chosen_rows)
    if chosen.empty:
        return

    max_cols = max(len(chosen.loc[chosen["flux"].eq(flux)]) for flux in FLUXES)
    fig, axes = plt.subplots(len(FLUXES), max_cols, figsize=(15.0, 6.6), squeeze=False)
    letter_idx = 0
    mesh = None
    for r, flux in enumerate(FLUXES):
        rows = chosen.loc[chosen["flux"].eq(flux)].reset_index(drop=True)
        row_flux_label(axes[r, 0], flux, x=-0.72)
        for c in range(max_cols):
            ax = axes[r, c]
            if c >= len(rows):
                ax.axis("off")
                continue
            row = rows.iloc[c]
            safe = row["partner"].replace("/", "_").replace("\\", "_")
            grid = pd.read_csv(surf_tab / f"{scale}_{flux}_NDVI_x_{safe}_surface.csv")
            mesh = draw_c_axis(
                ax,
                grid,
                label_map.get(row["partner"], row["partner_label"]),
                chr(ord("a") + letter_idx),
                show_xlabel=(r == len(FLUXES) - 1),
            )
            letter_idx += 1
    chosen.to_csv(fig_dir.parent / "tables" / f"{scale}_Figure_C_representative_group_interactions.csv", index=False)
    fig.subplots_adjust(left=0.085, right=0.86, bottom=0.09, top=0.97, wspace=0.42, hspace=0.28)
    if mesh is not None:
        cax = fig.add_axes([0.885, 0.26, 0.012, 0.48])
        fig.colorbar(mesh, cax=cax, label="Predicted ΔRMSE")
    fig.savefig(fig_dir / f"Figure_C_top_NDVI_interaction_surfaces_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_C_top_NDVI_interaction_surfaces_{scale}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_C_top_NDVI_interaction_surfaces_{scale}.png", dpi=600, bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_C_top_NDVI_interaction_surfaces_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)

    for flux in FLUXES:
        rows = chosen.loc[chosen["flux"].eq(flux)].reset_index(drop=True)
        if rows.empty:
            continue
        fig_one, axes_one = plt.subplots(1, len(rows), figsize=(11.0, 2.35), squeeze=False)
        one_mesh = None
        row_flux_label(axes_one[0, 0], flux, x=-0.58)
        for c, row in rows.iterrows():
            ax = axes_one[0, c]
            safe = row["partner"].replace("/", "_").replace("\\", "_")
            grid = pd.read_csv(surf_tab / f"{scale}_{flux}_NDVI_x_{safe}_surface.csv")
            one_mesh = draw_c_axis(
                ax,
                grid,
                label_map.get(row["partner"], row["partner_label"]),
                chr(ord("a") + c),
                show_xlabel=True,
            )
        if one_mesh is not None:
            fig_one.colorbar(one_mesh, ax=axes_one.ravel().tolist(), label="Predicted ΔRMSE", shrink=0.76, pad=0.018)
        fig_one.savefig(fig_dir / f"Nature_Figure_C_representative_NDVI_interactions_{scale}_{flux}.png", dpi=600, bbox_inches="tight")
        fig_one.savefig(fig_dir / f"Nature_Figure_C_representative_NDVI_interactions_{scale}_{flux}.pdf", bbox_inches="tight")
        plt.close(fig_one)


def draw_b_axis(
    ax: plt.Axes,
    model: RandomForestRegressor,
    d: pd.DataFrame,
    features: list[str],
    feat: str,
    label: str,
    letter: str,
    records: list[dict],
    scale: str,
    flux: str,
    cv_r2: float,
    show_ylabel: bool = False,
) -> tuple[object | None, object | None]:
    curve = partial_curve(model, d, features, feat)
    bins = observed_bins(d, feat)
    line_handle = None
    point_handle = None
    if not curve.empty:
        (line_handle,) = ax.plot(curve["x"], curve["y"], color="#222222", lw=1.2, label="RF partial response")
        tmp = curve.assign(scale=scale, flux=flux, feature=feat, label=label, cv_r2=cv_r2, n=len(d), n_sites=d["SITE_ID"].nunique())
        records.extend(tmp.to_dict(orient="records"))
    if not bins.empty:
        point_handle = ax.scatter(bins["x"], bins["y"], s=np.sqrt(bins["n"]) * 4.2, color="#D95F02", alpha=0.72, linewidth=0, label="Observed bins")
    ax.axhline(0, color="0.55", lw=0.55, ls="--")
    panel_letter(ax, letter, x=-0.2, y=1.08)
    ax.set_xlabel(label)
    ax.set_ylabel("ΔRMSE" if show_ylabel else "")
    clean_axis(ax, grid_axis="y")
    return line_handle, point_handle


def plot_figure_b(scale: str, model_df: pd.DataFrame, imp: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    fig_dir = ROOT / scale / "figures"
    tab_dir = ROOT / scale / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")
    features = features_from_importance(imp)
    flux_features = {flux: [f for f in top_features_for_flux(imp, flux, 5) if f in features] for flux in FLUXES}
    max_cols = max(len(v) for v in flux_features.values())
    records = []
    fig, axes = plt.subplots(len(FLUXES), max_cols, figsize=(8.8, 5.85), squeeze=False)
    letter_idx = 0
    line_handle = None
    point_handle = None

    for r, flux in enumerate(FLUXES):
        d = model_df.loc[model_df["flux"].eq(flux)].dropna(subset=["delta_RMSE"] + features).copy()
        if d.empty:
            continue
        cv_r2 = observed_cv_r2(d, features, seed=7100 + r)
        model = fit_rf(d, features, seed=7200 + r)
        row_flux_label(axes[r, 0], flux, x=-0.42)
        for c in range(max_cols):
            ax = axes[r, c]
            if c >= len(flux_features[flux]):
                ax.axis("off")
                continue
            feat = flux_features[flux][c]
            label = meta.loc[feat, "label"] if feat in meta.index else feat
            lh, ph = draw_b_axis(
                ax, model, d, features, feat, label, chr(ord("a") + letter_idx), records, scale, flux, cv_r2, show_ylabel=(c == 0)
            )
            letter_idx += 1
            line_handle = lh or line_handle
            point_handle = ph or point_handle

    if line_handle is not None and point_handle is not None:
        fig.legend([line_handle, point_handle], ["RF partial response", "Observed bins"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.035, 0.045, 1, 1))
    fig.savefig(fig_dir / f"Figure_B_core_partial_response_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_B_core_partial_response_{scale}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_B_core_partial_response_{scale}.png", dpi=600, bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_B_core_partial_response_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)

    for r, flux in enumerate(FLUXES):
        d = model_df.loc[model_df["flux"].eq(flux)].dropna(subset=["delta_RMSE"] + features).copy()
        if d.empty:
            continue
        cv_r2 = observed_cv_r2(d, features, seed=9100 + r)
        model = fit_rf(d, features, seed=9200 + r)
        fig_one, axes_one = plt.subplots(1, len(flux_features[flux]), figsize=(8.8, 2.25), squeeze=False)
        row_flux_label(axes_one[0, 0], flux, x=-0.43)
        one_line = None
        one_point = None
        for c, feat in enumerate(flux_features[flux]):
            ax = axes_one[0, c]
            label = meta.loc[feat, "label"] if feat in meta.index else feat
            lh, ph = draw_b_axis(ax, model, d, features, feat, label, chr(ord("a") + c), records, scale, flux, cv_r2, show_ylabel=(c == 0))
            one_line = lh or one_line
            one_point = ph or one_point
        if one_line is not None and one_point is not None:
            fig_one.legend([one_line, one_point], ["RF partial response", "Observed bins"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
        fig_one.tight_layout(rect=(0.035, 0.1, 1, 1))
        fig_one.savefig(fig_dir / f"Nature_Figure_B_top5_partial_response_{scale}_{flux}.png", dpi=600, bbox_inches="tight")
        fig_one.savefig(fig_dir / f"Nature_Figure_B_top5_partial_response_{scale}_{flux}.pdf", bbox_inches="tight")
        plt.close(fig_one)

    out = pd.DataFrame(records).drop_duplicates(subset=["scale", "flux", "feature", "x", "y"], keep="first")
    out.to_csv(tab_dir / f"{scale}_Figure_B_top5_partial_response_curves.csv", index=False)
    out.to_csv(tab_dir / f"{scale}_Figure_B_core_partial_response_curves.csv", index=False)
    return out


def draw_c_axis(
    ax: plt.Axes,
    grid: pd.DataFrame,
    y_label: str,
    letter: str,
    show_xlabel: bool,
) -> object:
    pivot = grid.pivot(index="y", columns="x", values="pred_delta_RMSE")
    mesh = ax.contourf(pivot.columns.values, pivot.index.values, pivot.values, levels=14, cmap="viridis")
    ax.contour(pivot.columns.values, pivot.index.values, pivot.values, colors="white", linewidths=0.22, alpha=0.45)
    panel_letter(ax, letter, x=-0.17, y=1.10, fontsize=12)
    ax.set_box_aspect(1)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_xlabel("NDVI heterogeneity" if show_xlabel else "", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10, labelpad=4.5)
    clean_axis(ax)
    return mesh


def plot_figure_c_top_summary(scale: str, summary: pd.DataFrame, surf_tab: Path, fig_dir: Path) -> None:
    imp_path = fig_dir.parent / "tables" / f"{scale}_relative_variable_importance.csv"
    imp = pd.read_csv(imp_path)
    label_map = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")["label"].to_dict()
    chosen_rows = []
    for flux in FLUXES:
        for partner in representative_ndvi_partners(imp, flux):
            row = summary.loc[summary["flux"].eq(flux) & summary["partner"].eq(partner)]
            if not row.empty:
                chosen_rows.append(row.iloc[0].to_dict())
    chosen = pd.DataFrame(chosen_rows)
    if chosen.empty:
        return

    max_cols = max(len(chosen.loc[chosen["flux"].eq(flux)]) for flux in FLUXES)
    fig, axes = plt.subplots(len(FLUXES), max_cols, figsize=(10.8, 7.1), squeeze=False)
    letter_idx = 0
    mesh = None
    for r, flux in enumerate(FLUXES):
        rows = chosen.loc[chosen["flux"].eq(flux)].reset_index(drop=True)
        row_flux_label(axes[r, 0], flux, x=-0.34)
        for c in range(max_cols):
            ax = axes[r, c]
            if c >= len(rows):
                ax.axis("off")
                continue
            row = rows.iloc[c]
            safe = row["partner"].replace("/", "_").replace("\\", "_")
            grid = pd.read_csv(surf_tab / f"{scale}_{flux}_NDVI_x_{safe}_surface.csv")
            mesh = draw_c_axis(
                ax,
                grid,
                label_map.get(row["partner"], row["partner_label"]),
                chr(ord("a") + letter_idx),
                show_xlabel=(r == len(FLUXES) - 1),
            )
            letter_idx += 1
    chosen.to_csv(fig_dir.parent / "tables" / f"{scale}_Figure_C_representative_group_interactions.csv", index=False)
    fig.subplots_adjust(left=0.075, right=0.86, bottom=0.08, top=0.965, wspace=0.34, hspace=0.24)
    if mesh is not None:
        cax = fig.add_axes([0.89, 0.08, 0.015, 0.885])
        fig.colorbar(mesh, cax=cax, label="ΔRMSE")
    fig.savefig(fig_dir / f"Figure_C_top_NDVI_interaction_surfaces_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_C_top_NDVI_interaction_surfaces_{scale}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_C_top_NDVI_interaction_surfaces_{scale}.png", dpi=600, bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_C_top_NDVI_interaction_surfaces_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)

    for flux in FLUXES:
        rows = chosen.loc[chosen["flux"].eq(flux)].reset_index(drop=True)
        if rows.empty:
            continue
        fig_one, axes_one = plt.subplots(1, len(rows), figsize=(9.2, 2.55), squeeze=False)
        one_mesh = None
        row_flux_label(axes_one[0, 0], flux, x=-0.34)
        for c, row in rows.iterrows():
            ax = axes_one[0, c]
            safe = row["partner"].replace("/", "_").replace("\\", "_")
            grid = pd.read_csv(surf_tab / f"{scale}_{flux}_NDVI_x_{safe}_surface.csv")
            one_mesh = draw_c_axis(
                ax,
                grid,
                label_map.get(row["partner"], row["partner_label"]),
                chr(ord("a") + c),
                show_xlabel=True,
            )
        fig_one.subplots_adjust(left=0.07, right=0.86, bottom=0.18, top=0.92, wspace=0.34)
        if one_mesh is not None:
            cax = fig_one.add_axes([0.89, 0.18, 0.015, 0.74])
            fig_one.colorbar(one_mesh, cax=cax, label="ΔRMSE")
        fig_one.savefig(fig_dir / f"Nature_Figure_C_representative_NDVI_interactions_{scale}_{flux}.png", dpi=600, bbox_inches="tight")
        fig_one.savefig(fig_dir / f"Nature_Figure_C_representative_NDVI_interactions_{scale}_{flux}.pdf", bbox_inches="tight")
        plt.close(fig_one)


def plot_figure_c_top_summary(scale: str, summary: pd.DataFrame, surf_tab: Path, fig_dir: Path) -> None:
    imp_path = fig_dir.parent / "tables" / f"{scale}_relative_variable_importance.csv"
    imp = pd.read_csv(imp_path)
    label_map = imp.drop_duplicates("analysis_variable").set_index("analysis_variable")["label"].to_dict()
    chosen_rows = []
    for flux in FLUXES:
        for partner in representative_ndvi_partners(imp, flux):
            row = summary.loc[summary["flux"].eq(flux) & summary["partner"].eq(partner)]
            if not row.empty:
                chosen_rows.append(row.iloc[0].to_dict())
    chosen = pd.DataFrame(chosen_rows)
    if chosen.empty:
        return

    max_cols = max(len(chosen.loc[chosen["flux"].eq(flux)]) for flux in FLUXES)
    fig, axes = plt.subplots(len(FLUXES), max_cols, figsize=(10.8, 7.1), squeeze=False)
    letter_idx = 0
    mesh = None
    for r, flux in enumerate(FLUXES):
        rows = chosen.loc[chosen["flux"].eq(flux)].reset_index(drop=True)
        row_flux_label(axes[r, 0], flux, x=-0.34)
        for c in range(max_cols):
            ax = axes[r, c]
            if c >= len(rows):
                ax.axis("off")
                continue
            row = rows.iloc[c]
            safe = row["partner"].replace("/", "_").replace("\\", "_")
            grid = pd.read_csv(surf_tab / f"{scale}_{flux}_NDVI_x_{safe}_surface.csv")
            mesh = draw_c_axis(
                ax,
                grid,
                label_map.get(row["partner"], row["partner_label"]),
                chr(ord("a") + letter_idx),
                show_xlabel=(r == len(FLUXES) - 1),
            )
            letter_idx += 1
    chosen.to_csv(fig_dir.parent / "tables" / f"{scale}_Figure_C_representative_group_interactions.csv", index=False)
    fig.subplots_adjust(left=0.075, right=0.86, bottom=0.08, top=0.965, wspace=0.34, hspace=0.24)
    if mesh is not None:
        cax = fig.add_axes([0.89, 0.08, 0.015, 0.885])
        cb = fig.colorbar(mesh, cax=cax)
        cb.set_label("ΔRMSE", fontsize=10)
        cb.ax.tick_params(labelsize=9)
    fig.savefig(fig_dir / f"Figure_C_top_NDVI_interaction_surfaces_{scale}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"Figure_C_top_NDVI_interaction_surfaces_{scale}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_C_top_NDVI_interaction_surfaces_{scale}.png", dpi=600, bbox_inches="tight")
    fig.savefig(fig_dir / f"Nature_Figure_C_top_NDVI_interaction_surfaces_{scale}.pdf", bbox_inches="tight")
    plt.close(fig)

    for flux in FLUXES:
        rows = chosen.loc[chosen["flux"].eq(flux)].reset_index(drop=True)
        if rows.empty:
            continue
        fig_one, axes_one = plt.subplots(1, len(rows), figsize=(9.2, 2.55), squeeze=False)
        one_mesh = None
        row_flux_label(axes_one[0, 0], flux, x=-0.34)
        for c, row in rows.iterrows():
            ax = axes_one[0, c]
            safe = row["partner"].replace("/", "_").replace("\\", "_")
            grid = pd.read_csv(surf_tab / f"{scale}_{flux}_NDVI_x_{safe}_surface.csv")
            one_mesh = draw_c_axis(
                ax,
                grid,
                label_map.get(row["partner"], row["partner_label"]),
                chr(ord("a") + c),
                show_xlabel=True,
            )
        fig_one.subplots_adjust(left=0.07, right=0.86, bottom=0.18, top=0.92, wspace=0.34)
        if one_mesh is not None:
            cax = fig_one.add_axes([0.89, 0.18, 0.015, 0.74])
            cb = fig_one.colorbar(one_mesh, cax=cax)
            cb.set_label("ΔRMSE", fontsize=10)
            cb.ax.tick_params(labelsize=9)
        fig_one.savefig(fig_dir / f"Nature_Figure_C_representative_NDVI_interactions_{scale}_{flux}.png", dpi=600, bbox_inches="tight")
        fig_one.savefig(fig_dir / f"Nature_Figure_C_representative_NDVI_interactions_{scale}_{flux}.pdf", bbox_inches="tight")
        plt.close(fig_one)


def main() -> None:
    apply_nature_style()
    all_c = []
    for scale in SCALES:
        model_df, imp, perf = load_inputs(scale)
        plot_figure_a_top10(scale, imp, perf)
        plot_figure_b(scale, model_df, imp, perf)

    if all_c:
        pd.concat(all_c, ignore_index=True).to_csv(ROOT / "all_scales_Figure_C_NDVI_x_partner_surface_summary.csv", index=False)

    manifest = {
        "root": str(ROOT),
        "scales": SCALES,
        "figure_A": "top-10 relative variable importance (%)",
        "figure_B": "RF partial-response curves for core variables with observed binned means",
        "figure_C": "RF NDVI x all-partner partial-response surfaces and top-surface summaries",
        "response": "scale-specific ΔRMSE = RMSE_static - RMSE_dynamic",
    }
    (ROOT / "figure_abc_multiscale_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()





