# for DL method
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.metrics import mean_squared_error, r2_score


def _resolve_prediction_files(result_outdir, run_id=0, save_prefix="DL"):
    """
    Resolve prediction file names.

    Priority:
    1) New-style files used by the current main script:
       - true.npy
       - pred.npy
       - site.npy (optional)
    2) Old-style files:
       - DL_true_0.npy
       - DL_pred_0.npy
    """
    true_file_new = os.path.join(result_outdir, "true.npy")
    pred_file_new = os.path.join(result_outdir, "pred.npy")
    site_file_new = os.path.join(result_outdir, "site.npy")

    true_file_old = os.path.join(result_outdir, f"{save_prefix}_true_{run_id}.npy")
    pred_file_old = os.path.join(result_outdir, f"{save_prefix}_pred_{run_id}.npy")
    site_file_old = os.path.join(result_outdir, f"{save_prefix}_site_{run_id}.npy")

    if os.path.exists(true_file_new) and os.path.exists(pred_file_new):
        print("[INFO] Using new-style prediction files")
        print(f"       true = {true_file_new}")
        print(f"       pred = {pred_file_new}")
        if os.path.exists(site_file_new):
            print(f"       site = {site_file_new}")
        return true_file_new, pred_file_new, (site_file_new if os.path.exists(site_file_new) else None)

    if os.path.exists(true_file_old) and os.path.exists(pred_file_old):
        print("[INFO] Using old-style prediction files")
        print(f"       true = {true_file_old}")
        print(f"       pred = {pred_file_old}")
        if os.path.exists(site_file_old):
            print(f"       site = {site_file_old}")
        return true_file_old, pred_file_old, (site_file_old if os.path.exists(site_file_old) else None)

    raise FileNotFoundError(
        "[FATAL] Prediction files not found.\n"
        f"Tried new-style:\n  {true_file_new}\n  {pred_file_new}\n"
        f"Tried old-style:\n  {true_file_old}\n  {pred_file_old}"
    )


def _safe_corrcoef(x, y):
    """Safe Pearson CC."""
    if len(x) < 2 or len(y) < 2:
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def _safe_r2(x, y):
    """Safe R2."""
    if len(x) < 2:
        return np.nan
    try:
        return r2_score(x, y)
    except Exception:
        return np.nan


def _safe_rmse(x, y):
    """Safe RMSE."""
    if len(x) == 0:
        return np.nan
    return np.sqrt(mean_squared_error(x, y))


def _safe_bias(x, y):
    """Bias = mean(true - pred)."""
    if len(x) == 0:
        return np.nan
    return np.mean(x - y)


def _make_density_style(hist, cmap_colors):
    """
    Build ListedColormap + BoundaryNorm based on histogram density.
    """
    max_density = hist.max()
    n_bins = len(cmap_colors)

    if max_density < 1:
        bounds = np.linspace(0, 1, n_bins + 1)
    else:
        bounds = np.logspace(np.log10(1), np.log10(max_density + 1), n_bins + 1)

    # Keep very sparse bins visually light
    pivot_idx = min(5, len(bounds) - 1)
    filtered_colors = [
        'white' if bound < bounds[pivot_idx] else color
        for color, bound in zip(cmap_colors, bounds[:len(cmap_colors)])
    ]

    cmap = ListedColormap(filtered_colors)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def evaluate_flux_model_results_single(
    result_outdir,
    evaluate_dir,
    run_id=0,
    variables=("NEE", "GPP", "RECO"),
    common_ranges=None,
    cmap_colors=None,
    bins_n=100,
    save_prefix="DL",
    mask_value=-9999.0,
):
    """
    Evaluate flux model results for a single run by comparing true and predicted arrays.

    Supports both:
      1) new-style files:
         true.npy / pred.npy / site.npy
      2) old-style files:
         DL_true_0.npy / DL_pred_0.npy / DL_site_0.npy

    Parameters
    ----------
    result_outdir : str
        Directory containing prediction arrays.
    evaluate_dir : str
        Directory to save evaluation figures and CSVs.
    run_id : int
        Run index used only for backward-compatible old file names and output naming.
    variables : tuple/list of str
        Default: ("NEE", "GPP", "RECO")
    common_ranges : dict
        Example:
        {"NEE": (-6,6), "GPP": (0,20), "RECO": (0,8)}
    cmap_colors : list
        Colors for hist2d.
    bins_n : int
        Number of bins for hist2d.
    save_prefix : str
        Prefix for saved outputs.
    mask_value : float
        Fill value to ignore in y_true.

    Returns
    -------
    df : pd.DataFrame
        Per-variable metrics.
    df_residual : pd.DataFrame
        Residual balance statistics for |RECO - GPP - NEE|.
    """
    os.makedirs(evaluate_dir, exist_ok=True)

    true_file, pred_file, site_file = _resolve_prediction_files(
        result_outdir=result_outdir,
        run_id=run_id,
        save_prefix=save_prefix
    )

    print(f"[INFO] Loading y_true from: {true_file}")
    print(f"[INFO] Loading y_pred from: {pred_file}")

    y_true = np.load(true_file, allow_pickle=True)
    y_pred = np.load(pred_file, allow_pickle=True)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"[FATAL] Shape mismatch: y_true.shape={y_true.shape}, y_pred.shape={y_pred.shape}"
        )

    if y_true.ndim != 3:
        raise ValueError(
            f"[FATAL] Expected 3D arrays (N, T, C), got y_true.ndim={y_true.ndim}, shape={y_true.shape}"
        )

    if y_true.shape[2] < len(variables):
        raise ValueError(
            f"[FATAL] Number of channels in prediction arrays ({y_true.shape[2]}) "
            f"is smaller than number of variables ({len(variables)})."
        )

    print(f"[INFO] y_true shape = {y_true.shape}")
    print(f"[INFO] y_pred shape = {y_pred.shape}")

    if site_file is not None:
        try:
            site_arr = np.load(site_file, allow_pickle=True)
            print(f"[INFO] site array loaded: {site_file}, shape={site_arr.shape}")
        except Exception as e:
            print(f"[WARNING] Failed to load site file: {site_file}. Error: {e}")

    if common_ranges is None:
        common_ranges = {"NEE": (-6, 6), "GPP": (0, 20), "RECO": (0, 8)}

    if cmap_colors is None:
        cmap_colors = [
            "white", "lightcyan", "lightblue", "deepskyblue", "blue",
            "mediumseagreen", "green", "greenyellow", "yellow", "gold",
            "orange", "tomato", "red", "firebrick", "darkred"
        ]

    rows = []
    residual_rows = []

    fig, axs = plt.subplots(1, len(variables), figsize=(5 * len(variables), 4))
    if len(variables) == 1:
        axs = [axs]

    first_bounds = None

    for j, var in enumerate(variables):
        x = y_true[:, :, j].ravel().astype(float)
        y = y_pred[:, :, j].ravel().astype(float)

        mask = (x != mask_value) & (~np.isnan(x)) & (~np.isnan(y))
        x = x[mask]
        y = y[mask]

        n_samples = len(x)
        if n_samples == 0:
            print(f"[WARNING] No valid samples for variable {var}. Skipping.")
            continue

        rmse = _safe_rmse(x, y)
        bias = _safe_bias(x, y)
        cc = _safe_corrcoef(x, y)
        r2 = _safe_r2(x, y)

        rows.append({
            "Run": run_id + 1,
            "Variable": var,
            "RMSE": rmse,
            "CC": cc,
            "R2": r2,
            "BIAS": bias,
            "n": n_samples
        })

        ax = axs[j]
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins_n)

        if first_bounds is None:
            _, _, first_bounds = _make_density_style(hist, cmap_colors)

        bounds = first_bounds
        pivot_idx = min(5, len(bounds) - 1)
        filtered_colors = [
            "white" if bound < bounds[pivot_idx] else color
            for color, bound in zip(cmap_colors, bounds[:len(cmap_colors)])
        ]
        cmap = ListedColormap(filtered_colors)
        norm = BoundaryNorm(bounds, cmap.N)

        ax.hist2d(x, y, bins=bins_n, cmap=cmap, norm=norm)

        if var in common_ranges:
            cmin, cmax = common_ranges[var]
            ax.plot([cmin, cmax], [cmin, cmax], color="#535CA8", linestyle="--", linewidth=1)
            ax.set_xlim(cmin, cmax)
            ax.set_ylim(cmin, cmax)
        else:
            xy_min = np.nanmin([np.nanmin(x), np.nanmin(y)])
            xy_max = np.nanmax([np.nanmax(x), np.nanmax(y)])
            ax.plot([xy_min, xy_max], [xy_min, xy_max], color="#535CA8", linestyle="--", linewidth=1)

        ax.set_xlabel("Flux tower")
        ax.set_ylabel("Model estimated")
        ax.set_title(var, fontsize=12)

        text = (
            f"RMSE={rmse:.2f}\n"
            f"CC={cc:.2f}\n"
            f"R2={r2:.2f}\n"
            f"Bias={bias:.2f}\n"
            f"n={n_samples}"
        )
        ax.text(
            0.05, 0.95, text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10, color="red"
        )

    # --------------------------------------------------
    # Residual balance: |RECO - GPP - NEE|
    # --------------------------------------------------
    if y_true.shape[2] >= 3:
        mask_all = (
            (y_true[:, :, 0] != mask_value) &
            (y_true[:, :, 1] != mask_value) &
            (y_true[:, :, 2] != mask_value) &
            (~np.isnan(y_true[:, :, 0])) &
            (~np.isnan(y_true[:, :, 1])) &
            (~np.isnan(y_true[:, :, 2])) &
            (~np.isnan(y_pred[:, :, 0])) &
            (~np.isnan(y_pred[:, :, 1])) &
            (~np.isnan(y_pred[:, :, 2]))
        )

        mask_flat = mask_all.ravel()
        y_true_valid = y_true.reshape(-1, y_true.shape[2])[mask_flat]
        y_pred_valid = y_pred.reshape(-1, y_pred.shape[2])[mask_flat]

        if len(y_true_valid) > 0:
            residual_true = np.abs(y_true_valid[:, 2] - y_true_valid[:, 1] - y_true_valid[:, 0])
            residual_pred = np.abs(y_pred_valid[:, 2] - y_pred_valid[:, 1] - y_pred_valid[:, 0])

            residual_rows.append({
                "Run": run_id + 1,
                "Type": "True",
                "Min": residual_true.min(),
                "Max": residual_true.max(),
                "Mean": residual_true.mean(),
                "Median": np.median(residual_true),
                "Std": residual_true.std(),
                "N": len(residual_true)
            })
            residual_rows.append({
                "Run": run_id + 1,
                "Type": "Pred",
                "Min": residual_pred.min(),
                "Max": residual_pred.max(),
                "Mean": residual_pred.mean(),
                "Median": np.median(residual_pred),
                "Std": residual_pred.std(),
                "N": len(residual_pred)
            })

    # --------------------------------------------------
    # Save figure
    # --------------------------------------------------
    fig.tight_layout()
    fig_path = os.path.join(evaluate_dir, f"{save_prefix}_scatter_run{run_id + 1}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------
    df = pd.DataFrame(rows)
    metrics_csv = os.path.join(evaluate_dir, f"{save_prefix}_metrics_run{run_id + 1}.csv")
    df.to_csv(metrics_csv, index=False)

    df_residual = pd.DataFrame(residual_rows)
    residual_csv = os.path.join(evaluate_dir, f"{save_prefix}_residual_run{run_id + 1}.csv")
    df_residual.to_csv(residual_csv, index=False)

    # Optional: also save a simple summary copy for quick checking
    if not df.empty:
        summary = df.groupby("Variable", sort=False).agg({
            "RMSE": "mean",
            "CC": "mean",
            "R2": "mean",
            "BIAS": "mean",
            "n": "sum"
        }).reset_index()
        summary_csv = os.path.join(evaluate_dir, f"{save_prefix}_metrics_summary_run{run_id + 1}.csv")
        summary.to_csv(summary_csv, index=False)
    else:
        summary = pd.DataFrame()

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------
    print(f"\n=== Evaluation Results for run {run_id + 1} ===")
    if df.empty:
        print("[WARNING] No valid metrics were computed.")
    else:
        for var in variables:
            subset = df[df["Variable"] == var]
            if subset.empty:
                continue
            rmse_mean = subset["RMSE"].mean()
            bias_mean = subset["BIAS"].mean()
            cc_mean = subset["CC"].mean()
            r2_mean = subset["R2"].mean()
            print(f"{var}: RMSE={rmse_mean:.3f}, Bias={bias_mean:.3f}, CC={cc_mean:.3f}, R2={r2_mean:.3f}")
    print("==============================================\n")

    print(f"[OK] Figure saved to:   {fig_path}")
    print(f"[OK] Metrics saved to:  {metrics_csv}")
    print(f"[OK] Residual saved to: {residual_csv}")
    if not summary.empty:
        print(f"[OK] Summary saved to:  {summary_csv}")

    return df, df_residual


# --------------------------------------------------
# Optional standalone test block
# --------------------------------------------------
if __name__ == "__main__":
    # Example:
    # result_outdir = "./data/v7_6/FFP0_dynamic_ffp/fold0/pred"
    # evaluate_dir  = "./data/v7_6/FFP0_dynamic_ffp/fold0/eval"
    #
    # evaluate_flux_model_results_single(
    #     result_outdir=result_outdir,
    #     evaluate_dir=evaluate_dir,
    #     run_id=0
    # )
    pass