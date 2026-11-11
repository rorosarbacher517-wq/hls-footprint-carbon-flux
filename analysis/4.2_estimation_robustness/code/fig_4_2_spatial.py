# -*- coding: utf-8 -*-
"""
Clean Figure 4 estimation robustness script.

Outputs:
1. Combined main figures:
   - NorthAmerica, categories with n_sites >= 10
   - Americas, categories with n_sites >= 10

2. Combined supplementary figures:
   - NorthAmerica, all categories
   - Americas, all categories

3. Single-panel figures:
   - maps
   - vertical main-style IGBP and climate panels
   - horizontal supplementary-style IGBP and climate panels

4. Category statistics CSV tables:
   - main n>=10 tables
   - supplementary all-category tables

Important:
- This script intentionally avoids non-ASCII characters in source code.
- Use "Koppen" instead of the special-character spelling.
- Use "deg" instead of degree symbol.
- Use "umol" instead of micro symbol.

Delta RMSE definition:
    DELTA_RMSE_FAT_TRANSFORMER = RMSE_static_Transformer - RMSE_FAT

Interpretation:
    Positive Delta RMSE means FAT has lower RMSE and performs better.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

warnings.filterwarnings("ignore")


# ============================================================
# 0. Plotting configuration
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

    # Large font version
    "font.size": 20,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 19,

    "legend.frameon": False,
    "axes.linewidth": 1.3,
    "lines.linewidth": 2.0,
    "patch.linewidth": 1.3,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.default": "regular",
})


def format_nature_axes(ax, hide_top: bool = True, hide_right: bool = True) -> None:
    """Apply simple clean axes."""
    if hide_top:
        ax.spines["top"].set_visible(False)
    if hide_right:
        ax.spines["right"].set_visible(False)


def add_panel_label(ax, label: str, x_offset: float) -> None:
    """Add panel label to the top-left of an axis."""
    ax.text(
        x_offset,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=32,
        fontweight="bold",
        va="top",
        ha="right",
    )


# ============================================================
# 1. User configuration
# ============================================================

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
import sys
sys.path.insert(0, str(PROJECT / "common"))
from figure_io import save_manuscript_png   # combined figures: PNG at print size (183 mm, 600 dpi)
OUTDIR = str(HERE.parent / "outputs" / "spatial" / "figures")

SITE_META_CSV = str(PROJECT / "data" / "site_metadata" / "site_multiscale_landcover.csv")

# Four-model held-out predictions, five folds (concatenated in fold order 0-4 = the former kfold5/pred files).
PRED_ROOT = Path(os.environ.get("PAPER_DATA_ROOT", PROJECT / "data" / "four_model_predictions"))
FAT_DIR = "FAT"                            # dynamic footprint aggregation
TRANSFORMER_DIR = "Uniform_Transformer"    # uniform-weighted aggregation (the baseline of Delta RMSE)
RF_DIR = "RF"                              # pre-aggregated random forest (only used as a validity filter)

MIN_SAMPLES_PER_SITE = 100
WIN_TIE_THRESHOLD = 0.05

# Map bounding boxes
NA_XLIM = (-170, -50)
NA_YLIM = (20, 80)

AMERICAS_XLIM = (-170, -30)
AMERICAS_YLIM = (-60, 85)

BIOME_ORDER = [
    "ENF", "DBF", "MF", "EBF", "DNF",
    "WSA", "SAV", "OSH", "CSH", "SHB",
    "GRA", "CRO", "WET", "URB", "BAR", "SNO", "WAT", "CVM",
    "Unknown",
]

CLIMATE_COL = "CLIM_CODE_USE"

CLIMATE_ORDER = [
    "Dfb", "Cfa", "Dfa", "BSk", "Csa", "ET", "Dfc",
    "Csb", "BWk", "BSh", "Dwc", "Dfd", "Dsb",
    "Cfb", "Cfc", "Aw", "Cwc", "Af", "Cwa", "Dwb", "BWh",
    "Dry_Fallback", "Temperate_Fallback", "Continental_Fallback",
    "Tropical_Fallback", "Polar_Fallback", "Unknown",
]

SITE_CLIMATE_IMPUTATIONS = {
    "US-Xst": {
        "climate": "Dfb",
        "source": "case-normalized AmeriFlux Clim match",
        "reference_site": "US-xST",
        "distance_km": 0.0,
        "note": "same coordinates as NEON Steigerwaldt Land Services",
        "lat": 45.5089,
        "lon": -89.5864,
        "igbp": "DBF",
    },
}

TARGET_TO_INDEX = {"NEE": 0, "GPP": 1, "RECO": 2}
TARGET_FLUXES = ["NEE", "GPP", "RECO"]

VALUE_COL = "DELTA_RMSE_FAT_TRANSFORMER"


# ============================================================
# 2. General utility functions
# ============================================================

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv_robust(path: str) -> pd.DataFrame:
    """Read a CSV file with common encodings."""
    last_err = None

    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as err:
            last_err = err

    raise RuntimeError(f"Could not read CSV with common encodings: {path}. Last error: {last_err}")


def safe_decode_value(val) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore")
    return str(val)


def clean_site_id_series(x: pd.Series) -> pd.Series:
    return x.astype(str).str.strip()


def clean_site_id_array(x: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(x).astype(str)).str.strip().to_numpy()


def clean_missing_str(x) -> str:
    """Normalize missing-like strings."""
    if pd.isna(x):
        return ""

    s = str(x).strip()

    if s.lower() in [
        "", "nan", "none", "null", "no_data", "nodata",
        "-9999", "na", "n/a",
    ]:
        return ""

    return s


def first_valid_value(series: pd.Series):
    """Return first non-empty value from a Series."""
    for v in series:
        s = clean_missing_str(v)
        if s != "":
            return s
    return np.nan


def first_valid_numeric(series: pd.Series):
    """Return first finite numeric value from a Series."""
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return np.nan

    return float(vals.iloc[0])


def mode_valid_value(series: pd.Series):
    """Return most frequent non-empty value from a Series."""
    vals = [clean_missing_str(v) for v in series]
    vals = [v for v in vals if v != ""]

    if len(vals) == 0:
        return np.nan

    return pd.Series(vals).mode().iloc[0]


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE after finite-value filtering."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        return np.nan

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def extract_site_ids_from_site_cube(site_arr: np.ndarray) -> np.ndarray:
    """Extract site IDs from a site.npy array."""
    if site_arr.ndim == 1:
        return clean_site_id_array(site_arr)

    return np.array(
        [safe_decode_value(site_arr[i, 0, 0]).strip() for i in range(site_arr.shape[0])],
        dtype=str,
    )


def flatten_predictions(
    pred: np.ndarray,
    true: np.ndarray,
    site_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten prediction arrays to:
        pred_flat: (N, n_targets)
        true_flat: (N, n_targets)
        site_flat: (N,)
    """
    site_ids_outer = extract_site_ids_from_site_cube(site_arr)

    if pred.ndim >= 3:
        n_targets = pred.shape[-1]
        repeat_factor = int(np.prod(pred.shape[1:-1]))

        return (
            pred.reshape(-1, n_targets),
            true.reshape(-1, n_targets),
            np.repeat(site_ids_outer, repeat_factor),
        )

    return pred, true, site_ids_outer


def load_model_arrays(model_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pred/true/site of one model, concatenating the five held-out folds in order."""
    preds, trues, sites = [], [], []
    for k in range(5):
        base = PRED_ROOT / model_dir / f"fold{k}"
        preds.append(np.load(base / "pred.npy", allow_pickle=True))
        trues.append(np.load(base / "true.npy", allow_pickle=True))
        sites.append(np.load(base / "site.npy", allow_pickle=True))
    pred, true, site = np.concatenate(preds), np.concatenate(trues), np.concatenate(sites)

    p, t, s = flatten_predictions(pred, true, site)
    return p, t, clean_site_id_array(s)


# ============================================================
# 3. Site metadata and climate-code handling
# ============================================================

def normalize_column_name(c) -> str:
    """
    Normalize column names without using non-ASCII source characters.

    The source CSV may include a degree symbol in the MAT column.
    This function normalizes it to "deg".
    """
    c_str = str(c).strip()
    c_safe = (
        c_str.replace("\u00b0", "deg")
             .replace("\u00b5", "u")
             .replace("\u03bc", "u")
             .replace("\xa0", " ")
             .strip()
    )
    return c_safe


def normalize_koppen_code(x) -> str:
    """
    Normalize Koppen-Geiger climate code.

    Examples:
        bsk -> BSk
        BSK -> BSk
        cfa -> Cfa
        ET  -> ET
    """
    s = clean_missing_str(x)

    if s == "":
        return "Unknown"

    s = s.strip()

    if s.upper() in ["ET", "EF"]:
        return s.upper()

    if len(s) >= 3:
        first = s[0].upper()

        if first == "B":
            # Arid classes: BWh, BWk, BSh, BSk
            return first + s[1].upper() + s[2].lower()

        return first + s[1].lower() + s[2].lower()

    return s


def infer_climate_from_mat_map(mat, map_mm) -> str:
    """
    Fallback when both Clim_fill and Clim are missing.

    This is a pragmatic backup, not a formal Koppen-Geiger classification.
    """
    try:
        mat = float(mat)
    except Exception:
        mat = np.nan

    try:
        map_mm = float(map_mm)
    except Exception:
        map_mm = np.nan

    if not np.isfinite(mat) and not np.isfinite(map_mm):
        return "Unknown"

    if np.isfinite(map_mm) and map_mm < 250:
        return "Dry_Fallback"

    if np.isfinite(mat):
        if mat >= 18:
            return "Tropical_Fallback"
        if mat < 0:
            return "Polar_Fallback"
        if mat < 5:
            return "Continental_Fallback"
        return "Temperate_Fallback"

    return "Unknown"


def build_site_metadata(site_meta_csv: str) -> pd.DataFrame:
    """
    Build site-level metadata.

    Climate priority:
        1. Clim_fill
        2. Clim
        3. fallback from MAT/MAP

    The input metadata may contain repeated rows by Target_Year.
    This function aggregates repeated rows to one row per SITE_ID.
    """
    df = read_csv_robust(site_meta_csv)

    # Rename columns robustly without relying on non-ASCII source text.
    col_rename = {}
    for c in df.columns:
        c_safe = normalize_column_name(c)

        if c_safe == "Site ID":
            col_rename[c] = "SITE_ID"
        elif c_safe == "Veg":
            col_rename[c] = "IGBP"
        elif c_safe == "Lat":
            col_rename[c] = "Lat"
        elif c_safe == "Long":
            col_rename[c] = "Lon"
        elif c_safe == "Clim":
            col_rename[c] = "Clim"
        elif c_safe == "Clim_fill":
            col_rename[c] = "Clim_fill"
        elif c_safe in ["MAT (degC)", "MAT (deg C)", "MAT (C)"]:
            col_rename[c] = "MAT_C"
        elif c_safe == "MAP (mm)":
            col_rename[c] = "MAP_mm"

    df = df.rename(columns=col_rename)

    if "SITE_ID" not in df.columns:
        raise ValueError("SITE_ID / Site ID column not found in site metadata CSV.")

    df["SITE_ID"] = clean_site_id_series(df["SITE_ID"])

    if "IGBP" not in df.columns:
        df["IGBP"] = "Unknown"

    if "Lat" not in df.columns or "Lon" not in df.columns:
        raise ValueError("Lat / Long columns are required for map panels.")

    if "Clim" not in df.columns:
        df["Clim"] = np.nan

    if "Clim_fill" not in df.columns:
        df["Clim_fill"] = np.nan

    if "MAT_C" not in df.columns:
        df["MAT_C"] = np.nan

    if "MAP_mm" not in df.columns:
        df["MAP_mm"] = np.nan

    clim_fill_clean = df["Clim_fill"].apply(clean_missing_str)
    clim_clean = df["Clim"].apply(clean_missing_str)

    df["CLIM_CODE_RAW"] = np.where(
        clim_fill_clean != "",
        clim_fill_clean,
        clim_clean,
    )

    df["CLIM_CODE_USE"] = df["CLIM_CODE_RAW"].apply(normalize_koppen_code)

    df["CLIM_FALLBACK"] = [
        infer_climate_from_mat_map(mat, map_mm)
        for mat, map_mm in zip(df["MAT_C"], df["MAP_mm"])
    ]

    df["CLIM_CODE_USE"] = np.where(
        df["CLIM_CODE_USE"] != "Unknown",
        df["CLIM_CODE_USE"],
        df["CLIM_FALLBACK"],
    )

    df["CLIM_SOURCE_USE"] = np.where(
        clim_fill_clean != "",
        "Clim_fill",
        np.where(clim_clean != "", "Clim", "MAT/MAP fallback"),
    )
    df.loc[df["CLIM_CODE_USE"] == "Unknown", "CLIM_SOURCE_USE"] = "Unknown"
    df["CLIM_REFERENCE_SITE"] = ""
    df["CLIM_REFERENCE_DISTANCE_KM"] = np.nan
    df["CLIM_IMPUTATION_NOTE"] = ""

    for site_id, info in SITE_CLIMATE_IMPUTATIONS.items():
        mask = df["SITE_ID"] == site_id
        if not mask.any():
            continue
        df.loc[mask, "CLIM_CODE_USE"] = normalize_koppen_code(info["climate"])
        df.loc[mask, "CLIM_SOURCE_USE"] = info["source"]
        df.loc[mask, "CLIM_REFERENCE_SITE"] = info["reference_site"]
        df.loc[mask, "CLIM_REFERENCE_DISTANCE_KM"] = info["distance_km"]
        df.loc[mask, "CLIM_IMPUTATION_NOTE"] = info["note"]

    # Aggregate repeated Target_Year rows to site level.
    agg_dict = {}
    for c in df.columns:
        if c == "SITE_ID":
            continue
        if c in [
            "IGBP", "Clim", "Clim_fill", "CLIM_CODE_RAW", "CLIM_CODE_USE",
            "CLIM_FALLBACK", "CLIM_SOURCE_USE", "CLIM_REFERENCE_SITE",
            "CLIM_IMPUTATION_NOTE",
        ]:
            agg_dict[c] = mode_valid_value
        elif c in ["Lat", "Lon", "Elev (m)", "MAT_C", "MAP_mm", "CLIM_REFERENCE_DISTANCE_KM"]:
            agg_dict[c] = first_valid_numeric
        else:
            agg_dict[c] = first_valid_value

    out = df.groupby("SITE_ID", as_index=False).agg(agg_dict)

    out["IGBP"] = out["IGBP"].fillna("Unknown").astype(str)
    out["CLIM_CODE_USE"] = out["CLIM_CODE_USE"].fillna("Unknown").astype(str)

    out["Lat"] = pd.to_numeric(out["Lat"], errors="coerce")
    out["Lon"] = pd.to_numeric(out["Lon"], errors="coerce")

    alias_rows = []
    for site_id, info in SITE_CLIMATE_IMPUTATIONS.items():
        if site_id in set(out["SITE_ID"].astype(str)):
            continue
        alias = {c: np.nan for c in out.columns}
        alias["SITE_ID"] = site_id
        alias["IGBP"] = info.get("igbp", "Unknown")
        alias["Lat"] = info.get("lat", np.nan)
        alias["Lon"] = info.get("lon", np.nan)
        alias["Clim"] = np.nan
        alias["Clim_fill"] = normalize_koppen_code(info["climate"])
        alias["CLIM_CODE_RAW"] = normalize_koppen_code(info["climate"])
        alias["CLIM_CODE_USE"] = normalize_koppen_code(info["climate"])
        alias["CLIM_FALLBACK"] = "Unknown"
        alias["CLIM_SOURCE_USE"] = info["source"]
        alias["CLIM_REFERENCE_SITE"] = info["reference_site"]
        alias["CLIM_REFERENCE_DISTANCE_KM"] = info["distance_km"]
        alias["CLIM_IMPUTATION_NOTE"] = info["note"]
        alias_rows.append(alias)

    if alias_rows:
        out = pd.concat([out, pd.DataFrame(alias_rows)], ignore_index=True)

    return out


# ============================================================
# 4. Metric computation
# ============================================================

def compute_metrics(
    fat_pred: np.ndarray,
    tr_pred: np.ndarray,
    rf_pred: np.ndarray,
    reference_true: np.ndarray,
    site_ids: np.ndarray,
    target_flux: str,
) -> pd.DataFrame:
    """
    Compute site-level RMSE.

    Delta definition:
        DELTA_RMSE_FAT_TRANSFORMER = RMSE_TR - RMSE_FAT
    """
    idx = TARGET_TO_INDEX[target_flux]

    yt = np.asarray(reference_true[:, idx], dtype=float)
    yf = np.asarray(fat_pred[:, idx], dtype=float)
    yt_tr = np.asarray(tr_pred[:, idx], dtype=float)
    y_rf = np.asarray(rf_pred[:, idx], dtype=float)

    df = pd.DataFrame({
        "SITE_ID": site_ids,
        "yt": yt,
        "yf": yf,
        "yt_tr": yt_tr,
        "yrf": y_rf,
    })

    df = df[
        (df["yt"] > -9000) &
        (df["yf"] > -9000) &
        (df["yt_tr"] > -9000) &
        (df["yrf"] > -9000)
    ].dropna()

    rows = []

    for site, g in df.groupby("SITE_ID"):
        if len(g) < MIN_SAMPLES_PER_SITE:
            continue

        rows.append({
            "SITE_ID": site,
            "N_SAMPLES": int(len(g)),
            "RMSE_FAT": compute_rmse(g["yt"].values, g["yf"].values),
            "RMSE_TR": compute_rmse(g["yt"].values, g["yt_tr"].values),
            "RMSE_RF": compute_rmse(g["yt"].values, g["yrf"].values),
            "TRUE_MEDIAN": float(np.median(g["yt"].values)),
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["DELTA_RMSE_FAT_TRANSFORMER"] = out["RMSE_TR"] - out["RMSE_FAT"]

    return out


def classify_win_tie_loss(delta_rmse: pd.Series, threshold: float) -> pd.Series:
    """Classify site-level Delta RMSE into win/tie/loss groups."""
    cls = pd.Series(index=delta_rmse.index, dtype=object)

    cls[delta_rmse > threshold] = r"$\Delta$RMSE > 0.05"
    cls[delta_rmse < -threshold] = r"$\Delta$RMSE < -0.05"
    cls[(delta_rmse >= -threshold) & (delta_rmse <= threshold)] = (
        r"|$\Delta$RMSE| $\leq$ 0.05"
    )

    return cls


# ============================================================
# 5. Category statistics tables
# ============================================================

def compute_tukey_whiskers(values: np.ndarray) -> Tuple[float, float]:
    """Compute Tukey whiskers using 1.5 * IQR rule."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan

    q25 = np.nanpercentile(values, 25)
    q75 = np.nanpercentile(values, 75)
    iqr = q75 - q25

    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    valid_low = values[values >= lower_bound]
    valid_high = values[values <= upper_bound]

    whisker_low = np.nanmin(valid_low) if valid_low.size > 0 else np.nanmin(values)
    whisker_high = np.nanmax(valid_high) if valid_high.size > 0 else np.nanmax(values)

    return float(whisker_low), float(whisker_high)


def summarize_category_stats(
    df: pd.DataFrame,
    region: str,
    figure_set: str,
    flux: str,
    group_type: str,
    group_col: str,
    category: str,
    threshold: float,
) -> dict:
    """Summarize site-level boxplot statistics for one category."""
    g = df[df[group_col].astype(str) == str(category)].copy()

    vals = pd.to_numeric(g[VALUE_COL], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]

    n_sites = int(g["SITE_ID"].nunique())

    if "N_SAMPLES" in g.columns:
        n_total_samples = int(pd.to_numeric(g["N_SAMPLES"], errors="coerce").fillna(0).sum())
    else:
        n_total_samples = np.nan

    base = {
        "region": region,
        "figure_set": figure_set,
        "flux": flux,
        "group_type": group_type,
        "category": category,
        "n_sites": n_sites,
        "n_total_samples": n_total_samples,
    }

    if vals.size == 0:
        base.update({
            "delta_mean": np.nan,
            "delta_median": np.nan,
            "delta_std": np.nan,
            "delta_q25": np.nan,
            "delta_q75": np.nan,
            "delta_iqr": np.nan,
            "delta_whisker_low": np.nan,
            "delta_whisker_high": np.nan,
            "delta_min": np.nan,
            "delta_max": np.nan,
            "n_positive": 0,
            "n_tie": 0,
            "n_negative": 0,
            "pct_positive": np.nan,
            "pct_tie": np.nan,
            "pct_negative": np.nan,
            "rmse_fat_mean": np.nan,
            "rmse_fat_median": np.nan,
            "rmse_static_mean": np.nan,
            "rmse_static_median": np.nan,
            "rmse_rf_mean": np.nan,
            "rmse_rf_median": np.nan,
        })
        return base

    q25 = float(np.nanpercentile(vals, 25))
    q75 = float(np.nanpercentile(vals, 75))
    whisker_low, whisker_high = compute_tukey_whiskers(vals)

    n_positive = int(np.sum(vals > threshold))
    n_negative = int(np.sum(vals < -threshold))
    n_tie = int(np.sum((vals >= -threshold) & (vals <= threshold)))

    denom = float(vals.size)

    rmse_fat = pd.to_numeric(g["RMSE_FAT"], errors="coerce") if "RMSE_FAT" in g.columns else pd.Series(dtype=float)
    rmse_static = pd.to_numeric(g["RMSE_TR"], errors="coerce") if "RMSE_TR" in g.columns else pd.Series(dtype=float)
    rmse_rf = pd.to_numeric(g["RMSE_RF"], errors="coerce") if "RMSE_RF" in g.columns else pd.Series(dtype=float)

    base.update({
        "delta_mean": float(np.nanmean(vals)),
        "delta_median": float(np.nanmedian(vals)),
        "delta_std": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0,
        "delta_q25": q25,
        "delta_q75": q75,
        "delta_iqr": float(q75 - q25),
        "delta_whisker_low": whisker_low,
        "delta_whisker_high": whisker_high,
        "delta_min": float(np.nanmin(vals)),
        "delta_max": float(np.nanmax(vals)),

        "n_positive": n_positive,
        "n_tie": n_tie,
        "n_negative": n_negative,
        "pct_positive": 100.0 * n_positive / denom,
        "pct_tie": 100.0 * n_tie / denom,
        "pct_negative": 100.0 * n_negative / denom,

        "rmse_fat_mean": float(np.nanmean(rmse_fat)) if len(rmse_fat) > 0 else np.nan,
        "rmse_fat_median": float(np.nanmedian(rmse_fat)) if len(rmse_fat) > 0 else np.nan,
        "rmse_static_mean": float(np.nanmean(rmse_static)) if len(rmse_static) > 0 else np.nan,
        "rmse_static_median": float(np.nanmedian(rmse_static)) if len(rmse_static) > 0 else np.nan,
        "rmse_rf_mean": float(np.nanmean(rmse_rf)) if len(rmse_rf) > 0 else np.nan,
        "rmse_rf_median": float(np.nanmedian(rmse_rf)) if len(rmse_rf) > 0 else np.nan,
    })

    return base


def save_category_stats_csv(
    dfs_plot: Dict[str, pd.DataFrame],
    region: str,
    figure_set: str,
    out_csv: str,
    min_sites_filter: int,
    threshold: float,
) -> None:
    """
    Save category-level statistics for IGBP and Koppen climate classes.

    min_sites_filter:
        10 for main figure tables.
        0 for supplementary all-category tables.
    """
    rows = []

    group_specs = [
        {"group_type": "IGBP", "group_col": "IGBP", "group_order": BIOME_ORDER},
        {"group_type": "Koppen climate class", "group_col": CLIMATE_COL, "group_order": CLIMATE_ORDER},
    ]

    for flux in TARGET_FLUXES:
        df_flux = dfs_plot.get(flux, pd.DataFrame()).copy()

        if df_flux.empty:
            continue

        for spec in group_specs:
            group_type = spec["group_type"]
            group_col = spec["group_col"]
            group_order = spec["group_order"]

            if group_col not in df_flux.columns:
                continue

            df_group = df_flux.copy()
            df_group[group_col] = df_group[group_col].fillna("Unknown").astype(str)

            if min_sites_filter > 0:
                df_group = df_group[df_group[group_col] != "Unknown"].copy()
                counts = df_group.groupby(group_col)["SITE_ID"].nunique()
                valid_categories = counts[counts >= min_sites_filter].index
                df_group = df_group[df_group[group_col].isin(valid_categories)].copy()

            if df_group.empty:
                continue

            ordered = [g for g in group_order if g in df_group[group_col].unique()]
            unordered = [g for g in df_group[group_col].unique() if g not in ordered]
            final_order = ordered + sorted(unordered)

            for category in final_order:
                rows.append(
                    summarize_category_stats(
                        df=df_group,
                        region=region,
                        figure_set=figure_set,
                        flux=flux,
                        group_type=group_type,
                        group_col=group_col,
                        category=category,
                        threshold=threshold,
                    )
                )

    out = pd.DataFrame(rows)

    if not out.empty:
        out["group_type_order"] = out["group_type"].map({
            "IGBP": 1,
            "Koppen climate class": 2,
        }).fillna(99)

        out["flux_order"] = out["flux"].map({
            "NEE": 1,
            "GPP": 2,
            "RECO": 3,
        }).fillna(99)

        out = out.sort_values(
            ["region", "figure_set", "flux_order", "group_type_order", "category"]
        ).drop(columns=["flux_order", "group_type_order"])

    out.to_csv(out_csv, index=False)
    print(f"Saved category statistics CSV: {out_csv}")


# ============================================================
# 6. Plotting helpers
# ============================================================

def get_group_order_and_data(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    group_order: List[str],
    min_sites_filter: int,
    drop_unknown_when_filter: bool,
):
    """Prepare group order, boxplot data, and site counts."""
    df_plot = df.copy()

    if group_col not in df_plot.columns:
        return df_plot.iloc[0:0], [], [], []

    df_plot[group_col] = df_plot[group_col].fillna("Unknown").astype(str)

    if min_sites_filter > 0:
        if drop_unknown_when_filter:
            df_plot = df_plot[df_plot[group_col] != "Unknown"].copy()

        counts_site = df_plot.groupby(group_col)["SITE_ID"].nunique()
        valid_groups = counts_site[counts_site >= min_sites_filter].index
        df_plot = df_plot[df_plot[group_col].isin(valid_groups)].copy()

    if df_plot.empty:
        return df_plot, [], [], []

    ordered = [g for g in group_order if g in df_plot[group_col].unique()]
    unordered = [g for g in df_plot[group_col].unique() if g not in ordered]
    final_order = ordered + sorted(unordered)

    data = [
        df_plot.loc[df_plot[group_col] == g, value_col].dropna().values
        for g in final_order
    ]

    counts = [
        df_plot.loc[df_plot[group_col] == g, "SITE_ID"].nunique()
        for g in final_order
    ]

    return df_plot, final_order, data, counts


def plot_vertical_boxplot(
    ax,
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    group_order: List[str],
    box_color: str,
    x_label: str,
    y_label: str = r"$\Delta$RMSE",
    min_sites_filter: int = 0,
    show_xlabel: bool = False,
    rotate_labels: int = 45,
    drop_unknown_when_filter: bool = True,
):
    """
    Vertical boxplot.

    No sample-size axis is shown. Sample sizes and boxplot statistics are
    exported to CSV tables.
    """
    df_plot, final_order, data, counts = get_group_order_and_data(
        df=df,
        group_col=group_col,
        value_col=value_col,
        group_order=group_order,
        min_sites_filter=min_sites_filter,
        drop_unknown_when_filter=drop_unknown_when_filter,
    )

    if df_plot.empty or len(final_order) == 0:
        ax.text(
            0.5,
            0.5,
            "No valid groups",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=18,
        )
        format_nature_axes(ax)
        return [], []

    x_pos = np.arange(1, len(final_order) + 1)

    bp = ax.boxplot(
        data,
        positions=x_pos,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.3),
        whiskerprops=dict(color="black", linewidth=1.1),
        capprops=dict(color="black", linewidth=1.1),
        boxprops=dict(edgecolor="black", linewidth=1.1),
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(box_color)
        patch.set_alpha(0.88)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        final_order,
        rotation=rotate_labels,
        ha="right" if rotate_labels != 0 else "center",
    )

    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    if show_xlabel:
        ax.set_xlabel(x_label)
    else:
        ax.set_xlabel("")

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.35,
        color="gray",
        alpha=0.22,
    )

    format_nature_axes(ax)

    return final_order, counts


def plot_map_panel(
    ax,
    df_site: pd.DataFrame,
    flux: str,
    region: str,
    threshold: float,
) -> None:
    """Plot map panel for one flux."""
    plot_df = df_site.copy()
    plot_df["WIN_CLASS"] = classify_win_tie_loss(plot_df[VALUE_COL], threshold)

    color_map = {
        r"$\Delta$RMSE > 0.05": "#d73027",
        r"|$\Delta$RMSE| $\leq$ 0.05": "#bdbdbd",
        r"$\Delta$RMSE < -0.05": "#4575b4",
    }

    for cls in [
        r"$\Delta$RMSE > 0.05",
        r"|$\Delta$RMSE| $\leq$ 0.05",
        r"$\Delta$RMSE < -0.05",
    ]:
        sub = plot_df[plot_df["WIN_CLASS"] == cls]
        pct = (100 * len(sub) / len(plot_df)) if len(plot_df) > 0 else 0

        ax.scatter(
            sub["Lon"],
            sub["Lat"],
            s=90,
            color=color_map.get(cls, "k"),
            edgecolor="black",
            linewidth=0.55,
            alpha=0.82,
            label=f"{cls} ({pct:.1f}%)",
        )

    ax.set_xlim(*(NA_XLIM if region == "NorthAmerica" else AMERICAS_XLIM))
    ax.set_ylim(*(NA_YLIM if region == "NorthAmerica" else AMERICAS_YLIM))

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    ax.set_ylabel(fr"$\mathbf{{{flux}}}$" + "\nLatitude")

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.35,
        color="gray",
        alpha=0.22,
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.75,
        facecolor="white",
        edgecolor="none",
        fontsize=17,
    )


def print_category_counts(dfs: Dict[str, pd.DataFrame], region: str) -> None:
    """Print IGBP and climate class site counts for QC."""
    print(f"\n===== Category counts for {region} =====")

    for flux, df in dfs.items():
        if df.empty:
            print(f"\n{flux}: empty")
            continue

        print(f"\n--- {flux} ---")

        print("IGBP site counts:")
        print(df.groupby("IGBP")["SITE_ID"].nunique().sort_values(ascending=False))

        print("\nClimate class site counts:")
        if CLIMATE_COL in df.columns:
            print(df.groupby(CLIMATE_COL)["SITE_ID"].nunique().sort_values(ascending=False))
        else:
            print(f"{CLIMATE_COL} not found")


# ============================================================
# 7. Combined figures
# ============================================================

def plot_figure_combined(
    dfs: Dict[str, pd.DataFrame],
    out_png: str,
    out_pdf: str,
    threshold: float,
    region: str,
    min_sites_filter: int,
    figure_tag: str,
) -> None:
    """
    3 x 3 layout:

    Row 1: NEE
    Row 2: GPP
    Row 3: RECO

    Col 1: Map
    Col 2: IGBP biome boxplots
    Col 3: Koppen-Geiger climate-class boxplots
    """
    if figure_tag == "supplementary":
        fig_size = (34, 25)
        width_ratios = [1.0, 2.35, 1.85]
        climate_rotate = 90
        biome_rotate = 45
    else:
        fig_size = (30, 24)
        width_ratios = [1.0, 2.25, 1.45]
        climate_rotate = 45
        biome_rotate = 45

    fig = plt.figure(figsize=fig_size)

    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=width_ratios,
        height_ratios=[1, 1, 1],
        hspace=0.46,
        wspace=0.38,
    )

    for row_idx, flux in enumerate(TARGET_FLUXES):
        df_raw = dfs[flux]

        if df_raw.empty:
            continue

        # For map panel, use IGBP-filtered subset in the main figure only.
        df_map = df_raw.copy()

        if min_sites_filter > 0:
            counts_igbp = df_map.groupby("IGBP")["SITE_ID"].nunique()
            valid_igbp = counts_igbp[counts_igbp >= min_sites_filter].index
            df_map = df_map[df_map["IGBP"].isin(valid_igbp)].copy()

        if df_map.empty:
            df_map = df_raw.copy()

        # Panel A/D/G: Map
        ax1 = fig.add_subplot(gs[row_idx, 0])
        add_panel_label(ax1, chr(97 + row_idx * 3), x_offset=-0.20)

        plot_map_panel(
            ax=ax1,
            df_site=df_map,
            flux=flux,
            region=region,
            threshold=threshold,
        )

        if row_idx == 2:
            ax1.set_xlabel("Longitude")

        # Panel B/E/H: IGBP
        ax2 = fig.add_subplot(gs[row_idx, 1])
        add_panel_label(ax2, chr(98 + row_idx * 3), x_offset=-0.085)

        plot_vertical_boxplot(
            ax=ax2,
            df=df_raw,
            group_col="IGBP",
            value_col=VALUE_COL,
            group_order=BIOME_ORDER,
            box_color="#f4a582",
            x_label="IGBP biome",
            y_label=r"$\Delta$RMSE",
            min_sites_filter=min_sites_filter,
            show_xlabel=(row_idx == 2),
            rotate_labels=biome_rotate,
            drop_unknown_when_filter=True,
        )

        # Panel C/F/I: climate
        ax3 = fig.add_subplot(gs[row_idx, 2])
        add_panel_label(ax3, chr(99 + row_idx * 3), x_offset=-0.19)

        plot_vertical_boxplot(
            ax=ax3,
            df=df_raw,
            group_col=CLIMATE_COL,
            value_col=VALUE_COL,
            group_order=CLIMATE_ORDER,
            box_color="#92c5de",
            x_label="Koppen-Geiger climate class",
            y_label=r"$\Delta$RMSE",
            min_sites_filter=min_sites_filter,
            show_xlabel=(row_idx == 2),
            rotate_labels=climate_rotate,
            drop_unknown_when_filter=True,
        )

    save_manuscript_png(fig, out_png, render_dpi=300)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined {figure_tag} figure:")
    print(f"  {out_png}")
    print(f"  {out_pdf}")


# ============================================================
# 8. Single-panel figures
# ============================================================

def save_single_map_panel(
    df: pd.DataFrame,
    flux: str,
    region: str,
    threshold: float,
    out_png: str,
    out_pdf: str,
) -> None:
    """Save one map panel as a standalone figure."""
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 8))

    plot_map_panel(
        ax=ax,
        df_site=df,
        flux=flux,
        region=region,
        threshold=threshold,
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{flux}: spatial distribution of site-level performance difference",
        fontsize=24,
        fontweight="bold",
        loc="left",
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_single_vertical_boxplot(
    df: pd.DataFrame,
    group_col: str,
    group_order: List[str],
    group_label: str,
    flux: str,
    out_png: str,
    out_pdf: str,
    box_color: str,
    min_sites_filter: int,
) -> None:
    """Save a standalone main-style vertical boxplot."""
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(15, 10))

    plot_vertical_boxplot(
        ax=ax,
        df=df,
        group_col=group_col,
        value_col=VALUE_COL,
        group_order=group_order,
        box_color=box_color,
        x_label=group_label,
        y_label=r"$\Delta$RMSE",
        min_sites_filter=min_sites_filter,
        show_xlabel=True,
        rotate_labels=45,
        drop_unknown_when_filter=True,
    )

    ax.set_title(
        f"{flux}: site-level performance difference by {group_label}",
        fontsize=24,
        fontweight="bold",
        loc="left",
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_supp_horizontal_boxplot_with_n_axis(
    df: pd.DataFrame,
    group_col: str,
    group_order: List[str],
    group_label: str,
    flux: str,
    out_png: str,
    out_pdf: str,
    box_color: str,
    min_sites_filter: int,
) -> None:
    """
    Save a supplementary-style horizontal boxplot.

    Left y-axis:
        category names

    Right y-axis:
        number of sites
    """
    if df.empty or group_col not in df.columns:
        return

    df_plot, final_order, data, counts = get_group_order_and_data(
        df=df,
        group_col=group_col,
        value_col=VALUE_COL,
        group_order=group_order,
        min_sites_filter=min_sites_filter,
        drop_unknown_when_filter=False,
    )

    if df_plot.empty or len(final_order) == 0:
        return

    y_pos = np.arange(1, len(final_order) + 1)

    fig_height = max(8, 0.48 * len(final_order) + 3.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    bp = ax.boxplot(
        data,
        vert=False,
        positions=y_pos,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.3),
        whiskerprops=dict(color="black", linewidth=1.1),
        capprops=dict(color="black", linewidth=1.1),
        boxprops=dict(edgecolor="black", linewidth=1.1),
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(box_color)
        patch.set_alpha(0.88)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(final_order)

    ax.set_xlabel(r"$\Delta$RMSE")
    ax.set_ylabel(group_label)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.35,
        color="gray",
        alpha=0.25,
    )

    format_nature_axes(ax)

    # Right y-axis: n sites.
    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels([str(n) for n in counts])
    ax_right.set_ylabel("Number of sites")

    ax_right.spines["top"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(axis="y", direction="out", length=5, width=1.1)

    ax.set_title(
        f"{flux}: all categories by {group_label}",
        fontsize=24,
        fontweight="bold",
        loc="left",
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_all_single_panels(
    dfs_plot: Dict[str, pd.DataFrame],
    region: str,
) -> None:
    """Save standalone panels for each flux and grouping."""
    single_dir = os.path.join(OUTDIR, "single_panels")
    ensure_outdir(single_dir)

    for flux in TARGET_FLUXES:
        df_flux = dfs_plot[flux]

        if df_flux.empty:
            continue

        save_single_map_panel(
            df=df_flux,
            flux=flux,
            region=region,
            threshold=WIN_TIE_THRESHOLD,
            out_png=os.path.join(single_dir, f"Single_Map_{region}_{flux}.png"),
            out_pdf=os.path.join(single_dir, f"Single_Map_{region}_{flux}.pdf"),
        )

        save_single_vertical_boxplot(
            df=df_flux,
            group_col="IGBP",
            group_order=BIOME_ORDER,
            group_label="IGBP biome",
            flux=flux,
            out_png=os.path.join(single_dir, f"Single_Main_{region}_{flux}_IGBP_n10.png"),
            out_pdf=os.path.join(single_dir, f"Single_Main_{region}_{flux}_IGBP_n10.pdf"),
            box_color="#f4a582",
            min_sites_filter=10,
        )

        save_single_vertical_boxplot(
            df=df_flux,
            group_col=CLIMATE_COL,
            group_order=CLIMATE_ORDER,
            group_label="Koppen-Geiger climate class",
            flux=flux,
            out_png=os.path.join(single_dir, f"Single_Main_{region}_{flux}_Climate_n10.png"),
            out_pdf=os.path.join(single_dir, f"Single_Main_{region}_{flux}_Climate_n10.pdf"),
            box_color="#92c5de",
            min_sites_filter=10,
        )

        save_supp_horizontal_boxplot_with_n_axis(
            df=df_flux,
            group_col="IGBP",
            group_order=BIOME_ORDER,
            group_label="IGBP biome",
            flux=flux,
            out_png=os.path.join(single_dir, f"Single_Supp_{region}_{flux}_IGBP_all_horizontal.png"),
            out_pdf=os.path.join(single_dir, f"Single_Supp_{region}_{flux}_IGBP_all_horizontal.pdf"),
            box_color="#f4a582",
            min_sites_filter=0,
        )

        save_supp_horizontal_boxplot_with_n_axis(
            df=df_flux,
            group_col=CLIMATE_COL,
            group_order=CLIMATE_ORDER,
            group_label="Koppen-Geiger climate class",
            flux=flux,
            out_png=os.path.join(single_dir, f"Single_Supp_{region}_{flux}_Climate_all_horizontal.png"),
            out_pdf=os.path.join(single_dir, f"Single_Supp_{region}_{flux}_Climate_all_horizontal.pdf"),
            box_color="#92c5de",
            min_sites_filter=0,
        )


# ============================================================
# 9. Main workflow
# ============================================================

def main() -> None:
    ensure_outdir(OUTDIR)

    print("Loading site metadata...")
    meta = build_site_metadata(SITE_META_CSV)

    print("\nSite metadata preview:")
    cols_show = [
        c for c in [
            "SITE_ID", "Lat", "Lon", "IGBP",
            "Clim", "Clim_fill", "CLIM_CODE_USE",
            "MAT_C", "MAP_mm",
        ]
        if c in meta.columns
    ]
    print(meta[cols_show].head())

    print("\nOverall detailed Koppen-Geiger classes by site:")
    print(meta["CLIM_CODE_USE"].value_counts(dropna=False))

    print("\nOverall IGBP classes by site:")
    print(meta["IGBP"].value_counts(dropna=False))

    site_meta_out = os.path.join(OUTDIR, "Fig4_v25_site_metadata_with_climate.csv")
    meta.to_csv(site_meta_out, index=False)
    print(f"\nSaved site metadata with final climate classes: {site_meta_out}")

    print("\nLoading model arrays...")
    f_p, f_t, f_s = load_model_arrays(FAT_DIR)
    t_p, t_t, t_s = load_model_arrays(TRANSFORMER_DIR)
    r_p, r_t, r_s = load_model_arrays(RF_DIR)

    print("\nArray shapes:")
    print("FAT pred:", f_p.shape, "true:", f_t.shape, "site:", f_s.shape)
    print("TR  pred:", t_p.shape, "true:", t_t.shape, "site:", t_s.shape)
    print("RF  pred:", r_p.shape, "true:", r_t.shape, "site:", r_s.shape)

    if not (f_p.shape == t_p.shape == r_p.shape == f_t.shape):
        raise ValueError(
            "Prediction/true array shapes are not aligned. "
            f"FAT={f_p.shape}, TR={t_p.shape}, RF={r_p.shape}, TRUE={f_t.shape}"
        )

    if len(f_s) != f_p.shape[0]:
        raise ValueError(
            f"Site ID length does not match flattened prediction length: "
            f"len(site)={len(f_s)}, n_pred={f_p.shape[0]}"
        )

    dfs_all = {}

    for target in TARGET_FLUXES:
        print(f"\nComputing site-level metrics for {target}...")

        metrics = compute_metrics(
            fat_pred=f_p,
            tr_pred=t_p,
            rf_pred=r_p,
            reference_true=f_t,
            site_ids=f_s,
            target_flux=target,
        )

        if metrics.empty:
            print(f"Warning: metrics for {target} are empty.")
            dfs_all[target] = pd.DataFrame()
            continue

        df_target = meta.merge(metrics, on="SITE_ID", how="inner")

        df_target = df_target[
            np.isfinite(df_target["Lat"]) &
            np.isfinite(df_target["Lon"])
        ].copy()

        dfs_all[target] = df_target

        print(f"{target}: n sites after merge = {df_target['SITE_ID'].nunique()}")

        print("IGBP counts:")
        print(df_target.groupby("IGBP")["SITE_ID"].nunique().sort_values(ascending=False))

        print("Climate counts:")
        print(df_target.groupby(CLIMATE_COL)["SITE_ID"].nunique().sort_values(ascending=False))

    for region in ["Americas", "NorthAmerica"]:
        print(f"\nPreparing region: {region}")

        dfs_plot = {}

        for target in TARGET_FLUXES:
            df = dfs_all[target]

            if df.empty:
                dfs_plot[target] = df
                continue

            if region == "NorthAmerica":
                mask = (
                    (df["Lon"] >= NA_XLIM[0]) &
                    (df["Lon"] <= NA_XLIM[1]) &
                    (df["Lat"] >= NA_YLIM[0]) &
                    (df["Lat"] <= NA_YLIM[1])
                )
            else:
                mask = (
                    (df["Lon"] >= AMERICAS_XLIM[0]) &
                    (df["Lon"] <= AMERICAS_XLIM[1]) &
                    (df["Lat"] >= AMERICAS_YLIM[0]) &
                    (df["Lat"] <= AMERICAS_YLIM[1])
                )

            dfs_plot[target] = df[mask].copy()

        print_category_counts(dfs_plot, region)

        if all(df.empty for df in dfs_plot.values()):
            print(f"Skipping {region}: no valid data.")
            continue

        # ------------------------------------------------------------
        # Category statistics tables
        # ------------------------------------------------------------
        out_stats_main_csv = os.path.join(
            OUTDIR,
            f"Fig4_v25_Main_category_stats_{region}_n10.csv",
        )

        save_category_stats_csv(
            dfs_plot=dfs_plot,
            region=region,
            figure_set="Main_n10",
            out_csv=out_stats_main_csv,
            min_sites_filter=10,
            threshold=WIN_TIE_THRESHOLD,
        )

        out_stats_supp_csv = os.path.join(
            OUTDIR,
            f"Fig4_v25_Supp_category_stats_{region}_all.csv",
        )

        save_category_stats_csv(
            dfs_plot=dfs_plot,
            region=region,
            figure_set="Supp_all",
            out_csv=out_stats_supp_csv,
            min_sites_filter=0,
            threshold=WIN_TIE_THRESHOLD,
        )

        # ------------------------------------------------------------
        # Main combined figure: n >= 10, no top n-axis
        # ------------------------------------------------------------
        out_main_png = os.path.join(
            OUTDIR,
            f"Fig4_v25_Main_{region}_combined_n10_no_top_n_axis.png",
        )
        out_main_pdf = os.path.join(
            OUTDIR,
            f"Fig4_v25_Main_{region}_combined_n10_no_top_n_axis.pdf",
        )

        print(f"\nPlotting main combined figure for {region}: n >= 10 categories")

        plot_figure_combined(
            dfs=dfs_plot,
            out_png=out_main_png,
            out_pdf=out_main_pdf,
            threshold=WIN_TIE_THRESHOLD,
            region=region,
            min_sites_filter=10,
            figure_tag="main",
        )

        # ------------------------------------------------------------
        # Supplementary combined figure: all categories, no top n-axis
        # ------------------------------------------------------------
        out_supp_png = os.path.join(
            OUTDIR,
            f"Fig4_v25_Supp_{region}_combined_all_classes_no_top_n_axis.png",
        )
        out_supp_pdf = os.path.join(
            OUTDIR,
            f"Fig4_v25_Supp_{region}_combined_all_classes_no_top_n_axis.pdf",
        )

        print(f"\nPlotting supplementary combined figure for {region}: all categories")

        plot_figure_combined(
            dfs=dfs_plot,
            out_png=out_supp_png,
            out_pdf=out_supp_pdf,
            threshold=WIN_TIE_THRESHOLD,
            region=region,
            min_sites_filter=0,
            figure_tag="supplementary",
        )

        # ------------------------------------------------------------
        # Single-panel figures
        # ------------------------------------------------------------
        print(f"\nSaving single-panel figures for {region}...")

        save_all_single_panels(
            dfs_plot=dfs_plot,
            region=region,
        )

    print("\nSuccess: v25 clean figures and category statistics tables generated.")


if __name__ == "__main__":
    main()
