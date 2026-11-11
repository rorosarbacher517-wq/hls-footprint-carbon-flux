#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Forest baseline with deterministic spatial aggregation.

Modified version:
1. Match DL inputs as closely as possible:
   - HLS uses only the first 6 bands
   - Flux/mete features include:
       DoY, Sensor, Hour,
       d2m_C, t2m_C, stl1_C, stl2_C, stl3_C, stl4_C,
       swvl1_pct, swvl2_pct, swvl3_pct, swvl4_pct,
       SSRD_Wm2, VPD_kPa
2. Keep preprocessing steps aligned with the DL pipeline:
   - same kfold split
   - same normalization
   - same preprocess_inputs
   - same center non-fill filtering
3. Save both fold-level and final kfold merged outputs/evaluation
   - fold*/pred/true.npy, pred.npy, site.npy
   - fold*/eval/
   - kfold5/pred/ and kfold5/eval/
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
import json
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor

from fair_interfaces import validate_predictions, audit_fold, host_inputs, aggregate_reference
import train_test
import Evaluate_DL_RF
import Model_build

from model_heads_ffp import resolve_mode, FFP_HEADS_VERSION

base_name = os.path.basename(__file__).replace(".py", "")

TARGET_NAME_TO_INDEX = {
    "NEE": 7,
    "GPP": 8,
    "RECO": 9,
}
CHANNEL_ORDER = ["NEE", "GPP", "RECO"]


# =========================================================
# Utilities
# =========================================================
def to_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else x


def cast_selected_cols_inplace(site_like, cols_to_cast, dtype=np.float32):
    arr = to_numpy(site_like)
    out = np.array(arr, copy=True)
    for j in cols_to_cast:
        out[:, :, j] = out[:, :, j].astype(dtype)
    return out


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path, obj):
    # Output interface only: avoid truncated histories when LR is np.float32.
    from pathlib import Path
    def json_default(value):
        if isinstance(value, np.generic):return value.item()
        if isinstance(value, np.ndarray):return value.tolist()
        raise TypeError('Not JSON serializable: '+type(value).__name__)
    serialized=json.dumps(obj,indent=2,default=json_default)
    destination=Path(path)
    temporary=destination.with_name(destination.name+'.tmp')
    temporary.write_text(serialized,encoding='utf-8')
    temporary.replace(destination)



def write_txt_lines(path, mapping):
    with open(path, "w", encoding="utf-8") as f:
        for k, v in mapping.items():
            f.write(f"{k}: {v}\n")

def preprocess_inputs(HLS_train_norm1, flux_train_norm0, FFP_train):
    site_names = flux_train_norm0[:, :, 0]
    flux_numeric = flux_train_norm0[:, :, 1:].astype(np.float32)
    flux_numeric_tensor = tf.convert_to_tensor(flux_numeric, dtype=tf.float32)
    HLS_tensor = tf.convert_to_tensor(HLS_train_norm1, dtype=tf.float32)
    FFP_tensor = tf.convert_to_tensor(FFP_train, dtype=tf.float32)
    return site_names, HLS_tensor, flux_numeric_tensor, FFP_tensor

# =========================================================
# Flux column names after preprocess_inputs()
# =========================================================
FLUX_NUMERIC_COLUMN_NAMES = [
    "Lat", "Lon", "Year", "DoY", "HLS_Hour", "Sensor", "Hour",
    "NEE_VUT_REF", "GPP_NT_VUT_REF", "RECO_NT_VUT_REF",
    "NEE_VUT_REF_QC", "USTAR", "SW_IN_F_MDS", "SW_IN_F_MDS_QC",
    "TA_F_MDS", "TA_F_MDS_QC", "TS_F_MDS_1", "TS_F_MDS_1_QC",
    "P_F", "P_F_QC", "RH_x", "RH_QC", "VPD_F", "VPD_F_QC",
    "PA_F", "PA_F_QC", "WS_F", "WS_F_QC", "WD", "WD_F_QC",
    "SWC_F_MDS_1", "SWC_F_MDS_1_QC", "CO2_F_MDS", "CO2_F_MDS_QC",
    "LE_F_MDS", "LE_F_MDS_QC", "H_F_MDS", "H_F_MDS_QC",
    "era5_lat", "era5_lon", "sdfor", "u10", "v10", "d2m", "t2m",
    "blh", "zust", "iews", "ie", "inss", "ishf", "stl1", "stl2",
    "stl3", "stl4", "sp", "swvl1", "swvl2", "swvl3", "swvl4",
    "tp", "sshf", "ssrd", "RH", "VPD_kPa", "WS_ms", "WD_deg", "L_m",
    "t2m_C", "d2m_C", "stl1_C", "stl2_C", "stl3_C", "stl4_C",
    "sp_kPa", "swvl1_pct", "swvl2_pct", "swvl3_pct", "swvl4_pct",
    "P_mm", "SSRD_Wm2", "SSHF_Wm2", "time_utc", "time_local",
    "MH", "MH_QC", "MH_source", "IGBP", "CH", "CH_QC", "CH_source",
    "V_SIGMA", "V_SIGMA_QC", "h_aero", "d", "z0", "zm", "lambda_rs",
    "ha_QC", "h_aero_smooth", "d_smooth", "z0_smooth", "zm_smooth",
    "z_over_L", "Hour_sin", "Hour_cos", "WD_sin", "WD_cos", "sigma_v_rf"
]


def get_fair_rf_flux_feature_indices():
    """
    Match RF inputs to the DL model as closely as possible.

    Included:
    - DoY
    - Sensor (already encoded as 0/1)
    - Hour
    - selected meteorological variables from DL n_index
    """
    fair_idx = [
        3,   # DoY
        5,   # Sensor
        6,   # Hour
        69,  # d2m_C
        68,  # t2m_C
        70,  # stl1_C
        71,  # stl2_C
        72,  # stl3_C
        73,  # stl4_C
        75,  # swvl1_pct
        76,  # swvl2_pct
        77,  # swvl3_pct
        78,  # swvl4_pct
        80,  # SSRD_Wm2
        64   # VPD_kPa
    ]
    return fair_idx


def debug_column_validity(X_3d, cols, name, fill_value=-9999.0, max_print=500):
    arr = np.array(X_3d, dtype=np.float32, copy=False)
    flat = arr.reshape(-1, arr.shape[-1])

    print(f"\n[DEBUG] Validity summary for {name}:")
    print(f"[DEBUG] total flattened rows = {flat.shape[0]}")
    print(f"[DEBUG] checking {len(cols)} columns")

    bad_cols = []
    for c in cols[:max_print]:
        col = flat[:, c]
        invalid = np.sum((~np.isfinite(col)) | (col == fill_value))
        valid = col.shape[0] - invalid
        col_name = FLUX_NUMERIC_COLUMN_NAMES[c] if c < len(FLUX_NUMERIC_COLUMN_NAMES) else f"col_{c}"
        print(f"  col={c:3d} | name={col_name:15s} | invalid={invalid:8d} | valid={valid:8d}")
        if valid == 0:
            bad_cols.append(c)

    if len(cols) > max_print:
        print(f"[DEBUG] ... truncated, only first {max_print} columns printed")

    if len(bad_cols) > 0:
        print("[DEBUG] fully invalid columns:")
        for c in bad_cols:
            col_name = FLUX_NUMERIC_COLUMN_NAMES[c] if c < len(FLUX_NUMERIC_COLUMN_NAMES) else f"col_{c}"
            print(f"  {c}: {col_name}")
    else:
        print("[DEBUG] fully invalid columns: []")


def auto_filter_flux_feature_indices(flux_numeric, candidate_idx, fill_value=-9999.0,
                                     min_valid_ratio=0.05):
    arr = np.array(flux_numeric, dtype=np.float32, copy=False)
    flat = arr.reshape(-1, arr.shape[-1])

    keep_idx = []
    drop_idx = []
    ratio_map = {}

    for c in candidate_idx:
        col = flat[:, c]
        valid = np.isfinite(col) & (col != fill_value)
        ratio = np.mean(valid)
        ratio_map[c] = float(ratio)

        if ratio > min_valid_ratio:
            keep_idx.append(c)
        else:
            drop_idx.append(c)

    print("\n[DEBUG] auto_filter_flux_feature_indices")
    print("[DEBUG] min_valid_ratio:", min_valid_ratio)

    print("[DEBUG] kept columns:")
    for c in keep_idx:
        col_name = FLUX_NUMERIC_COLUMN_NAMES[c] if c < len(FLUX_NUMERIC_COLUMN_NAMES) else f"col_{c}"
        print(f"  {c:3d} | {col_name:15s} | valid_ratio={ratio_map[c]:.4f}")

    print("[DEBUG] dropped columns:")
    for c in drop_idx:
        col_name = FLUX_NUMERIC_COLUMN_NAMES[c] if c < len(FLUX_NUMERIC_COLUMN_NAMES) else f"col_{c}"
        print(f"  {c:3d} | {col_name:15s} | valid_ratio={ratio_map[c]:.4f}")

    return keep_idx, drop_idx, ratio_map


def make_valid_row_mask(X_flat, Y_flat, x_cols_to_check=None, fill_value=-9999.0):
    X_flat = np.array(X_flat, dtype=np.float32, copy=False)
    Y_flat = np.array(Y_flat, dtype=np.float32, copy=False)

    if x_cols_to_check is not None:
        X_check = X_flat[:, x_cols_to_check]
    else:
        X_check = X_flat

    valid_x = (
        np.all(np.isfinite(X_check), axis=1) &
        np.all(X_check != fill_value, axis=1)
    )

    valid_y = (
        np.all(np.isfinite(Y_flat), axis=1) &
        np.all(Y_flat != fill_value, axis=1)
    )

    return valid_x & valid_y


# =========================================================
# Spatial aggregation helpers
# =========================================================
def get_hls_valid_mask(hls, n_bands=6, mask_value=-9999.0):
    hls = to_numpy(hls)
    sub = hls[:, :, :, :n_bands]
    return np.all(np.isfinite(sub) & (sub != mask_value), axis=-1)


def center_block_mask(H, W, k):
    mask = np.zeros((H, W), dtype=np.float32)
    ch = H // 2
    cw = W // 2
    half = k // 2
    top = ch - half
    left = cw - half
    mask[top:top + k, left:left + k] = 1.0
    return mask


def gaussian_mask(H, W, sigma=0.35):
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dy = yy - cy
    dx = xx - cx
    r = np.sqrt(dx ** 2 + dy ** 2)
    r_norm = r / (r.max() + 1e-12)
    g = np.exp(-(r_norm ** 2) / (2.0 * sigma * sigma))
    return g.astype(np.float32)


def normalize_2d_weights(weights2d, valid2d):
    w = weights2d.astype(np.float32) * valid2d.astype(np.float32)
    s = w.sum()
    if s <= 0:
        w = valid2d.astype(np.float32)
        s = w.sum()
    if s <= 0:
        return w
    return w / s


def aggregate_static_hls(hls, valid_mask, spatial_weights):
    hls = to_numpy(hls)
    valid_mask = to_numpy(valid_mask)
    N, H, W, C = hls.shape
    out = np.full((N, C), np.nan, dtype=np.float32)

    for i in range(N):
        w = normalize_2d_weights(spatial_weights, valid_mask[i])
        for c in range(C):
            band = hls[i, :, :, c]
            band_valid = np.isfinite(band) & valid_mask[i]
            if np.sum(w * band_valid.astype(np.float32)) <= 0:
                out[i, c] = np.nan
            else:
                ww = normalize_2d_weights(w, band_valid)
                out[i, c] = np.nansum(ww * band)

    return out


def aggregate_dynamic_ffp_hls(hls, valid_mask, ffp):
    hls = to_numpy(hls)
    valid_mask = to_numpy(valid_mask)
    ffp = to_numpy(ffp)
    N, H, W, C = hls.shape
    T = ffp.shape[1]
    out = np.full((N, T, C), np.nan, dtype=np.float32)
    center1 = center_block_mask(H, W, 1)

    for i in range(N):
        for t in range(T):
            ffp_w = np.array(ffp[i, t], dtype=np.float32)
            ffp_valid = np.isfinite(ffp_w)
            valid = valid_mask[i] & ffp_valid

            ffp_w = np.where(np.isfinite(ffp_w) & (ffp_w > 0), ffp_w, 0.0)

            if np.sum(ffp_w * valid.astype(np.float32)) <= 0:
                base_w = normalize_2d_weights(center1, valid_mask[i])
            else:
                base_w = normalize_2d_weights(ffp_w, valid)

            for c in range(C):
                band = hls[i, :, :, c]
                band_valid = np.isfinite(band) & valid_mask[i]
                if np.sum(base_w * band_valid.astype(np.float32)) <= 0:
                    out[i, t, c] = np.nan
                else:
                    ww = normalize_2d_weights(base_w, band_valid)
                    out[i, t, c] = np.nansum(ww * band)

    return out


def aggregate_hls_by_mode(hls, ffp, mode_name, n_times, n_bands=6, mask_value=-9999.0):
    hls = to_numpy(hls)
    ffp = to_numpy(ffp)
    N, H, W, C = hls.shape
    valid_mask = get_hls_valid_mask(hls, n_bands=n_bands, mask_value=mask_value)

    if mode_name == "dynamic_ffp":
        return aggregate_reference(hls, ffp)

    if mode_name == "uniform_valid":
        spatial_w = np.ones((H, W), dtype=np.float32)
    elif mode_name == "center1":
        spatial_w = center_block_mask(H, W, 1)
    elif mode_name == "center3":
        spatial_w = center_block_mask(H, W, 3)
    elif mode_name == "center5":
        spatial_w = center_block_mask(H, W, 5)
    elif mode_name == "gaussian_static":
        spatial_w = gaussian_mask(H, W, sigma=0.35)
    elif mode_name == "gaussian_narrow":
        spatial_w = gaussian_mask(H, W, sigma=0.20)
    elif mode_name == "gaussian_wide":
        spatial_w = gaussian_mask(H, W, sigma=0.55)
    elif mode_name in ["wind_distance", "param_driven"]:
        raise ValueError(
            f"Mode '{mode_name}' is not supported in pure RF because it requires a learned weighting head."
        )
    else:
        raise ValueError(f"Unsupported mode_name: {mode_name}")

    agg_static = aggregate_static_hls(hls, valid_mask, spatial_w)
    agg_time = np.repeat(agg_static[:, np.newaxis, :], n_times, axis=1)
    return agg_time.astype(np.float32)


# =========================================================
# RF feature builder with COMMON 3-target mask
# =========================================================
def build_rf_features_common(HLS_norm1, flux_numeric, FFP_array, mode_name,
                             target_slice=(7, 10), n_bands=6, mask_value=-9999.0,
                             flux_feature_indices=None,
                             auto_drop_bad_flux_cols=True,
                             min_valid_ratio=0.05,
                             debug=True):
    # IMPORTANT:
    # Keep only the first 6 HLS bands to align with the DL model.
    HLS_norm1 = to_numpy(HLS_norm1).astype(np.float32)[..., :n_bands]
    flux_numeric = to_numpy(flux_numeric).astype(np.float32)
    FFP_array = to_numpy(FFP_array).astype(np.float32)

    N, T, F = flux_numeric.shape

    hls_agg = aggregate_hls_by_mode(
        hls=HLS_norm1,
        ffp=FFP_array,
        mode_name=mode_name,
        n_times=T,
        n_bands=n_bands,
        mask_value=mask_value
    )  # [N,T,6]

    n_hls_total = hls_agg.shape[-1]

    # COMMON 3-target Y
    y_raw = flux_numeric[:, :, target_slice[0]:target_slice[1]]  # [N,T,3]

    if flux_feature_indices is None:
        flux_feature_indices = get_fair_rf_flux_feature_indices()

    flux_feature_indices = [
        c for c in flux_feature_indices
        if c < F and c not in range(target_slice[0], target_slice[1])
    ]

    if debug:
        print("\n[build_rf_features_common] Requested fair flux feature indices:")
        for c in flux_feature_indices:
            col_name = FLUX_NUMERIC_COLUMN_NAMES[c] if c < len(FLUX_NUMERIC_COLUMN_NAMES) else f"col_{c}"
            print(f"  {c:3d} | {col_name}")

    if debug:
        debug_column_validity(
            flux_numeric,
            cols=flux_feature_indices + list(range(target_slice[0], target_slice[1])),
            name="flux_numeric (selected feature cols + COMMON 3-target cols)",
            fill_value=mask_value
        )

    if auto_drop_bad_flux_cols:
        flux_feature_indices, dropped_idx, ratio_map = auto_filter_flux_feature_indices(
            flux_numeric,
            candidate_idx=flux_feature_indices,
            fill_value=mask_value,
            min_valid_ratio=min_valid_ratio
        )
    else:
        dropped_idx = []
        ratio_map = {}

    if len(flux_feature_indices) == 0:
        raise ValueError("No usable fair RF flux feature columns remain after filtering.")

    flux_feat_raw = flux_numeric[:, :, flux_feature_indices]

    X_raw = np.concatenate([hls_agg, flux_feat_raw], axis=-1)
    X_flat_raw = X_raw.reshape(-1, X_raw.shape[-1]).astype(np.float32)
    Y_flat_raw = y_raw.reshape(-1, y_raw.shape[-1]).astype(np.float32)

    # Only first 6 HLS bands are present in hls_agg
    hls_band_cols = list(range(n_hls_total))
    flux_feat_cols = list(range(n_hls_total, X_flat_raw.shape[-1]))
    cols_to_check = hls_band_cols + flux_feat_cols

    print(f"\n[build_rf_features_common] X cols total={X_flat_raw.shape[-1]}")
    print(f"[build_rf_features_common] checked cols={len(cols_to_check)}")
    print(f"[build_rf_features_common] HLS checked bands: 0-{n_hls_total - 1}")
    print(f"[build_rf_features_common] flux checked cols in X: {flux_feat_cols}")

    print("[build_rf_features_common] selected original flux feature indices:")
    for c in flux_feature_indices:
        col_name = FLUX_NUMERIC_COLUMN_NAMES[c] if c < len(FLUX_NUMERIC_COLUMN_NAMES) else f"col_{c}"
        print(f"  {c:3d} | {col_name}")

    print("[build_rf_features_common] dropped original flux feature indices:")
    for c in dropped_idx:
        col_name = FLUX_NUMERIC_COLUMN_NAMES[c] if c < len(FLUX_NUMERIC_COLUMN_NAMES) else f"col_{c}"
        print(f"  {c:3d} | {col_name}")

    valid_row_mask = make_valid_row_mask(
        X_flat_raw, Y_flat_raw,
        x_cols_to_check=cols_to_check,
        fill_value=mask_value
    )

    X_flat_full = np.where(X_flat_raw == mask_value, np.nan, X_flat_raw)
    Y_flat_full = np.where(Y_flat_raw == mask_value, np.nan, Y_flat_raw)

    if debug:
        valid_x_only = (
            np.all(np.isfinite(X_flat_raw[:, cols_to_check]), axis=1) &
            np.all(X_flat_raw[:, cols_to_check] != mask_value, axis=1)
        )

        valid_y_only = (
            np.all(np.isfinite(Y_flat_raw), axis=1) &
            np.all(Y_flat_raw != mask_value, axis=1)
        )

        print(f"[DEBUG] rows passing X check only: {np.sum(valid_x_only)} / {len(valid_x_only)}")
        print(f"[DEBUG] rows passing Y check only: {np.sum(valid_y_only)} / {len(valid_y_only)}")
        print(f"[DEBUG] rows passing both X/Y    : {np.sum(valid_row_mask)} / {len(valid_row_mask)}")

    X_shape_info = {
        "N": N,
        "T": T,
        "D": X_raw.shape[-1],
        "target_dim": y_raw.shape[-1],
        "selected_flux_feature_indices": flux_feature_indices,
        "dropped_flux_feature_indices": dropped_idx,
        "selected_flux_feature_names": [
            FLUX_NUMERIC_COLUMN_NAMES[c] if c < len(FLUX_NUMERIC_COLUMN_NAMES) else f"col_{c}"
            for c in flux_feature_indices
        ],
        "dropped_flux_feature_names": [
            FLUX_NUMERIC_COLUMN_NAMES[c] if c < len(FLUX_NUMERIC_COLUMN_NAMES) else f"col_{c}"
            for c in dropped_idx
        ],
        "n_hls_features_used": int(n_hls_total),
    }

    return X_flat_full, Y_flat_full, valid_row_mask, X_shape_info


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    if os.environ.get('STRICT_FAIR_RELEASE') != 'strict_v6_tf215_dualgpu_fourmodel':
        raise RuntimeError('Use the package run.py launcher: frozen controls are mandatory.')

    N_ESTIMATORS = 200
    MAX_DEPTH = 10
    MIN_SAMPLES_LEAF = 10
    N_JOBS = -1
    RANDOM_STATE = 42

    y_SCALE = 1
    ITERS = 1

    N_SPLITS = 5
    SEED = 42
    FOLD_ONLY = 0
    FOLD_LIST = None   # e.g. '0,1' passed by run.py: several folds in one process

    FFP_MODE = 0
    VERSION = "strict_v6_tf215_dualgpu_fourmodel"

    if len(sys.argv) > 1:
        N_ESTIMATORS = int(sys.argv[1])
    if len(sys.argv) > 2:
        md = sys.argv[2]
        MAX_DEPTH = None if md.lower() == "none" else int(md)
    if len(sys.argv) > 3:
        MIN_SAMPLES_LEAF = int(sys.argv[3])
    if len(sys.argv) > 4:
        FOLD_LIST = [int(x) for x in str(sys.argv[4]).split(',')]
        FOLD_ONLY = FOLD_LIST[0] if len(FOLD_LIST) == 1 else -2
    if len(sys.argv) > 5:
        try:
            FFP_MODE = int(sys.argv[5])
        except ValueError:
            FFP_MODE = sys.argv[5]
    if len(sys.argv) > 6:
        VERSION = str(sys.argv[6])

    TARGET_VARS = ["NEE", "GPP", "RECO"]

    # ---------------------------------------------------------
    # Unified output root for RF results, kfold splits, and
    # normalization statistics. This keeps RF/CNN/FAT using the
    # same CV artifacts when TRAIN_DATA_ROOT is shared.
    # ---------------------------------------------------------
    DATA_ROOT = os.environ.get("TRAIN_DATA_ROOT", "./data")
    ensure_dir(DATA_ROOT)

    mode_name = resolve_mode(FFP_MODE)
    if mode_name in ["wind_distance", "param_driven"]:
        raise ValueError(
            f"RF runner does not support mode '{mode_name}'. Use deterministic aggregation modes only."
        )

    if isinstance(FFP_MODE, int):
        mode_dir_name = f"FFP{FFP_MODE}_{mode_name}"
        priority_group = "main" if FFP_MODE in [0, 1, 2, 3] else "supplementary"
    else:
        mode_dir_name = f"{mode_name}"
        priority_group = "unknown"

    base_tag_nofold = (
        f"{base_name}"
        f".nest{N_ESTIMATORS}"
        f".depth{MAX_DEPTH}"
        f".leaf{MIN_SAMPLES_LEAF}"
    )

    version_root = ensure_dir(os.path.join(DATA_ROOT, "RF", VERSION))
    mode_root = ensure_dir(os.path.join(version_root, mode_dir_name))

    FAIR_RF_FLUX_FEATURE_IDXS = get_fair_rf_flux_feature_indices()

    mode_info = {
        "version": VERSION,
        "DATA_ROOT": DATA_ROOT,
        "dataset_variant": "remove_selected_sites",
        "base_script": base_name,
        "base_tag_nofold": base_tag_nofold,
        "FFP_input": FFP_MODE,
        "mode_name": mode_name,
        "priority_group": priority_group,
        "ffp_heads_version": FFP_HEADS_VERSION,
        "model_type": "RandomForestRegressor_input_aggregation_single_target_common3mask",
        "N_ESTIMATORS": N_ESTIMATORS,
        "MAX_DEPTH": MAX_DEPTH,
        "MIN_SAMPLES_LEAF": MIN_SAMPLES_LEAF,
        "N_JOBS": N_JOBS,
        "RANDOM_STATE": RANDOM_STATE,
        "N_SPLITS": N_SPLITS,
        "SEED": SEED,
        "FOLD_ONLY": FOLD_ONLY,
        "y_SCALE": y_SCALE,
        "ITERS": ITERS,
        "target_vars": TARGET_VARS,
        "channel_order": CHANNEL_ORDER,
        "common_target_slice": [7, 10],
        "n_hls_bands_used": 6,
        "aligned_flux_feature_indices": FAIR_RF_FLUX_FEATURE_IDXS,
        "aligned_flux_feature_names": [
            FLUX_NUMERIC_COLUMN_NAMES[c] for c in FAIR_RF_FLUX_FEATURE_IDXS
        ],
        "notes": (
            "Aligned RF version: uses first 6 HLS bands only, and uses "
            "DoY + Sensor + Hour + selected mete variables to match the DL setup as closely as possible. "
            "Fold-level outputs/evaluation and final kfold merged outputs/evaluation are saved."
        ),
    }
    write_json(os.path.join(mode_root, "mode_spec.json"), mode_info)
    write_txt_lines(os.path.join(mode_root, "mode_info.txt"), mode_info)

    inputdata_path = os.environ.get(
        "INPUTDATA_PATH",
        './data/input_data',
    )
    x1_ref_array = np.load(os.path.join(inputdata_path, "HLS_matched_v11_remove_selected_sites.npy"), allow_pickle=True)
    x2_y_array = np.load(os.path.join(inputdata_path, "merged_flux_era5_ffp_array_v11_remove_selected_sites.npy"), allow_pickle=True)
    ffp_array = np.load(os.path.join(inputdata_path, "FFP_hls_aligned_v11_50x50_timefilled_remove_selected_sites.npy"), allow_pickle=True)

    if FOLD_LIST is not None and len(FOLD_LIST) > 1:
        fold_list = sorted(set(FOLD_LIST))
    elif FOLD_ONLY < 0:
        fold_list = list(range(N_SPLITS))
    else:
        fold_list = [FOLD_ONLY]

    kfold_true_list = []
    kfold_pred_list = []
    kfold_site_list = []

    from fair_interfaces import protect_inputs, release_fold_memory
    protect_inputs(globals())
    for fold in fold_list:
        release_fold_memory(globals())
        print(f"\n========== Running RF version={VERSION}, mode={mode_name}, fold {fold} / {N_SPLITS-1} ==========")

        # Keep fold model dir for saved models/specs if needed
        fold_root = ensure_dir(os.path.join(mode_root, f"fold{fold}"))
        combined_model_dir = ensure_dir(os.path.join(fold_root, "model"))

        split_dir = ensure_dir(os.path.join(DATA_ROOT, "split_kfold"))
        index_train, index_test = train_test.get_kfold_siteid(
            x2_y_array, n_splits=N_SPLITS, fold_id=fold, seed=SEED, split_dir=split_dir
        )

        norm_dir = ensure_dir(os.path.join(DATA_ROOT, "norm_stats_kfold"))
        HLS_train_norm0, HLS_test_norm0 = train_test.get_HLS_normalization_joint(
            x1_ref_array, x2_y_array, index_train, index_test,
            is_joint_norm=True,
            fold_id=fold,
            output_dir=norm_dir
        )

        flux_norm_dir = ensure_dir(os.path.join(DATA_ROOT, "flux_norm_stats_kfold"))
        flux_train_norm0, flux_test_norm0 = train_test.normalize_flux_train_test(
            x2_y_array, index_train, index_test,
            use_mete=True, use_xy=True, use_sensor=False,
            fold_id=fold,
            save_dir=flux_norm_dir
        )

        HLS_train_norm1 = np.transpose(HLS_train_norm0, (0, 2, 3, 1))
        HLS_test_norm1 = np.transpose(HLS_test_norm0, (0, 2, 3, 1))

        index_train_non_fill = (HLS_train_norm1[:, 23:27, 23:27, 0] != -9999.).sum(axis=(1, 2)) == 4 * 4
        index_test_non_fill = (HLS_test_norm1[:, 23:27, 23:27, 0] != -9999.).sum(axis=(1, 2)) == 4 * 4

        HLS_train_norm1 = HLS_train_norm1[index_train_non_fill]
        HLS_test_norm1 = HLS_test_norm1[index_test_non_fill]
        flux_train_norm0 = flux_train_norm0[index_train_non_fill]
        flux_test_norm0 = flux_test_norm0[index_test_non_fill]

        FFP_train = ffp_array[index_train][index_train_non_fill]
        FFP_test = ffp_array[index_test][index_test_non_fill]

        flux_test_site_mete = x2_y_array[index_test][index_test_non_fill]

        site_train_names, HLS_train_tensor, flux_train_numeric_tensor, FFP_train_tensor =\
            host_inputs(Model_build.preprocess_inputs, HLS_train_norm1, flux_train_norm0, FFP_train)

        site_test_names, HLS_test_tensor, flux_test_numeric_tensor, FFP_test_tensor =\
            host_inputs(Model_build.preprocess_inputs, HLS_test_norm1, flux_test_norm0, FFP_test)

        audit_fold(fold, index_train, index_test, index_train_non_fill, index_test_non_fill,
                   flux_train_numeric_tensor, flux_test_numeric_tensor)
        # Match Keras validation_split=.04 before flattening records.
        fit_end = int(int(HLS_train_tensor.shape[0]) * 0.96)
        HLS_train_tensor = HLS_train_tensor[:fit_end]
        flux_train_numeric_tensor = flux_train_numeric_tensor[:fit_end]
        FFP_train_tensor = FFP_train_tensor[:fit_end]

        print("\n[Aligned with DL preprocess_inputs]")
        print("HLS_train_tensor shape:", HLS_train_tensor.shape)
        print("flux_train_numeric_tensor shape:", flux_train_numeric_tensor.shape)
        print("FFP_train_tensor shape:", FFP_train_tensor.shape)
        print("HLS_test_tensor shape:", HLS_test_tensor.shape)
        print("flux_test_numeric_tensor shape:", flux_test_numeric_tensor.shape)
        print("FFP_test_tensor shape:", FFP_test_tensor.shape)

        X_train_full_common, y_train_full_common, train_valid_mask_common, train_shape_info = build_rf_features_common(
            HLS_norm1=HLS_train_tensor,
            flux_numeric=flux_train_numeric_tensor,
            FFP_array=FFP_train_tensor,
            mode_name=mode_name,
            target_slice=(7, 10),
            n_bands=6,
            mask_value=-9999.0,
            flux_feature_indices=FAIR_RF_FLUX_FEATURE_IDXS,
            auto_drop_bad_flux_cols=False,
            min_valid_ratio=0.05,
            debug=True
        )

        # IMPORTANT BUG FIX:
        # Feature columns must be selected from the TRAIN fold only.
        # The TEST fold must use exactly the same selected columns; otherwise
        # train/test feature matrices may have different columns or column order.
        train_selected_flux_feature_indices = train_shape_info["selected_flux_feature_indices"]

        X_test_full_common, y_test_full_common, test_valid_mask_common, test_shape_info = build_rf_features_common(
            HLS_norm1=HLS_test_tensor,
            flux_numeric=flux_test_numeric_tensor,
            FFP_array=FFP_test_tensor,
            mode_name=mode_name,
            target_slice=(7, 10),
            n_bands=6,
            mask_value=-9999.0,
            flux_feature_indices=train_selected_flux_feature_indices,
            auto_drop_bad_flux_cols=False,
            min_valid_ratio=0.05,
            debug=True
        )

        if train_shape_info["selected_flux_feature_indices"] != test_shape_info["selected_flux_feature_indices"]:
            raise RuntimeError(
                "Train/test RF flux feature columns are inconsistent after fixing test columns. "
                f"train={train_shape_info['selected_flux_feature_indices']} "
                f"test={test_shape_info['selected_flux_feature_indices']}"
            )

        print("\n[COMMON MASK SUMMARY]")
        print("valid train rows:", np.sum(train_valid_mask_common), "/", len(train_valid_mask_common))
        print("valid test rows :", np.sum(test_valid_mask_common), "/", len(test_valid_mask_common))

        X_train_common = X_train_full_common[train_valid_mask_common]
        X_test_common = X_test_full_common[test_valid_mask_common]
        Y_train_common = y_train_full_common[train_valid_mask_common]
        Y_test_common = y_test_full_common[test_valid_mask_common]

        if X_train_common.shape[1] != X_test_common.shape[1]:
            raise RuntimeError(
                f"Train/test RF feature dimension mismatch: "
                f"X_train={X_train_common.shape}, X_test={X_test_common.shape}. "
                "This usually means feature columns were not fixed from the train fold."
            )

        print("X_train_common shape:", X_train_common.shape)
        print("Y_train_common shape:", Y_train_common.shape)
        print("X_test_common shape :", X_test_common.shape)
        print("Y_test_common shape :", Y_test_common.shape)

        if X_train_common.shape[0] == 0:
            raise ValueError(f"No valid training rows for fold {fold}, mode={mode_name}")
        if X_test_common.shape[0] == 0:
            raise ValueError(f"No valid testing rows for fold {fold}, mode={mode_name}")

        N_test = test_shape_info["N"]
        T_test = test_shape_info["T"]

        combined_y_true = np.full((N_test, T_test, 3), np.nan, dtype=np.float32)
        combined_y_pred = np.full((N_test, T_test, 3), np.nan, dtype=np.float32)
        selected_feature_summary = {}

        for target_var in TARGET_VARS:
            target_model_dir = ensure_dir(os.path.join(combined_model_dir, target_var))

            target_ch = CHANNEL_ORDER.index(target_var)
            target_index = TARGET_NAME_TO_INDEX[target_var]

            y_train = Y_train_common[:, target_ch:target_ch+1]
            y_test_valid = Y_test_common[:, target_ch:target_ch+1]

            X_train = X_train_common
            X_test_valid = X_test_common

            print(f"\n------ Target {target_var} (index={target_index}, channel={target_ch}) ------")
            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)
            print("X_test_valid shape:", X_test_valid.shape)
            print("y_test_valid shape:", y_test_valid.shape)
            print("y_train min/max:", np.nanmin(y_train), np.nanmax(y_train))
            print("y_test_valid min/max:", np.nanmin(y_test_valid), np.nanmax(y_test_valid))
            print("X_train NaN count:", np.sum(np.isnan(X_train)))
            print("y_train NaN count:", np.sum(np.isnan(y_train)))

            selected_feature_summary[target_var] = {
                "selected_flux_feature_indices": train_shape_info.get("selected_flux_feature_indices", []),
                "selected_flux_feature_names": train_shape_info.get("selected_flux_feature_names", []),
                "dropped_flux_feature_indices": train_shape_info.get("dropped_flux_feature_indices", []),
                "dropped_flux_feature_names": train_shape_info.get("dropped_flux_feature_names", []),
                "valid_train_rows_common": int(np.sum(train_valid_mask_common)),
                "valid_test_rows_common": int(np.sum(test_valid_mask_common)),
                "n_hls_features_used": int(train_shape_info.get("n_hls_features_used", 6)),
            }

            for run_id in range(ITERS):
                print(f"\n[Version {VERSION}] [Mode {mode_name}] [Fold {fold}] [Target {target_var}] Run {run_id}")

                rf = RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_depth=MAX_DEPTH,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    n_jobs=N_JOBS,
                    random_state=RANDOM_STATE + fold + run_id
                )

                rf.fit(X_train, y_train.ravel())

                y_pred_valid = np.asarray(rf.predict(X_test_valid), dtype=np.float32).reshape(-1, 1) / y_SCALE
                y_true_valid = np.asarray(y_test_valid, dtype=np.float32).reshape(-1, 1) / y_SCALE
                validate_predictions(y_true_valid, y_pred_valid)

                y_pred_full = np.full((N_test * T_test, 1), np.nan, dtype=np.float32)
                y_true_full = np.full((N_test * T_test, 1), np.nan, dtype=np.float32)

                y_pred_full[test_valid_mask_common] = y_pred_valid
                y_true_full[test_valid_mask_common] = y_true_valid

                y_pred_single = y_pred_full.reshape(N_test, T_test, 1)
                y_true_single = y_true_full.reshape(N_test, T_test, 1)

                print("Pred min/max:", np.nanmin(y_pred_single), np.nanmax(y_pred_single))
                print("True min/max:", np.nanmin(y_true_single), np.nanmax(y_true_single))

                if run_id == 0:
                    combined_y_true[:, :, target_ch] = y_true_single[:, :, 0]
                    combined_y_pred[:, :, target_ch] = y_pred_single[:, :, 0]

                joblib.dump(rf, os.path.join(target_model_dir, "rf_model.joblib"))

                spec = dict(
                    version=VERSION,
                    base_script=base_name,
                    mode_name=mode_name,
                    priority_group=priority_group,
                    FFP=FFP_MODE,
                    ffp_heads_version=FFP_HEADS_VERSION,
                    fold=int(fold),
                    run_id=int(run_id),
                    target_index=int(target_index),
                    target_name=target_var,
                    channel_index=int(target_ch),
                    model_type="RandomForestRegressor_input_aggregation_single_target_common3mask",
                    n_estimators=int(N_ESTIMATORS),
                    max_depth=MAX_DEPTH,
                    min_samples_leaf=int(MIN_SAMPLES_LEAF),
                    n_jobs=int(N_JOBS) if isinstance(N_JOBS, int) else N_JOBS,
                    random_state=int(RANDOM_STATE + fold + run_id),
                    x_train_shape=list(X_train.shape),
                    y_train_shape=list(y_train.shape),
                    x_test_shape=list(X_test_valid.shape),
                    y_test_shape=list(y_test_valid.shape),
                    common_valid_train_rows=int(np.sum(train_valid_mask_common)),
                    common_valid_test_rows=int(np.sum(test_valid_mask_common)),
                    fair_rf_flux_feature_indices=train_shape_info.get("selected_flux_feature_indices", []),
                    fair_rf_flux_feature_names=train_shape_info.get("selected_flux_feature_names", []),
                    dropped_flux_feature_indices=train_shape_info.get("dropped_flux_feature_indices", []),
                    dropped_flux_feature_names=train_shape_info.get("dropped_flux_feature_names", []),
                    n_hls_features_used=int(train_shape_info.get("n_hls_features_used", 6)),
                )
                write_json(os.path.join(target_model_dir, "spec.json"), spec)

                summary_info = {
                    "model_type": "RandomForestRegressor_input_aggregation_single_target_common3mask",
                    "mode_name": mode_name,
                    "priority_group": priority_group,
                    "target_name": target_var,
                    "target_index": int(target_index),
                    "channel_index": int(target_ch),
                    "n_estimators": N_ESTIMATORS,
                    "max_depth": MAX_DEPTH,
                    "min_samples_leaf": MIN_SAMPLES_LEAF,
                    "x_train_shape": X_train.shape,
                    "y_train_shape": y_train.shape,
                    "x_test_shape": X_test_valid.shape,
                    "y_test_shape": y_test_valid.shape,
                    "common_valid_train_rows": int(np.sum(train_valid_mask_common)),
                    "common_valid_test_rows": int(np.sum(test_valid_mask_common)),
                    "n_hls_features_used": int(train_shape_info.get("n_hls_features_used", 6)),
                }
                write_txt_lines(os.path.join(target_model_dir, "summary.txt"), summary_info)

                try:
                    np.save(os.path.join(target_model_dir, "feature_importance.npy"), rf.feature_importances_)
                except Exception as e:
                    print(f"[Warning] Failed to save feature_importance.npy for fold {fold}, target {target_var}: {e}")

        combined_spec = {
            "version": VERSION,
            "DATA_ROOT": DATA_ROOT,
            "dataset_variant": "remove_selected_sites",
            "mode_name": mode_name,
            "fold": int(fold),
            "target_vars": TARGET_VARS,
            "channel_order": CHANNEL_ORDER,
            "merged_output_shape": list(combined_y_pred.shape),
            "common_valid_train_rows": int(np.sum(train_valid_mask_common)),
            "common_valid_test_rows": int(np.sum(test_valid_mask_common)),
            "notes": (
                "Fold-level RF models and fold-level pred/true/site/eval are saved. "
                "Final kfold merged outputs are also saved."
            ),
            "selected_feature_summary": selected_feature_summary,
        }
        write_json(os.path.join(combined_model_dir, "spec.json"), combined_spec)
        write_txt_lines(os.path.join(combined_model_dir, "summary.txt"), combined_spec)

        kfold_true_list.append(combined_y_true.astype(np.float32))
        kfold_pred_list.append(combined_y_pred.astype(np.float32))

        site_block = flux_test_site_mete[:, :, :8]
        site_block_fixed = cast_selected_cols_inplace(
            site_block,
            cols_to_cast=range(1, 8),
            dtype=np.float32
        )
        kfold_site_list.append(site_block_fixed)

        # --------------------------------------------------
        # Save fold-level predictions/evaluation to match FAT/CNN layout
        # --------------------------------------------------
        fold_pred_dir = ensure_dir(os.path.join(fold_root, "pred"))
        fold_eval_dir = ensure_dir(os.path.join(fold_root, "eval"))

        np.save(os.path.join(fold_pred_dir, "true.npy"), combined_y_true.astype(np.float32))
        np.save(os.path.join(fold_pred_dir, "pred.npy"), combined_y_pred.astype(np.float32))
        np.save(os.path.join(fold_pred_dir, "site.npy"), site_block_fixed)

        Evaluate_DL_RF.evaluate_flux_model_results_single(
            result_outdir=fold_pred_dir,
            evaluate_dir=fold_eval_dir,
            run_id=0
        )

        print("Fold model artifacts saved to:", fold_root)
        print("Fold predictions/evaluation saved to:", fold_pred_dir, fold_eval_dir)

    if len(fold_list) == N_SPLITS and len(kfold_true_list) > 0:   # merged table only when all folds ran in this process
        print("\n========== KFold aggregated evaluation ==========")

        y_true_all = np.concatenate(kfold_true_list, axis=0)
        y_pred_all = np.concatenate(kfold_pred_list, axis=0)
        site_all = np.concatenate(kfold_site_list, axis=0)

        kfold_root = ensure_dir(os.path.join(mode_root, f"kfold{N_SPLITS}"))
        kpred_dir = ensure_dir(os.path.join(kfold_root, "pred"))
        keval_dir = ensure_dir(os.path.join(kfold_root, "eval"))
        kmodel_dir = ensure_dir(os.path.join(kfold_root, "model"))

        np.save(os.path.join(kpred_dir, "true.npy"), y_true_all)
        np.save(os.path.join(kpred_dir, "pred.npy"), y_pred_all)
        np.save(os.path.join(kpred_dir, "site.npy"), site_all)

        Evaluate_DL_RF.evaluate_flux_model_results_single(
            result_outdir=kpred_dir,
            evaluate_dir=keval_dir,
            run_id=0
        )

        kfold_spec = {
            "version": VERSION,
            "DATA_ROOT": DATA_ROOT,
            "dataset_variant": "remove_selected_sites",
            "mode_name": mode_name,
            "n_splits": N_SPLITS,
            "target_vars": TARGET_VARS,
            "channel_order": CHANNEL_ORDER,
            "merged_output_shape": list(y_pred_all.shape),
            "n_hls_bands_used": 6,
            "aligned_flux_feature_indices": FAIR_RF_FLUX_FEATURE_IDXS,
            "aligned_flux_feature_names": [
                FLUX_NUMERIC_COLUMN_NAMES[c] for c in FAIR_RF_FLUX_FEATURE_IDXS
            ],
            "notes": (
                "Final kfold outputs merged from independently trained single-target RF "
                "models with COMMON 3-target valid mask. Fold-level and merged kfold "
                "predictions/evaluation are saved."
            )
        }
        write_json(os.path.join(kmodel_dir, "spec.json"), kfold_spec)
        write_txt_lines(os.path.join(kmodel_dir, "summary.txt"), kfold_spec)

        print("\n[KFold Summary Saved]")
        print("  pred:", kpred_dir)
        print("  eval:", keval_dir)
    else:
        print("\n[INFO] Only one fold was run, so no kfold merged outputs were generated.")
        print("Set FOLD_ONLY = -1 to run all 5 folds and save the final merged kfold results.")
