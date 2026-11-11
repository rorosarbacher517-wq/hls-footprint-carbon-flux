#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd


def _safe_int_series(x, fill=-1):
    """Robust int conversion that won't depend on pandas nullable Int64 dtype."""
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    s = s.fillna(fill).astype(np.int64)
    return s


def build_record_meta_from_flux(flux_arr):
    """
    Build per-record meta from FLUX object array:
      global_index, SITE_ID, Year, DoY, Lat, Lon, HLS_Hour, Sensor
    Uses safe numeric conversion (no pandas nullable Int64).
    """
    N = int(flux_arr.shape[0])
    F = int(flux_arr.shape[2]) if flux_arr.ndim == 3 else 0

    site_id = np.asarray(flux_arr[:, 0, 0], dtype=str) if F > 0 else np.array(["NA"] * N, dtype=str)

    lat = pd.to_numeric(pd.Series(flux_arr[:, 0, 1] if F > 1 else [np.nan] * N), errors="coerce").to_numpy()
    lon = pd.to_numeric(pd.Series(flux_arr[:, 0, 2] if F > 2 else [np.nan] * N), errors="coerce").to_numpy()

    year = _safe_int_series(flux_arr[:, 0, 3] if F > 3 else [np.nan] * N, fill=-1)
    doy  = _safe_int_series(flux_arr[:, 0, 4] if F > 4 else [np.nan] * N, fill=-1)

    hls_hour = pd.to_numeric(pd.Series(flux_arr[:, 0, 5] if F > 5 else [np.nan] * N), errors="coerce").to_numpy()
    sensor   = _safe_int_series(flux_arr[:, 0, 6] if F > 6 else [np.nan] * N, fill=-1)

    df = pd.DataFrame({
        "global_index": np.arange(N, dtype=np.int64),
        "SITE_ID": site_id,
        "Year": year.values if hasattr(year, "values") else np.asarray(year, dtype=np.int64),
        "DoY":  doy.values if hasattr(doy, "values") else np.asarray(doy, dtype=np.int64),
        "Lat":  lat,
        "Lon":  lon,
        "HLS_Hour": hls_hour,
        "Sensor": sensor.values if hasattr(sensor, "values") else np.asarray(sensor, dtype=np.int64),
    })
    return df


def unique_sites_first_appearance(site_ids_vec):
    seen = set()
    out = []
    for s in site_ids_vec.tolist():
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def load_site_fold_map(site_fold_csv):
    """
    CSV must contain columns: SITE_ID, fold
    fold in {0..4,5}. fold==5 means 'not in any fold', but per your rule we map it to fold 2 for model/stats.
    Returns dict SITE_ID -> fold_raw (0..5)
    """
    df = pd.read_csv(site_fold_csv)
    if "SITE_ID" not in df.columns or "fold" not in df.columns:
        raise KeyError("site_fold_csv must contain SITE_ID and fold columns.")
    mp = {}
    for _, r in df.iterrows():
        mp[str(r["SITE_ID"])] = int(r["fold"])
    return mp


def fold_for_inference(site_id, site2fold_raw, default_fold_for_5=2):
    """
    If site fold is 0..4 => use that.
    If fold==5 or missing => use default_fold_for_5 (your rule: 5 -> fold2).
    """
    fr = site2fold_raw.get(str(site_id), 5)
    if fr == 5:
        return int(default_fold_for_5)
    return int(fr)


def resolve_model_paths_for_fold(ffp_mode, fold_use, model_root):
    """
    Your naming pattern (example given):
      Pro_cnn_transformer_HLS_carbon_v7_6...fold0.spec.json
      Pro_cnn_transformer_HLS_carbon_v7_6...fold0.weights.h5

    We will search within model_root for:
      *fold{fold_use}.spec.json
      *fold{fold_use}.weights.h5
    And prefer exact match for FFP mode if model_root is already under FFP1/FFP2.
    """
    fold_use = int(fold_use)
    if not os.path.isdir(model_root):
        raise FileNotFoundError(f"model_root not found: {model_root}")

    spec_pat = re.compile(rf".*fold{fold_use}\.spec\.json$", re.IGNORECASE)
    w_pat    = re.compile(rf".*fold{fold_use}\.weights\.h5$", re.IGNORECASE)

    specs = [os.path.join(model_root, f) for f in os.listdir(model_root) if spec_pat.match(f)]
    wts   = [os.path.join(model_root, f) for f in os.listdir(model_root) if w_pat.match(f)]

    if not specs:
        raise FileNotFoundError(f"No spec.json for fold{fold_use} in {model_root}")
    if not wts:
        raise FileNotFoundError(f"No weights.h5 for fold{fold_use} in {model_root}")

    # If multiple matches, pick the shortest (often most specific) then lexicographic
    specs = sorted(specs, key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
    wts   = sorted(wts,   key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))

    return specs[0], wts[0]


def resolve_hls_stats_paths_foldaware(fold_use, hls_stats_root):
    """
    Your HLS stats folder is FLAT:
      L8_mean_std_fold0.csv
      S2_mean_std_fold0.csv
      joint_mean_std_fold0.csv
    """
    fold_use = int(fold_use)
    l8 = os.path.join(hls_stats_root, f"L8_mean_std_fold{fold_use}.csv")
    s2 = os.path.join(hls_stats_root, f"S2_mean_std_fold{fold_use}.csv")
    jt = os.path.join(hls_stats_root, f"joint_mean_std_fold{fold_use}.csv")

    if not os.path.exists(l8):
        raise FileNotFoundError(f"Missing: {l8}")
    if not os.path.exists(s2):
        raise FileNotFoundError(f"Missing: {s2}")
    # joint is optional if is_joint_norm=False, so???
    return l8, s2, jt


def resolve_flux_stats_paths_foldaware(fold_use, flux_stats_root):
    """
    Your flux stats folder is FLAT:
      mete_features_mean_std_fold0.csv
      xy_features_mean_std_fold0.csv
      sensor_feature_mean_std_fold0.csv
    """
    fold_use = int(fold_use)
    mete = os.path.join(flux_stats_root, f"mete_features_mean_std_fold{fold_use}.csv")
    xy   = os.path.join(flux_stats_root, f"xy_features_mean_std_fold{fold_use}.csv")
    sen  = os.path.join(flux_stats_root, f"sensor_feature_mean_std_fold{fold_use}.csv")

    if not os.path.exists(mete):
        raise FileNotFoundError(f"Missing: {mete}")
    if not os.path.exists(xy):
        raise FileNotFoundError(f"Missing: {xy}")
    if not os.path.exists(sen):
        raise FileNotFoundError(f"Missing: {sen}")
    return mete, xy, sen
