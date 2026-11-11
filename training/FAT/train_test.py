#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import csv
import numpy as np
import pandas as pd

MASK_VALUE = -9999

S30_band_list = [
    "BLUE", "GREEN", "RED", "NIR1",
    "SWIR1", "SWIR2", "VZA", "VAA", "SZA", "SAA", "FMASK",
    "COASTAL-AEROSOL", "CIRRUS", "TIR1", "TIR2", "RED-EDGE1",
    "RED-EDGE2", "RED-EDGE3", "NIR-Broad", "WATER-VAPOR"
]

# valid band indices (keep your original)
L8_valid_bands = list(range(15))               # 0-14
S2_valid_bands = list(range(13)) + list(range(15, 20))  # 0-12 and 15-19


# ============================================================
# Helper: extract one valid SITE_ID per sample (your logic)
# ============================================================
def _extract_first_valid_siteid_per_sample(y_array, mask_value=-9999.0):
    """
    y_array: (N,T,F)
    SITE_ID is y_array[:,:,0]
    Return: site_ids_per_sample (N,) with -9999 for invalid
    """
    site_ids_per_sample = []
    for i in range(y_array.shape[0]):
        valid_ids = y_array[i, :, 0]
        valid_ids = valid_ids[valid_ids != mask_value]
        if len(valid_ids) == 0:
            site_ids_per_sample.append(mask_value)
        else:
            site_ids_per_sample.append(valid_ids[0])
    return np.array(site_ids_per_sample, dtype=object)


# ============================================================
# (keep) random_split helper (your legacy)
# ============================================================
def random_split(total_n, split_n, y_array):
    """
    create split index by splitting total_n into train and test based on split_n
    test: orders == 0
    train: order != 0
    """
    sample_n = np.ceil(total_n / split_n).astype(int)
    SPLIT_DIR = "./region model/split/"
    os.makedirs(SPLIT_DIR, exist_ok=True)

    file_index = os.path.join(SPLIT_DIR, f"index.total_n{total_n}.for.random.txt")
    if os.path.isfile(file_index):
        print('File already exists! ' + file_index)
        dat = np.loadtxt(open(file_index, "rb"), dtype='<U10', delimiter=",", skiprows=1)
        orders = dat.astype(np.int64)
    else:
        header = 'order'
        orders = 0
        for i in range(split_n):
            if i == 0:
                orders = np.repeat(i, sample_n)
            else:
                orders = np.concatenate((orders, np.repeat(i, sample_n)))
        orders = orders[range(total_n)]
        np.random.shuffle(orders)
        np.savetxt(file_index, orders, fmt="%s", header=header, delimiter=",")

    # map site->order, and write back into y_array[:,:,0]
    site_order_dict = {}
    for i, order in enumerate(orders):
        site = y_array[i, 0, 0]
        site_order_dict[site] = order

    for i in range(y_array.shape[0]):
        site = y_array[i, 0, 0]
        y_array[i, :, 0] = site_order_dict[site]
    return y_array


# ============================================================
# (keep) your current site-level train/test split (single split)
# ============================================================
def get_training_test_siteid(y_array, pecentage=0.2):
    """
    Split by SITE_ID (site-level), keep your exact extraction logic:
    - per sample: take first valid site_id across 48 steps (ignore -9999)
    - split by unique sites
    - save train/test site locations (site_id, lat, lon)
    """
    site_ids_per_sample = _extract_first_valid_siteid_per_sample(y_array, mask_value=MASK_VALUE)

    unique_ids = np.unique(site_ids_per_sample[site_ids_per_sample != MASK_VALUE])
    total_n = len(unique_ids)

    sample_n = math.ceil(total_n * pecentage)
    split_n = math.ceil(total_n / sample_n)

    split_dir = "./data/split/"
    os.makedirs(split_dir, exist_ok=True)
    file_index = os.path.join(split_dir, f"split.total_n{total_n}.for.random_split_field.txt")

    if os.path.isfile(file_index):
        print(f'file already exists: {file_index}')
        orders = np.loadtxt(file_index, dtype=np.int64, delimiter=",", skiprows=1)
    else:
        print(f'Generating file {file_index}')
        np.random.seed(42)
        orders = np.concatenate([np.repeat(i, sample_n) for i in range(split_n)])[:total_n]
        np.random.shuffle(orders)
        np.savetxt(file_index, orders, fmt="%s", header="order", delimiter=",")

    test_sites = unique_ids[orders == 0]

    index_test = np.isin(site_ids_per_sample, test_sites)
    index_train = (~index_test) & (site_ids_per_sample != MASK_VALUE)

    feature_names = ["site_id", "lat", "lon"]

    def extract_valid_info(sample):
        sub = sample[:, [0, 1, 2]]
        mask = np.any(sub != MASK_VALUE, axis=1)
        if not np.any(mask):
            return np.array([MASK_VALUE, MASK_VALUE, MASK_VALUE], dtype=object)
        return sub[mask][0]

    train_info = np.array([extract_valid_info(s) for s in y_array[index_train]])
    test_info = np.array([extract_valid_info(s) for s in y_array[index_test]])

    df_train = pd.DataFrame(train_info, columns=feature_names).drop_duplicates()
    df_test = pd.DataFrame(test_info, columns=feature_names).drop_duplicates()

    train_csv = os.path.join(split_dir, "train_sites_loc.csv")
    test_csv = os.path.join(split_dir, "test_sites_loc.csv")
    df_train.to_csv(train_csv, index=False, encoding="utf-8")
    df_test.to_csv(test_csv, index=False, encoding="utf-8")

    print(f"Train info saved: {train_csv} ({len(df_train)} unique)")
    print(f"Test  info saved: {test_csv} ({len(df_test)} unique)")
    print(f"Train unique sites: {df_train['site_id'].nunique()}")
    print(f"Test  unique sites: {df_test['site_id'].nunique()}")

    return index_train, index_test


# ============================================================
# NEW: 5-fold site-level split (GroupKFold-like, deterministic)
# ============================================================
def get_kfold_siteid(y_array, n_splits=5, fold_id=0, seed=42, split_dir="./data/split_kfold"):
    """
    Site-level KFold split (each site belongs to exactly one fold).

    Writes:
      - site_to_fold.k{n_splits}.seed{seed}.csv
      - train_sites_fold{fold_id}.csv / test_sites_fold{fold_id}.csv
      - index_train_fold{fold_id}.npy / index_test_fold{fold_id}.npy  (optional convenience)

    Returns:
      index_train (N,) bool
      index_test  (N,) bool
    """
    os.makedirs(split_dir, exist_ok=True)
    fold_id = int(fold_id)
    if fold_id < 0 or fold_id >= int(n_splits):
        raise ValueError(f"fold_id must be in [0, {n_splits-1}]")

    site_ids_per_sample = _extract_first_valid_siteid_per_sample(y_array, mask_value=MASK_VALUE)
    valid_mask = (site_ids_per_sample != MASK_VALUE)

    unique_sites = np.unique(site_ids_per_sample[valid_mask])
    map_path = os.path.join(split_dir, f"site_to_fold.k{n_splits}.seed{seed}.csv")

    if os.path.exists(map_path):
        df = pd.read_csv(map_path)
        site_to_fold = {row["site_id"]: int(row["fold"]) for _, row in df.iterrows()}
    else:
        rng = np.random.default_rng(seed)
        sites = unique_sites.astype(object).copy()
        rng.shuffle(sites)
        folds = np.arange(len(sites)) % int(n_splits)
        df = pd.DataFrame({"site_id": sites, "fold": folds})
        df.to_csv(map_path, index=False)
        site_to_fold = {row["site_id"]: int(row["fold"]) for _, row in df.iterrows()}
        print("[KFold] saved mapping:", map_path)

    sample_fold = np.full(y_array.shape[0], -1, dtype=int)
    for i in range(y_array.shape[0]):
        if not valid_mask[i]:
            continue
        sample_fold[i] = site_to_fold.get(site_ids_per_sample[i], -1)

    index_test = (sample_fold == fold_id)
    index_train = valid_mask & (sample_fold != fold_id)

    train_sites = np.unique(site_ids_per_sample[index_train])
    test_sites = np.unique(site_ids_per_sample[index_test])

    pd.DataFrame({"site_id": train_sites}).to_csv(os.path.join(split_dir, f"train_sites_fold{fold_id}.csv"), index=False)
    pd.DataFrame({"site_id": test_sites}).to_csv(os.path.join(split_dir, f"test_sites_fold{fold_id}.csv"), index=False)

    # save indices (optional but handy)
    np.save(os.path.join(split_dir, f"index_train_fold{fold_id}.npy"), index_train)
    np.save(os.path.join(split_dir, f"index_test_fold{fold_id}.npy"), index_test)

    print(f"[KFold] fold={fold_id}/{n_splits-1} "
          f"| train_sites={len(train_sites)} test_sites={len(test_sites)} "
          f"| train_samples={index_train.sum()} test_samples={index_test.sum()}")

    return index_train, index_test


# ============================================================
# HLS normalization utils (your style)
# ============================================================
def bandwise_mean_std(data, mask, valid_bands):
    mean_list = []
    std_list = []
    for b in range(data.shape[1]):
        if b in valid_bands:
            vals = data[:, b, :, :][mask[:, b, :, :]]
            if len(vals) > 0:
                mean_list.append(vals.mean())
                std_list.append(vals.std())
            else:
                mean_list.append(np.nan)
                std_list.append(np.nan)
        else:
            mean_list.append(np.nan)
            std_list.append(np.nan)
    return np.array(mean_list), np.array(std_list)


def get_HLS_normalization_joint(HLS_array, FLUX_array, index_train, index_test,
                                is_joint_norm=False,
                                fold_id=None,
                                output_dir="./data/norm_stats"):
    """
    Your original joint function, but:
    - add fold_id / output_dir
    - save stats filenames with fold suffix when fold_id is not None
    """
    os.makedirs(output_dir, exist_ok=True)

    sensor_col = FLUX_array[:, :, 6].astype(int)
    L8_index = np.where(sensor_col == 0)[0]
    S2_index = np.where(sensor_col == 1)[0]

    train_idx = np.where(index_train)[0]
    test_idx = np.where(index_test)[0]

    train_g2l = {g: i for i, g in enumerate(train_idx)}
    test_g2l = {g: i for i, g in enumerate(test_idx)}

    L8_train_idx = np.intersect1d(train_idx, L8_index)
    L8_test_idx = np.intersect1d(test_idx, L8_index)
    S2_train_idx = np.intersect1d(train_idx, S2_index)
    S2_test_idx = np.intersect1d(test_idx, S2_index)

    input_l8_train = HLS_array[L8_train_idx].astype(np.float32)
    input_l8_test = HLS_array[L8_test_idx].astype(np.float32)
    input_s2_train = HLS_array[S2_train_idx].astype(np.float32)
    input_s2_test = HLS_array[S2_test_idx].astype(np.float32)

    l8_mask_train = input_l8_train != MASK_VALUE
    l8_mask_test = input_l8_test != MASK_VALUE
    s2_mask_train = input_s2_train != MASK_VALUE
    s2_mask_test = input_s2_test != MASK_VALUE

    # separate stats
    l8_mean, l8_std = bandwise_mean_std(input_l8_train, l8_mask_train, L8_valid_bands)
    s2_mean, s2_std = bandwise_mean_std(input_s2_train, s2_mask_train, S2_valid_bands)

    # optional joint stats for shared bands
    joint_bands = sorted(set(L8_valid_bands).intersection(set(S2_valid_bands)))
    joint_mean, joint_std = None, None
    if is_joint_norm:
        joint_train_data = np.concatenate(
            [input_l8_train[:, joint_bands, :, :], input_s2_train[:, joint_bands, :, :]], axis=0
        )
        joint_mask = np.concatenate(
            [l8_mask_train[:, joint_bands, :, :], s2_mask_train[:, joint_bands, :, :]], axis=0
        )
        # here "valid_bands" becomes 0..len(joint_bands)-1 for the joint_train_data
        joint_mean, joint_std = bandwise_mean_std(joint_train_data, joint_mask, list(range(len(joint_bands))))

    suffix = f"_fold{fold_id}" if fold_id is not None else ""
    pd.DataFrame({"band": S30_band_list, "mean": l8_mean, "std": l8_std})\
        .to_csv(os.path.join(output_dir, f"L8_mean_std{suffix}.csv"), index=False)
    pd.DataFrame({"band": S30_band_list, "mean": s2_mean, "std": s2_std})\
        .to_csv(os.path.join(output_dir, f"S2_mean_std{suffix}.csv"), index=False)
    if is_joint_norm:
        pd.DataFrame({
            "band": [S30_band_list[b] for b in joint_bands],
            "mean": joint_mean,
            "std": joint_std
        }).to_csv(os.path.join(output_dir, f"Joint_mean_std{suffix}.csv"), index=False)

    # output arrays are train/test local-indexed like your original
    train_norm = np.full((len(train_idx), 20, HLS_array.shape[2], HLS_array.shape[3]), MASK_VALUE, dtype=np.float32)
    test_norm = np.full((len(test_idx), 20, HLS_array.shape[2], HLS_array.shape[3]), MASK_VALUE, dtype=np.float32)

    # L8 single norm
    for b in L8_valid_bands:
        vals_train = input_l8_train[:, b, :, :]
        mask_train = l8_mask_train[:, b, :, :]
        norm_train_band = np.full_like(vals_train, MASK_VALUE)
        norm_train_band[mask_train] = (vals_train[mask_train] - l8_mean[b]) / (l8_std[b] if l8_std[b] != 0 else 1.0)
        train_norm[[train_g2l[g] for g in L8_train_idx], b, :, :] = norm_train_band

        vals_test = input_l8_test[:, b, :, :]
        mask_test = l8_mask_test[:, b, :, :]
        norm_test_band = np.full_like(vals_test, MASK_VALUE)
        norm_test_band[mask_test] = (vals_test[mask_test] - l8_mean[b]) / (l8_std[b] if l8_std[b] != 0 else 1.0)
        test_norm[[test_g2l[g] for g in L8_test_idx], b, :, :] = norm_test_band

    # S2 single norm
    for b in S2_valid_bands:
        vals_train = input_s2_train[:, b, :, :]
        mask_train = s2_mask_train[:, b, :, :]
        norm_train_band = np.full_like(vals_train, MASK_VALUE)
        norm_train_band[mask_train] = (vals_train[mask_train] - s2_mean[b]) / (s2_std[b] if s2_std[b] != 0 else 1.0)
        train_norm[[train_g2l[g] for g in S2_train_idx], b, :, :] = norm_train_band

        vals_test = input_s2_test[:, b, :, :]
        mask_test = s2_mask_test[:, b, :, :]
        norm_test_band = np.full_like(vals_test, MASK_VALUE)
        norm_test_band[mask_test] = (vals_test[mask_test] - s2_mean[b]) / (s2_std[b] if s2_std[b] != 0 else 1.0)
        test_norm[[test_g2l[g] for g in S2_test_idx], b, :, :] = norm_test_band

    # joint overwrite (optional)
    if is_joint_norm and joint_mean is not None and joint_std is not None:
        for i, b in enumerate(joint_bands):
            # L8
            vals_train = input_l8_train[:, b, :, :]
            mask_train = l8_mask_train[:, b, :, :]
            norm_train_band = np.full_like(vals_train, MASK_VALUE)
            norm_train_band[mask_train] = (vals_train[mask_train] - joint_mean[i]) / (joint_std[i] if joint_std[i] != 0 else 1.0)
            train_norm[[train_g2l[g] for g in L8_train_idx], b, :, :] = norm_train_band

            vals_test = input_l8_test[:, b, :, :]
            mask_test = l8_mask_test[:, b, :, :]
            norm_test_band = np.full_like(vals_test, MASK_VALUE)
            norm_test_band[mask_test] = (vals_test[mask_test] - joint_mean[i]) / (joint_std[i] if joint_std[i] != 0 else 1.0)
            test_norm[[test_g2l[g] for g in L8_test_idx], b, :, :] = norm_test_band

            # S2
            vals_train = input_s2_train[:, b, :, :]
            mask_train = s2_mask_train[:, b, :, :]
            norm_train_band = np.full_like(vals_train, MASK_VALUE)
            norm_train_band[mask_train] = (vals_train[mask_train] - joint_mean[i]) / (joint_std[i] if joint_std[i] != 0 else 1.0)
            train_norm[[train_g2l[g] for g in S2_train_idx], b, :, :] = norm_train_band

            vals_test = input_s2_test[:, b, :, :]
            mask_test = s2_mask_test[:, b, :, :]
            norm_test_band = np.full_like(vals_test, MASK_VALUE)
            norm_test_band[mask_test] = (vals_test[mask_test] - joint_mean[i]) / (joint_std[i] if joint_std[i] != 0 else 1.0)
            test_norm[[test_g2l[g] for g in S2_test_idx], b, :, :] = norm_test_band

    return train_norm, test_norm


# ============================================================
# Flux normalization (your version, fold-aware file outputs, shape-safe)
# ============================================================
def normalize_flux_train_test(x2_y_array, index_train, index_test,
                              use_mete=True, use_xy=True, use_sensor=True,
                              mask_value=-9999,
                              fold_id=None,
                              save_dir="./data/flux_norm_stats"):
    """
    Keep your logic:
    - SITE_ID is string/object in column 0 -> not normalized
    - numeric part: x2_y_array[:,:,1:].astype(float)
    - normalize selected groups based on feature name lists
    - return object arrays with same last-dim as original x2_y_array

    Enhancement:
    - fold-aware csv file names (mete_features_mean_std_fold{fold}.csv, etc.)
    - final shape uses x2_y_array.shape[2] (no hard-coded 110)
    """
    feature_names = [
        'SITE_ID','Lat','Lon','Year','DoY','HLS_Hour','Sensor','Hour','NEE_VUT_REF','GPP_NT_VUT_REF','RECO_NT_VUT_REF','NEE_VUT_REF_QC','USTAR','SW_IN_F_MDS','SW_IN_F_MDS_QC',
        'TA_F_MDS','TA_F_MDS_QC','TS_F_MDS_1','TS_F_MDS_1_QC','P_F','P_F_QC','RH_x','RH_QC','VPD_F','VPD_F_QC','PA_F','PA_F_QC','WS_F','WS_F_QC','WD','WD_F_QC','SWC_F_MDS_1',
        'SWC_F_MDS_1_QC','CO2_F_MDS','CO2_F_MDS_QC','LE_F_MDS','LE_F_MDS_QC','H_F_MDS','H_F_MDS_QC','era5_lat','era5_lon','sdfor','u10','v10','d2m','t2m','blh','zust','iews',
        'ie','inss','ishf','stl1','stl2','stl3','stl4','sp','swvl1','swvl2','swvl3','swvl4','tp','sshf','ssrd','RH','VPD_kPa','WS_ms','WD_deg','L_m','t2m_C','d2m_C','stl1_C',
        'stl2_C','stl3_C','stl4_C','sp_kPa','swvl1_pct','swvl2_pct','swvl3_pct','swvl4_pct','P_mm','SSRD_Wm2','SSHF_Wm2','time_utc','time_local','MH','MH_QC','MH_source','IGBP',
        'CH','CH_QC','CH_source','V_SIGMA','V_SIGMA_QC','h_aero','d','z0','zm','lambda_rs','ha_QC','h_aero_smooth','d_smooth','z0_smooth','zm_smooth','z_over_L','Hour_sin','Hour_cos',
        'WD_sin','WD_cos','sigma_v_rf'
    ]

    # SITE_ID string/object
    site_ids = x2_y_array[:, :, 0]

    # numeric data
    numeric_data = x2_y_array[:, :, 1:].astype(np.float32)  # (N,48,F-1)

    feature_names_numeric = feature_names[1:]  # remove SITE_ID

    mete_features = ['t2m_C', 'd2m_C', 'u10', 'v10', 'sp_kPa', 'stl1_C', 'stl2_C', 'stl3_C', 'stl4_C',
                     'swvl1_pct', 'swvl2_pct', 'swvl3_pct', 'swvl4_pct', 'P_mm', 'SSRD_Wm2',
                     'WS_ms', 'WD_deg', 'VPD_kPa', 'RH', 'MH']
    xy_features = ['Lat', 'Lon']
    sensor_feature = 'Sensor'

    mete_idx = [feature_names_numeric.index(f) for f in mete_features if f in feature_names_numeric]
    xy_idx = [feature_names_numeric.index(f) for f in xy_features if f in feature_names_numeric]
    sensor_idx = feature_names_numeric.index(sensor_feature) if sensor_feature in feature_names_numeric else None

    train_data = numeric_data[index_train]
    test_data = numeric_data[index_test]

    train_norm = train_data.copy()
    test_norm = test_data.copy()

    def compute_mean_std(data_1d):
        valid = data_1d != mask_value
        if np.any(valid):
            mean = float(data_1d[valid].mean())
            std = float(data_1d[valid].std())
            std = std if std > 1e-6 else 1.0
        else:
            mean, std = 0.0, 1.0
        return mean, std

    def normalize_feature(data, mean, std):
        normed = np.full_like(data, mask_value, dtype=np.float32)
        valid = data != mask_value
        normed[valid] = (data[valid] - mean) / std
        return normed

    def normalize_group(indices):
        group_stats = []
        for i in indices:
            vals_train = train_norm[:, :, i].reshape(-1)
            mean, std = compute_mean_std(vals_train)
            group_stats.append({"feature": feature_names_numeric[i], "mean": mean, "std": std})

            train_norm[:, :, i] = normalize_feature(train_norm[:, :, i], mean, std)
            test_norm[:, :, i] = normalize_feature(test_norm[:, :, i], mean, std)
        return group_stats

    mete_stats, xy_stats, sensor_stats = [], [], []

    if use_mete and len(mete_idx) > 0:
        mete_stats = normalize_group(mete_idx)
    if use_xy and len(xy_idx) > 0:
        xy_stats = normalize_group(xy_idx)
    if use_sensor and (sensor_idx is not None):
        vals_train = train_norm[:, :, sensor_idx].reshape(-1)
        mean, std = compute_mean_std(vals_train)
        sensor_stats = [{"feature": feature_names_numeric[sensor_idx], "mean": mean, "std": std}]
        train_norm[:, :, sensor_idx] = normalize_feature(train_norm[:, :, sensor_idx], mean, std)
        test_norm[:, :, sensor_idx] = normalize_feature(test_norm[:, :, sensor_idx], mean, std)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        suffix = f"_fold{fold_id}" if fold_id is not None else ""
        if mete_stats:
            pd.DataFrame(mete_stats).to_csv(os.path.join(save_dir, f"mete_features_mean_std{suffix}.csv"), index=False)
        if xy_stats:
            pd.DataFrame(xy_stats).to_csv(os.path.join(save_dir, f"xy_features_mean_std{suffix}.csv"), index=False)
        if sensor_stats:
            pd.DataFrame(sensor_stats).to_csv(os.path.join(save_dir, f"sensor_feature_mean_std{suffix}.csv"), index=False)

    # rebuild object arrays with same feature count as x2_y_array
    N_train = train_norm.shape[0]
    N_test = test_norm.shape[0]
    F_total = x2_y_array.shape[2]

    train_final = np.empty((N_train, x2_y_array.shape[1], F_total), dtype=object)
    test_final = np.empty((N_test, x2_y_array.shape[1], F_total), dtype=object)

    train_final[:, :, 0] = site_ids[index_train]
    test_final[:, :, 0] = site_ids[index_test]

    train_final[:, :, 1:] = train_norm
    test_final[:, :, 1:] = test_norm

    return train_final, test_final
