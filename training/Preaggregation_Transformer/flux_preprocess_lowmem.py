"""Memory-bounded fold normalization for the numeric flux/ERA5 features."""

import os
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    'SITE_ID','Lat','Lon','Year','DoY','HLS_Hour','Sensor','Hour','NEE_VUT_REF','GPP_NT_VUT_REF','RECO_NT_VUT_REF','NEE_VUT_REF_QC','USTAR','SW_IN_F_MDS','SW_IN_F_MDS_QC',
    'TA_F_MDS','TA_F_MDS_QC','TS_F_MDS_1','TS_F_MDS_1_QC','P_F','P_F_QC','RH_x','RH_QC','VPD_F','VPD_F_QC','PA_F','PA_F_QC','WS_F','WS_F_QC','WD','WD_F_QC','SWC_F_MDS_1',
    'SWC_F_MDS_1_QC','CO2_F_MDS','CO2_F_MDS_QC','LE_F_MDS','LE_F_MDS_QC','H_F_MDS','H_F_MDS_QC','era5_lat','era5_lon','sdfor','u10','v10','d2m','t2m','blh','zust','iews',
    'ie','inss','ishf','stl1','stl2','stl3','stl4','sp','swvl1','swvl2','swvl3','swvl4','tp','sshf','ssrd','RH','VPD_kPa','WS_ms','WD_deg','L_m','t2m_C','d2m_C','stl1_C',
    'stl2_C','stl3_C','stl4_C','sp_kPa','swvl1_pct','swvl2_pct','swvl3_pct','swvl4_pct','P_mm','SSRD_Wm2','SSHF_Wm2','time_utc','time_local','MH','MH_QC','MH_source','IGBP',
    'CH','CH_QC','CH_source','V_SIGMA','V_SIGMA_QC','h_aero','d','z0','zm','lambda_rs','ha_QC','h_aero_smooth','d_smooth','z0_smooth','zm_smooth','z_over_L','Hour_sin','Hour_cos',
    'WD_sin','WD_cos','sigma_v_rf'
]

METE_FEATURES = [
    't2m_C', 'd2m_C', 'u10', 'v10', 'sp_kPa', 'stl1_C', 'stl2_C',
    'stl3_C', 'stl4_C', 'swvl1_pct', 'swvl2_pct', 'swvl3_pct',
    'swvl4_pct', 'P_mm', 'SSRD_Wm2', 'WS_ms', 'WD_deg', 'VPD_kPa', 'RH', 'MH'
]


def _object_rows_to_float(source, indices, chunk_size=256):
    indices = np.asarray(indices, dtype=np.int64)
    output = np.empty(
        (len(indices), source.shape[1], source.shape[2] - 1), dtype=np.float32
    )
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        output[start:stop] = np.asarray(
            source[indices[start:stop], :, 1:], dtype=np.float32
        )
    return output


def normalize_flux_train_test_lowmem(source, train_indices, test_indices,
                                     mask_value=-9999.0, fold_id=None,
                                     save_dir=None, chunk_size=256):
    """Return float32 numeric arrays without rebuilding object arrays."""
    train = _object_rows_to_float(source, train_indices, chunk_size=chunk_size)
    test = _object_rows_to_float(source, test_indices, chunk_size=chunk_size)
    numeric_names = FEATURE_NAMES[1:]
    mete_idx = [numeric_names.index(name) for name in METE_FEATURES]
    xy_idx = [numeric_names.index('Lat'), numeric_names.index('Lon')]
    sensor_idx = numeric_names.index('Sensor')

    def normalize_indices(indices):
        stats = []
        for feature_idx in indices:
            values = train[:, :, feature_idx]
            valid = np.isfinite(values) & (values != mask_value)
            if np.any(valid):
                mean = float(values[valid].mean())
                std = float(values[valid].std())
                if std <= 1e-6:
                    std = 1.0
            else:
                mean, std = 0.0, 1.0

            for array in (train, test):
                column = array[:, :, feature_idx]
                column_valid = np.isfinite(column) & (column != mask_value)
                column[column_valid] = (column[column_valid] - mean) / std
                column[~column_valid] = mask_value
            stats.append({"feature": numeric_names[feature_idx], "mean": mean, "std": std})
        return stats

    mete_stats = normalize_indices(mete_idx)
    xy_stats = normalize_indices(xy_idx)
    sensor_stats = normalize_indices([sensor_idx])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        suffix = f"_fold{fold_id}" if fold_id is not None else ""
        pd.DataFrame(mete_stats).to_csv(
            os.path.join(save_dir, f"mete_features_mean_std{suffix}.csv"), index=False
        )
        pd.DataFrame(xy_stats).to_csv(
            os.path.join(save_dir, f"xy_features_mean_std{suffix}.csv"), index=False
        )
        pd.DataFrame(sensor_stats).to_csv(
            os.path.join(save_dir, f"sensor_feature_mean_std{suffix}.csv"), index=False
        )
    return train, test
