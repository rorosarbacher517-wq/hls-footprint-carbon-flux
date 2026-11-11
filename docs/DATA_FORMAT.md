# Data format and organization / 数据格式与组织

```text
data/input_data/
  HLS_matched_v11_remove_selected_sites.npy
  FFP_hls_aligned_v11_50x50_timefilled_remove_selected_sites.npy
  merged_flux_era5_ffp_array_v11_remove_selected_sites.npy
analysis/data/four_model_predictions/
  FAT/fold0/{true,pred,site}.npy
  Uniform_Transformer/fold0/{true,pred,site}.npy
  Preaggregation_Transformer/fold0/{true,pred,site}.npy
  RF/fold0/{true,pred,site}.npy
  ... fold1 through fold4 for each model
```

| File | Shape | dtype | Meaning |
|---|---|---|---|
| HLS | `[N,20,50,50]` | float32 | One matched daily image patch per sample; channel-first; 30 m grid |
| FFP | `[N,48,50,50]` | float32 | 48 half-hourly footprint fields on the same spatial grid |
| flux/environment | `[N,48,110]` | object | Site identifiers plus numeric metadata, labels and covariates |
| true / pred | `[N_fold,48,3]` | floating point | Last axis is NEE, GPP, RECO |

Sample index N must align across all three input arrays. The 48 slots refer to the same matched day and the same half-hour ordering. Imagery is not 48 independent satellite acquisitions. Spatial row/column orientation must be identical between image and footprint. Missing values use -9999 where required by the source preprocessing.

`examples/flux_columns.csv` gives the actual 0-based index of every column. Column 0 is SITE_ID, 6 Sensor, 7 Hour, 8 NEE, 9 GPP, 10 RECO. Sensor is raw categorical 0/1, not a standardized continuous variable. Inspect `training/FAT/train_test.py::S30_band_list` for channel order and sensor-specific channel availability. Flux labels use the input study's CO2 flux units; do not mix half-hour mean rates with daily totals.

The object array requires `allow_pickle=True` in the original loader: load only trusted arrays. The synthetic generator writes this legacy format for interface demonstration. Real input hashes and site split masks are intentionally not replaced with synthetic identities.

The generated example uses fabricated site IDs and no private measurements. For ancillary analyses, place land-cover, site-day driver and VIF tables at the portable defaults in `portable_paths.json`, or use the scripts' exposed CLI/environment options. These ancillary tables are not derivable from the three example arrays alone.

## Frozen split files (not distributed)

Supply `training/reference/split_kfold/index_train_fold{k}.npy` and `index_test_fold{k}.npy` for k=0..4 from the authorized research dataset. Each is a Boolean mask of length N. Original effective-index NPZ files are not distributed. Do not use the synthetic sample with the original frozen input hashes.
