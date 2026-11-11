"""FAT (V6 re-run, mode 0 = dynamic footprint) 30 m pixel products for one site-day.

Runs the fold-aware streaming prediction script of the original code base (patch products: pixel flux, footprint weights, tower value) with
  - the model / head modules of FOUR_MODEL_TF215_DUALGPU_V6/FAT (put first on sys.path),
  - the V6 FAT weights, spec and normalisation statistics of results_v6_tf215/FAT,
  - the numpy pickle compatibility shim (the flux array was written with numpy 2).
The V6 weights were trained with TF 2.15 and are read here with the local TF 2.8 environment; the tower-scale values are checked against the
stored V6 predictions in fig_4_4_case_study.py (case_consistency_check.csv).
Usage: python run_fat_pixel_prediction.py SITE YEAR DOY     (python of the 'tensorflow' conda environment)
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V6 = Path('./training')
OLD_ROOT = Path('./data/external/HLS_ERA5_FLUX_FFP_CV5_FINAL')
OLD_DATA = OLD_ROOT / "data_faircompare_remove_selected_sites"
RES = V6 / "results_v6_tf215" / "FAT"
sys.path.insert(0, str(OLD_ROOT))
sys.path.insert(0, str(V6 / "FAT"))                  # V6 Model_build / model_heads_ffp take precedence
sys.path.insert(0, str(V6))
from numpy_pickle_compat import install_numpy_pickle_compat  # noqa: E402

install_numpy_pickle_compat()

# Keras 2.15 (training) knows BatchNormalization(synchronized=...); Keras 2.8 does not.  Synchronised statistics only matter for training,
# so the argument is dropped here (inference with the stored moving averages is unchanged).
import tensorflow as tf  # noqa: E402

_BN = tf.keras.layers.BatchNormalization
_bn_init = _BN.__init__


def _bn_init_compat(self, *args, **kwargs):
    kwargs.pop("synchronized", None)
    _bn_init(self, *args, **kwargs)


_BN.__init__ = _bn_init_compat
site, year, doy = sys.argv[1:4]
OUT = HERE.parent / "data" / "FAT_pixel_predictions" / site
sys.argv = ["x", "--ffp-modes", "0", "--site-fold-csv", str(OLD_DATA / "split_kfold" / "site_to_fold.k5.seed42.csv"), "--default-fold-for-5", "2",
            "--model-base-root", str(RES / "strict_v6_tf215_dualgpu_fourmodel"), "--hls-stats-root", str(RES / "norm_stats_kfold"),
            "--flux-stats-root", str(OLD_DATA / "flux_norm_stats_kfold"),          # identical to the V6 mete/xy statistics (checked); also holds the sensor file
            "--hls-npy", str(OLD_DATA / "input_data" / "HLS_matched_v11_remove_selected_sites.npy"),
            "--flux-npy", str(OLD_DATA / "input_data" / "merged_flux_era5_ffp_array_v11_remove_selected_sites.npy"),
            "--ffp-npy", str(OLD_DATA / "input_data" / "FFP_hls_aligned_v11_50x50_timefilled_remove_selected_sites.npy"),
            "--outdir", str(OUT), "--site-id", site, "--year", year, "--doy", doy, "--chunk-records", "4", "--batch-size", "1",
            "--is-joint-norm", "--save-patch-products", "--save-ffp-input", "--save-meta", "--save-extra-products", "--sample-check", "--quiet", "--overwrite"]
runpy.run_path(str(OLD_ROOT / "Prediction_streaming_cv5_foldaware_patch_FINAL.py"), run_name="__main__")
