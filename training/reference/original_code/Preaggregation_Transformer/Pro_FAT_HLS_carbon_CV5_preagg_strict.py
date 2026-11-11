#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Input-side dynamic-footprint aggregation comparison runner.

Mode numbering is aligned with:
- Main-text priority first
- Supplementary sensitivity modes second

Output layout:

./data/<VERSION>/<MODE>/
    mode_info.txt
    mode_spec.json

    fold0/
        model/
            weights.h5
            spec.json
            summary.txt
            history.json
        pred/
            true.npy
            pred.npy
            site.npy
        eval/
            ...

    fold1/
        ...

    kfold5/
        pred/
            true.npy
            pred.npy
            site.npy
        eval/
            ...
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# TensorFlow 2.8 deterministic sparse GPU updates are incompatible with the
# negative sensor-embedding indices used by the reference FAT preprocessing.
# Seeds remain fixed, but deterministic kernels must stay disabled locally.
os.environ["TF_DETERMINISTIC_OPS"] = "0"

import sys
import json
import gc
import hashlib
import random
import shutil
import numpy as np
import importlib
import pandas as pd
import tensorflow as tf

import train_test
import Model_build_preagg_fairfix as Model_build
import myloss
import Model_history_preagg as Model_history
import Evaluate_DL
from preagg_ffp import band_mean_std_memmap, center_valid_mask_memmap
from flux_preprocess_lowmem import normalize_flux_train_test_lowmem

PREAGG_VERSION = "v2_strict_fair_dynamic_ffp"

STRICT_REFERENCE = {
    "EPOCH": 30,
    "LEARNING_RATE": 0.0001,
    "BATCH_SIZE": 16,
    "L2": 0.1,
    "units": 64,
    "drop": 0.1,
    "METHOD": 2,
    "n_head": 4,
    "layer_cnn": 2,
    "layern": 4,
    "N_SPLITS": 5,
    "SEED": 42,
}

base_name = os.path.basename(__file__).replace(".py", "")


def to_numpy(x):
    """Convert tf.Tensor -> numpy, otherwise return as-is."""
    return x.numpy() if hasattr(x, "numpy") else x


def cast_selected_cols_inplace(site_like, cols_to_cast, dtype=np.float32):
    """
    Keep string/object columns unchanged, but cast only selected numeric columns.

    Parameters
    ----------
    site_like : np.ndarray or tf.Tensor
        Shape (N, T, K). May contain string columns, so dtype can be object.
    cols_to_cast : iterable of int
        Column indices to cast.
    dtype : numpy dtype
        Target dtype for selected numeric columns.

    Returns
    -------
    out : np.ndarray
        Same shape/content as input, but selected columns cast to `dtype`.
    """
    arr = to_numpy(site_like)
    out = np.array(arr, copy=True)

    for j in cols_to_cast:
        out[:, :, j] = out[:, :, j].astype(dtype)

    return out


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_global_seed(seed):
    """Make a fold reproducible whether it is run alone or in a five-fold job."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def validate_reference_split(map_path, y_array, n_splits):
    """Reject a stale or incomplete split map before any model is trained."""
    df = pd.read_csv(map_path)
    required = {"site_id", "fold"}
    if not required.issubset(df.columns):
        raise ValueError(f"Reference split must contain columns {sorted(required)}: {map_path}")
    if df["site_id"].duplicated().any():
        raise ValueError(f"Reference split contains duplicate site IDs: {map_path}")

    observed_folds = set(df["fold"].astype(int).tolist())
    expected_folds = set(range(n_splits))
    if observed_folds != expected_folds:
        raise ValueError(
            f"Reference split folds differ: observed={sorted(observed_folds)}, "
            f"expected={sorted(expected_folds)}"
        )

    sample_sites = train_test._extract_first_valid_siteid_per_sample(
        y_array, mask_value=-9999.0
    )
    data_sites = set(np.unique(sample_sites[sample_sites != -9999.0]).astype(str))
    map_sites = set(df["site_id"].astype(str))
    if data_sites != map_sites:
        missing = sorted(data_sites - map_sites)
        extra = sorted(map_sites - data_sites)
        raise ValueError(
            "Reference split does not exactly match the input dataset. "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return sha256_file(map_path), len(map_sites)


def validate_strict_hyperparameters(actual):
    mismatches = {
        key: (STRICT_REFERENCE[key], actual[key])
        for key in STRICT_REFERENCE
        if actual[key] != STRICT_REFERENCE[key]
    }
    if mismatches:
        details = ", ".join(
            f"{key}: expected={expected}, actual={actual_value}"
            for key, (expected, actual_value) in mismatches.items()
        )
        raise ValueError(
            "Strict-fair configuration check failed. " + details
        )


def write_json(path, obj):
    def json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=json_default)


def write_txt_lines(path, mapping):
    with open(path, "w", encoding="utf-8") as f:
        for k, v in mapping.items():
            f.write(f"{k}: {v}\n")


def resolve_input_file(input_dir, preferred_name, fallback_name):
    """Resolve the selected-site-filtered dataset first, then legacy v11."""
    preferred_path = os.path.join(input_dir, preferred_name)
    fallback_path = os.path.join(input_dir, fallback_name)
    if os.path.isfile(preferred_path):
        return preferred_path
    if os.path.isfile(fallback_path):
        return fallback_path
    raise FileNotFoundError(
        "Neither supported input file exists:\n"
        f"  preferred: {preferred_path}\n"
        f"  fallback : {fallback_path}"
    )


def install_numpy_pickle_compat():
    """Allow NumPy 1.x to read object arrays pickled by NumPy 2.x.

    NumPy 2 serializes some objects with the private module path
    ``numpy._core.multiarray``. TensorFlow 2.8 requires NumPy 1.x, where the
    equivalent path is ``numpy.core.multiarray``. Registering aliases is safer
    than upgrading NumPy and breaking the local TensorFlow environment.
    """
    if hasattr(np, "_core"):
        return
    import numpy.core as numpy_core
    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", numpy_core.multiarray)
    sys.modules.setdefault("numpy._core._multiarray_umath", numpy_core._multiarray_umath)
    sys.modules.setdefault("numpy._core.numeric", numpy_core.numeric)


if __name__ == "__main__":

    # ------------------------
    # Default hyperparams
    # ------------------------
    EPOCH = 30
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 16
    L2 = 0.1
    units = 64
    drop = 0.1
    METHOD = 2
    n_head = 4
    layer_cnn = 2
    layern = 4
    y_SCALE = 1
    ITERS = 1

    # ------------------------
    # KFold config
    # ------------------------
    N_SPLITS = 5
    SEED = 42
    FOLD_ONLY = 0   # default run fold 0; set -1 to run all folds sequentially

    # ------------------------
    # This comparison has one fixed mode: dynamic FFP aggregation before the model.
    # ------------------------
    FFP_MODE = "preagg_dynamic_ffp"

    # ------------------------
    # Version config
    # ------------------------
    VERSION = "v7_6_preagg_strict_fair_seed42"

    # args:
    # 1:EPOCH 2:LR 3:BS 4:L2 5:units 6:layer_cnn 7:layern 8:fold_id 9:FFP_MODE 10:VERSION
    if len(sys.argv) > 7:
        EPOCH = int(sys.argv[1])
        LEARNING_RATE = float(sys.argv[2])
        BATCH_SIZE = int(sys.argv[3])
        L2 = float(sys.argv[4])
        units = int(sys.argv[5])
        layer_cnn = int(sys.argv[6])
        layern = int(sys.argv[7])

    if len(sys.argv) > 8:
        FOLD_ONLY = int(sys.argv[8])

    if len(sys.argv) > 9:
        try:
            FFP_MODE = int(sys.argv[9])
        except ValueError:
            FFP_MODE = sys.argv[9]

    if len(sys.argv) > 10:
        VERSION = str(sys.argv[10])

    validate_strict_hyperparameters({
        "EPOCH": EPOCH,
        "LEARNING_RATE": LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "L2": L2,
        "units": units,
        "drop": drop,
        "METHOD": METHOD,
        "n_head": n_head,
        "layer_cnn": layer_cnn,
        "layern": layern,
        "N_SPLITS": N_SPLITS,
        "SEED": SEED,
    })

    myloss = importlib.reload(myloss)
    loss = myloss.mask_loss_physical_gpp(maskValue=-9999.0, lambda1=1, beta=0)

    # ------------------------
    # Base tag (used only for metadata)
    # ------------------------
    base_tag_nofold = (
        f"{base_name}"
        f".epoch{EPOCH:03d}"
        f".rate{LEARNING_RATE:.5f}"
        f".batch{BATCH_SIZE:03d}"
        f".L{L2:.5f}"
    )

    # ------------------------
    # Unified root config
    # ------------------------
    DATA_ROOT = os.environ.get("TRAIN_DATA_ROOT", "./data")
    ensure_dir(DATA_ROOT)

    # ------------------------
    # Resolve mode and create mode root
    # ------------------------
    mode_name = "preagg_dynamic_ffp"
    mode_dir_name = mode_name
    priority_group = "comparison"

    version_root = ensure_dir(os.path.join(DATA_ROOT, VERSION))
    mode_root = ensure_dir(os.path.join(version_root, mode_dir_name))

    mode_info = {
        "version": VERSION,
        "DATA_ROOT": DATA_ROOT,
        "base_script": base_name,
        "base_tag_nofold": base_tag_nofold,
        "FFP_input": FFP_MODE,
        "mode_name": mode_name,
        "priority_group": priority_group,
        "preagg_version": PREAGG_VERSION,
        "N_SPLITS": N_SPLITS,
        "SEED": SEED,
        "FOLD_ONLY": FOLD_ONLY,
        "EPOCH": EPOCH,
        "LEARNING_RATE": LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "L2": L2,
        "units": units,
        "drop": drop,
        "METHOD": METHOD,
        "n_head": n_head,
        "layer_cnn": layer_cnn,
        "layern": layern,
        "y_SCALE": y_SCALE,
        "ITERS": ITERS,
        "model_type": "FAT_CV5_input_side_preaggregation",
        "notes": "Dynamic FFP aggregation is applied to normalized 50x50 HLS before the temporal Transformer."
    }
    write_json(os.path.join(mode_root, "mode_spec.json"), mode_info)
    write_txt_lines(os.path.join(mode_root, "mode_info.txt"), mode_info)

    # ------------------------
    # Load data
    # ------------------------
    inputdata_path = os.environ.get(
        "INPUTDATA_PATH",
        './data/external/HLS_FLUX_ERA5_matched_filtering',
    )
    install_numpy_pickle_compat()

    hls_path = resolve_input_file(
        inputdata_path,
        "HLS_matched_v11_remove_selected_sites.npy",
        "HLS_matched_v11.npy",
    )
    flux_path = resolve_input_file(
        inputdata_path,
        "merged_flux_era5_ffp_array_v11_remove_selected_sites.npy",
        "merged_flux_era5_ffp_array_v11.npy",
    )
    ffp_path = resolve_input_file(
        inputdata_path,
        "FFP_hls_aligned_v11_50x50_timefilled_remove_selected_sites.npy",
        "FFP_hls_aligned_v11_50x50_timefilled.npy",
    )
    print("[Input] HLS :", hls_path)
    print("[Input] Flux:", flux_path)
    print("[Input] FFP :", ffp_path)
    # HLS and FFP are large numeric arrays. Memory mapping avoids allocating
    # roughly 6 GiB + 13.5 GiB before fold processing even begins.
    x1_ref_array = np.load(hls_path, allow_pickle=True, mmap_mode="r")
    x2_y_array = np.load(flux_path, allow_pickle=True)
    ffp_array = np.load(ffp_path, allow_pickle=True, mmap_mode="r")

    if not (x1_ref_array.shape[0] == x2_y_array.shape[0] == ffp_array.shape[0]):
        raise ValueError(
            "Input sample counts differ: "
            f"HLS={x1_ref_array.shape[0]}, Flux={x2_y_array.shape[0]}, FFP={ffp_array.shape[0]}"
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    reference_split_dir = os.environ.get(
        "REFERENCE_SPLIT_DIR", os.path.join(script_dir, "reference_split")
    )
    reference_map = os.path.join(
        reference_split_dir, f"site_to_fold.k{N_SPLITS}.seed{SEED}.csv"
    )
    if not os.path.isfile(reference_map):
        raise FileNotFoundError(f"FAT reference split is missing: {reference_map}")
    split_sha256, split_site_count = validate_reference_split(
        reference_map, x2_y_array, N_SPLITS
    )
    split_dir = ensure_dir(os.path.join(DATA_ROOT, "split_kfold_strict_fat_reference"))
    active_map = os.path.join(split_dir, os.path.basename(reference_map))
    if os.path.isfile(active_map) and sha256_file(active_map) != split_sha256:
        raise ValueError(f"Existing active split differs from FAT reference: {active_map}")
    if not os.path.isfile(active_map):
        shutil.copy2(reference_map, active_map)
    print(f"[Strict] FAT split: {reference_map}")
    print(f"[Strict] split SHA-256: {split_sha256}")
    print(f"[Strict] sites: {split_site_count}")

    mode_info.update({
        "strict_fair": True,
        "reference_fat_mode": "v7_6_remove_selected_sites_fairfix/FFP0_dynamic_ffp",
        "reference_split_file": reference_map,
        "reference_split_sha256": split_sha256,
        "reference_site_count": split_site_count,
        "deterministic_ops": False,
        "deterministic_ops_note": (
            "Disabled for TensorFlow 2.8/TFA AdamW sparse Embedding compatibility; "
            "Python, NumPy and TensorFlow seeds remain fixed."
        ),
        "fold_model_seed_rule": "SEED + fold_id",
    })
    write_json(os.path.join(mode_root, "mode_spec.json"), mode_info)
    write_txt_lines(os.path.join(mode_root, "mode_info.txt"), mode_info)

    # ------------------------
    # Fold plan
    # ------------------------
    if FOLD_ONLY < 0:
        fold_list = list(range(N_SPLITS))
    else:
        fold_list = [FOLD_ONLY]

    # For kfold aggregated evaluation
    kfold_true_list = []
    kfold_pred_list = []
    kfold_site_list = []

    # ------------------------
    # Run folds sequentially
    # ------------------------
    for fold in fold_list:
        print(f"\n========== Running version={VERSION}, mode={mode_name}, fold {fold} / {N_SPLITS-1} ==========")
        fold_model_seed = SEED + fold
        set_global_seed(fold_model_seed)
        print(f"[Strict] fold model seed: {fold_model_seed}")

        # fold directories
        fold_root = ensure_dir(os.path.join(mode_root, f"fold{fold}"))
        model_dir = ensure_dir(os.path.join(fold_root, "model"))
        pred_dir  = ensure_dir(os.path.join(fold_root, "pred"))
        eval_dir  = ensure_dir(os.path.join(fold_root, "eval"))

        # ------------------------
        # Split (KFold by SITE_ID)
        # ------------------------
        index_train, index_test = train_test.get_kfold_siteid(
            x2_y_array, n_splits=N_SPLITS, fold_id=fold, seed=SEED, split_dir=split_dir
        )

        # ------------------------
        # HLS normalization statistics (streaming, train fold only)
        # ------------------------
        norm_dir = ensure_dir(os.path.join(DATA_ROOT, "norm_stats_kfold"))
        train_indices_all = np.flatnonzero(index_train)
        test_indices_all = np.flatnonzero(index_test)
        hls_mean, hls_std, hls_count = band_mean_std_memmap(
            x1_ref_array, train_indices_all, n_bands=6, mask_value=-9999.0
        )
        stats_path = os.path.join(norm_dir, f"Preagg6_mean_std_fold{fold}.csv")
        np.savetxt(
            stats_path,
            np.column_stack([np.arange(6), hls_mean, hls_std, hls_count]),
            delimiter=",", header="band_index,mean,std,valid_count", comments="",
        )
        print("[HLS stats]", stats_path)
        print("[HLS mean]", hls_mean)
        print("[HLS std ]", hls_std)

        # ------------------------
        # Flux normalization
        # ------------------------
        flux_norm_dir = ensure_dir(os.path.join(DATA_ROOT, "flux_norm_stats_kfold"))
        flux_train_norm0, flux_test_norm0 = normalize_flux_train_test_lowmem(
            x2_y_array, train_indices_all, test_indices_all,
            fold_id=fold,
            save_dir=flux_norm_dir
        )

        # ------------------------
        # Same central 4x4 non-fill filter, evaluated from the HLS memmap
        # ------------------------
        index_train_non_fill = center_valid_mask_memmap(
            x1_ref_array, train_indices_all, mask_value=-9999.0
        )
        index_test_non_fill = center_valid_mask_memmap(
            x1_ref_array, test_indices_all, mask_value=-9999.0
        )
        flux_train_norm0 = flux_train_norm0[index_train_non_fill]
        flux_test_norm0  = flux_test_norm0[index_test_non_fill]

        # Keep only global row indices for the memmapped FFP source. Boolean
        # slicing here would materialize 10+ GiB for each training fold.
        train_global_indices = train_indices_all[index_train_non_fill]
        test_global_indices = test_indices_all[index_test_non_fill]

        flux_test_site_mete = x2_y_array[test_global_indices, :, :8]

        # ------------------------
        # preprocess_inputs
        # ------------------------
        site_train_names, HLS_train_tensor, flux_train_numeric_tensor =\
            Model_build.preprocess_inputs(
                x1_ref_array, flux_train_norm0, ffp_array,
                hls_indices=train_global_indices,
                ffp_indices=train_global_indices,
                hls_mean=hls_mean, hls_std=hls_std,
                flux_is_numeric=True,
            )

        site_test_names, HLS_test_tensor, flux_test_numeric_tensor =\
            Model_build.preprocess_inputs(
                x1_ref_array, flux_test_norm0, ffp_array,
                hls_indices=test_global_indices,
                ffp_indices=test_global_indices,
                hls_mean=hls_mean, hls_std=hls_std,
                flux_is_numeric=True,
            )

        print("HLS pre-aggregated train shape:", HLS_train_tensor.shape)
        print("HLS pre-aggregated test shape :", HLS_test_tensor.shape)
        if HLS_train_tensor.shape[1] != flux_train_numeric_tensor.shape[1]:
            raise ValueError("Pre-aggregated HLS and flux time dimensions differ in training data.")
        if HLS_test_tensor.shape[1] != flux_test_numeric_tensor.shape[1]:
            raise ValueError("Pre-aggregated HLS and flux time dimensions differ in test data.")

        train_n = HLS_train_tensor.shape[0]
        per_epoch = max(1, train_n // BATCH_SIZE)

        # ------------------------
        # Training loop
        # ------------------------
        # If ITERS > 1, later runs overwrite files in the same fold directory.
        # This script is optimized for the common case ITERS=1.
        for run_id in range(ITERS):
            print(f"\n[Version {VERSION}] [Mode {mode_name}] [Fold {fold}] Run {run_id}")

            Model_build_r = importlib.reload(Model_build)

            model = Model_build_r.get_transformer_preagg(
                layer_cnn=layer_cnn, layern=layern, decoder_layer_n=2, units=units,
                n_times=flux_train_numeric_tensor.shape[1],
                n_bands=HLS_train_tensor.shape[2], n_head=n_head, drop=drop, n_out=3,
                n_flux_features=flux_test_numeric_tensor.shape[2],
                n_index=[69, 68, 70, 71, 72, 73, 75, 76, 77, 78, 80, 64],
                is_batch=True, mask_value=-9999.0,
                is_mete=True, is_sensor=True, is_xxy=False, is_doy=True, is_hour=True,
                final_relu=False, is_decoder=False
            )

            if run_id == 0 and fold == fold_list[0]:
                model.summary()

            Model_history_r = importlib.reload(Model_history)
            model_history = Model_history_r.my_train_1schedule(
                model,
                HLS_train_tensor, flux_train_numeric_tensor,
                epochs=EPOCH,
                start_rate=LEARNING_RATE,
                loss=loss,
                per_epoch=per_epoch,
                split_epoch=5,
                option=METHOD,
                decay=L2,
                batch_size=BATCH_SIZE,
                validation_split=0.04,
                hold_epoch=0,
                reduce_epoch=False
            )

            y_pred = model.predict(
                [HLS_test_tensor, flux_test_numeric_tensor],
                verbose=2
            ) / y_SCALE

            y_true = flux_test_numeric_tensor[:, :, 7:10] / y_SCALE

            # ------------------------
            # Save predictions
            # ------------------------
            site_block = flux_test_site_mete[:, :, :8]
            site_block_fixed = cast_selected_cols_inplace(
                site_block,
                cols_to_cast=range(1, 8),
                dtype=np.float32
            )
            np.save(os.path.join(pred_dir, "true.npy"), to_numpy(y_true).astype(np.float32))
            np.save(os.path.join(pred_dir, "pred.npy"), to_numpy(y_pred).astype(np.float32))
            np.save(os.path.join(pred_dir, "site.npy"), site_block_fixed)

            # ------------------------
            # Evaluate
            # ------------------------
            Evaluate_DL.evaluate_flux_model_results_single(
                result_outdir=pred_dir,
                evaluate_dir=eval_dir,
                run_id=0
            )

            # ------------------------
            # Collect for kfold summary (use run_id==0)
            # ------------------------
            if run_id == 0:
                kfold_true_list.append(to_numpy(y_true).astype(np.float32))
                kfold_pred_list.append(to_numpy(y_pred).astype(np.float32))

                site_block = flux_test_site_mete[:, :, :8]
                site_block_fixed = cast_selected_cols_inplace(
                    site_block,
                    cols_to_cast=range(1, 8),
                    dtype=np.float32
                )
                kfold_site_list.append(site_block_fixed)

            # ------------------------
            # Export model artifacts
            # ------------------------
            model.save_weights(os.path.join(model_dir, "weights.h5"))

            spec = dict(
                version=VERSION,
                base_script=base_name,
                mode_name=mode_name,
                priority_group=priority_group,
                FFP=FFP_MODE,
                preagg_version=PREAGG_VERSION,

                fold=int(fold),
                run_id=int(run_id),

                layer_cnn=int(layer_cnn),
                layern=int(layern),
                decoder_layer_n=2,
                units=int(units),

                n_times=int(flux_train_numeric_tensor.shape[1]),
                n_hls_features=int(HLS_train_tensor.shape[2]),
                n_bands=int(HLS_train_tensor.shape[2]),
                source_window=50,
                input_window=1,
                n_head=int(n_head),
                drop=float(drop),
                n_out=3,
                n_flux_features=int(flux_test_numeric_tensor.shape[2]),

                n_index=[69, 68, 70, 71, 72, 73, 75, 76, 77, 78, 80, 64],

                is_batch=True,
                mask_value=-9999.0,
                is_mete=True,
                is_sensor=True,
                is_xxy=False,
                is_doy=True,
                is_hour=True,
                final_relu=False,
                is_decoder=False,
                aggregation="dynamic_ffp_before_model",
            )
            write_json(os.path.join(model_dir, "spec.json"), spec)

            with open(os.path.join(model_dir, "summary.txt"), "w", encoding="utf-8") as f:
                model.summary(print_fn=lambda x: f.write(x + "\n"))

            try:
                history_obj = getattr(model_history, "history", None)
                if history_obj is not None:
                    write_json(os.path.join(model_dir, "history.json"), history_obj)
            except Exception as e:
                print(f"[Warning] Failed to save history.json for fold {fold}: {e}")

            print("Saved fold artifacts to:", fold_root)

    # ------------------------
    # KFold aggregated evaluation
    # ------------------------
    if len(fold_list) > 1 and len(kfold_true_list) > 0:
        print("\n========== KFold aggregated evaluation ==========")

        y_true_all = np.concatenate(kfold_true_list, axis=0)
        y_pred_all = np.concatenate(kfold_pred_list, axis=0)
        site_all   = np.concatenate(kfold_site_list, axis=0)

        kfold_root = ensure_dir(os.path.join(mode_root, f"kfold{N_SPLITS}"))
        kpred_dir  = ensure_dir(os.path.join(kfold_root, "pred"))
        keval_dir  = ensure_dir(os.path.join(kfold_root, "eval"))

        np.save(os.path.join(kpred_dir, "true.npy"), y_true_all)
        np.save(os.path.join(kpred_dir, "pred.npy"), y_pred_all)
        np.save(os.path.join(kpred_dir, "site.npy"), site_all)

        Evaluate_DL.evaluate_flux_model_results_single(
            result_outdir=kpred_dir,
            evaluate_dir=keval_dir,
            run_id=0
        )

        print("\n[KFold Summary Saved]")
        print("  pred:", kpred_dir)
        print("  eval:", keval_dir)
