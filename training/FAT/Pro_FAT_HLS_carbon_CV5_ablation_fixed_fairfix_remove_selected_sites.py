#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared-backbone spatial-weighting ablation runner.

This version preserves the existing weights.h5 + spec.json workflow and also
saves a full Keras model artifact for easier reload/inference.
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
import json
import numpy as np
import importlib

from fair_interfaces import validate_predictions, seed_fold, audit_fold, host_inputs, aggregate_reference
import train_test
import Model_build
import myloss
import Model_history
import Evaluate_DL

from model_heads_ffp import resolve_mode, FFP_HEADS_VERSION

base_name = os.path.basename(__file__).replace(".py", "")


def to_numpy(x):
    """Convert tf.Tensor -> numpy, otherwise return as-is."""
    return x.numpy() if hasattr(x, "numpy") else x


def cast_selected_cols_inplace(site_like, cols_to_cast, dtype=np.float32):
    """
    Keep string/object columns unchanged, but cast only selected numeric columns.
    """
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


def save_model_artifacts(model, model_dir):
    """
    Save both weights-only checkpoint and full model.

    weights.h5:
        Robust backup. Requires rebuilding the same architecture before loading.

    model.keras:
        Full Keras model including architecture + weights. Preferred for reload.

    saved_model:
        TensorFlow SavedModel / export directory when supported.
    """
    weights_path = os.path.join(model_dir, "weights.h5")
    model_keras_path = os.path.join(model_dir, "model.keras")
    saved_model_dir = os.path.join(model_dir, "saved_model")

    model.save_weights(weights_path)
    print("[SAVE] weights:", weights_path)

    try:
        try:
            model.save(model_keras_path, include_optimizer=False)
        except TypeError:
            model.save(model_keras_path)
        print("[SAVE] full model:", model_keras_path)
    except Exception as e:
        print(f"[Warning] Failed to save full .keras model: {e}")

    try:
        if hasattr(model, "export"):
            model.export(saved_model_dir)
        else:
            model.save(saved_model_dir)
        print("[SAVE] SavedModel/export:", saved_model_dir)
    except Exception as e:
        print(f"[Warning] Failed to save SavedModel/export directory: {e}")


if __name__ == "__main__":
    if os.environ.get('STRICT_FAIR_RELEASE') != 'strict_v6_tf215_dualgpu_fourmodel':
        raise RuntimeError('Use the package run.py launcher: frozen controls are mandatory.')
    # ------------------------
    # Default hyperparams
    # ------------------------
    # Standalone defaults MUST match run.py/config.json for reproducibility.
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
    FOLD_ONLY = 0
    FOLD_LIST = None   # e.g. '0,1' passed by run.py: several folds in one process

    # ------------------------
    # FFP / mode config
    # ------------------------
    FFP_MODE = 0

    # ------------------------
    # Version config
    # ------------------------
    VERSION = "strict_v6_tf215_dualgpu_fourmodel"

    # args:
    # 1:EPOCH 2:LR 3:BS 4:L2 5:units 6:layer_cnn 7:layern
    # 8:fold_id 9:FFP_MODE 10:VERSION
    if len(sys.argv) > 7:
        EPOCH = int(sys.argv[1])
        LEARNING_RATE = float(sys.argv[2])
        BATCH_SIZE = int(sys.argv[3])
        L2 = float(sys.argv[4])
        units = int(sys.argv[5])
        layer_cnn = int(sys.argv[6])
        layern = int(sys.argv[7])

    if len(sys.argv) > 8:
        FOLD_LIST = [int(x) for x in str(sys.argv[8]).split(',')]
        FOLD_ONLY = FOLD_LIST[0] if len(FOLD_LIST) == 1 else -2

    if len(sys.argv) > 9:
        try:
            FFP_MODE = int(sys.argv[9])
        except ValueError:
            FFP_MODE = sys.argv[9]

    if len(sys.argv) > 10:
        VERSION = str(sys.argv[10])

    myloss = importlib.reload(myloss)
    loss = myloss.mask_loss_physical_gpp(maskValue=-9999.0, lambda1=1, beta=0)

    base_tag_nofold = (
        f"{base_name}"
        f".epoch{EPOCH:03d}"
        f".rate{LEARNING_RATE:.5f}"
        f".batch{BATCH_SIZE:03d}"
        f".L{L2:.5f}"
    )

    DATA_ROOT = os.environ.get("TRAIN_DATA_ROOT", "./data")
    ensure_dir(DATA_ROOT)

    mode_name = resolve_mode(FFP_MODE)
    if isinstance(FFP_MODE, int):
        mode_dir_name = f"FFP{FFP_MODE}_{mode_name}"
        priority_group = "main" if FFP_MODE <= 4 else "supplementary"
    else:
        mode_dir_name = f"{mode_name}"
        priority_group = "unknown"

    version_root = ensure_dir(os.path.join(DATA_ROOT, VERSION))
    mode_root = ensure_dir(os.path.join(version_root, mode_dir_name))

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
        "model_type": ("FAT_CV5_ablation" if FFP_MODE == 0 else "Uniform_Transformer_CV5_ablation"),
        "notes": "Two-model controlled comparison: FAT vs half-hourly Uniform Transformer.",
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
        seed_fold(SEED + fold)
        print(f"\n========== Running version={VERSION}, mode={mode_name}, fold {fold} / {N_SPLITS-1} ==========")

        fold_root = ensure_dir(os.path.join(mode_root, f"fold{fold}"))
        model_dir = ensure_dir(os.path.join(fold_root, "model"))
        pred_dir = ensure_dir(os.path.join(fold_root, "pred"))
        eval_dir = ensure_dir(os.path.join(fold_root, "eval"))

        split_dir = ensure_dir(os.path.join(DATA_ROOT, "split_kfold"))
        index_train, index_test = train_test.get_kfold_siteid(
            x2_y_array,
            n_splits=N_SPLITS,
            fold_id=fold,
            seed=SEED,
            split_dir=split_dir,
        )

        norm_dir = ensure_dir(os.path.join(DATA_ROOT, "norm_stats_kfold"))
        HLS_train_norm0, HLS_test_norm0 = train_test.get_HLS_normalization_joint(
            x1_ref_array,
            x2_y_array,
            index_train,
            index_test,
            is_joint_norm=True,
            fold_id=fold,
            output_dir=norm_dir,
        )

        flux_norm_dir = ensure_dir(os.path.join(DATA_ROOT, "flux_norm_stats_kfold"))
        flux_train_norm0, flux_test_norm0 = train_test.normalize_flux_train_test(
            x2_y_array,
            index_train,
            index_test,
            use_mete=True,
            use_xy=True,
            use_sensor=False,  # Sensor is a categorical embedding index (0/1).
            fold_id=fold,
            save_dir=flux_norm_dir,
        )

        HLS_train_norm1 = np.transpose(HLS_train_norm0, (0, 2, 3, 1))
        HLS_test_norm1 = np.transpose(HLS_test_norm0, (0, 2, 3, 1))

        index_train_non_fill = (
            (HLS_train_norm1[:, 23:27, 23:27, 0] != -9999.).sum(axis=(1, 2)) == 4 * 4
        )
        index_test_non_fill = (
            (HLS_test_norm1[:, 23:27, 23:27, 0] != -9999.).sum(axis=(1, 2)) == 4 * 4
        )

        HLS_train_norm1 = HLS_train_norm1[index_train_non_fill]
        HLS_test_norm1 = HLS_test_norm1[index_test_non_fill]
        flux_train_norm0 = flux_train_norm0[index_train_non_fill]
        flux_test_norm0 = flux_test_norm0[index_test_non_fill]

        FFP_train = ffp_array[index_train][index_train_non_fill]
        FFP_test = ffp_array[index_test][index_test_non_fill]

        flux_train_site_mete = x2_y_array[index_train][index_train_non_fill]
        flux_test_site_mete = x2_y_array[index_test][index_test_non_fill]

        site_train_names, HLS_train_tensor, flux_train_numeric_tensor, FFP_train_tensor = (
            host_inputs(Model_build.preprocess_inputs, HLS_train_norm1, flux_train_norm0, FFP_train)
        )
        site_test_names, HLS_test_tensor, flux_test_numeric_tensor, FFP_test_tensor = (
            host_inputs(Model_build.preprocess_inputs, HLS_test_norm1, flux_test_norm0, FFP_test)
        )

        audit_fold(fold, index_train, index_test, index_train_non_fill, index_test_non_fill,
                   flux_train_numeric_tensor, flux_test_numeric_tensor)

        train_n = HLS_train_tensor.shape[0]
        per_epoch = max(1, train_n // BATCH_SIZE)

        for run_id in range(ITERS):
            print(f"\n[Version {VERSION}] [Mode {mode_name}] [Fold {fold}] Run {run_id}")

            from distributed_runtime import get_strategy
            with get_strategy().scope():
                Model_build_r = importlib.reload(Model_build)
                model = Model_build_r.get_transformer_cnn_at(
                    layer_cnn=layer_cnn,
                    layern=layern,
                    decoder_layer_n=2,
                    units=units,
                    n_times=flux_train_numeric_tensor.shape[1],
                    n_hls_features=HLS_train_tensor.shape[3],
                    n_bands=6,
                    n_window=50,
                    n_head=n_head,
                    drop=drop,
                    n_out=3,
                    n_flux_features=flux_test_numeric_tensor.shape[2],
                    n_index=[69, 68, 70, 71, 72, 73, 75, 76, 77, 78, 80, 64],
                    FFP=FFP_MODE,
                    is_batch=True,
                    mask_value=-9999.0,
                    is_mete=True,
                    is_sensor=True,
                    is_xxy=False,
                    is_doy=True,
                    is_hour=True,
                    final_relu=False,
                    is_decoder=False,
                    is_degrade=False,
                    radius=0.03,
                )

                if run_id == 0 and fold == fold_list[0]:
                    model.summary()

                Model_history_r = importlib.reload(Model_history)
                model_history = Model_history_r.my_train_1schedule(
                    model,
                    HLS_train_tensor,
                    flux_train_numeric_tensor,
                    FFP_train_tensor,
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
                    reduce_epoch=False,
                )

            from batching import predict_in_batches
            y_pred = predict_in_batches(
                model,
                [to_numpy(HLS_test_tensor), to_numpy(flux_test_numeric_tensor), to_numpy(FFP_test_tensor)],
            ) / y_SCALE
            y_true = flux_test_numeric_tensor[:, :, 7:10] / y_SCALE
            validate_predictions(y_true, y_pred)

            site_block = flux_test_site_mete[:, :, :8]
            site_block_fixed = cast_selected_cols_inplace(
                site_block,
                cols_to_cast=range(1, 8),
                dtype=np.float32,
            )
            np.save(os.path.join(pred_dir, "true.npy"), to_numpy(y_true).astype(np.float32))
            np.save(os.path.join(pred_dir, "pred.npy"), to_numpy(y_pred).astype(np.float32))
            np.save(os.path.join(pred_dir, "site.npy"), site_block_fixed)

            Evaluate_DL.evaluate_flux_model_results_single(
                result_outdir=pred_dir,
                evaluate_dir=eval_dir,
                run_id=0,
            )

            if run_id == 0:
                kfold_true_list.append(to_numpy(y_true).astype(np.float32))
                kfold_pred_list.append(to_numpy(y_pred).astype(np.float32))
                kfold_site_list.append(site_block_fixed)

            # ------------------------
            # Export model artifacts
            # ------------------------
            save_model_artifacts(model, model_dir)

            spec = dict(
                version=VERSION,
                base_script=base_name,
                mode_name=mode_name,
                priority_group=priority_group,
                FFP=FFP_MODE,
                ffp_heads_version=FFP_HEADS_VERSION,
                fold=int(fold),
                run_id=int(run_id),
                layer_cnn=int(layer_cnn),
                layern=int(layern),
                decoder_layer_n=2,
                units=int(units),
                n_times=int(flux_train_numeric_tensor.shape[1]),
                n_hls_features=int(HLS_train_tensor.shape[3]),
                n_bands=6,
                n_window=50,
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
                is_degrade=False,
                radius=0.03,
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

    if len(fold_list) == N_SPLITS and len(kfold_true_list) > 0:   # merged table only when all folds ran in this process
        print("\n========== KFold aggregated evaluation ==========")

        y_true_all = np.concatenate(kfold_true_list, axis=0)
        y_pred_all = np.concatenate(kfold_pred_list, axis=0)
        site_all = np.concatenate(kfold_site_list, axis=0)

        kfold_root = ensure_dir(os.path.join(mode_root, f"kfold{N_SPLITS}"))
        kpred_dir = ensure_dir(os.path.join(kfold_root, "pred"))
        keval_dir = ensure_dir(os.path.join(kfold_root, "eval"))

        np.save(os.path.join(kpred_dir, "true.npy"), y_true_all)
        np.save(os.path.join(kpred_dir, "pred.npy"), y_pred_all)
        np.save(os.path.join(kpred_dir, "site.npy"), site_all)

        Evaluate_DL.evaluate_flux_model_results_single(
            result_outdir=kpred_dir,
            evaluate_dir=keval_dir,
            run_id=0,
        )

        print("\n[KFold Summary Saved]")
        print("  pred:", kpred_dir)
        print("  eval:", keval_dir)

