"""Transformer model for input-side dynamic-footprint HLS aggregation."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import BatchNormalization, Concatenate, Dense, Embedding

from model_heads_ffp import COL
from Model_build_fairfix import make_decoder, transformer_block
from preagg_ffp import aggregate_dynamic_ffp_hls, normalize_preaggregated_hls


def preprocess_inputs(HLS_train_norm1, flux_train_norm0, FFP_train,
                      n_bands=6, mask_value=-9999.0, ffp_indices=None,
                      hls_indices=None, hls_mean=None, hls_std=None,
                      flux_is_numeric=False):
    """Create time-varying 1x1 HLS inputs while preserving site/flux layout."""
    if flux_is_numeric:
        site_names = None
        flux_numeric = np.asarray(flux_train_norm0, dtype=np.float32)
    else:
        site_names = flux_train_norm0[:, :, 0]
        flux_numeric = flux_train_norm0[:, :, 1:].astype(np.float32)
    hls_preagg_raw = aggregate_dynamic_ffp_hls(
        HLS_train_norm1, FFP_train, n_bands=n_bands,
        mask_value=mask_value, output_mask_value=mask_value,
        ffp_indices=ffp_indices, hls_indices=hls_indices,
    )
    if hls_mean is None or hls_std is None:
        hls_preagg = hls_preagg_raw
    else:
        hls_preagg = normalize_preaggregated_hls(
            hls_preagg_raw, hls_mean, hls_std, mask_value=mask_value
        )
    return (
        site_names,
        tf.convert_to_tensor(hls_preagg, dtype=tf.float32),
        tf.convert_to_tensor(flux_numeric, dtype=tf.float32),
    )


def _feature_embedding(values, units, layer_cnn, final_relu=False):
    """Dense equivalent of the original per-pixel stack of 1x1 convolutions."""
    x = values
    for layer_idx in range(layer_cnn):
        x = Dense(units, activation=None)(x)
        if layer_idx != layer_cnn - 1:
            x = layers.Activation("relu")(x)
            x = BatchNormalization()(x)
    if final_relu:
        x = layers.Activation("relu")(x)
    return x


def _make_covariate_embedding(units, layer_cnn=2, final_relu=False):
    block = tf.keras.Sequential()
    for layer_idx in range(layer_cnn - 1):
        block.add(Dense(units, activation=None, use_bias=layer_idx > 0))
        block.add(layers.Activation("relu"))
        block.add(BatchNormalization())
    block.add(Dense(units, activation=None, use_bias=layer_cnn > 1))
    if final_relu:
        block.add(layers.Activation("relu"))
    return block


def get_transformer_preagg(layer_cnn=2, layern=4, decoder_layer_n=2, units=64,
                           n_times=48, n_bands=6, n_head=4, drop=0.1,
                           n_out=3, n_flux_features=81,
                           n_index=None, is_batch=True, mask_value=-9999.0,
                           is_mete=True, is_sensor=True, is_xxy=False,
                           is_doy=True, is_hour=True, final_relu=False,
                           is_decoder=False):
    """Build the pre-aggregated counterpart of the fair FAT model.

    Spatial prediction and output-side weighting are removed. All temporal
    Transformer settings and output heads retain the original definitions.
    """
    del is_xxy  # Kept in the signature for experiment-spec compatibility.
    if n_index is None:
        n_index = [69, 68, 70, 71, 72, 73, 75, 76, 77, 78, 80, 64]

    hls_input = Input(shape=(n_times, n_bands), name="HLS_preagg_input")
    flux_input = Input(shape=(n_times, n_flux_features), name="Flux_input")

    hls_valid = tf.reduce_all(
        tf.math.is_finite(hls_input) & tf.not_equal(hls_input, mask_value),
        axis=-1, keepdims=True,
    )
    hls_clean = tf.where(hls_valid, hls_input, tf.zeros_like(hls_input))
    x = _feature_embedding(hls_clean, units, layer_cnn, final_relu=False)

    x_hour = flux_input[:, :, COL["Hour"]:COL["Hour"] + 1]
    x_doy = flux_input[:, :, COL["DoY"]:COL["DoY"] + 1]
    xmete = tf.gather(flux_input, indices=n_index, axis=-1)
    xsensor = flux_input[:, :, COL["Sensor"]]

    if is_doy:
        x = x + _make_covariate_embedding(units, layer_cnn, final_relu)(x_doy)
    if is_hour:
        x = x + _make_covariate_embedding(units, layer_cnn, final_relu)(x_hour)
    if is_sensor:
        sensor_embedding = tf.keras.Sequential([Embedding(input_dim=2, output_dim=units)])
        for _ in range(layer_cnn - 1):
            sensor_embedding.add(layers.Activation("relu"))
            sensor_embedding.add(Dense(units, activation=None))
        if final_relu:
            sensor_embedding.add(layers.Activation("relu"))
        x = x + sensor_embedding(xsensor)
    if is_mete:
        x = x + _make_covariate_embedding(units, layer_cnn, final_relu)(xmete)

    # Match the original temporal attention behavior. Samples have already
    # passed the same centre-valid filter and FFP fallback as the RF pipeline.
    padding_mask = x[:, :, :1] > -99999.0
    z = x
    for _ in range(layern):
        z = transformer_block(
            queryx=z, x=z, padding_mask=padding_mask,
            units=units, n_head=n_head, drop=drop, is_batch=is_batch,
        )

    outputs = []
    for output_idx in range(n_out):
        z_out = make_decoder(units, decoder_layer_n)(z) if is_decoder else z
        activation = "linear" if output_idx == 0 else "exponential"
        outputs.append(Dense(1, activation=activation)(z_out))

    prediction = Concatenate(axis=-1, name="flux_prediction")(outputs)
    return Model([hls_input, flux_input], prediction, name="FAT_preagg_dynamic_ffp")
