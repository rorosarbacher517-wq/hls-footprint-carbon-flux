import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Dense, Concatenate, Dropout, LayerNormalization, Embedding

print("XLA_FLAGS =", os.environ.get("XLA_FLAGS"))
print("CUDA_HOME =", os.environ.get("CUDA_HOME"))

import RadialConv2D
import MaskedAveragePooling2D

from model_heads_ffp import (
    COL,
    resolve_mode,
    build_weight_head,
    FFP_HEADS_VERSION,
)


def preprocess_inputs(HLS_train_norm1, flux_train_norm0, FFP_train):
    site_names = flux_train_norm0[:, :, 0]
    flux_numeric = flux_train_norm0[:, :, 1:].astype(np.float32)
    flux_numeric_tensor = tf.convert_to_tensor(flux_numeric, dtype=tf.float32)
    HLS_tensor = tf.convert_to_tensor(HLS_train_norm1, dtype=tf.float32)
    FFP_tensor = tf.convert_to_tensor(FFP_train, dtype=tf.float32)
    return site_names, HLS_tensor, flux_numeric_tensor, FFP_tensor


def postprocess_outputs(site_names, model_output):
    if isinstance(model_output, tf.Tensor):
        model_output = model_output.numpy()
    site_names_expanded = np.expand_dims(site_names, axis=-1)
    final_data = np.concatenate([site_names_expanded, model_output], axis=-1)
    return final_data


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation="relu"),
        tf.keras.layers.Dense(d_model)
    ])


def create_hls_mask(HLS, mask_value=-9999.0):
    """Valid pixel mask for HLS bands.

    A pixel is valid only when all selected bands are finite and not equal to
    mask_value. This is slightly safer than checking only -9999 because NaN/Inf
    can otherwise leak through multiplication by the mask.
    """
    valid_mask = tf.reduce_all(
        tf.math.is_finite(HLS) & tf.not_equal(HLS, mask_value),
        axis=-1,
        keepdims=True,
    )
    return valid_mask


def make_embedding(units, layer_cnn=2, final_relu=False):
    layers_list = []
    for i in range(layer_cnn - 1):
        layers_list.append(Dense(units, activation=None, use_bias=i > 0))
        layers_list.append(layers.Activation("relu"))
        layers_list.append(layers.BatchNormalization(synchronized=True))

    layers_list.append(Dense(units, activation=None, use_bias=layer_cnn > 1))
    if final_relu:
        layers_list.append(layers.Activation("relu"))

    return tf.keras.Sequential(layers_list)


def make_decoder(units, layer_cnn=1):
    layers_list = []
    for _ in range(layer_cnn):
        layers_list.append(Dense(units, activation=None))
        layers_list.append(layers.Activation("relu"))
        layers_list.append(layers.BatchNormalization(synchronized=True))
    return tf.keras.Sequential(layers_list)


def build_cnn_block_2nd(x, hls_mask_float, units, layer_cnn=2, drop=0.1, is_degrade=True):
    if is_degrade:
        x, hls_mask_float = MaskedAveragePooling2D.masked_avg_pool2d(x, hls_mask_float)

    for i in range(layer_cnn):
        x = layers.Conv2D(units, kernel_size=(1, 1), activation=None)(x)
        if i != layer_cnn - 1:
            x = layers.Activation("relu")(x)
            x = layers.BatchNormalization(synchronized=True)(x)

    return x, hls_mask_float


def transformer_block(queryx, x, padding_mask, units, n_head, drop=0.1, is_batch=True):
    attn_output, _ = layers.MultiHeadAttention(
        key_dim=units // n_head,
        num_heads=n_head
    )(
        query=queryx,
        value=x,
        key=x,
        return_attention_scores=True,
        attention_mask=padding_mask
    )

    if drop > 0:
        attn_output = Dropout(drop)(attn_output)

    out1 = queryx + attn_output
    if is_batch:
        out1 = LayerNormalization(epsilon=1e-6)(out1)

    ffn_output = point_wise_feed_forward_network(units, units * 4)(out1)
    if drop > 0:
        ffn_output = Dropout(drop)(ffn_output)

    out2 = out1 + ffn_output
    if is_batch:
        out2 = LayerNormalization(epsilon=1e-6)(out2)

    return out2


def build_shared_backbone(inputs,
                          layer_cnn=2, layern=3, decoder_layer_n=2, units=64,
                          n_times=48, n_hls_features=20, n_bands=6, n_window=50,
                          n_head=4, drop=0.1, n_out=3, n_flux_features=39,
                          n_index=None, is_batch=True, mask_value=-9999.0,
                          is_mete=False, is_sensor=True, is_xxy=False, is_doy=True,
                          is_hour=False, final_relu=False, is_decoder=False,
                          is_degrade=True):
    if n_index is None:
        n_index = [12, 14, 16, 20, 22, 26, 28, 30]

    hls_input = inputs[0]
    flux_input = inputs[1]
    ffp_input = inputs[2]

    hls_sub = hls_input[:, :, :, 0:n_bands]
    hls_mask = create_hls_mask(hls_sub, mask_value)
    hls_mask_float = tf.cast(hls_mask, tf.float32)
    # Use tf.where instead of multiplication so NaN/Inf invalid pixels do not
    # propagate as NaN * 0.
    hls_masked = tf.where(hls_mask, hls_sub, tf.zeros_like(hls_sub))

    hls_feat, hls_mask_float = build_cnn_block_2nd(
        hls_masked, hls_mask_float, units, layer_cnn, drop, is_degrade
    )

    H = tf.shape(hls_feat)[1]
    W = tf.shape(hls_feat)[2]
    T = n_times

    x = tf.repeat(hls_feat[:, :, :, tf.newaxis, :], repeats=T, axis=3)

    x_hour = flux_input[:, :, COL["Hour"]:COL["Hour"] + 1]
    x_doy = flux_input[:, :, COL["DoY"]:COL["DoY"] + 1]
    xmete = tf.gather(flux_input, indices=n_index, axis=-1)
    xsensor = flux_input[:, :, COL["Sensor"]]

    embedding_doy = make_embedding(units, layer_cnn=layer_cnn, final_relu=final_relu)
    embedding_hour = make_embedding(units, layer_cnn=layer_cnn, final_relu=final_relu)
    embedding_m = make_embedding(units, layer_cnn=layer_cnn, final_relu=final_relu)

    embedding_s = tf.keras.Sequential([Embedding(input_dim=2, output_dim=units)])
    for _ in range(layer_cnn - 1):
        embedding_s.add(layers.Activation("relu"))
        embedding_s.add(Dense(units, activation=None))
    if final_relu:
        embedding_s.add(layers.Activation("relu"))

    if is_doy:
        x = x + embedding_doy(x_doy)[:, tf.newaxis, tf.newaxis, :, :]
    if is_hour:
        x = x + embedding_hour(x_hour)[:, tf.newaxis, tf.newaxis, :, :]
    if is_sensor:
        x = x + embedding_s(xsensor)[:, tf.newaxis, tf.newaxis, :, :]
    if is_mete:
        x = x + embedding_m(xmete)[:, tf.newaxis, tf.newaxis, :, :]

    x_flat = tf.reshape(x, [-1, T, tf.shape(x)[-1]])
    padding_mask = x_flat[:, :, :1] > -99999

    z_flat = x_flat
    for _ in range(layern):
        z_flat = transformer_block(
            queryx=z_flat, x=z_flat, padding_mask=padding_mask,
            units=units, n_head=n_head, drop=drop, is_batch=is_batch
        )

    outputs = []
    for ii in range(n_out):
        if is_decoder:
            z1 = make_decoder(units, decoder_layer_n)(z_flat)
            if ii == 0:
                y = Dense(1, activation="linear")(z1)
            else:
                y = Dense(1, activation="exponential")(z1)
        else:
            if ii == 0:
                y = Dense(1, activation="linear")(z_flat)
            else:
                y = Dense(1, activation="exponential")(z_flat)
        outputs.append(y)

    pixel_pred = Concatenate(axis=-1)(outputs)

    pixel_pred_4d = tf.reshape(pixel_pred, [-1, H * W, T, n_out])
    pixel_pred_bt_hw_o = tf.transpose(pixel_pred_4d, perm=[0, 2, 1, 3])

    return {
        "hls_mask_float": hls_mask_float,
        "pixel_pred_bt_hw_o": pixel_pred_bt_hw_o,
        "flux_input": flux_input,
        "ffp_input": ffp_input,
        "H": H,
        "W": W,
        "T": T,
        "n_out": n_out,
    }


def get_transformer_cnn_at(layer_cnn=2, layern=3, decoder_layer_n=2, units=64, n_times=48,
                           n_hls_features=20, n_bands=6, n_window=50, n_head=4, drop=0.1,
                           n_out=3, n_flux_features=39,
                           n_index=[12, 14, 16, 20, 22, 26, 28, 30],
                           FFP=0, is_batch=True, mask_value=-9999.0,
                           is_mete=False, is_sensor=True, is_xxy=False, is_doy=True, is_hour=False,
                           final_relu=False, is_decoder=False, is_degrade=True, radius=0.5,
                           output_kind="weighted"):
    """
    Main-text priority:
      0 -> dynamic_ffp
      1 -> uniform_valid
      2 -> center3
      3 -> gaussian_static
      4 -> wind_distance

    Supplementary:
      5 -> center1
      6 -> center5
      7 -> gaussian_narrow
      8 -> gaussian_wide
      9 -> param_driven
    """
    mode = resolve_mode(FFP)
    output_kind = str(output_kind).lower()
    if output_kind not in ("weighted", "pixel_center", "pixel_map"):
        raise ValueError(
            f"Unsupported output_kind={output_kind!r}. "
            "Use 'weighted', 'pixel_center', or 'pixel_map'."
        )

    inputs = [
        Input(shape=(n_window, n_window, n_hls_features), name="HLS_input"),
        Input(shape=(n_times, n_flux_features), name="Flux_input"),
        Input(shape=(n_times, n_window, n_window), name="FFP_input")
    ]

    backbone = build_shared_backbone(
        inputs,
        layer_cnn=layer_cnn, layern=layern, decoder_layer_n=decoder_layer_n,
        units=units, n_times=n_times, n_hls_features=n_hls_features,
        n_bands=n_bands, n_window=n_window, n_head=n_head, drop=drop,
        n_out=n_out, n_flux_features=n_flux_features, n_index=n_index,
        is_batch=is_batch, mask_value=mask_value, is_mete=is_mete,
        is_sensor=is_sensor, is_xxy=is_xxy, is_doy=is_doy, is_hour=is_hour,
        final_relu=final_relu, is_decoder=is_decoder, is_degrade=is_degrade
    )

    attn_bt_1_hw, attn_bt_h_w = build_weight_head(
        mode=mode,
        backbone_dict=backbone,
        units=units,
        is_degrade=is_degrade
    )

    pixel_pred = backbone["pixel_pred_bt_hw_o"]
    weighted_output = tf.matmul(attn_bt_1_hw, pixel_pred)[:, :, 0, :]

    model = Model(inputs, weighted_output, name=f"FAT_{mode}")
    return model