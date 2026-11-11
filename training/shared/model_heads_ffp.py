import math
import tensorflow as tf
import MaskedAveragePooling2D

FFP_HEADS_VERSION = "v4_strict_corrected_dynamic_normalization"

WIND_DIRECTION_mean = 202.68707
WIND_DIRECTION_std = 95.78515

COL = {
    "DoY": 3,
    "Sensor": 5,
    "Hour": 6,
    "blh": 45,
    "zust": 46,
    "RH": 63,
    "VPD_kPa": 64,
    "WS_ms": 65,
    "WD_deg": 66,
    "L_m": 67,
    "t2m_C": 68,
    "d2m_C": 69,
    "stl1_C": 70,
    "stl2_C": 71,
    "stl3_C": 72,
    "stl4_C": 73,
    "sp_kPa": 74,
    "swvl1_pct": 75,
    "swvl2_pct": 76,
    "swvl3_pct": 77,
    "swvl4_pct": 78,
    "P_mm": 79,
    "SSRD_Wm2": 80,
    "SSHF_Wm2": 81,
    "V_SIGMA": 91,
    "h_aero": 93,
    "d": 94,
    "z0": 95,
    "zm": 96,
    "lambda_rs": 97,
    "Hour_sin": 104,
    "Hour_cos": 105,
    "WD_sin": 106,
    "WD_cos": 107,
    "sigma_v_rf": 108,
}

# main-text priority first, supplementary later
MODE_MAP = {
    0: "dynamic_ffp",      # main
    1: "uniform_valid",    # main
    2: "center3",          # main
    3: "gaussian_static",  # main
    4: "wind_distance",    # main
    5: "center1",          # supp
    6: "center5",          # supp
    7: "gaussian_narrow",  # supp
    8: "gaussian_wide",    # supp
    9: "param_driven",     # supp
}

VALID_MODES = set(MODE_MAP.values())

FFP_PARAM_IDX = [
    COL["blh"], COL["zust"], COL["WS_ms"], COL["WD_deg"],
    COL["L_m"], COL["z0"], COL["zm"], COL["sigma_v_rf"]
]


def resolve_mode(mode_or_code):
    if isinstance(mode_or_code, str):
        if mode_or_code not in VALID_MODES:
            raise ValueError(
                f"Unsupported mode string: {mode_or_code}. "
                f"Valid modes: {sorted(VALID_MODES)}"
            )
        return mode_or_code

    if mode_or_code not in MODE_MAP:
        raise ValueError(
            f"Unsupported FFP code: {mode_or_code}. "
            f"Supported numeric codes: {sorted(MODE_MAP.keys())}"
        )
    return MODE_MAP[mode_or_code]


def build_relative_spatial_features(H, W):
    cy = (tf.cast(H, tf.float32) - 1.0) / 2.0
    cx = (tf.cast(W, tf.float32) - 1.0) / 2.0

    y = tf.range(H, dtype=tf.float32)
    x = tf.range(W, dtype=tf.float32)
    yy, xx = tf.meshgrid(y, x, indexing="ij")

    dy = yy - cy
    dx = xx - cx

    r = tf.sqrt(dx * dx + dy * dy)
    r_max = tf.reduce_max(r)
    r_norm = tf.math.divide_no_nan(r, r_max)

    az = (tf.atan2(dy, dx) + math.pi / 2.0) * 180.0 / math.pi
    az = tf.where(az < 0.0, az + 360.0, az)

    cos_az = tf.cos(az * math.pi / 180.0)
    sin_az = tf.sin(az * math.pi / 180.0)

    feat = tf.stack([
        1.0 - r_norm,
        (az - WIND_DIRECTION_mean) / WIND_DIRECTION_std,
        cos_az,
        sin_az
    ], axis=-1)

    return {
        "dx": dx,
        "dy": dy,
        "r": r,
        "r_norm": r_norm,
        "az_deg": az,
        "cos_az": cos_az,
        "sin_az": sin_az,
        "feat": feat,
    }


def normalize_attention_map(weight_map, eps=None):
    """Nonnegative, finite weights normalized without epsilon shrinkage.

    Scale first to avoid overflow when summing very large positive weights.
    eps is retained only as a compatibility argument; it never changes mass.
    """
    weight_map=tf.where(tf.math.is_finite(weight_map),weight_map,tf.zeros_like(weight_map))
    weight_map=tf.maximum(weight_map,0.0)
    maximum=tf.reduce_max(weight_map,axis=(2,3),keepdims=True)
    scaled=tf.math.divide_no_nan(weight_map,maximum)
    denom=tf.reduce_sum(scaled,axis=(2,3),keepdims=True)
    return tf.math.divide_no_nan(scaled,denom)


def valid_mask_as_bt_hw(hls_mask_float, T):
    mask_hw = hls_mask_float[..., 0]
    mask_hw = mask_hw[:, tf.newaxis, :, :]
    mask_hw = tf.tile(mask_hw, [1, T, 1, 1])
    return mask_hw


def make_center_k_pattern(H, W, k, dtype):
    H = tf.cast(H, tf.int32)
    W = tf.cast(W, tf.int32)

    base = tf.zeros([H, W], dtype=dtype)

    ch = H // 2
    cw = W // 2
    half = k // 2

    top = ch - half
    left = cw - half

    rr = tf.range(top, top + k)
    cc = tf.range(left, left + k)
    rr, cc = tf.meshgrid(rr, cc, indexing="ij")

    idx = tf.stack([tf.reshape(rr, [-1]), tf.reshape(cc, [-1])], axis=1)
    updates = tf.ones([k * k], dtype=dtype)

    base = tf.tensor_scatter_nd_update(base, idx, updates)
    return base[tf.newaxis, tf.newaxis, :, :]


def masked_uniform_attention(hls_mask_float, T):
    mask_hw = valid_mask_as_bt_hw(hls_mask_float, T)
    return normalize_attention_map(mask_hw)


def masked_center_attention(hls_mask_float, T, center_k=3):
    B = tf.shape(hls_mask_float)[0]
    H = tf.shape(hls_mask_float)[1]
    W = tf.shape(hls_mask_float)[2]

    center_pattern = make_center_k_pattern(H, W, center_k, tf.float32)
    center_pattern = tf.tile(center_pattern, [B, T, 1, 1])

    mask_hw = valid_mask_as_bt_hw(hls_mask_float, T)
    masked_value = center_pattern * mask_hw

    sum_hw = tf.reduce_sum(masked_value, axis=(2, 3), keepdims=True)
    no_valid = tf.equal(sum_hw, 0.0)

    masked_value = tf.where(no_valid, mask_hw, masked_value)
    return normalize_attention_map(masked_value)


def masked_gaussian_attention(hls_mask_float, T, r_norm, sigma=0.35):
    B = tf.shape(hls_mask_float)[0]

    g = tf.exp(-tf.square(r_norm) / (2.0 * sigma * sigma))
    g = g[tf.newaxis, tf.newaxis, :, :]
    g = tf.tile(g, [B, T, 1, 1])

    mask_hw = valid_mask_as_bt_hw(hls_mask_float, T)
    masked_value = g * mask_hw

    sum_hw = tf.reduce_sum(masked_value, axis=(2, 3), keepdims=True)
    no_valid = tf.equal(sum_hw, 0.0)

    masked_value = tf.where(no_valid, mask_hw, masked_value)
    return normalize_attention_map(masked_value)


def masked_attention_with_fallback(ffp_value, hls_ffp_mask, fallback_mode="center1", fallback_valid_mask=None):
    ffp_value=tf.convert_to_tensor(ffp_value,dtype=tf.float32)
    hls_ffp_mask=tf.cast(hls_ffp_mask,tf.float32)
    masked_value=ffp_value*hls_ffp_mask
    no_valid=~tf.reduce_any(masked_value>0,axis=(2,3),keepdims=True)
    H=tf.shape(masked_value)[2];W=tf.shape(masked_value)[3]
    if fallback_mode=="center1":fallback=make_center_k_pattern(H,W,1,masked_value.dtype)
    elif fallback_mode=="center3":fallback=make_center_k_pattern(H,W,3,masked_value.dtype)
    elif fallback_mode=="uniform":fallback=tf.ones([1,1,H,W],dtype=masked_value.dtype)
    else:raise ValueError("Unsupported fallback_mode: "+str(fallback_mode))
    valid=tf.cast(fallback_valid_mask if fallback_valid_mask is not None else hls_ffp_mask,tf.float32)
    fallback=fallback*valid
    empty=~tf.reduce_any(fallback>0,axis=(2,3),keepdims=True)
    fallback=tf.where(empty,valid,fallback)
    chosen=tf.where(no_valid,fallback,masked_value)
    # No-support records are rejected at the shared data interface before model use.
    return normalize_attention_map(chosen)



def build_true_dynamic_ffp_attention(ffp_input, hls_mask_float, is_degrade=False,
                                     fallback_mode="center1"):
    ffp_value = ffp_input
    ffp_mask = tf.cast(
        tf.math.is_finite(ffp_input) & (ffp_input > 0.0) & (ffp_input != -9999.0),
        tf.float32
    )
    ffp_value = tf.where(ffp_mask > 0, ffp_input, tf.zeros_like(ffp_input))

    if is_degrade:
        ffp_value, ffp_mask = MaskedAveragePooling2D.masked_avg_pool2d2(
            ffp_input, ffp_mask,
            data_format="NCHW", pool_size=2, strides=2, padding="VALID"
        )

    hls_mask_expanded = tf.expand_dims(hls_mask_float[..., 0], axis=1)
    hls_ffp_mask = tf.logical_and(
        tf.cast(hls_mask_expanded, tf.bool),
        tf.cast(ffp_mask, tf.bool)
    )

    return masked_attention_with_fallback(
        ffp_value,
        hls_ffp_mask,
        fallback_mode=fallback_mode,
        fallback_valid_mask=hls_mask_expanded
    )


def build_wind_distance_attention(flux_input, hls_mask_float, spatial, H, W, T, units):
    x_ws = flux_input[:, :, COL["WS_ms"]:COL["WS_ms"] + 1]
    x_wd = flux_input[:, :, COL["WD_deg"]:COL["WD_deg"] + 1]

    angle = (x_wd * WIND_DIRECTION_std + WIND_DIRECTION_mean) * math.pi / 180.0

    q_feat = tf.concat([
        x_ws,
        x_wd,
        tf.cos(angle),
        tf.sin(angle)
    ], axis=-1)

    k_feat = spatial["feat"]

    q = tf.keras.layers.Dense(units, activation="relu")(q_feat)
    q = tf.keras.layers.Dense(units, activation=None)(q)

    k = tf.keras.layers.Dense(units, activation="relu")(k_feat)
    k = tf.keras.layers.Dense(units, activation=None)(k)

    k = tf.reshape(k, [1, 1, H * W, units])
    q = q[:, :, tf.newaxis, :]

    logits = tf.reduce_sum(q * k, axis=-1)

    valid_mask = hls_mask_float[..., 0]
    valid_mask = tf.reshape(valid_mask, [tf.shape(valid_mask)[0], 1, H * W])
    valid_mask = tf.tile(valid_mask, [1, T, 1])

    very_neg = tf.constant(-1e9, dtype=logits.dtype)
    logits = tf.where(valid_mask > 0, logits, very_neg)

    attn = tf.nn.softmax(logits, axis=-1)
    attn_bt_1_hw = attn[:, :, tf.newaxis, :]
    attn0 = tf.reshape(attn, [-1, T, H, W])

    return attn_bt_1_hw, attn0


def build_param_driven_attention(flux_input, hls_mask_float, spatial, H, W, T, units):
    q_feat = tf.gather(flux_input, indices=FFP_PARAM_IDX, axis=-1)

    wd_pos = FFP_PARAM_IDX.index(COL["WD_deg"])
    x_wd = q_feat[:, :, wd_pos:wd_pos + 1]
    angle = (x_wd * WIND_DIRECTION_std + WIND_DIRECTION_mean) * math.pi / 180.0

    q_feat = tf.concat([
        q_feat,
        tf.cos(angle),
        tf.sin(angle)
    ], axis=-1)

    k_feat = spatial["feat"]

    q = tf.keras.layers.Dense(units, activation="relu")(q_feat)
    q = tf.keras.layers.Dense(units, activation=None)(q)

    k = tf.keras.layers.Dense(units, activation="relu")(k_feat)
    k = tf.keras.layers.Dense(units, activation=None)(k)

    k = tf.reshape(k, [1, 1, H * W, units])
    q = q[:, :, tf.newaxis, :]

    logits = tf.reduce_sum(q * k, axis=-1)

    valid_mask = hls_mask_float[..., 0]
    valid_mask = tf.reshape(valid_mask, [tf.shape(valid_mask)[0], 1, H * W])
    valid_mask = tf.tile(valid_mask, [1, T, 1])

    very_neg = tf.constant(-1e9, dtype=logits.dtype)
    logits = tf.where(valid_mask > 0, logits, very_neg)

    attn = tf.nn.softmax(logits, axis=-1)
    attn_bt_1_hw = attn[:, :, tf.newaxis, :]
    attn0 = tf.reshape(attn, [-1, T, H, W])

    return attn_bt_1_hw, attn0


def build_weight_head(mode, backbone_dict, units=64, is_degrade=False):
    mode = resolve_mode(mode)

    hls_mask_float = backbone_dict["hls_mask_float"]
    flux_input = backbone_dict["flux_input"]
    ffp_input = backbone_dict["ffp_input"]
    H = backbone_dict["H"]
    W = backbone_dict["W"]
    T = backbone_dict["T"]

    spatial = build_relative_spatial_features(H, W)

    if mode == "dynamic_ffp":
        attn0 = build_true_dynamic_ffp_attention(
            ffp_input=ffp_input,
            hls_mask_float=hls_mask_float,
            is_degrade=is_degrade,
            fallback_mode="center1"
        )

    elif mode == "uniform_valid":
        attn0 = masked_uniform_attention(hls_mask_float, T)

    elif mode == "center3":
        attn0 = masked_center_attention(hls_mask_float, T, center_k=3)

    elif mode == "gaussian_static":
        attn0 = masked_gaussian_attention(hls_mask_float, T, spatial["r_norm"], sigma=0.35)

    elif mode == "wind_distance":
        return build_wind_distance_attention(
            flux_input=flux_input,
            hls_mask_float=hls_mask_float,
            spatial=spatial,
            H=H, W=W, T=T,
            units=units
        )

    elif mode == "center1":
        attn0 = masked_center_attention(hls_mask_float, T, center_k=1)

    elif mode == "center5":
        attn0 = masked_center_attention(hls_mask_float, T, center_k=5)

    elif mode == "gaussian_narrow":
        attn0 = masked_gaussian_attention(hls_mask_float, T, spatial["r_norm"], sigma=0.20)

    elif mode == "gaussian_wide":
        attn0 = masked_gaussian_attention(hls_mask_float, T, spatial["r_norm"], sigma=0.55)

    elif mode == "param_driven":
        return build_param_driven_attention(
            flux_input=flux_input,
            hls_mask_float=hls_mask_float,
            spatial=spatial,
            H=H, W=W, T=T,
            units=units
        )

    else:
        raise ValueError(f"Unsupported weighting mode: {mode}")

    attn_bt_1_hw = tf.reshape(attn0, [-1, T, 1, H * W])
    return attn_bt_1_hw, attn0