"""Half-hourly height-based uniform footprint support for the controlled ablation.

For every 30-minute timestep independently:
1. effective height z_eff is taken from raw ``zm`` (numeric column 96);
2. if raw ``zm`` is invalid/non-positive at that timestep, the same-timestep
   ``zm_smooth`` (numeric column 102) is used;
3. half-side = 100 * z_eff;
4. the half-side is snapped to the nearest complete 30 m pixel boundary,
   with exact half-way ties rounded outward;
5. the square is intersected with the existing 50x50 HLS patch and with the
   six-band HLS validity mask;
6. weights are exactly uniform over the remaining support and sum to one.

The result is therefore a true [N, T, H, W] half-hourly weight field. No
record-level median height is used, and the same spatial weight map is never
silently copied across all 48 timesteps.
"""
import warnings
import numpy as np

PIXEL_M = 30.0
FETCH_FACTOR = 100.0
SNAP_TOL_PIXELS = 1e-6
ZM_INDEX = 96
ZM_SMOOTH_INDEX = 102
EXPECTED_TIMESTEPS = 48


def effective_timestep_height_numpy(flux_numeric):
    """Return [N,T] effective heights using same-timestep fallback only."""
    flux = np.asarray(flux_numeric)
    if flux.ndim != 3:
        raise ValueError(f"flux_numeric must be [N,T,F], got {flux.shape}")
    if flux.shape[1] != EXPECTED_TIMESTEPS:
        raise ValueError(
            f"Half-hourly uniform geometry requires T={EXPECTED_TIMESTEPS}, got T={flux.shape[1]}"
        )

    z = np.asarray(flux[..., ZM_INDEX], dtype=np.float32)
    smooth = np.asarray(flux[..., ZM_SMOOTH_INDEX], dtype=np.float32)
    raw_good = np.isfinite(z) & (z > 0.0)
    smooth_good = np.isfinite(smooth) & (smooth > 0.0)
    use_smooth = ~raw_good
    z_eff = np.where(raw_good, z, smooth)
    good = raw_good | smooth_good
    z_eff = np.where(good, z_eff, np.nan)
    return z_eff.astype(np.float32), use_smooth, good


# Backward-compatible name; semantics are now explicitly timestep-level.
def effective_record_height_numpy(flux_numeric):
    """Backward-compatible alias returning ([N,T] heights, use_smooth [N,T])."""
    height, use_smooth, _ = effective_timestep_height_numpy(flux_numeric)
    return height, use_smooth


def snap_half_side_numpy(length_m, axis_size):
    """Nearest symmetric grid edge, ties outward; retain at least central cells.

    Even grids (50x50): grid-center is between the two central pixels and the
    minimum nonempty half-side is 30 m. Odd grids use half-pixel offsets.
    """
    length = np.asarray(length_m, dtype=np.float32)
    offset = np.float32(0.5 if axis_size % 2 else 0.0)
    steps = np.floor(length / PIXEL_M - offset + np.float32(0.5) + SNAP_TOL_PIXELS)
    steps = np.maximum(steps, 0 if axis_size % 2 else 1)
    return (steps + offset) * PIXEL_M


def square_support_numpy(height_m, height, width):
    """Build [N,T,H,W] half-hourly square support from [N,T] heights."""
    height_m = np.asarray(height_m, dtype=np.float32)
    if height_m.ndim != 2:
        raise ValueError(f"height_m must be [N,T], got {height_m.shape}")
    length = FETCH_FACTOR * height_m
    hy = snap_half_side_numpy(length, height)
    hx = snap_half_side_numpy(length, width)

    dy = np.abs(
        (np.arange(height, dtype=np.float32) - (height - 1) / 2.0) * PIXEL_M
    )[None, None, :, None]
    dx = np.abs(
        (np.arange(width, dtype=np.float32) - (width - 1) / 2.0) * PIXEL_M
    )[None, None, None, :]

    support = (dy < hy[..., None, None]) & (dx < hx[..., None, None])
    return support, hy, hx


def height_uniform_weights_numpy(valid_mask, flux_numeric):
    """Return exact [N,T,H,W] half-hourly height-uniform weights."""
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 3:
        raise ValueError(f"valid_mask must be [N,H,W], got {valid.shape}")
    if valid.shape[1] != 50 or valid.shape[2] != 50:
        raise ValueError(f"Expected 50x50 HLS patch, got {valid.shape[1:]}")

    height_m, use_smooth, good_height = effective_timestep_height_numpy(flux_numeric)
    if np.any(~good_height):
        bad = np.argwhere(~good_height)
        first = tuple(int(v) for v in bad[0]) if len(bad) else None
        raise ValueError(
            "Grid-uniform: missing positive finite effective height at one or more "
            f"half-hours; no sample/timestep dropping is allowed. First bad (N,T)={first}"
        )

    n, h, w = valid.shape
    if height_m.shape[0] != n:
        raise ValueError(f"N mismatch: valid_mask={n}, flux={height_m.shape[0]}")

    support, hy, hx = square_support_numpy(height_m, h, w)
    weights = support & valid[:, None, :, :]
    total = weights.sum(axis=(2, 3), keepdims=True)
    if np.any(total == 0):
        bad = np.argwhere(total[..., 0, 0] == 0)
        first = tuple(int(v) for v in bad[0]) if len(bad) else None
        raise ValueError(
            "Grid-uniform: no valid HLS pixel inside one or more snapped half-hourly "
            f"squares; no whole-patch fallback or timestep dropping is allowed. First bad={first}"
        )

    clipped = (hy > h * PIXEL_M / 2.0) | (hx > w * PIXEL_M / 2.0)
    if np.any(clipped):
        warnings.warn(
            "Grid-uniform: %d/%d half-hour windows extend beyond the original patch; "
            "support is square INTERSECT patch."
            % (int(clipped.sum()), int(clipped.size)),
            RuntimeWarning,
        )

    out = weights.astype(np.float32) / total.astype(np.float32)
    assert np.all(np.isfinite(out))
    return out


def aggregate_height_uniform_hls(hls, valid_mask, flux_numeric, n_times=None):
    """Aggregate HLS to [N,T,C] using timestep-specific uniform weights."""
    hls = np.asarray(hls, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    flux = np.asarray(flux_numeric)
    if hls.ndim != 4 or valid.ndim != 3 or flux.ndim != 3:
        raise ValueError(f"Expected HLS [N,H,W,C], valid [N,H,W], flux [N,T,F]; got {hls.shape}, {valid.shape}, {flux.shape}")
    if n_times is not None and int(n_times) != flux.shape[1]:
        raise ValueError(f"n_times={n_times} disagrees with flux T={flux.shape[1]}")

    if hls.shape[:3] != valid.shape:
        raise ValueError(f'HLS/valid shape mismatch: {hls.shape[:3]} vs {valid.shape}')
    weights = height_uniform_weights_numpy(valid, flux)
    safe = np.where(valid[..., None], hls, 0.0)
    pooled = np.einsum('nthw,nhwc->ntc', weights, safe, optimize=True, dtype=np.float32)
    return pooled.astype(np.float32)


def make_tensorflow_layer():
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package='HeightUniformGridStrict')
    class HeightUniformAttention(tf.keras.layers.Layer):
        """Differentiable layer producing timestep-specific [B,T,H,W] weights."""

        def call(self, inputs):
            mask, flux = inputs
            mask = tf.cast(mask[..., 0], tf.float32)  # [B,H,W]
            z = tf.cast(flux[..., ZM_INDEX], tf.float32)  # [B,T]
            smooth = tf.cast(flux[..., ZM_SMOOTH_INDEX], tf.float32)

            raw_good = tf.math.is_finite(z) & (z > 0.0)
            smooth_good = tf.math.is_finite(smooth) & (smooth > 0.0)
            z_eff = tf.where(raw_good, z, smooth)
            good_height = raw_good | smooth_good

            count_bad = tf.reduce_sum(tf.cast(~good_height, tf.int32))
            height_check = tf.debugging.assert_equal(
                count_bad, 0,
                message='Grid-uniform: every 30-min timestep must have a positive finite effective height.'
            )
            with tf.control_dependencies([height_check]):
                length = FETCH_FACTOR * z_eff

            h = tf.shape(mask)[1]
            w = tf.shape(mask)[2]

            def snapped_half(length_bt, size):
                odd = tf.math.floormod(size, 2)
                offset = tf.cast(odd, tf.float32) * 0.5
                steps = tf.floor(
                    length_bt / PIXEL_M - offset + 0.5 + SNAP_TOL_PIXELS
                )
                minimum = tf.cast(1 - odd, tf.float32)
                return (tf.maximum(steps, minimum) + offset) * PIXEL_M

            hy = snapped_half(length, h)  # [B,T]
            hx = snapped_half(length, w)  # [B,T]

            dy = tf.abs(
                (tf.cast(tf.range(h), tf.float32) - (tf.cast(h, tf.float32) - 1.0) / 2.0)
                * PIXEL_M
            )[tf.newaxis, tf.newaxis, :, tf.newaxis]
            dx = tf.abs(
                (tf.cast(tf.range(w), tf.float32) - (tf.cast(w, tf.float32) - 1.0) / 2.0)
                * PIXEL_M
            )[tf.newaxis, tf.newaxis, tf.newaxis, :]

            support = (
                (dy < hy[:, :, tf.newaxis, tf.newaxis]) &
                (dx < hx[:, :, tf.newaxis, tf.newaxis])
            )
            weights = tf.cast(support, tf.float32) * mask[:, tf.newaxis, :, :]
            total = tf.reduce_sum(weights, axis=(2, 3), keepdims=True)
            support_check = tf.debugging.assert_positive(
                total,
                message='Grid-uniform: no valid HLS pixel exists inside a snapped half-hour square.'
            )
            with tf.control_dependencies([support_check]):
                weights = tf.math.divide_no_nan(weights, total)
            return weights

    return HeightUniformAttention
