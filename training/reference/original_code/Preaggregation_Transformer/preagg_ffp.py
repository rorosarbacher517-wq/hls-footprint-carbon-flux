"""Input-side footprint aggregation shared by the pre-aggregated DL experiment.

The rules intentionally match the dynamic-FFP branch in the fair RF runner:
finite positive footprint weights, intersection with the six-band HLS validity
mask, per-time/per-band renormalization, and a centre-pixel fallback.
"""

import numpy as np


def _to_numpy(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _center_block_mask(height, width, size=1):
    mask = np.zeros((height, width), dtype=np.float32)
    top = height // 2 - size // 2
    left = width // 2 - size // 2
    mask[top:top + size, left:left + size] = 1.0
    return mask


def normalize_2d_weights(weights, valid_mask):
    """Normalize non-negative weights over valid pixels, with uniform fallback."""
    weights = np.asarray(weights, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    normalized = np.where(np.isfinite(weights) & (weights > 0) & valid_mask, weights, 0.0)
    total = normalized.sum(dtype=np.float64)
    if total <= 0:
        normalized = valid_mask.astype(np.float32)
        total = normalized.sum(dtype=np.float64)
    if total <= 0:
        return normalized
    return normalized / np.float32(total)


def center_valid_mask_memmap(hls_source, sample_indices, mask_value=-9999.0,
                             row_slice=slice(23, 27), col_slice=slice(23, 27),
                             chunk_size=512):
    """Evaluate the original central 4x4 filter without loading a fold."""
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    result = np.zeros(len(sample_indices), dtype=bool)
    for start in range(0, len(sample_indices), chunk_size):
        stop = min(start + chunk_size, len(sample_indices))
        block = np.asarray(
            hls_source[sample_indices[start:stop], 0, row_slice, col_slice],
            dtype=np.float32,
        )
        result[start:stop] = np.all(np.isfinite(block) & (block != mask_value), axis=(1, 2))
    return result


def band_mean_std_memmap(hls_source, train_indices, n_bands=6,
                         mask_value=-9999.0, chunk_size=128):
    """Two-pass, train-fold-only statistics for the shared HLS bands."""
    train_indices = np.asarray(train_indices, dtype=np.int64)
    sums = np.zeros(n_bands, dtype=np.float64)
    counts = np.zeros(n_bands, dtype=np.int64)

    for start in range(0, len(train_indices), chunk_size):
        idx = train_indices[start:start + chunk_size]
        block = np.asarray(hls_source[idx, :n_bands], dtype=np.float32)
        valid = np.isfinite(block) & (block != mask_value)
        sums += np.where(valid, block, 0.0).sum(axis=(0, 2, 3), dtype=np.float64)
        counts += valid.sum(axis=(0, 2, 3), dtype=np.int64)

    if np.any(counts == 0):
        raise ValueError(f"No valid training pixels for HLS bands {np.flatnonzero(counts == 0).tolist()}")
    means = sums / counts
    squared = np.zeros(n_bands, dtype=np.float64)

    for start in range(0, len(train_indices), chunk_size):
        idx = train_indices[start:start + chunk_size]
        block = np.asarray(hls_source[idx, :n_bands], dtype=np.float32)
        valid = np.isfinite(block) & (block != mask_value)
        delta = block.astype(np.float64) - means[np.newaxis, :, np.newaxis, np.newaxis]
        squared += np.where(valid, delta * delta, 0.0).sum(axis=(0, 2, 3), dtype=np.float64)

    stds = np.sqrt(squared / counts)
    stds = np.where(stds > 0, stds, 1.0)
    return means.astype(np.float32), stds.astype(np.float32), counts


def normalize_preaggregated_hls(values, means, stds, mask_value=-9999.0):
    """Apply train-fold HLS statistics after linear footprint aggregation."""
    values = np.asarray(values, dtype=np.float32)
    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)
    valid = np.isfinite(values) & (values != mask_value)
    output = np.full(values.shape, mask_value, dtype=np.float32)
    normalized = (values - means[np.newaxis, np.newaxis, :]) / stds[np.newaxis, np.newaxis, :]
    output[valid] = normalized[valid]
    return output


def aggregate_dynamic_ffp_hls(hls, ffp, n_bands=6, mask_value=-9999.0,
                              output_mask_value=-9999.0, ffp_indices=None,
                              hls_indices=None):
    """Aggregate a 50x50 HLS patch to one feature vector per footprint time.

    Parameters
    ----------
    hls : array-like, shape (N, H, W, C)
        Fold-normalized HLS data. Only the first ``n_bands`` are used.
    ffp : array-like, shape (N, T, H, W)
        Dynamic footprint maps.

    Returns
    -------
    np.ndarray, shape (N, T, n_bands)
        The 1x1-equivalent, time-varying HLS input.
    """
    hls = _to_numpy(hls)
    # Preserve np.memmap so the 13.5 GiB FFP array is not copied into RAM.
    ffp = _to_numpy(ffp)

    if hls.ndim != 4 or ffp.ndim != 4:
        raise ValueError(f"Expected HLS [N,H,W,C] and FFP [N,T,H,W], got {hls.shape} and {ffp.shape}")
    if hls_indices is None:
        hls_indices = np.arange(hls.shape[0], dtype=np.int64)
    else:
        hls_indices = np.asarray(hls_indices, dtype=np.int64)
    n_selected = len(hls_indices)
    if ffp_indices is None:
        ffp_indices = hls_indices.copy()
    else:
        ffp_indices = np.asarray(ffp_indices, dtype=np.int64)
    if len(ffp_indices) != n_selected:
        raise ValueError(
            f"ffp_indices length {len(ffp_indices)} does not match selected HLS samples {n_selected}"
        )
    if np.any(hls_indices < 0) or np.any(hls_indices >= hls.shape[0]):
        raise IndexError("hls_indices contains an index outside the HLS source array")
    if np.any(ffp_indices < 0) or np.any(ffp_indices >= ffp.shape[0]):
        raise IndexError("ffp_indices contains an index outside the FFP source array")
    is_nchw = hls.shape[2:4] == ffp.shape[2:4] and hls.shape[1] >= n_bands
    is_nhwc = hls.shape[1:3] == ffp.shape[2:4] and hls.shape[3] >= n_bands
    if not (is_nchw or is_nhwc):
        raise ValueError(f"HLS/FFP dimensions are not aligned: {hls.shape} versus {ffp.shape}")

    n_samples = n_selected
    height, width = ffp.shape[2], ffp.shape[3]
    n_times = ffp.shape[1]
    output = np.full((n_samples, n_times, n_bands), output_mask_value, dtype=np.float32)

    # This matches Model_build_fairfix.create_hls_mask: all selected bands must
    # be finite and non-fill for a pixel to contribute.
    center1 = _center_block_mask(height, width, size=1)

    for sample_idx in range(n_samples):
        hls_source_idx = int(hls_indices[sample_idx])
        source_idx = int(ffp_indices[sample_idx])
        # Source HLS is [N,C,H,W]; already-prepared input is [N,H,W,C].
        if is_nchw:
            hls_sample = np.transpose(
                np.asarray(hls[hls_source_idx, :n_bands], dtype=np.float32), (1, 2, 0)
            )
        else:
            hls_sample = np.asarray(hls[hls_source_idx, ..., :n_bands], dtype=np.float32)
        valid_hw = np.all(
            np.isfinite(hls_sample) & (hls_sample != mask_value), axis=-1
        )
        footprints = np.asarray(ffp[source_idx], dtype=np.float32)
        joint_valid = (
            np.isfinite(footprints)
            & (footprints > 0)
            & (footprints != mask_value)
            & valid_hw[np.newaxis, :, :]
        )
        weights = np.where(joint_valid, footprints, 0.0).astype(np.float32, copy=False)
        weight_sums = weights.sum(axis=(1, 2), dtype=np.float64)

        # Identical fallback to the RF dynamic mode. If the centre is invalid,
        # normalize_2d_weights falls back again to all valid HLS pixels.
        fallback = normalize_2d_weights(center1, valid_hw)
        no_weight = weight_sums <= 0
        if np.any(no_weight):
            weights[no_weight] = fallback
            weight_sums[no_weight] = weights[no_weight].sum(axis=(1, 2), dtype=np.float64)

        valid_times = weight_sums > 0
        if not np.any(valid_times):
            continue
        weights[valid_times] /= weight_sums[valid_times, np.newaxis, np.newaxis].astype(np.float32)
        safe_hls = np.where(valid_hw[..., np.newaxis], hls_sample, 0.0)
        output[sample_idx, valid_times] = np.einsum(
            "thw,hwc->tc", weights[valid_times], safe_hls,
            optimize=True, dtype=np.float32,
        )

    return output
