#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How different are the two aggregation rules?  Half-hourly distance between the FAT footprint weights and the
uniform-window weights of the (new) uniform-weighted Transformer.

For every held-out record (one HLS-matched day) and half-hour t, on the 50x50 HLS patch:
  FAT weights      f_t : dynamic footprint (FFP) restricted to valid six-band HLS pixels and renormalised, with the
                         model's fallback (centre pixel, then all valid pixels) when no footprint pixel is valid;
  Uniform weights  u_t : 100 x z_eff square (z_eff = zm, same-half-hour zm_smooth fallback), snapped to 30 m pixel
                         edges, intersected with the valid pixels, equal weights  (common/height_uniform.py = the rule
                         the uniform-weighted Transformer was trained with).
Measures:
  tv        = 0.5 * sum |f_t - u_t|            total-variation distance in [0, 1]   (0: identical weights)
  outside   = sum of f_t over pixels with u_t = 0   share of FAT's weight lying outside the uniform window
  neff_f / neff_u = 1 / sum w^2   effective number of pixels of each rule (how concentrated the weights are)
  shift_m   = distance (m) between the weighted centroids of f_t and u_t   (footprint displaced from the window centre)
All are independent of the vegetation and of the model predictions: they describe the aggregation rules only.

Inputs : model input arrays (HLS, FFP, merged flux) from the input root + frozen record order of the held-out folds
Outputs: ../../data/weight_difference_halfhour.npz  (keys: SITE_ID, Year, DoY, HLS_Hour, tv, outside, neff_f, neff_u, shift_m; arrays [N,48])
Run    : python weight_difference.py [--input-root DIR] [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "common"))
from height_uniform import height_uniform_weights_numpy  # noqa: E402

V6 = Path('./training')
DEFAULT_INPUT = Path('./data/input_data')
OUT = PROJECT / "data" / "weight_difference_halfhour.npz"
CHUNK = 64


def fat_weights(ffp, valid):
    """ffp [n,48,50,50] raw footprint, valid [n,50,50] -> weights [n,48,50,50] (sum 1 over the patch)."""
    f = np.asarray(ffp, dtype=np.float32)
    ok = np.isfinite(f) & (f > 0) & (f != -9999.0)
    w = np.where(ok & valid[:, None], f, 0.0).astype(np.float64)
    empty = ~(w > 0).any(axis=(2, 3), keepdims=True)                      # no valid footprint pixel
    centre = np.zeros((1, 1, 50, 50))
    centre[0, 0, 25, 25] = 1.0
    fb = centre * valid[:, None]                                          # centre-pixel fallback (if valid)
    fb_empty = ~(fb > 0).any(axis=(2, 3), keepdims=True)
    fb = np.where(fb_empty, valid[:, None].astype(np.float64), fb)        # else all valid pixels
    w = np.where(empty, fb, w)
    return w / w.sum(axis=(2, 3), keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--limit", type=int, default=0, help="only the first N records (timing / test)")
    a = ap.parse_args()

    sys.path.insert(0, str(V6))
    from numpy_pickle_compat import install_numpy_pickle_compat
    install_numpy_pickle_compat()

    idx = np.concatenate([np.load(V6 / "reference" / f"fold{k}_effective_indices.npz")["test"] for k in range(5)])
    site = np.concatenate([np.load(PROJECT / "data" / "four_model_predictions" / "FAT" / f"fold{k}" / "site.npy", allow_pickle=True)[:, 0, :]
                           for k in range(5)])
    n = len(idx) if not a.limit else a.limit
    print("records:", len(idx), "using", n)
    hls = np.load(a.input_root / "HLS_matched_v11_remove_selected_sites.npy", mmap_mode="r")
    ffp = np.load(a.input_root / "FFP_hls_aligned_v11_50x50_timefilled_remove_selected_sites.npy", mmap_mode="r")
    flux = np.load(a.input_root / "merged_flux_era5_ffp_array_v11_remove_selected_sites.npy", allow_pickle=True)

    # identity check: the frozen indices must point at the same site / year / day as the saved predictions
    chk = np.asarray(flux[idx[:n], 0, 0]).astype(str)
    if not np.array_equal(chk, site[:n, 0].astype(str)):
        raise RuntimeError("record order does not match the saved predictions")

    tv = np.full((n, 48), np.nan, np.float32)
    outside = np.full((n, 48), np.nan, np.float32)
    neff_f = np.full((n, 48), np.nan, np.float32)
    neff_u = np.full((n, 48), np.nan, np.float32)
    shift_m = np.full((n, 48), np.nan, np.float32)
    yy, xx = np.meshgrid(np.arange(50, dtype=np.float64), np.arange(50, dtype=np.float64), indexing="ij")
    t0 = time.time()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        rows = idx[s:e]
        order = np.argsort(rows)                                          # sorted reads from the memmaps
        r = rows[order]
        h6 = np.asarray(hls[r, :6], dtype=np.float32)                     # [m,6,50,50]
        valid = np.all(np.isfinite(h6) & (h6 != -9999.0), axis=1)         # [m,50,50]
        f = np.asarray(ffp[r])                                            # [m,48,50,50]
        fl = np.asarray(flux[r][:, :, 1:], dtype=np.float32)              # numeric columns
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        valid, f, fl = valid[inv], f[inv], fl[inv]
        good = valid.any(axis=(1, 2))
        if not good.all():
            raise RuntimeError("record without any valid HLS pixel")
        w_f = fat_weights(f, valid)
        w_u = height_uniform_weights_numpy(valid, fl).astype(np.float64)
        tv[s:e] = (0.5 * np.abs(w_f - w_u).sum(axis=(2, 3))).astype(np.float32)
        outside[s:e] = np.where(w_u == 0, w_f, 0.0).sum(axis=(2, 3)).astype(np.float32)
        neff_f[s:e] = (1.0 / (w_f ** 2).sum(axis=(2, 3))).astype(np.float32)
        neff_u[s:e] = (1.0 / (w_u ** 2).sum(axis=(2, 3))).astype(np.float32)
        cf = np.stack([(w_f * yy).sum(axis=(2, 3)), (w_f * xx).sum(axis=(2, 3))])
        cu = np.stack([(w_u * yy).sum(axis=(2, 3)), (w_u * xx).sum(axis=(2, 3))])
        shift_m[s:e] = (30.0 * np.sqrt(((cf - cu) ** 2).sum(axis=0))).astype(np.float32)
        if (s // CHUNK) % 20 == 0:
            print(f"  {e}/{n}  {time.time() - t0:.0f}s", flush=True)

    if not a.limit:
        np.savez_compressed(OUT, SITE_ID=site[:, 0].astype(str), Year=site[:, 3].astype(float).astype(int),
                            DoY=site[:, 4].astype(float).astype(int), HLS_Hour=site[:, 5].astype(float).astype(int),
                            tv=tv, outside=outside, neff_f=neff_f, neff_u=neff_u, shift_m=shift_m)
        print("saved", OUT)
    print("N_eff FAT median %.1f | uniform median %.1f | shift_m median %.0f p10 %.0f p90 %.0f" % (
        np.nanmedian(neff_f), np.nanmedian(neff_u), np.nanmedian(shift_m), np.nanpercentile(shift_m, 10), np.nanpercentile(shift_m, 90)))
    print("TV: mean %.3f  median %.3f  p10 %.3f  p90 %.3f | outside: mean %.3f" % (
        np.nanmean(tv), np.nanmedian(tv), np.nanpercentile(tv, 10), np.nanpercentile(tv, 90), np.nanmean(outside)))


if __name__ == "__main__":
    main()
