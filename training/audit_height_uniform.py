"""Read-only preflight audit for the 48 half-hour height-uniform geometry.

This audit must be run against the real input directory before production training.
It verifies per-timestep effective-height availability, the MH-d identity, the
frozen site split, and the same center 4x4 record-retention rule used by the
four-model runners. Optional CSV output is one row per record-half-hour.
"""
import csv
from pathlib import Path
import numpy as np
from height_uniform import FETCH_FACTOR, PIXEL_M, EXPECTED_TIMESTEPS, square_support_numpy, snap_half_side_numpy, effective_timestep_height_numpy

RAW_ZM = 97          # numeric column 96 after SITE_ID is removed
RAW_ZM_SMOOTH = 103  # numeric column 102 after SITE_ID is removed
RAW_MH = 85
RAW_D = 95
HEIGHT_IDENTITY_ATOL_M = 1e-4  # 0.1 mm: accommodates float32 subtraction
HEIGHT_IDENTITY_RTOL = 1e-6


def audit(input_root, split_root, records_csv=None):
    input_root, split_root = Path(input_root), Path(split_root)
    flux = np.load(str(input_root/'merged_flux_era5_ffp_array_v11_remove_selected_sites.npy'), allow_pickle=True)
    hls = np.load(str(input_root/'HLS_matched_v11_remove_selected_sites.npy'), mmap_mode='r')
    n, t, f = flux.shape
    if t != EXPECTED_TIMESTEPS:
        raise ValueError(f'Expected {EXPECTED_TIMESTEPS} half-hours per record, got {t}')
    if hls.shape[0] != n:
        raise ValueError(f'HLS/flux record mismatch: {hls.shape[0]} vs {n}')

    z_raw = np.asarray(flux[..., RAW_ZM], dtype=np.float32)
    z_smooth = np.asarray(flux[..., RAW_ZM_SMOOTH], dtype=np.float32)
    raw_good = np.isfinite(z_raw) & (z_raw > 0)
    smooth_good = np.isfinite(z_smooth) & (z_smooth > 0)
    z_eff = np.where(raw_good, z_raw, z_smooth).astype(np.float32)
    good = raw_good | smooth_good
    use_smooth = ~raw_good & smooth_good

    mh = np.asarray(flux[..., RAW_MH], np.float32)
    d = np.asarray(flux[..., RAW_D], np.float32)
    identity_mask = raw_good & np.isfinite(mh) & np.isfinite(d) & (mh > 0) & (d >= 0)
    residual = np.abs(z_raw[identity_mask] - (mh[identity_mask] - d[identity_mask]))

    if not np.any(identity_mask):
        raise ValueError('Height identity cannot be verified: no valid zm/MH/d triples')
    expected_height = mh[identity_mask] - d[identity_mask]
    close = np.isclose(z_raw[identity_mask], expected_height,
                       atol=HEIGHT_IDENTITY_ATOL_M, rtol=HEIGHT_IDENTITY_RTOL)
    if not np.all(close):
        first = tuple(int(v) for v in np.argwhere(identity_mask)[np.flatnonzero(~close)[0]])
        raise ValueError(f'Height identity mismatch zm != MH-d: first={first}, max_residual_m={float(residual.max())}')

    if np.any(~good):
        bad = np.argwhere(~good)
        first = tuple(int(v) for v in bad[0])
        raise ValueError(f'No effective height at half-hour(s); no timestep dropping allowed. First bad={first}')

    half_side = FETCH_FACTOR * z_eff
    h, w = hls.shape[2:]
    hy = snap_half_side_numpy(half_side, h)
    hx = snap_half_side_numpy(half_side, w)
    clipped = (hy > h*PIXEL_M/2) | (hx > w*PIXEL_M/2)

    kept_all = np.zeros(n, dtype=bool)
    retained_support_ok = np.ones((n,t), dtype=bool)
    valid_support_pixels = np.zeros((n,t), dtype=np.int32)
    total_support_pixels = np.zeros((n,t), dtype=np.int32)

    for start in range(0, n, 128):
        stop = min(start+128, n)
        block = np.asarray(hls[start:stop, :6])
        valid = np.all(np.isfinite(block) & (block != -9999), axis=1)
        centre = block[:,0,23:27,23:27]
        keep = np.all(np.isfinite(centre) & (centre != -9999), axis=(1,2))
        kept_all[start:stop] = keep
        support, _, _ = square_support_numpy(z_eff[start:stop], h, w)
        vp = np.sum(valid[:,None,:,:] & support, axis=(2,3))
        tp = np.sum(support, axis=(2,3))
        valid_support_pixels[start:stop] = vp.astype(np.int32)
        total_support_pixels[start:stop] = tp.astype(np.int32)
        retained_support_ok[start:stop] = vp > 0

    retained_bad_height = int(np.sum(kept_all[:,None] & ~good))
    retained_no_support = int(np.sum(kept_all[:,None] & ~retained_support_ok))
    if retained_bad_height or retained_no_support:
        raise ValueError(f'Retained half-hour geometry invalid: missing_height={retained_bad_height}, no_valid_support={retained_no_support}')

    with (split_root/'site_to_fold.k5.seed42.csv').open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
        mapping = {r['site_id']:int(r['fold']) for r in rows}
        if len(mapping) != len(rows) or any(k not in range(5) for k in mapping.values()):
            raise ValueError('Invalid frozen site mapping: duplicate site or fold outside 0..4')
    sites=[]
    for row in flux[...,0]:
        usable=row[row!=-9999.0]
        sites.append(usable[0] if len(usable) else -9999.0)
    sites=np.asarray(sites,dtype=object)
    folds=np.asarray([mapping.get(s,-1) for s in sites])
    mismatch=int(np.sum((folds<0)&(sites!=-9999.0)))
    for k in range(5):
        for label,selected in [('test',folds==k),('train',(sites!=-9999.0)&(folds!=k))]:
            old=np.load(str(split_root/('index_%s_fold%d.npy'%(label,k))))
            mismatch += int(np.count_nonzero(old != selected)) if old.shape==selected.shape else len(sites)

    if mismatch:
        raise ValueError(f'Frozen site/fold mismatch: {mismatch} inconsistent entries')

    if records_csv is not None:
        with Path(records_csv).open('w',newline='',encoding='utf-8-sig') as fcsv:
            writer=csv.writer(fcsv)
            writer.writerow(['record_index','half_hour_index','site','year','doy','effective_height_m','height_source','requested_half_side_m','snapped_half_y_m','snapped_half_x_m','clipped_after_snap','original_filter_kept','support_pixels','valid_support_pixels'])
            for i in range(n):
                for j in range(t):
                    writer.writerow([i,j,sites[i],flux[i,0,4],flux[i,0,3],float(z_eff[i,j]),'zm_smooth' if use_smooth[i,j] else 'zm',float(half_side[i,j]),float(hy[i,j]),float(hx[i,j]),int(clipped[i,j]),int(kept_all[i]),int(total_support_pixels[i,j]),int(valid_support_pixels[i,j])])

    return {
        'records':int(n),
        'half_hours_per_record':int(t),
        'half_hour_timesteps':int(n*t),
        'half_hours_using_same_timestep_zm_smooth':int(use_smooth.sum()),
        'half_hours_without_effective_height':int((~good).sum()),
        'requested_half_side_m_quantiles':np.percentile(half_side,[0,25,50,75,100]).tolist(),
        'snapped_half_side_m_quantiles':np.percentile(hy,[0,25,50,75,100]).tolist(),
        'max_abs_snap_error_m':float(np.max(np.abs(hy-half_side))),
        'half_hours_clipped_after_snap':int(clipped.sum()),
        'retained_half_hours_clipped_after_snap':int(np.sum(clipped & kept_all[:,None])),
        'retained_records':int(kept_all.sum()),
        'retained_half_hours_without_support':retained_no_support,
        'height_identity_checked_values':int(identity_mask.sum()),
        'raw_height_values_without_checkable_MH_d':int(np.sum(raw_good & ~identity_mask)),
        'height_identity_atol_m':HEIGHT_IDENTITY_ATOL_M,
        'height_identity_rtol':HEIGHT_IDENTITY_RTOL,
        'max_abs_zm_minus_MH_minus_d':float(residual.max()) if len(residual) else None,
        'fold_mismatch_count':mismatch,
        'geometry':'square; half-side=100*effective height independently at each 30-min timestep; raw zm=MH-d; same-timestep zm_smooth fallback; nearest full-pixel edges; ties outward; original geometric centre; intersect original 50x50 patch; no spatial fallback',
    }


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Audit the 48 half-hour height-uniform footprint geometry on real inputs.")
    parser.add_argument("--input-root", required=True, type=Path, help="Directory containing the three shared input NPY files")
    parser.add_argument("--split-root", required=True, type=Path, help="Directory containing frozen k-fold index files and site_to_fold CSV")
    parser.add_argument("--csv", type=Path, default=None, help="Optional per-record/per-half-hour geometry CSV")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()
    report = audit(args.input_root, args.split_root, args.csv)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
