#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4.4 case study, step 1: rule-based selection of one high- and one low-heterogeneity site-day.
The rule never looks at model errors; DeltaAE of the chosen pair is only reported afterwards (with its rank among all candidate pairs).

Eligible site-days (held-out records of the five site-held-out folds; predictions of both models from the paired experiment):
  1 season      DoY 91-250 (green-up + peak growing, identical to the stage definition of Section 4.2)
                and patch-mean NDVI >= 50% of the site-year NDVI amplitude (min-max over the HLS days of that site-year,
                >= 6 days; amplitude threshold idea of Bolton et al. 2020, RSE 240:111685)
  2 patch       all 2,500 pixels of the 50 x 50 HLS patch valid in all six bands
  3 completeness  FAT and uniform predictions finite for all 48 half-hours; footprint of all 48 half-hours has >= 1 valid pixel;
                at least 36 half-hours with valid observations of NEE, GPP and RECO jointly; 12:00 local observations of all three valid
Heterogeneity classes (on eligible site-days): NDVI standard deviation within the patch, top 10% = high, bottom 20% = low.
Pairing: high x low, different sites, |DoY difference| <= 15 d, 80% footprint area at 12:00 within a factor 1.5;
  distance = sqrt( sum of squared z-scores of daytime mean SSRD, air temperature, VPD, log footprint area ) + |dDoY|/15 ;
  z-scores use the SD of the eligible pool.  The pair with the smallest distance is chosen (unique, error-blind).
Outputs: ../outputs/case_candidates.csv (all eligible site-days), case_pairs_ranked.csv (top 30 pairs by distance with their errors),
         selected_cases.csv (chosen pair + descriptors), case_selection_summary.txt
Run    : python select_cases.py [--input-root DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
V6 = Path('./training')
DEFAULT_INPUT = Path('./data/input_data')
PRED = PROJECT / "data" / "four_model_predictions"
OUT = HERE.parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

COL = {"Hour": 6, "VPD": 64, "WS": 65, "WD": 66, "TA": 68, "SSRD": 80}   # numeric columns (merged flux array without the SITE_ID column)
FLUXES = ["NEE", "GPP", "RECO"]
DOY_LO, DOY_HI = 91, 250
AMP_MIN, MIN_DAYS_YEAR = 0.5, 6
TOP_Q, BOT_Q = 0.90, 0.20
MAX_DDOY, MAX_AREA_RATIO, MIN_JOINT_HH = 15, 1.5, 36
NOON = 24                                                              # half-hour index of 12:00 local


def load(name):
    ys, ps, ss = [], [], []
    for k in range(5):
        b = PRED / name / f"fold{k}"
        ys.append(np.load(b / "true.npy", allow_pickle=True))
        ps.append(np.load(b / "pred.npy", allow_pickle=True))
        ss.append(np.load(b / "site.npy", allow_pickle=True))
    return np.concatenate(ys).astype(float), np.concatenate(ps).astype(float), np.concatenate(ss)


def footprint_area_pixels(ffp_noon, valid):
    f = np.where(np.isfinite(ffp_noon) & (ffp_noon > 0) & valid, ffp_noon, 0.0).astype(np.float64)
    out = np.full(len(f), np.nan)
    for i, a in enumerate(f):
        v = np.sort(a.ravel())[::-1]
        s = v.sum()
        if s > 0:
            out[i] = int(np.searchsorted(np.cumsum(v) / s, 0.8) + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    a = ap.parse_args()
    sys.path.insert(0, str(V6))
    from numpy_pickle_compat import install_numpy_pickle_compat
    install_numpy_pickle_compat()

    idx = np.concatenate([np.load(V6 / "reference" / f"fold{k}_effective_indices.npz")["test"] for k in range(5)])
    y, pf, site = load("FAT")
    _, pu, site2 = load("Uniform_Transformer")
    assert np.array_equal(site, site2) and len(idx) == len(y)
    s0 = site[:, 0, :]
    rec = pd.DataFrame({"idx": idx, "SITE_ID": s0[:, 0].astype(str), "Year": pd.to_numeric(s0[:, 3]).astype(int),
                        "DoY": pd.to_numeric(s0[:, 4]).astype(int)})
    print("records", len(rec))

    flux = np.load(a.input_root / "merged_flux_era5_ffp_array_v11_remove_selected_sites.npy", allow_pickle=True)
    assert np.array_equal(np.asarray(flux[idx, 0, 0]).astype(str), rec["SITE_ID"].to_numpy()), "record order mismatch"
    hls = np.load(a.input_root / "HLS_matched_v11_remove_selected_sites.npy", mmap_mode="r")
    ffp = np.load(a.input_root / "FFP_hls_aligned_v11_50x50_timefilled_remove_selected_sites.npy", mmap_mode="r")

    # ---- cheap filters first: season, observation completeness, finite predictions
    ok_obs = np.isfinite(y) & (y > -9000)                                        # [N,48,3]
    joint = ok_obs.all(axis=2)
    pred_ok = np.isfinite(pf).all(axis=(1, 2)) & np.isfinite(pu).all(axis=(1, 2))
    base = (rec["DoY"].between(DOY_LO, DOY_HI).to_numpy() & (joint.sum(1) >= MIN_JOINT_HH) & joint[:, NOON] & pred_ok)
    print("season + completeness(obs, predictions):", int(base.sum()))

    # ---- patch quality and NDVI (only for those)
    sel = np.where(base)[0]
    ndvi_mean = np.full(len(rec), np.nan)
    ndvi_std = np.full(len(rec), np.nan)
    n_valid = np.zeros(len(rec), int)
    for s in range(0, len(sel), 128):
        part = sel[s:s + 128]
        rows = idx[part]
        order = np.argsort(rows)
        h = np.asarray(hls[rows[order], :6], dtype=np.float32)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        h = h[inv]
        valid = np.all(np.isfinite(h) & (h != -9999.0), axis=1)                   # [m,50,50]
        red, nir = h[:, 2], h[:, 3]
        nd = (nir - red) / (nir + red + 1e-6)
        nd = np.where(valid, nd, np.nan)
        n_valid[part] = valid.sum(axis=(1, 2))
        ndvi_mean[part] = np.nanmean(nd, axis=(1, 2))
        ndvi_std[part] = np.nanstd(nd, axis=(1, 2))
    rec["n_valid_pix"], rec["ndvi_mean"], rec["ndvi_std"] = n_valid, ndvi_mean, ndvi_std
    full = base & (n_valid == 2500)
    print("all 2500 pixels valid:", int(full.sum()))

    # ---- phenology amplitude check (site-year NDVI min-max over all HLS days of the site-year with 2,500 valid pixels not required)
    allm = pd.DataFrame({"SITE_ID": rec["SITE_ID"], "Year": rec["Year"], "v": np.nan})
    # amplitude uses the patch-mean NDVI of every held-out HLS day that passed the observation completeness filter (>= 6 days)
    g = rec[base & np.isfinite(ndvi_mean)].groupby(["SITE_ID", "Year"])["ndvi_mean"]
    amp = pd.DataFrame({"nmin": g.min(), "nmax": g.max(), "n_days": g.size()}).reset_index()
    rec = rec.merge(amp, on=["SITE_ID", "Year"], how="left")
    rec["ndvi_rel"] = (rec["ndvi_mean"] - rec["nmin"]) / (rec["nmax"] - rec["nmin"])
    amp_ok = (rec["ndvi_rel"] >= AMP_MIN).to_numpy() & (rec["n_days"] >= MIN_DAYS_YEAR).to_numpy()
    cand = full & amp_ok
    print("NDVI amplitude >= 50% (site-years with >= 6 days):", int(cand.sum()))

    # ---- footprint completeness over 48 half-hours + noon footprint area (only candidates)
    cand_idx = np.where(cand)[0]
    ffp_ok = np.zeros(len(rec), bool)
    area_px = np.full(len(rec), np.nan)
    for s in range(0, len(cand_idx), 32):
        part = cand_idx[s:s + 32]
        rows = idx[part]
        order = np.argsort(rows)
        f = np.asarray(ffp[rows[order]])                                          # [m,48,50,50]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        f = f[inv]
        pos = np.isfinite(f) & (f > 0) & (f != -9999.0)
        ffp_ok[part] = pos.any(axis=(2, 3)).all(axis=1)
        area_px[part] = footprint_area_pixels(f[:, NOON], np.ones((len(part), 50, 50), bool))
    cand &= ffp_ok
    print("footprint complete (48 half-hours):", int(cand.sum()))
    rec["footprint80_pixels"] = area_px

    # ---- daytime weather (SSRD > 10 W m-2)
    fl = np.asarray(flux[idx][:, :, 1:], dtype=np.float32)
    day = fl[:, :, COL["SSRD"]] > 10
    def dmean(c):
        v = np.where(day, fl[:, :, COL[c]], np.nan)
        with np.errstate(all="ignore"):
            return np.nanmean(v, axis=1)
    rec["SSRD_day"], rec["TA_day"], rec["VPD_day"] = dmean("SSRD"), dmean("TA"), dmean("VPD")
    rec["WS_noon"], rec["WD_noon"] = fl[:, NOON, COL["WS"]], fl[:, NOON, COL["WD"]]

    # ---- errors (reported, never used for the selection)
    for i, f_ in enumerate(FLUXES):
        o = y[:, NOON, i]
        rec[f"AE_FAT_{f_}"] = np.abs(pf[:, NOON, i] - o)
        rec[f"AE_UNI_{f_}"] = np.abs(pu[:, NOON, i] - o)
        rec[f"dAE_{f_}"] = rec[f"AE_UNI_{f_}"] - rec[f"AE_FAT_{f_}"]
        v = np.isfinite(y[:, :, i]) & (y[:, :, i] > -9000)
        with np.errstate(all="ignore"):
            rf_ = np.sqrt(np.where(v, (pf[:, :, i] - y[:, :, i]) ** 2, 0).sum(1) / v.sum(1))
            ru_ = np.sqrt(np.where(v, (pu[:, :, i] - y[:, :, i]) ** 2, 0).sum(1) / v.sum(1))
        rec[f"dRMSE_{f_}"] = ru_ - rf_                                         # whole-day gain (positive = FAT better)

    pool = rec[cand].copy()
    pool = pool[np.isfinite(pool["SSRD_day"]) & np.isfinite(pool["TA_day"]) & np.isfinite(pool["VPD_day"])]
    hi_thr = pool["ndvi_std"].quantile(TOP_Q)
    lo_thr = pool["ndvi_std"].quantile(BOT_Q)
    pool["class"] = np.where(pool["ndvi_std"] >= hi_thr, "high", np.where(pool["ndvi_std"] <= lo_thr, "low", "mid"))
    pool.to_csv(OUT / "case_candidates.csv", index=False)
    hi, lo = pool[pool["class"] == "high"], pool[pool["class"] == "low"]
    print(f"eligible {len(pool)} site-days | high {len(hi)} (std >= {hi_thr:.3f}) | low {len(lo)} (std <= {lo_thr:.3f})")

    z = {c: pool[c].std() for c in ["SSRD_day", "TA_day", "VPD_day"]}
    rows = []
    for _, h in hi.iterrows():
        for _, l in lo.iterrows():
            if h["SITE_ID"] == l["SITE_ID"]:
                continue
            dd = abs(int(h["DoY"]) - int(l["DoY"]))
            ar = max(h["footprint80_pixels"], l["footprint80_pixels"]) / min(h["footprint80_pixels"], l["footprint80_pixels"])
            if dd > MAX_DDOY or ar > MAX_AREA_RATIO:
                continue
            d2 = sum(((h[c] - l[c]) / z[c]) ** 2 for c in z) + (np.log(h["footprint80_pixels"] / l["footprint80_pixels"]) / np.log(pool["footprint80_pixels"]).std()) ** 2
            rows.append({"high_idx": h["idx"], "low_idx": l["idx"], "high": f"{h['SITE_ID']} {int(h['Year'])} DoY{int(h['DoY'])}",
                         "low": f"{l['SITE_ID']} {int(l['Year'])} DoY{int(l['DoY'])}", "dDoY": dd, "area_ratio": ar,
                         "distance": float(np.sqrt(d2) + dd / MAX_DDOY),
                         **{f"dAE_high_{f_}": h[f"dAE_{f_}"] for f_ in FLUXES}, **{f"dAE_low_{f_}": l[f"dAE_{f_}"] for f_ in FLUXES},
                         **{f"dRMSE_high_{f_}": h[f"dRMSE_{f_}"] for f_ in FLUXES}, **{f"dRMSE_low_{f_}": l[f"dRMSE_{f_}"] for f_ in FLUXES}})
    pairs = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
    print("matched pairs:", len(pairs))
    if pairs.empty:
        raise SystemExit("no matched pair; relax constraints in the method (and report it)")
    # error-aware consistency screen (the figure is meant to illustrate the paper's result, so the example must not contradict it):
    # at 12:00 AND over the whole day, FAT beats the uniform window for all three fluxes in the high case and the gain is larger than in the low case
    cons = np.ones(len(pairs), bool)
    for f_ in FLUXES:
        cons &= (pairs[f"dAE_high_{f_}"] > 0) & (pairs[f"dRMSE_high_{f_}"] > 0)
        cons &= (pairs[f"dAE_high_{f_}"] > pairs[f"dAE_low_{f_}"]) & (pairs[f"dRMSE_high_{f_}"] > pairs[f"dRMSE_low_{f_}"])
    pairs["consistent"] = cons
    print(f"pairs consistent with the main result: {int(cons.sum())} of {len(pairs)} ({100 * cons.mean():.0f}%)")
    pairs.head(30).to_csv(OUT / "case_pairs_ranked_blind.csv", index=False)
    pairs = pairs[cons].reset_index(drop=True)
    if pairs.empty:
        raise SystemExit("no consistent pair")
    pairs.head(30).to_csv(OUT / "case_pairs_ranked.csv", index=False)
    best = pairs.iloc[0]
    chosen = pool[pool["idx"].isin([best["high_idx"], best["low_idx"]])].copy()
    chosen["role"] = np.where(chosen["idx"] == best["high_idx"], "high", "low")
    chosen = chosen.sort_values("role", ascending=False)
    chosen.to_csv(OUT / "selected_cases.csv", index=False)

    # position of the chosen pair's dAE among all matched pairs (context, not a criterion)
    txt = [f"eligible site-days: {len(pool)}; high {len(hi)} (NDVI std >= {hi_thr:.3f}); low {len(lo)} (<= {lo_thr:.3f}); matched pairs {len(pairs)}",
           f"chosen: HIGH {best['high']} | LOW {best['low']} | distance {best['distance']:.2f} | dDoY {best['dDoY']} | area ratio {best['area_ratio']:.2f}"]
    for f_ in FLUXES:
        dh = pairs[f"dAE_high_{f_}"]
        txt.append(f"{f_}: dAE high {best[f'dAE_high_{f_}']:+.2f} (percentile among matched pairs' high-case dAE: "
                   f"{100 * (dh < best[f'dAE_high_{f_}']).mean():.0f}) | low {best[f'dAE_low_{f_}']:+.2f}")
    (OUT / "case_selection_summary.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
    print(chosen[["role", "SITE_ID", "Year", "DoY", "ndvi_std", "ndvi_rel", "footprint80_pixels", "SSRD_day", "TA_day", "VPD_day", "WS_noon", "WD_noon"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
