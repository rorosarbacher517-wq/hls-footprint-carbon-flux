#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4.3 (random-forest importance, original logic): site-day table of the FAT gain with its drivers, using the NEW paired results.

Gain per HLS-matched day and flux:  delta_RMSE_day = RMSE(uniform-weighted Transformer, V6) - RMSE(FAT)   (positive = FAT better)
computed from the held-out predictions of the five site-held-out folds (../../data/four_model_predictions).
Drivers (daily medians of meteorology, soil, footprint parameters, NDVI heterogeneity, land-cover edge density, ...) do not depend on the
models and are taken unchanged from the original driver table (records are matched on SITE_ID, Year, DoY, HLS_Hour, flux).

Output: ../data/site_day_delta_rmse_with_drivers_V6.csv   (same layout as the original site_day_delta_rmse_with_drivers.csv)
Run   : python build_site_day_table.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "data" / "four_model_predictions"
OLD = Path('./data/external/site_day_delta_rmse_with_drivers.csv')
OUT = HERE.parent / "data" / "site_day_delta_rmse_with_drivers_V6.csv"
FLUXES = ["NEE", "GPP", "RECO"]
KEYS = ["SITE_ID", "Year", "DoY", "HLS_Hour"]


def load(name):
    ys, ps, ss = [], [], []
    for k in range(5):
        b = DATA / name / f"fold{k}"
        ys.append(np.load(b / "true.npy", allow_pickle=True))
        ps.append(np.load(b / "pred.npy", allow_pickle=True))
        ss.append(np.load(b / "site.npy", allow_pickle=True))
    return np.concatenate(ys).astype(float), np.concatenate(ps).astype(float), np.concatenate(ss)


def main():
    y, pf, site = load("FAT")
    _, pu, site2 = load("Uniform_Transformer")
    assert np.array_equal(site, site2), "FAT / uniform records are not aligned"
    s0 = site[:, 0, :]
    rec = pd.DataFrame({"SITE_ID": s0[:, 0].astype(str), "Year": pd.to_numeric(s0[:, 3]).astype(int),
                        "DoY": pd.to_numeric(s0[:, 4]).astype(int), "HLS_Hour": pd.to_numeric(s0[:, 5]).astype(int)})
    rows = []
    for i, flux in enumerate(FLUXES):
        obs = y[:, :, i]
        ef, eu = pf[:, :, i] - obs, pu[:, :, i] - obs
        valid = np.isfinite(obs) & (obs > -9000) & np.isfinite(ef) & np.isfinite(eu)
        n = valid.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rf = np.sqrt(np.where(valid, ef ** 2, 0).sum(1) / n)
            ru = np.sqrt(np.where(valid, eu ** 2, 0).sum(1) / n)
        t = rec.copy()
        t["flux"], t["n_valid_halfhours"] = flux, n
        t["rmse_dynamic_day"], t["rmse_static_day"] = rf, ru
        rows.append(t[n > 0])
    new = pd.concat(rows, ignore_index=True)
    new["delta_RMSE_day"] = new["rmse_static_day"] - new["rmse_dynamic_day"]
    new["relative_delta_RMSE_day"] = new["delta_RMSE_day"] / new["rmse_static_day"]

    old = pd.read_csv(OLD)
    drivers = old.drop(columns=["record_id", "n_valid_halfhours", "rmse_dynamic_day", "rmse_static_day", "delta_RMSE_day", "relative_delta_RMSE_day"])
    drivers = drivers.drop_duplicates(subset=KEYS + ["flux"])
    out = new.merge(drivers, on=KEYS + ["flux"], how="inner")
    out.insert(0, "record_id", out.groupby(KEYS, sort=False).ngroup())
    out.to_csv(OUT, index=False)
    print("new gain records:", len(new), "| with drivers:", len(out), "| records:", out["record_id"].nunique(), "| sites:", out["SITE_ID"].nunique())
    print(out.groupby("flux")["delta_RMSE_day"].agg(["median", "mean"]).round(3))


if __name__ == "__main__":
    main()
