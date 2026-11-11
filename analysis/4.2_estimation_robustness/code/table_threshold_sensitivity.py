#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threshold-sensitivity table (Table S4 in the current manuscript numbering): sensitivity of the site-level improvement classes to the near-tie threshold.

Site-level Delta RMSE^k = RMSE_Uniform^k - RMSE_FAT^k  (k = NEE, GPP, RECO; umol CO2 m-2 s-1).
Positive values = lower RMSE for FAT.  Sites: all sites with >= 100 valid held-out samples and finite coordinates
(identical to the sites used in the Fig. 4.2 maps / box plots, because this script imports the same functions).

Classes for a threshold t:   FAT better: Delta > t     near tie: |Delta| <= t     Uniform better: Delta < -t

Inputs : ../../data/four_model_predictions (FAT, Uniform_Transformer, RF), ../../data/site_metadata
Outputs: ../outputs/Table_threshold_sensitivity.{csv,md}   (paper layout)
         ../outputs/Table_threshold_sensitivity_detailed.csv (counts, IQR)
Run    : python table_S5_threshold_sensitivity.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs" / "spatial"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.0, 0.025, 0.05, 0.1]
BASELINE_COLUMN = "Uniform better (%)"   # the paper draft called this column "Fixed better (%)"

# Reuse the exact site-level computation of the figure script.
spec = importlib.util.spec_from_file_location("fig42", HERE / "fig_4_2_spatial.py")
fig42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig42)


def site_table():
    meta = fig42.build_site_metadata(fig42.SITE_META_CSV)
    fat = fig42.load_model_arrays(fig42.FAT_DIR)
    uni = fig42.load_model_arrays(fig42.TRANSFORMER_DIR)
    rf = fig42.load_model_arrays(fig42.RF_DIR)
    tables = {}
    for flux in fig42.TARGET_FLUXES:
        d = fig42.compute_metrics(fat[0], uni[0], rf[0], fat[1], fat[2], flux)
        df = meta.merge(d, on="SITE_ID", how="inner")
        df = df[np.isfinite(df["Lat"]) & np.isfinite(df["Lon"])].copy()
        tables[flux] = df
    return tables


def main():
    tables = site_table()
    paper, detailed = [], []
    for flux, df in tables.items():
        delta = df[fig42.VALUE_COL].astype(float).to_numpy()
        n = len(delta)
        q1, q3 = np.percentile(delta, [25, 75])
        for t in THRESHOLDS:
            pos = int(np.sum(delta > t))
            tie = int(np.sum(np.abs(delta) <= t)) if t > 0 else 0
            neg = n - pos - tie
            paper.append({
                "Flux": flux, "Near-tie threshold": f"{t:g}", "n sites": n,
                "FAT better (%)": f"{100 * pos / n:.2f}", "Near tie (%)": f"{100 * tie / n:.2f}",
                BASELINE_COLUMN: f"{100 * neg / n:.2f}",
                "Mean ΔRMSE": f"{delta.mean():.2f}", "Median ΔRMSE": f"{np.median(delta):.2f}",
            })
            detailed.append({
                "Flux": flux, "threshold": t, "n_sites": n, "n_FAT_better": pos, "n_near_tie": tie,
                "n_baseline_better": neg, "pct_FAT_better": 100 * pos / n, "pct_near_tie": 100 * tie / n,
                "pct_baseline_better": 100 * neg / n, "mean_delta_RMSE": delta.mean(),
                "median_delta_RMSE": float(np.median(delta)), "delta_RMSE_IQR": q3 - q1,
                "definition": "delta_RMSE = RMSE_Uniform - RMSE_FAT; positive = FAT lower RMSE",
            })
        # near-tie handling at t=0: exact zeros (practically none) are counted as 'baseline better' above;
        # report them explicitly so the three classes always sum to n.
        zeros = int(np.sum(delta == 0))
        if zeros:
            print(f"[note] {flux}: {zeros} sites have Delta RMSE exactly 0")

    pd.DataFrame(paper).to_csv(OUT / "Table_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(detailed).to_csv(OUT / "Table_threshold_sensitivity_detailed.csv", index=False, encoding="utf-8-sig")
    cols = list(paper[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in paper]
    foot = ("\nΔRMSE^k = RMSE_Uniform^k − RMSE_FAT^k in μmol CO₂ m⁻² s⁻¹, k ∈ {NEE, GPP, RECO}. "
            "Positive values indicate lower RMSE for FAT.\n")
    (OUT / "Table_threshold_sensitivity.md").write_text("\n".join(md) + "\n" + foot, encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
