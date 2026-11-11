#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Old FAT (HLS_ERA5_FLUX_FFP_CV5_FINAL, TF 2.8) versus re-run FAT (V6, TF 2.15), everything else identical
(new uniform-weighted Transformer V6, pre-aggregation Transformer and RF unchanged).
Reads the outputs of PAPER_RESULTS_V6 (old FAT) and PAPER_RESULTS_V6_FATV6 (this folder) and writes COMPARE_old_vs_V6FAT.md.
Run: python compare_old_vs_V6FAT.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

NEW = Path(__file__).resolve().parent
OLD = NEW.parent / "PAPER_RESULTS_V6"
FL = ["NEE", "GPP", "RECO"]


def r(a, b, f):
    return pd.read_csv(OLD / f), pd.read_csv(NEW / f)


def md(df):
    return df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)


def main():
    out = ["# Old FAT vs V6 FAT (same data, same other models)", ""]

    # 4.1
    o, n = r(0, 0, "4.1_accuracy_assessment/outputs/pooled_metrics.csv")
    rows = []
    for m in ["FAT", "Uniform-weighted Transformer", "UNIFORM"]:
        pass
    key = lambda d: d[d["model"].isin(["FAT"])]
    rows = []
    for fl in FL:
        a, b = key(o)[key(o)["flux"] == fl].iloc[0], key(n)[key(n)["flux"] == fl].iloc[0]
        rows.append({"flux": fl, "R2 old": round(a.R2, 3), "R2 V6": round(b.R2, 3), "RMSE old": round(a.RMSE, 3), "RMSE V6": round(b.RMSE, 3),
                     "Bias old": round(a.Bias, 3), "Bias V6": round(b.Bias, 3)})
    out += ["## 4.1 FAT, pooled five folds", md(pd.DataFrame(rows)), ""]
    allm = pd.concat([o.assign(run="old FAT"), n.assign(run="V6 FAT")])
    out += ["Pooled metrics of all models (the three non-FAT models are the same in both runs):", md(allm.round(3)), ""]

    # 4.2
    o, n = r(0, 0, "4.2_estimation_robustness/outputs/spatial/Table_threshold_sensitivity.csv")
    c = [x for x in o.columns if x.startswith("Median")][0]
    cm = [x for x in o.columns if x.startswith("Mean")][0]
    t = pd.DataFrame({"flux": o["Flux"], "threshold": o["Near-tie threshold"], "FAT better % old": o["FAT better (%)"], "FAT better % V6": n["FAT better (%)"],
                      "uniform better % old": o["Uniform better (%)"], "uniform better % V6": n["Uniform better (%)"],
                      "median dRMSE old": o[c], "median dRMSE V6": n[c], "mean dRMSE old": o[cm], "mean dRMSE V6": n[cm]})
    out += ["## 4.2 site-level comparison (210 sites)", md(t), ""]
    o, n = r(0, 0, "4.2_estimation_robustness/outputs/temporal/summary_temporal_delta_rmse.csv")
    f = lambda d: d[(d["aggregation"] == "unit_first") & (d["group_col"] == "season")][["group", "var", "delta_rmse", "ci_low", "ci_high"]]
    s = f(o).merge(f(n), on=["group", "var"], suffixes=(" old", " V6"))
    out += ["## 4.2 temporal: median dRMSE by phenological stage", md(s.round(3)), ""]
    f = lambda d: d[(d["aggregation"] == "unit_first") & (d["group_col"] == "hour_bin")]
    try:
        a, b = f(o), f(n)
        out += ["## 4.2 temporal: peak hour (max median dRMSE)", md(pd.DataFrame({
            "flux": FL,
            "old": [a[a["var"] == v].sort_values("delta_rmse").iloc[-1][["group", "delta_rmse"]].tolist() for v in FL],
            "V6": [b[b["var"] == v].sort_values("delta_rmse").iloc[-1][["group", "delta_rmse"]].tolist() for v in FL]})), ""]
    except Exception as e:  # pragma: no cover
        out += [f"(hourly peak skipped: {e})", ""]

    # 4.3 correlation / terciles
    o, n = r(0, 0, "4.3_gain_correlates/outputs/heterogeneity_terciles_annual.csv")
    t = pd.DataFrame({"flux": FL})
    for tag, d in (("old", o), ("V6", n)):
        h = d[d["group"] == "High"].set_index("flux")
        t[f"high-low {tag}"] = [round(h.loc[x, "high_minus_low"], 3) for x in FL]
        t[f"95% CI {tag}"] = [f"[{h.loc[x, 'diff_ci_low']:.2f}, {h.loc[x, 'diff_ci_high']:.2f}]" for x in FL]
    out += ["## 4.3 annual gain, highest minus lowest third of NDVI heterogeneity", md(t), ""]
    o, n = r(0, 0, "4.3_gain_correlates/outputs/fig6_null_precision.csv")
    out += ["## 4.3 significant (FDR q<0.05) correlations with the relative gain / max |rho|", md(pd.concat([o.assign(run="old FAT"), n.assign(run="V6 FAT")]).round(3)), ""]
    o, n = r(0, 0, "4.3_gain_correlates/outputs/fig6_correlations_annual.csv")
    rows = []
    for oc in ["nrmse_d", "nrmse_s", "rel"]:
        for fl in FL:
            a = o[(o.outcome == oc) & (o.flux == fl) & (o.factor == "NDVI_STD_DAY")].iloc[0]
            b = n[(n.outcome == oc) & (n.flux == fl) & (n.factor == "NDVI_STD_DAY")].iloc[0]
            rows.append({"outcome": oc, "flux": fl, "rho old": round(a.rho, 3), "q old": round(a.q, 3), "rho V6": round(b.rho, 3), "q V6": round(b.q, 3)})
    out += ["## 4.3 NDVI heterogeneity: Spearman rho with FAT error (nrmse_d), uniform error (nrmse_s), relative gain (rel)", md(pd.DataFrame(rows)), ""]
    for tag, base in (("old", OLD), ("V6", NEW)):
        c = pd.read_csv(base / "4.3_gain_correlates/outputs/correlation_all_scales.csv")
        c = c[c["scale"] == "year"]
        p = c.pivot(index="label", columns="flux", values="rho")[FL]
        q = c.pivot(index="label", columns="flux", values="q")[FL]
        out += [f"## 4.3 Spearman rho with the annual gain (dRMSE), {tag} FAT; rows with q<0.05 in any flux marked",
                md(p.assign(sig=(q < 0.05).any(axis=1).map({True: "*", False: ""})).round(2).reset_index()), ""]

    # RF
    o, n = r(0, 0, "4.3_rf_importance/outputs/year/tables/year_rf_cross_site_performance.csv")
    out += ["## 4.3 random forest (annual), held-out-site R2", md(pd.concat([o.assign(run="old FAT"), n.assign(run="V6 FAT")])[["run", "flux", "n", "n_sites", "cv_r2", "cv_rmse"]].round(3)), ""]
    for tag, base in (("old", OLD), ("V6", NEW)):
        v = pd.read_csv(base / "4.3_rf_importance/outputs/year/tables/year_relative_variable_importance.csv")
        out += [f"Top permutation importances, {tag} FAT:"]
        for fl in FL:
            g = v[v.flux == fl].sort_values("relative_importance_percent", ascending=False).head(5)
            out += [f"- {fl}: " + "; ".join(f"{a} {b:.0f}%" for a, b in zip(g["label"], g["relative_importance_percent"]))]
        out += [""]

    # 4.4
    for tag, base in (("old", OLD), ("V6", NEW)):
        p = base / "4.4_case_study/outputs/case_selection_summary.txt"
        out += [f"## 4.4 case selection, {tag} FAT", "```", p.read_text(encoding="utf-8"), "```", ""]
        t = pd.read_csv(base / "4.4_case_study/outputs/Table_case_1200.csv")
        out += [md(t.round(2)), ""]
    (NEW / "COMPARE_old_vs_V6FAT.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out)[:6000])


if __name__ == "__main__":
    main()
