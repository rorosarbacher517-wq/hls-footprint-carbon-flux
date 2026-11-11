#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Positive control for the factor analysis: do the factors relate to each model's OWN error?

Weather, soil and turbulence variables are inputs of both models, so they change the absolute error of FAT and of the
uniform-weighted Transformer alike (e.g. warm, humid conditions = large fluxes = large errors) and largely cancel in the
difference (the gain).  This script reports Spearman rho of every factor with RMSE_FAT, RMSE_Uniform and the gain.
Outputs: ../outputs/factor_vs_model_rmse.csv  (scales: monthly, annual; fluxes NEE, GPP, RECO)
Run    : python factor_vs_model_rmse.py
"""
import importlib.util
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g", HERE / "gain_correlates.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

d = g.daily_gain_table()
rows = []
for scale in ["monthly", "year"]:
    t = g.aggregate(d, scale)
    for fl in g.FLUXES:
        x = t[t["flux"] == fl]
        for c in g.FCOLS:
            ok = x[[c, "rmse_d", "rmse_s", "delta"]].dropna()
            rows.append({"scale": scale, "flux": fl, "factor": g.LABEL[c], "group": g.GROUP[c], "n": len(ok),
                         "rho_RMSE_FAT": spearmanr(ok[c], ok["rmse_d"])[0], "rho_RMSE_uniform": spearmanr(ok[c], ok["rmse_s"])[0],
                         "rho_gain": spearmanr(ok[c], ok["delta"])[0]})
out = pd.DataFrame(rows)
out.to_csv(g.OUT / "factor_vs_model_rmse.csv", index=False)
s = out.groupby(["scale", "flux"])[["rho_RMSE_FAT", "rho_RMSE_uniform", "rho_gain"]].agg(lambda v: v.abs().mean()).round(2)
print("mean |rho| over the 20 factors:\n", s.to_string())
