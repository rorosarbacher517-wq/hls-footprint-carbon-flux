#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4.4 case study, step 2: figures for the selected high-/low-heterogeneity site-days (see select_cases.py).

Fig_case_main      2 rows (high / low heterogeneity) x 4 columns: NDVI of the 50x50 HLS patch | FAT footprint weights (12:00 local) |
                   uniform-window weights (new height-based rule, 12:00 local) | tower-scale bars (observed, FAT, uniform) for NEE, GPP, RECO
Fig_case_diurnal   2 rows (sites) x 3 columns (NEE, GPP, RECO): half-hourly observed, FAT and uniform-weighted estimates for the day
Fig_case_regional_{site}_{flux}  one figure per site and flux: FAT 30 m pixel estimates at the 24 full hours, 4 rows x 6 columns

Data   : FAT pixel products (patch_pre = 30 m pixel flux, w_map = FAT weights, site_post = tower-scale FAT estimate) from
         run_fat_pixel_prediction.py (old FAT weights of the held-out fold; to be replaced when FAT is re-run in V6);
         uniform-weighted Transformer (V6) tower-scale predictions from data/four_model_predictions; observations: true.npy.
Note   : north is up (row 0 of the HLS/footprint rasters is the northern edge; checked against the wind direction over 400 random records).
         FAT values in this figure come from the pixel-level re-run, so that the tower value equals the weighted sum of the maps shown;
         they differ slightly from the evaluation arrays (see README: max |difference| per site-day).
Run    : python fig_4_4_case_study.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "common"))
from figure_io import save_manuscript_png  # noqa: E402
from height_uniform import height_uniform_weights_numpy  # noqa: E402

V6 = Path('./training')
INPUT = Path('./data/input_data')
OUT = HERE.parent / "outputs"
DATA = HERE.parent / "data" / "FAT_pixel_predictions"
PRED = PROJECT / "data" / "four_model_predictions"
FLUXES = ["NEE", "GPP", "RECO"]
UNIT = r"$\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$"
NOON = 24
EXTENT = [-750, 750, -750, 750]
COL_OBS, COL_FAT, COL_UNI = "#222222", "#238B45", "0.58"
LANDCOVER = Path('./data/input_data/site_yearly_landcover_tracked.csv')
RGB_MAX, RGB_GAMMA = 0.18, 0.7                                                  # fixed true-colour stretch for both sites

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42})


def save_png(fig, path, **kw):
    """save_manuscript_png with retries (the file may be briefly locked while a viewer has it open)"""
    import time
    for t in range(8):
        try:
            return save_manuscript_png(fig, path, **kw)
        except OSError:
            time.sleep(2)
    raise OSError(f"could not write {path}")


# ------------------------------------------------------------------ data
def load_case(row):
    site, year, doy, idx = row["SITE_ID"], int(row["Year"]), int(row["DoY"]), int(row["idx"])
    base = DATA / site / site / "FFP0"
    ld = lambda n: np.load(base / f"pred.FFP0.{n}.{site}.npy")[0]
    c = {"site": site, "year": year, "doy": doy, "idx": idx, "role": row["role"], "ndvi_std": float(row["ndvi_std"]),
         "patch_pre": ld("patch_pre"), "w_fat": ld("w_map"), "fat": ld("site_post"), "obs": ld("true")}
    # uniform-weighted Transformer (V6): stored tower-scale prediction of the held-out fold
    for k in range(5):
        s = np.load(PRED / "Uniform_Transformer" / f"fold{k}" / "site.npy", allow_pickle=True)
        m = np.where((s[:, 0, 0] == site) & (pd.to_numeric(pd.Series(s[:, 0, 3])).to_numpy() == year)
                     & (pd.to_numeric(pd.Series(s[:, 0, 4])).to_numpy() == doy))[0]
        if len(m):
            c["uni"] = np.load(PRED / "Uniform_Transformer" / f"fold{k}" / "pred.npy")[m[0]].astype(float)
            c["fat_eval"] = np.load(PRED / "FAT" / f"fold{k}" / "pred.npy")[m[0]].astype(float)
            c["obs_eval"] = np.load(PRED / "FAT" / f"fold{k}" / "true.npy")[m[0]].astype(float)
            break
    # NDVI, valid mask, uniform window weights
    from numpy_pickle_compat import install_numpy_pickle_compat
    install_numpy_pickle_compat()
    hls = np.load(INPUT / "HLS_matched_v11_remove_selected_sites.npy", mmap_mode="r")
    h = np.asarray(hls[idx, :6], dtype=np.float32)
    valid = np.all(np.isfinite(h) & (h != -9999.0), axis=0)
    red, nir = h[2], h[3]
    c["ndvi"] = np.where(valid, (nir - red) / (nir + red + 1e-6), np.nan)
    flux = np.load(INPUT / "merged_flux_era5_ffp_array_v11_remove_selected_sites.npy", allow_pickle=True)
    fl = np.asarray(flux[idx, :, 1:], dtype=np.float32)
    c["w_uni"] = height_uniform_weights_numpy(valid[None], fl[None])[0].astype(float)
    c["ws"], c["wd"] = float(fl[NOON, 65]), float(fl[NOON, 66])
    zm = float(fl[NOON, 96])
    c["z_eff"] = zm if np.isfinite(zm) and zm > 0 else float(fl[NOON, 102])
    rgb = np.stack([h[2], h[1], h[0]], axis=-1).astype(float)                  # red, green, blue (HLS reflectance)
    rgb = np.where(np.isfinite(rgb) & (rgb > -1), rgb, np.nan)
    c["rgb"] = np.clip(np.nan_to_num(rgb, nan=1.0) / RGB_MAX, 0, 1) ** RGB_GAMMA
    lc = pd.read_csv(LANDCOVER)
    t = lc[(lc["Site_ID"] == site) & (lc["Year"] == year)]
    t = t if len(t) else lc[lc["Site_ID"] == site]
    c["igbp"] = str(t["Original_IGBP"].iloc[0])
    c["label"] = f"{site} ({c['igbp']})"
    c["date"] = f"{year}, DoY {doy}"
    return c


def mask_obs(a):
    a = a.astype(float).copy()
    a[(~np.isfinite(a)) | (a < -9000)] = np.nan
    return a


def contour_level(w, frac=0.8):
    v = np.sort(w[w > 0])[::-1]
    return v[np.searchsorted(np.cumsum(v) / v.sum(), frac)] if v.size else None


def neff(w):
    return 1.0 / np.nansum(w ** 2)


def draw_weights(ax, w, vmax=None, cmap="YlOrRd", box=False):
    wm = np.where(w > 0, w / w.max(), np.nan)                                  # relative weight, 1 = largest weight of the panel
    im = ax.imshow(wm, cmap=cmap, norm=PowerNorm(0.5, vmin=0, vmax=1), extent=EXTENT, origin="upper", interpolation="nearest")
    lev = contour_level(w)
    if lev is not None:
        ax.contour(np.linspace(-735, 735, 50), np.linspace(735, -735, 50), w, levels=[lev * 0.999], colors="#111111", linewidths=0.9)
    ax.plot(0, 0, marker="+", color="k", ms=6, mew=1.1)
    return im


def style_map(ax, show_y):
    ax.set_xlim(-750, 750)
    ax.set_ylim(-750, 750)
    ax.set_xticks([-500, 0, 500])
    ax.set_yticks([-500, 0, 500] if show_y else [])
    ax.tick_params(length=2.5, pad=1.5)
    ax.set_aspect("equal")


# ------------------------------------------------------------------ main figure
def fig_main(cases):
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.15), gridspec_kw={"width_ratios": [1, 1, 1, 1, 2.3], "wspace": 0.24, "hspace": 0.3})
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.2, top=0.985)
    for r, c in enumerate(cases):
        # true-colour HLS composite
        ax = axes[r, 0]
        ax.imshow(c["rgb"], extent=EXTENT, origin="upper", interpolation="nearest")
        ax.plot(0, 0, marker="+", color="w", ms=6, mew=1.1)
        style_map(ax, True)
        ax.set_ylabel("North–south (m)")
        # NDVI
        ax = axes[r, 1]
        im1 = ax.imshow(c["ndvi"], cmap="YlGn", vmin=0, vmax=1, extent=EXTENT, origin="upper", interpolation="nearest")
        ax.plot(0, 0, marker="+", color="k", ms=6, mew=1.1)
        style_map(ax, False)
        ax.text(0.04, 0.04, f"NDVI$_{{sd}}$ = {c['ndvi_std']:.2f}", transform=ax.transAxes, fontsize=6.2, va="bottom", ha="left", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
        # FAT footprint
        ax = axes[r, 2]
        draw_weights(ax, c["w_fat"])
        style_map(ax, False)
        ax.text(0.04, 0.04, f"WS = {c['ws']:.1f} m s$^{{-1}}$\nWD = {c['wd']:.0f}°", transform=ax.transAxes, fontsize=6.2, va="bottom", ha="left",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
        # uniform window
        ax = axes[r, 3]
        im3 = draw_weights(ax, c["w_uni"])
        style_map(ax, False)
        ax.text(0.04, 0.04, f"$z_m$ = {c['z_eff']:.1f} m\n100$z_m$ = {100 * c['z_eff']:.0f} m", transform=ax.transAxes, fontsize=6.2, va="bottom", ha="left",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
        # bars with explicit labels
        ax = axes[r, 4]
        obs, fat, uni = (mask_obs(c["obs"])[NOON], c["fat"][NOON], c["uni"][NOON])
        x = np.arange(3)
        wd = 0.27
        for k, (vals, col, lab) in enumerate([(obs, COL_OBS, "Observed"), (fat, COL_FAT, "FAT"), (uni, COL_UNI, "Uniform-weighted\nTransformer")]):
            ax.bar(x + (k - 1) * wd, vals, wd * 0.92, color=col, label=lab if r == 0 else None)
        ymin, ymax = np.nanmin(np.r_[obs, fat, uni, 0]), np.nanmax(np.r_[obs, fat, uni, 0])
        span = ymax - ymin
        ax.set_ylim(ymin - 0.04 * span if ymin < 0 else 0, ymax + 0.62 * span)
        for i in range(3):
            ae_f, ae_u = round(abs(fat[i] - obs[i]), 2), round(abs(uni[i] - obs[i]), 2)      # differences are formed from the printed (rounded) values
            y0 = ymax + 0.04 * span
            # three right-aligned lines: AE_FAT, AE_Uni., dAE (dAE formed from the printed values)
            rows = [(f"AE$_{{FAT}}$ = {ae_f:.2f}", COL_FAT, 2), (f"AE$_{{Uni.}}$ = {ae_u:.2f}", "0.40", 1), (f"$\Delta$AE = {ae_u - ae_f:+.2f}", "#111111", 0)]
            for txt, col, j in rows:
                ax.text(i + 0.43, y0 + j * 0.14 * span, txt, ha="right", va="bottom", fontsize=5.2, color=col, fontweight="bold" if j == 0 else None)
        ax.axhline(0, color="#333333", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(FLUXES)
        ax.set_ylabel(UNIT, fontsize=6.5, labelpad=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", lw=0.3, alpha=0.4)
        ax.set_axisbelow(True)
        for j in range(5):
            ax_ = axes[r, j]
            ax_.text(0.04 if j < 4 else -0.22, 0.96 if j < 4 else 1.0, "abcdefghij"[r * 5 + j], transform=ax_.transAxes, fontsize=9, fontweight="bold", va="top",
                     bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8) if j < 4 else None)
        axes[r, 0].text(-0.64, 0.5, f"{c['label']}\n{c['date']}", transform=axes[r, 0].transAxes, rotation=90, ha="center", va="center", fontsize=7.2, fontweight="bold")
        if r == 1:
            for j in range(4):
                axes[r, j].set_xlabel("East–west (m)")
    fig.canvas.draw()
    for r in range(2):                                                         # bar panels span the same height as the map panels
        axes[r, 3].apply_aspect()
        pm, pb = axes[r, 3].get_position(), axes[r, 4].get_position()
        axes[r, 4].set_position([pb.x0 + 0.035, pm.y0, pb.width - 0.035, pm.height])      # extra gap for the y label
    h, l = axes[0, 4].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6, loc="lower right", ncol=1, bbox_to_anchor=(0.995, 0.0), handlelength=1.1, labelspacing=0.25)
    p1, p2, p3 = axes[1, 1].get_position(), axes[1, 2].get_position(), axes[1, 3].get_position()
    cb = fig.colorbar(im1, cax=fig.add_axes([p1.x0 + 0.005, 0.045, p1.width - 0.01, 0.016]), orientation="horizontal")
    cb.set_label("NDVI", fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=6, length=2)
    cb2 = fig.colorbar(im3, cax=fig.add_axes([p2.x0 + 0.005, 0.045, p3.x1 - p2.x0 - 0.01, 0.016]), orientation="horizontal")
    cb2.set_label("Relative weight (1 = largest weight of the panel)", fontsize=6.5, labelpad=1)
    cb2.ax.tick_params(labelsize=6, length=2)
    save_png(fig, OUT / "Fig_case_main.png", width_in=7.2, render_dpi=600)
    fig.savefig(OUT / "Fig_case_main.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ diurnal curves
def fig_diurnal(cases, shared=False):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex=True, sharey="col" if shared else False, gridspec_kw={"wspace": 0.34, "hspace": 0.16})
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.17, top=0.95)
    hh = np.arange(48) / 2.0
    for r, c in enumerate(cases):
        for i, flux in enumerate(FLUXES):
            ax = axes[r, i]
            ax.plot(hh, mask_obs(c["obs"])[:, i], color=COL_OBS, lw=1.0, marker="o", ms=1.8, label="Observed")
            ax.plot(hh, c["fat"][:, i], color=COL_FAT, lw=1.3, label="FAT")
            ax.plot(hh, c["uni"][:, i], color=COL_UNI, lw=1.3, ls="--", label="Uniform-weighted Transformer")
            ax.axvline(12, color="#B22222", lw=0.8, ls=":")
            if flux == "NEE":
                ax.axhline(0, color="#333333", lw=0.6)
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", lw=0.3, alpha=0.4)
            if r == 0:
                ax.set_title(flux, fontsize=8, fontweight="bold")
            if r == 1:
                ax.set_xlabel("Local time (h)")
            if i == 0:
                ax.set_ylabel(f"{c['label']}\n{c['date']}\n{UNIT}", fontsize=6.6)
            else:
                ax.set_ylabel(UNIT, fontsize=6.6)
            ax.text(0.02, 1.02, "abcdef"[r * 3 + i], transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005), handlelength=2.0, columnspacing=2.0)
    name = "Fig_case_diurnal_sharedY" if shared else "Fig_case_diurnal"
    save_png(fig, OUT / f"{name}.png", width_in=7.2, render_dpi=600)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ regional (patch-wide) FAT maps, hourly
def fig_regional(c, i, flux):
    pre = c["patch_pre"][:, :, :, i]                                           # [48,50,50]
    hours = np.arange(0, 48, 2)
    vals = pre[hours]
    if flux == "NEE":
        v = np.nanpercentile(np.abs(vals), 99)
        norm, cmap = TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v), "RdBu_r"        # source (NEE > 0) red, sink (NEE < 0) blue
        lab = f"{c['label']}, {c['date']}   NEE ({UNIT})   sink  ◄  0  ►  source"
    else:
        hi = np.nanpercentile(vals, 99)
        norm, cmap = plt.Normalize(0, hi), "viridis"
        lab = f"{c['label']}, {c['date']}   {flux} ({UNIT})"
    fig, axes = plt.subplots(4, 6, figsize=(7.2, 5.0), gridspec_kw={"wspace": 0.06, "hspace": 0.2})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.975, bottom=0.12)
    for k, h in enumerate(hours):
        ax = axes[k // 6, k % 6]
        im = ax.imshow(pre[h], cmap=cmap, norm=norm, extent=EXTENT, origin="upper", interpolation="nearest")
        lev = contour_level(c["w_fat"][h])
        if lev is not None:
            ax.contour(np.linspace(-735, 735, 50), np.linspace(735, -735, 50), c["w_fat"][h], levels=[lev * 0.999], colors="k", linewidths=0.55)
        ax.plot(0, 0, marker="+", color="k", ms=4, mew=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_title(f"{h // 2:02d}:00", fontsize=6.5, pad=1.5)
    x0, x1 = axes[3, 0].get_position().x0, axes[3, 5].get_position().x1
    cb = fig.colorbar(im, cax=fig.add_axes([x0, 0.055, x1 - x0, 0.016]), orientation="horizontal")
    cb.set_label(lab, fontsize=6.8, labelpad=1.5)
    cb.ax.tick_params(labelsize=6, length=2)
    name = f"Fig_case_regional_{c['site']}_{flux}"
    save_png(fig, OUT / f"{name}.png", width_in=7.2, render_dpi=500)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    sel = pd.read_csv(OUT / "selected_cases.csv")
    sel = sel.sort_values("role", ascending=True)                              # "high" before "low"
    cases = [load_case(r) for _, r in sel.iterrows()]
    rows = []
    for c in cases:
        d_eval = np.nanmax(np.abs(c["fat"] - c["fat_eval"]))
        rows.append({"site": c["site"], "role": c["role"], "max_abs_diff_pixelrun_vs_evaluation": d_eval,
                     "sum_w_fat_noon": c["w_fat"][NOON].sum(), "sum_w_uni_noon": c["w_uni"][NOON].sum()})
        c["w_fat_all"], c["w_uni_all"] = c["w_fat"], c["w_uni"]
    pd.DataFrame(rows).to_csv(OUT / "case_consistency_check.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    # panels use the 12:00 weights
    mains = []
    for c in cases:
        m = dict(c)
        m["w_fat"], m["w_uni"] = c["w_fat"][NOON], c["w_uni"][NOON]
        mains.append(m)
    # per-half-hour diurnal: tower values are [48,3]
    fig_main(mains)
    fig_diurnal(cases)
    fig_diurnal(cases, shared=True)
    for c in cases:
        for i, flux in enumerate(FLUXES):
            fig_regional(c, i, flux)
    # numbers for the text / table
    tab = []
    for c in cases:
        for i, flux in enumerate(FLUXES):
            o, f, u = mask_obs(c["obs"])[NOON, i], c["fat"][NOON, i], c["uni"][NOON, i]
            tab.append({"role": c["role"], "site": c["site"], "year": c["year"], "doy": c["doy"], "flux": flux, "obs_1200": o, "FAT_1200": f,
                        "uniform_1200": u, "AE_FAT": abs(f - o), "AE_uniform": abs(u - o), "dAE": abs(u - o) - abs(f - o)})
    pd.DataFrame(tab).round(3).to_csv(OUT / "Table_case_1200.csv", index=False)
    print(pd.DataFrame(tab).round(2).to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(V6))
    main()
