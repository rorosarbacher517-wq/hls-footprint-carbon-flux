"""Schematic (NOT real data) of the proposed analysis of FAT gains: mismatch definition + three steps."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(3)
fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.2), gridspec_kw={"hspace": 0.38, "wspace": 0.28})

# A: definition
a = ax[0, 0]; a.set_aspect("equal"); a.set_xlim(0, 8); a.set_ylim(0, 8); a.axis("off")
ndvi = np.clip(0.35 + 0.5 * (np.arange(8)[None, :] / 7) + 0.18 * rng.standard_normal((8, 8)) + 0.12 * (np.arange(8)[:, None] > 4), 0.1, 0.95)
for i in range(8):
    for j in range(8):
        a.add_patch(Rectangle((j, i), 1, 1, fc=plt.cm.YlGn(ndvi[i, j]), ec="white", lw=0.6))
a.add_patch(Rectangle((2.2, 2.2), 3.6, 3.6, fc="none", ec="#1f4e9c", lw=2.4, ls="--"))
a.add_patch(Ellipse((5.4, 4.3), 5.0, 2.2, angle=18, fc="#d6301f", alpha=0.32, ec="#d6301f", lw=2.2))
a.plot(4.0, 4.0, marker="^", ms=13, color="k", zorder=5)
a.text(4.0, 3.5, "tower", ha="center", fontsize=10)
a.text(0.1, 8.25, "A  What is the mismatch D?", fontsize=13, fontweight="bold")
a.text(0.0, -0.25, "blue dashed square = fixed window (Uniform)\nred ellipse = dynamic footprint (FAT)\nD = |mean NDVI(footprint) − mean NDVI(window)|\n(each half-hour; averaged over a day / a year)",
       fontsize=10.5, va="top")

# B: step 1
b = ax[0, 1]
q = np.arange(1, 6); y = np.array([0.02, 0.05, 0.09, 0.15, 0.24]); e = np.array([0.04, 0.04, 0.05, 0.06, 0.08])
b.errorbar(q, y, yerr=e, fmt="o-", color="#d6301f", capsize=4, lw=2, ms=8)
b.axhline(0, color="k", lw=0.8); b.set_xticks(q); b.set_xticklabels(["Q1\nlow D", "Q2", "Q3", "Q4", "Q5\nhigh D"])
b.set_ylabel("mean ΔRMSE (FAT gain)"); b.set_xlabel("mismatch D, five groups of site-years")
b.set_title("B  Step 1: dose–response curve", loc="left", fontsize=13, fontweight="bold")
b.text(0.03, 0.95, "all sites pooled; bars = 95% CI\n(resampling whole sites)\nlarger D → larger gain?", transform=b.transAxes, va="top", fontsize=10.5)

# C: step 2
c = ax[1, 0]
names = ["Mismatch D", "Atmospheric stability", "Growing-season strength", "Soil moisture (control)", "Air temperature (control)"]
eff = np.array([0.065, 0.012, 0.020, -0.004, 0.008]); lo = eff - np.array([0.03, 0.03, 0.03, 0.03, 0.03]); hi = eff + np.array([0.03, 0.03, 0.03, 0.03, 0.03])
ypos = np.arange(len(names))[::-1]
for yy, ef, l, h, n in zip(ypos, eff, lo, hi, names):
    col = "#d6301f" if l > 0 else "#777777"
    c.plot([l, h], [yy, yy], color=col, lw=2.5); c.plot(ef, yy, "o", color=col, ms=9)
c.axvline(0, color="k", lw=0.9, ls="--"); c.set_yticks(ypos); c.set_yticklabels(names)
c.set_xlabel("change in ΔRMSE per +1 SD of the variable")
c.set_title("C  Step 2: effect size (one regression)", loc="left", fontsize=13, fontweight="bold")
c.text(0.98, 0.04, "red = interval excludes 0\ncontrols are adjusted for,\nnot ranked", transform=c.transAxes, ha="right", fontsize=10.5)

# D: step 3
d = ax[1, 1]
cols = ["#1f77b4", "#2ca02c", "#9467bd"]
for k, col in enumerate(cols):
    x = rng.uniform(0.1 + 0.05 * k, 0.6 + 0.1 * k, 18); base = 0.02 + 0.12 * k
    yv = base + 0.22 * (x - x.mean()) + 0.035 * rng.standard_normal(18)
    d.scatter(x, yv, s=18, color=col, alpha=0.7)
    p = np.polyfit(x, yv, 1); xs = np.array([x.min(), x.max()]); d.plot(xs, np.polyval(p, xs), color=col, lw=2.2)
xs = np.linspace(0.1, 0.8, 5); d.plot(xs, 0.12 + 0.0 * xs, color="#999999", lw=2, ls=":")
d.text(0.8, 0.095, "placebo: shuffled\nfootprint → flat", color="#777777", ha="right", fontsize=10)
d.set_xlabel("daily mismatch D"); d.set_ylabel("daily ΔRMSE")
d.set_title("D  Step 3: within each site, then pooled", loc="left", fontsize=13, fontweight="bold")
d.text(0.03, 0.97, "each colour = one site (days as points)\nsame site → same vegetation, climate,\ninstrument; only the footprint changes", transform=d.transAxes, va="top", fontsize=10.2)
fig.suptitle("Schematic of the proposed analysis (illustrative values, not results)", fontsize=14, y=0.98)
fig.savefig("schematic_gain_analysis.png", dpi=170, bbox_inches="tight", facecolor="white")
