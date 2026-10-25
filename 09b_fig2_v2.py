# -*- coding: utf-8 -*-
"""
09b_fig2_v2.py
=====================================================================
Figure 2 重绘（v2）
改动理由：
  1. 原图只展示三类解，且把"置信区间跨过无效线"当作结论标注；
     如今四类解（纯 BIC 选模）在 M4 仍显著，必须并列展示。
  2. 删除任何解释性标注，只呈现两个模型下的调整后效应与不确定性。
布局：A 轨迹（竖跨）｜B 三类解 M1–M4｜C 四类解 M1–M4
覆盖输出 fig2_trajectory_increment.png / .pdf
=====================================================================
"""
import re
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator, FixedLocator

OUT = os.environ.get("INSPIRE_OUT", "output")
FIG = os.path.join(OUT, "fig")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_MAIN, C_ORNG, C_ALT = "#2E5A88", "#C98A2E", "#B5442E"

pat = re.compile(r"([\d.]+)\s*\(([\d.]+)-([\d.]+)\)(.*)")


def parse(s):
    m = pat.search(str(s))
    if not m:
        return np.nan, np.nan, np.nan, ""
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4).strip()


def draw_panel(ax, df, terms, labels, cols, ylab, ylim):
    models = ["M1", "M2", "M3", "M4"]
    x = np.arange(len(models))
    w = 0.34 if len(terms) > 1 else 0.0
    for i, (t, lab, c) in enumerate(zip(terms, labels, cols)):
        est, lo, hi = [], [], []
        for m in models:
            r = df[(df.model == m) & (df.term == t)]
            e, l, h, _ = parse(r.est.iloc[0]) if len(r) else (np.nan,) * 4
            est.append(e); lo.append(l); hi.append(h)
        est, lo, hi = map(np.array, (est, lo, hi))
        xs = x + (i - (len(terms) - 1) / 2) * w if w else x
        ax.errorbar(xs, est, yerr=[est - lo, hi - est], fmt="o", ms=4.6,
                    color=c, mec="white", mew=0.7, lw=1.2, capsize=2.6, label=lab)
    ax.axhline(1.0, color="#666666", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["M1", "M2", "M3", "M4"], fontsize=6.8)
    ax.set_xlim(-0.55, 3.55)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(FixedLocator([0.15, 0.25, 0.4, 0.6, 1.0]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(ylab, fontsize=7.4)
    ax.tick_params(axis="y", labelsize=6.8)


fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], hspace=0.62, wspace=0.34)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])

# ---- A 轨迹 ----
trj = pd.read_csv(os.path.join(OUT, "lcmm_candidate_traj.csv"))
trj = trj[(trj.form == "quad") & (trj.ng == 3)]
colmap = {1: C_MAIN, 2: C_ORNG, 3: C_ALT}
namemap = {1: "C1 high-decline (3.90%)", 2: "C2 recovering (8.17%)",
           3: "C3 low-flat (16.51%)"}
for k in [1, 2, 3]:
    z = trj[trj["class"] == k].sort_values("time")
    ax1.plot(z.time, z.mean_pni, "-o", color=colmap[k], lw=1.6, ms=4.2,
             mec="white", mew=0.7, label=namemap[k])
ax1.set_xlabel("Postoperative day (binned)", fontsize=7.6)
ax1.set_ylabel("Fitted PNI", fontsize=7.6)
ax1.set_xticks([0, 2, 4, 6])
ax1.set_xlim(-0.4, 6.9)
ax1.set_ylim(31.5, 48.5)
ax1.tick_params(labelsize=6.8)
ax1.legend(loc="upper left", fontsize=5.9, frameon=False,
           handlelength=1.3, borderaxespad=0.2)
ax1.set_title("A   Three-class solution", loc="left", fontsize=8.2, fontweight="bold")
ax1.text(0.02, 0.02, "n=6,036 (75.5%); entropy 0.416\npercentages are class-specific AKI",
         transform=ax1.transAxes, fontsize=5.8, color="#666666")

# ---- B 三类解 ----
adj3 = pd.read_csv(os.path.join(OUT, "lcmm_adjusted_ors.csv"))
adj3 = adj3[(adj3.form == "quad") & (adj3.ng == 3)]
if "OR_CI" in adj3.columns:
    adj3 = adj3.rename(columns={"OR_CI": "est"})
draw_panel(ax2, adj3, ["cl1", "cl2"], ["Class 1", "Class 2"], [C_MAIN, C_ORNG],
           "OR vs class 3", (0.13, 1.45))
ax2.legend(loc="lower right", fontsize=5.8, frameon=False, borderaxespad=0.2)
ax2.set_title("B   Class contrasts, three-class solution", loc="left",
              fontsize=8.2, fontweight="bold")

# ---- C 四类解（BIC 选模）----
adj4 = pd.read_csv(os.path.join(OUT, "bic_model_m1m4.csv"))
draw_panel(ax3, adj4, ["cl2", "cl3"], ["Class 2", "Class 3"], [C_MAIN, C_ORNG],
           "OR vs class 1", (0.13, 1.45))
# 四类解为探索性对照，不再高亮任何单个“显著”对比（避免暗示确认性结论）
ax3.legend(loc="lower right", fontsize=5.8, frameon=False, borderaxespad=0.2)
ax3.set_title("C   Class contrasts, four-class solution (BIC only)", loc="left",
              fontsize=8.2, fontweight="bold")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig2_trajectory_increment.{ext}"), dpi=300,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已重绘 fig2_trajectory_increment（三面板，含 BIC 选模对照）")
