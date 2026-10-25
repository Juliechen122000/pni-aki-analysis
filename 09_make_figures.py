# -*- coding: utf-8 -*-
"""
09_make_figures.py  (v2)
=====================================================================
生成投稿用四张主图（300 dpi，PNG + PDF，全英文标注）
v2 修复：对数轴次要刻度标签、图例遮挡、标签重叠、dumbbell 视觉权重
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
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_MAIN = "#2E5A88"
C_ALT = "#B5442E"
C_ORNG = "#C98A2E"
C_GREY = "#6E6E6E"
C_LIGHT = "#C9D6E4"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  已保存", name)


# =====================================================================
# Fig 1  主关联森林图
# =====================================================================
rows = [
    ("INSPIRE  Model A (minimal)",  1.613, 1.492, 1.745, C_MAIN, "s"),
    ("INSPIRE  Model B (primary)",  1.466, 1.349, 1.593, C_MAIN, "s"),
    ("MOVER  Model A (complete case)", 1.461, 1.310, 1.628, C_ALT, "o"),
    ("MOVER  Model A (imputed, m=20)", 1.444, 1.310, 1.591, C_ALT, "o"),
    ("INSPIRE  serum albumin (Model B comparator)", 1.436, 1.330, 1.551, C_GREY, "D"),
]
fig, ax = plt.subplots(figsize=(7.0, 3.0))
ypos = np.arange(len(rows))[::-1]
for y, (lab, orr, lo, hi, c, mk) in zip(ypos, rows):
    ax.plot([lo, hi], [y, y], color=c, lw=1.4, solid_capstyle="butt", zorder=2)
    ax.plot(orr, y, marker=mk, ms=5.5, color=c, mec="white", mew=0.8, zorder=3)
    ax.text(1.02, y, f"{orr:.2f} ({lo:.2f}\u2013{hi:.2f})", va="center",
            ha="left", fontsize=7.5, color="#222222",
            transform=ax.get_yaxis_transform())
ax.axvline(1.0, color="#666666", ls="--", lw=0.8, zorder=1)
ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_xscale("log")
ax.set_xlim(1.18, 1.90)
ax.xaxis.set_major_locator(FixedLocator([1.2, 1.4, 1.6, 1.8]))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:.1f}"))
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Adjusted odds ratio for AKI per 1-SD lower preoperative marker", fontsize=8)
save(fig, "fig1_association_forest")


# =====================================================================
# Fig 2  轨迹梯度与增量归零
# =====================================================================
trj = pd.read_csv(os.path.join(OUT, "lcmm_candidate_traj.csv"))
trj = trj[(trj.form == "quad") & (trj.ng == 3)]
adj = pd.read_csv(os.path.join(OUT, "lcmm_adjusted_ors.csv"))
adj = adj[(adj.form == "quad") & (adj.ng == 3)]
pat = re.compile(r"([\d.]+)\s*\(([\d.]+)-([\d.]+)\)")


def parse(s):
    m = pat.search(str(s))
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (np.nan,) * 3


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1),
                               gridspec_kw={"width_ratios": [1.0, 1.05]})

colmap = {1: C_MAIN, 2: C_ORNG, 3: C_ALT}
namemap = {1: "C1 high-decline (AKI 3.90%)",
           2: "C2 recovering (8.17%)",
           3: "C3 low-flat (16.51%)"}
for k in [1, 2, 3]:
    z = trj[trj["class"] == k].sort_values("time")
    ax1.plot(z.time, z.mean_pni, "-o", color=colmap[k], lw=1.6, ms=4.2,
             mec="white", mew=0.7, label=namemap[k])
ax1.set_xlabel("Postoperative day (binned)", fontsize=8)
ax1.set_ylabel("Fitted PNI", fontsize=8)
ax1.set_xticks([0, 2, 4, 6])
ax1.set_xlim(-0.4, 6.9)
ax1.set_ylim(31.5, 48.5)
ax1.legend(loc="upper left", fontsize=5.9, frameon=False,
           handlelength=1.3, borderaxespad=0.2)
ax1.set_title("A   Trajectory classes", loc="left", fontsize=8.5, fontweight="bold")
ax1.text(0.02, 0.02, "n=6,036 (75.5%); entropy 0.416", transform=ax1.transAxes,
         fontsize=6.0, color="#666666")

models = ["M1", "M2", "M3", "M4"]
mlab = ["M1", "M2\n+preop\nPNI", "M3\n+full\nadj.", "M4\npostop\nvalue"]
x = np.arange(len(models))
w = 0.33
for i, (cls, cname) in enumerate([("cl1", "Class 1"), ("cl2", "Class 2")]):
    est, lo, hi = [], [], []
    for m in models:
        r = adj[(adj.model == m) & (adj.term == cls)]
        e, l, h = parse(r.OR_CI.iloc[0]) if len(r) else (np.nan,) * 3
        est.append(e); lo.append(l); hi.append(h)
    est, lo, hi = map(np.array, (est, lo, hi))
    col = C_MAIN if cls == "cl1" else C_ORNG
    ax2.errorbar(x + (i - 0.5) * w, est, yerr=[est - lo, hi - est],
                 fmt="o", ms=4.5, color=col, mec="white", mew=0.7,
                 lw=1.2, capsize=2.5, label=cname)
ax2.axhline(1.0, color="#666666", ls="--", lw=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(mlab, fontsize=6.2)
ax2.set_xlim(-0.6, 3.6)
ax2.set_yscale("log")
ax2.set_ylim(0.13, 1.45)
ax2.yaxis.set_major_locator(FixedLocator([0.15, 0.25, 0.4, 0.6, 1.0]))
ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, p: f"{v:g}"))
ax2.yaxis.set_minor_locator(NullLocator())
ax2.yaxis.set_minor_formatter(NullFormatter())
ax2.set_ylabel("Odds ratio vs C3 (95% CI)", fontsize=8)
ax2.legend(loc="lower right", fontsize=6.3, frameon=False, borderaxespad=0.3)
ax2.set_title("B   Class effect attenuates to the null", loc="left",
              fontsize=8.5, fontweight="bold")

fig.tight_layout(w_pad=2.0)
save(fig, "fig2_trajectory_increment")


# =====================================================================
# Fig 3  检测密度梯度（两国）
# =====================================================================
sb = pd.read_csv(os.path.join(OUT, "selection_bias_results.csv"))
gi = sb[sb.analysis.astype(str).str.startswith("E_quintile_gradient_INSPIRE")]
gm = sb[sb.analysis.astype(str).str.startswith("E_quintile_gradient_MOVER")]


def rate(df):
    v = []
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        r = df[df.model.astype(str).str.endswith(q)]
        v.append(float(r.OR.iloc[0]) * 100 if len(r) else np.nan)
    return np.array(v)


ri, rm = rate(gi), rate(gm)
xi = np.arange(5)

fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.plot(xi, ri, "-s", color=C_MAIN, lw=1.8, ms=5.5, mec="white", mew=0.8,
        label="INSPIRE (Republic of Korea), n=7,995")
ax.plot(xi, rm, "-o", color=C_ALT, lw=1.8, ms=5.5, mec="white", mew=0.8,
        label="MOVER (United States), n=1,338")
for x, y in zip(xi, ri):
    if x == 0:
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(-15, 3), ha="right", fontsize=6.8, color=C_MAIN)
    else:
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=6.8, color=C_MAIN)
for x, y in zip(xi, rm):
    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=6.8, color=C_ALT)
ax.set_xticks(range(5))
ax.set_xticklabels(["Q1\n2 vs 1", "Q2\n4 vs 3", "Q3\n5 vs 6",
                    "Q4\n6 vs 8", "Q5\n8 vs 17"], fontsize=7)
ax.set_xlim(-0.45, 4.55)
ax.set_xlabel("Within-cohort quintile of creatinine testing density\n"
              "(median number of creatinine measurements, INSPIRE vs MOVER)",
              fontsize=7.6)
ax.set_ylabel("Detected AKI (%)", fontsize=8)
ax.set_ylim(-2.5, 72)
ax.legend(loc="upper left", fontsize=7, frameon=False, borderaxespad=0.3)
save(fig, "fig3_surveillance_gradient")


# =====================================================================
# Fig 4  校正前后衰减（dumbbell）
# =====================================================================
pairs = [
    ("INSPIRE  Model A (minimal)", 1.613, 1.433, "\u221211.2%", C_MAIN),
    ("INSPIRE  Model B (primary)", 1.466, 1.350, "\u22127.9%", C_MAIN),
    ("MOVER  Model A (complete case)", 1.461, 1.329, "\u22129.0%", C_ALT),
]

fig, ax = plt.subplots(figsize=(7.0, 3.0))
ypos = np.arange(len(pairs))[::-1]
for y, (lab, b, a, pct, c) in zip(ypos, pairs):
    ax.plot([a, b], [y, y], color=c, lw=1.1, zorder=2)
    ax.plot(b, y, "o", ms=6.2, color=c, mec="white", mew=0.9, zorder=4)
    ax.plot(a, y, "o", ms=6.2, color="white", mec=c, mew=1.5, zorder=4)
    ax.text(a - 0.012, y, f"{pct}", va="center", ha="right",
            fontsize=7.2, color=c, zorder=5)

# IPW 参考区间：底部独立条，避免遮挡主体
ybar = -0.62
ax.plot([1.4477, 1.5221], [ybar, ybar], color=C_LIGHT, lw=5.0,
        solid_capstyle="butt", zorder=1)
for v in (1.4477, 1.5221):
    ax.plot([v, v], [ybar - 0.08, ybar + 0.08], color="#8FA6BF", lw=0.9, zorder=1)
ax.plot(1.5117, ybar, "|", ms=7, color=C_MAIN, mew=1.2, zorder=2)
ax.text(1.40, ybar, "IPW reweighting, six schemes:", fontsize=6.2,
        color="#31465E", ha="right", va="center")
ax.text(1.529, ybar, "1.45\u20131.52", fontsize=6.2, color="#31465E",
        ha="left", va="center")

ax.axvline(1.0, color="#666666", ls="--", lw=0.8)
ax.set_yticks(ypos)
ax.set_yticklabels([p[0] for p in pairs], fontsize=7.5)
ax.set_xlim(1.19, 1.70)
ax.set_ylim(-1.1, len(pairs) - 0.45)
ax.set_xscale("log")
ax.xaxis.set_major_locator(FixedLocator([1.2, 1.3, 1.4, 1.5, 1.6, 1.7]))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:.1f}"))
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Odds ratio per 1-SD lower preoperative PNI", fontsize=8)
ax.scatter([], [], s=26, color=C_GREY, edgecolors="white", label="unadjusted")
ax.scatter([], [], s=26, facecolors="white", edgecolors=C_GREY,
           linewidths=1.4, label="adjusted for testing density")
ax.legend(loc="lower right", fontsize=6.6, frameon=False, scatterpoints=1,
          handletextpad=0.5, borderaxespad=0.3)
save(fig, "fig4_attenuation")

print("\n完成，输出目录：", FIG)
