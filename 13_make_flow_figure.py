# -*- coding: utf-8 -*-
"""
13_make_flow_figure.py
双栏 STROBE 受试者流程图（INSPIRE 左 / MOVER 右）
规格与现有主图一致：300 dpi、Arial、全英文
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
})

OUT = os.environ.get("OUT_DIR", "./figures")
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.2, 7.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

FS = 7.2
EDGE = "#333333"


def box(x, y, w, h, text, fs=FS, bold=False, fc="#FFFFFF"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.06,rounding_size=0.10",
                                linewidth=0.8, edgecolor=EDGE, facecolor=fc))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.40, color="#111111")


def arrow(x, y1, y2, label=None):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=EDGE,
                                shrinkA=0, shrinkB=0))
    if label:
        ax.text(x + 0.14, (y1 + y2) / 2, label, ha="left", va="center",
                fontsize=6.0, color="#555555", linespacing=1.20)


def header(x, text):
    ax.text(x, 9.72, text, ha="center", va="center", fontsize=8.4,
            fontweight="bold", color="#111111")


# ---------------- 列标题 ----------------
header(2.55, "INSPIRE (derivation)")
header(7.45, "MOVER (replication)")

# ---------------- 左栏 ----------------
LX, LW, BH = 2.55, 3.9, 0.62
ys = [9.05, 8.05, 7.05, 6.05, 5.05, 4.05, 3.05, 1.95]

box(LX, ys[0], LW, BH, "Operations in source database\nn = 130,960", bold=True)
box(LX, ys[1], LW, BH, "Abdominal surgery\n(ICD-10-PCS 0D or 0F)\nn = 17,796")
box(LX, ys[2], LW, BH, "Operative duration ≥120 min\nn = 10,442")
box(LX, ys[3], LW, BH, "Index procedure, complete timestamps\nn = 10,051")
box(LX, ys[4], LW, BH, "Preoperative PNI and creatinine\navailable\nn = 8,595")
box(LX, ys[5], LW, BH, "Baseline creatinine <3.0 mg/dL\nn = 8,451")
box(LX, ys[6], LW, BH, "No liver transplantation\nn = 7,995")
box(LX, ys[7], LW, 0.78, "Complete covariate data\nn = 7,889 analysed\n739 with AKI (9.37%)",
    bold=True, fc="#F2F2F2")

EXCL = {0: "113,164 not\nabdominal surgery",
        1: "7,354 under 120 min",
        2: "391 not index / incomplete",
        3: "1,456 PNI or creatinine\nnot available",
        4: "144 creatinine\n≥3.0 mg/dL",
        5: "456 liver\ntransplantation",
        6: "106 incomplete\ncovariates"}
for i in range(len(ys) - 1):
    y_top = ys[i] - BH / 2 - 0.03
    y_bot = ys[i + 1] + (0.78 / 2 if i + 1 == 7 else BH / 2) + 0.03
    arrow(LX, y_top, y_bot, EXCL.get(i))

# ---------------- 右栏 ----------------
RX, RW = 7.45, 3.9
RYS = [9.05, 7.80, 6.75, 5.45, 4.30]
box(RX, RYS[0], RW, BH, "Operations in source database\n64,354 (39,683 patients)", bold=True)
box(RX, RYS[1], RW, BH, "Abdominal surgery\n(ICD-10-PCS 0D or 0F), ≥120 min\nn = 9,022 (5,710 patients)")
box(RX, RYS[2], RW, BH, "First operation per patient\nn = 5,710")
box(RX, RYS[3], RW, BH, "Preoperative PNI and creatinine\navailable\nn = 1,432")
box(RX, RYS[4], RW, 0.78, "n = 1,338 analysed\n352 with AKI (26.31%)", bold=True, fc="#F2F2F2")

REX = {0: "55,332 not abdominal\nor under 120 min",
       1: "3,312 repeat\noperations",
       2: "4,278 PNI or\ncreatinine not available",
       3: "94 creatinine\n≥3.0 mg/dL"}
for i in range(len(RYS) - 1):
    y_top = RYS[i] - BH / 2 - 0.03
    y_bot = RYS[i + 1] + (0.78 / 2 if i + 1 == len(RYS) - 1 else BH / 2) + 0.03
    arrow(RX, y_top, y_bot, REX.get(i))

# 右栏说明
ax.text(RX, 2.95,
        "EPIC extract contained no usable age variable;\nadult eligibility could not be independently verified.\nOpen approach and emergency status were also\nunavailable, so cross-cohort comparisons used a\ncommon model without these three variables.",
        ha="center", va="center", fontsize=6.4, color="#555555", linespacing=1.40)

# 左栏说明（底部）
ax.text(LX, 1.05,
        "INSPIRE: adults (≥18 yr); MOVER: age unavailable\nAKI: 0.3 mg/dL rise within 48 h; ratio criteria within 7 d",
        ha="center", va="center", fontsize=6.4, color="#555555", linespacing=1.35)

fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Figure1_flow.{ext}"), dpi=300,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("已生成:", os.path.join(OUT, "Figure1_flow.png / .pdf"))
