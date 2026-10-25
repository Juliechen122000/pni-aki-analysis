import os
# -*- coding: utf-8 -*-
"""
08b_multimarker_compare.py
=====================================================================
围手术期「营养 vs 炎症」指标头对头比较：谁真正携带 AKI 风险信号？

基线模型 B0 = 年龄 + 性别 + ASA + 手术时长 + 术前Cr + 开放入路 + 急诊
对每个指标单独做 B0 + z（z 已统一方向：越差 = z 越大）

营养侧（低=差）：PNI(Onodera)、简化PNI、白蛋白、ALC、GNRI、总蛋白
炎症侧（高=差）：NLR、PLR、SII、CRP

关键设计（避免不可比）：
  Set A 核心集（不含 CRP，覆盖率 ~99.7%）—— 所有模型在同一 common sample 上拟合
  Set B 含 CRP（覆盖率 51.8%）—— 敏感性，检验炎症是否在白蛋白之上有独立增量
  另做双指标模型：B0 + 白蛋白 + NLR / CRP，看炎症的增量是否独立于营养

指标：ΔAUC、IDI、连续NRI、分类NRI@20%、dNB@10%，bootstrap B=500 分层重采样
=====================================================================
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

OUT = os.environ.get("INSPIRE_OUT", "output")
SD_PNI = 6.4452
B = 500
rng = np.random.default_rng(20261024)

# ---------------- 1. 数据与指标构建 ----------------
d = pd.read_csv(f"{OUT}/p0_aki_inspire_harmonized.csv", low_memory=False)
mk = pd.read_csv(f"{OUT}/p0_multimarker_preop.csv", low_memory=False)
d = d.merge(mk, on="subject_id", how="left")

d = d[(d.eskd == 0) & (d.ltx == 0) & (d.elig == True)].copy()
d["open_"] = (d.pcs.astype(str).str[4] == "0").astype(int)
d["male"] = (d.sex == "M").astype(int)

d["alc"] = d.wbc * d.lym / 100.0
d["nlr"] = d.pre_seg / d.lym
d["plr"] = d.pre_platelet / d.alc
d["sii"] = d.pre_platelet * d.nlr
d["tp"] = d.pre_total_protein
d["crp"] = d.pre_crp

# GNRI: 14.89*alb(g/dL) + 41.7*min(W/Wideal, 1)
h = pd.to_numeric(d.height, errors="coerce")
w = pd.to_numeric(d.weight, errors="coerce")
ideal = np.where(d.male == 1, h - 100 - (h - 150) / 4.0, h - 100 - (h - 150) / 2.5)
ideal = np.where((ideal > 0) & np.isfinite(ideal), ideal, np.nan)
d["gnri"] = 14.89 * d.alb + 41.7 * np.minimum(w / ideal, 1.0)

B0 = ["age", "male", "asa", "op_duration", "cr", "open_", "emop"]

# 指标：(列名, 中文名, 方向) 方向 -1 = 低值更差，+1 = 高值更差
MARKERS = [
    ("pni_onodera", "PNI (Onodera)", -1),
    ("pni_simple", "PNI (simplified)", -1),
    ("alb", "Albumin", -1),
    ("gnri", "GNRI", -1),
    ("tp", "Total protein", -1),
    ("alc", "ALC", -1),
    ("nlr", "NLR", +1),
    ("plr", "PLR", +1),
    ("sii", "SII", +1),
    ("crp", "CRP", +1),
]

# 统一方向标准化：z 越大 = 越差
for col, _, direc in MARKERS:
    x = pd.to_numeric(d[col], errors="coerce")
    x = x.mask(x <= 0) if col in ("nlr", "plr", "sii", "crp") else x
    d["z_" + col] = direc * (x - x.mean()) / x.std()

print("=" * 78)
print("指标可得性（分母 7,995）")
print("=" * 78)
for col, nm, _ in MARKERS:
    n = d[col].notna().sum()
    print(f"  {nm:18s} {n:5d} ({100*n/7995:5.1f}%)  "
          f"中位 {pd.to_numeric(d[col], errors='coerce').median():8.2f}")

# ---------------- 2. 工具 ----------------
def auc(yy, p):
    yy = np.asarray(yy); p = np.asarray(p)
    n1 = int(yy.sum()); n0 = len(yy) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = stats.rankdata(p)
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def nri_cont(yy, po, pn):
    ev = yy == 1; ne = yy == 0
    return ((pn[ev] > po[ev]).mean() - (pn[ev] < po[ev]).mean()) + \
           ((pn[ne] < po[ne]).mean() - (pn[ne] > po[ne]).mean())

def nri_cat(yy, po, pn, t1=0.10, t2=0.20):
    f = lambda p: np.where(p < t1, 0, np.where(p < t2, 1, 2))
    co, cn = f(po), f(pn); ev = yy == 1; ne = yy == 0
    return ((cn[ev] > co[ev]).mean() - (cn[ev] < co[ev]).mean()) + \
           ((cn[ne] < co[ne]).mean() - (cn[ne] > co[ne]).mean())

def idi(yy, po, pn):
    ev = yy == 1; ne = yy == 0
    return (pn[ev].mean() - po[ev].mean()) - (pn[ne].mean() - po[ne].mean())

def nb(yy, p, pt):
    tp = ((p >= pt) & (yy == 1)).sum() / len(yy)
    fp = ((p >= pt) & (yy == 0)).sum() / len(yy)
    return tp - fp * pt / (1 - pt)

def catci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 20: return (np.nan, np.nan, np.nan)
    return (np.nanmean(x), np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5))

# ---------------- 3. 分套分析 ----------------
def run_set(name, marker_list, D):
    y = D.aki_unified.astype(int).values
    print(f"\n{'='*78}\n{name}  n={len(D)}  events={int(y.sum())} "
          f"({100*y.mean():.2f}%)  EPV≈{y.sum()/ (len(B0)+1):.1f}\n{'='*78}")
    X0 = sm.add_constant(D[B0].astype(float), has_constant="add")
    m0 = sm.Logit(y, X0).fit(disp=0)
    p0 = m0.predict(X0)
    a0 = auc(y, p0)
    print(f"B0 baseline AUC = {a0:.4f}")

    rows = []
    for col, nm, _ in marker_list:
        X = sm.add_constant(D[B0 + ["z_" + col]].astype(float), has_constant="add")
        m = sm.Logit(y, X).fit(disp=0)
        p = m.predict(X)
        b, se = m.params["z_" + col], m.bse["z_" + col]
        rows.append(dict(set=name, marker=nm, col=col,
                         or_sd=np.exp(b), or_lo=np.exp(b - 1.96 * se),
                         or_hi=np.exp(b + 1.96 * se), p_marker=m.pvalues["z_" + col],
                         auc=auc(y, p), d_auc=auc(y, p) - a0,
                         idi=idi(y, p0, p), nri_cont=nri_cont(y, p0, p),
                         nri_cat20=nri_cat(y, p0, p),
                         dnb10=nb(y, p, .10) - nb(y, p0, .10)))
    return rows, y, p0, a0, D

def boot_set(rows, y, D, marker_list, a0):
    ev_idx = np.where(y == 1)[0]; ne_idx = np.where(y == 0)[0]
    acc = {r["col"]: {k: [] for k in ("d_auc", "idi", "nri_cont", "nri_cat20", "dnb10")}
           for r in rows}
    for b in range(B):
        idx = np.concatenate([rng.choice(ev_idx, len(ev_idx), True),
                              rng.choice(ne_idx, len(ne_idx), True)])
        yb = y[idx]
        try:
            X0 = sm.add_constant(D[B0].astype(float), has_constant="add").iloc[idx]
            pb0 = sm.Logit(yb, X0).fit(disp=0).predict(X0)
        except Exception:
            continue
        a0b = auc(yb, pb0)
        for col, nm, _ in marker_list:
            try:
                X = sm.add_constant(D[B0 + ["z_" + col]].astype(float),
                                    has_constant="add").iloc[idx]
                pb = sm.Logit(yb, X).fit(disp=0).predict(X)
            except Exception:
                continue
            acc[col]["d_auc"].append(auc(yb, pb) - a0b)
            acc[col]["idi"].append(idi(yb, pb0, pb))
            acc[col]["nri_cont"].append(nri_cont(yb, pb0, pb))
            acc[col]["nri_cat20"].append(nri_cat(yb, pb0, pb))
            acc[col]["dnb10"].append(nb(yb, pb, .10) - nb(yb, pb0, .10))
        if (b + 1) % 100 == 0:
            print(f"  boot {b+1}/{B}", flush=True)
    for r in rows:
        for k in ("d_auc", "idi", "nri_cont", "nri_cat20", "dnb10"):
            m, lo, hi = catci(acc[r["col"]][k])
            r[k + "_lo"], r[k + "_hi"] = lo, hi
    return rows

# --- Set A：核心（不含 CRP）---
core = [m for m in MARKERS if m[0] != "crp"]
A = d.dropna(subset=B0 + ["z_" + m[0] for m in core] + ["aki_unified"]).copy()
rowsA, yA, p0A, a0A, DA = run_set("SetA_core_noCRP", core, A)
rowsA = boot_set(rowsA, yA, DA, core, a0A)

# --- Set B：含 CRP ---
allm = MARKERS
Bd = d.dropna(subset=B0 + ["z_" + m[0] for m in allm] + ["aki_unified"]).copy()
rowsB, yB, p0B, a0B, DB = run_set("SetB_withCRP", allm, Bd)
rowsB = boot_set(rowsB, yB, DB, allm, a0B)

R = pd.DataFrame(rowsA + rowsB)
R.to_csv(f"{OUT}/multimarker_compare.csv", index=False)

# ---------------- 4. 双指标：炎症是否独立于营养 ----------------
print("\n" + "=" * 78)
print("双指标模型：炎症指标是否在白蛋白之上有独立增量")
print("=" * 78)
DUAL = []
for pair in [("alb", "nlr"), ("alb", "crp"), ("gnri", "nlr"),
             ("alb", "sii"), ("alb", "plr")]:
    base_cols = B0 + ["z_" + pair[0]]
    Dd = d.dropna(subset=B0 + ["z_" + pair[0], "z_" + pair[1], "aki_unified"])
    yd = Dd.aki_unified.astype(int).values
    Xa = sm.add_constant(Dd[base_cols].astype(float), has_constant="add")
    pa = sm.Logit(yd, Xa).fit(disp=0).predict(Xa)
    Xb = sm.add_constant(Dd[base_cols + ["z_" + pair[1]]].astype(float),
                         has_constant="add")
    mb = sm.Logit(yd, Xb).fit(disp=0)
    pb = mb.predict(Xb)
    k = "z_" + pair[1]
    DUAL.append(dict(pair=f"{pair[0]} + {pair[1]}", n=len(Dd), events=int(yd.sum()),
                     or_added=np.exp(mb.params[k]),
                     or_lo=np.exp(mb.params[k] - 1.96 * mb.bse[k]),
                     or_hi=np.exp(mb.params[k] + 1.96 * mb.bse[k]),
                     p_added=mb.pvalues[k],
                     d_auc=auc(yd, pb) - auc(yd, pa),
                     idi=idi(yd, pa, pb),
                     nri_cont=nri_cont(yd, pa, pb)))
    print(f"  {pair[0]:6s} + {pair[1]:4s}  n={len(Dd):5d}  "
          f"added OR={np.exp(mb.params[k]):.2f} p={mb.pvalues[k]:.3g}  "
          f"ΔAUC={auc(yd,pb)-auc(yd,pa):+.5f}  IDI={idi(yd,pa,pb):+.5f}")

pd.DataFrame(DUAL).to_csv(f"{OUT}/multimarker_dual.csv", index=False)

# ---------------- 5. 排行榜 ----------------
for st in ["SetA_core_noCRP", "SetB_withCRP"]:
    s = R[R.set == st].sort_values("d_auc", ascending=False)
    print(f"\n{'='*78}\n排行榜 {st}\n{'='*78}")
    print(s[["marker", "or_sd", "or_lo", "or_hi", "p_marker",
             "auc", "d_auc", "d_auc_lo", "d_auc_hi", "idi", "idi_lo", "idi_hi",
             "nri_cat20", "dnb10"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n已写出 multimarker_compare.csv / multimarker_dual.csv")
