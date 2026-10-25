import os
# -*- coding: utf-8 -*-
"""15_alb_vs_pni_dauc.py
白蛋白 vs PNI 的直接比较：ΔAUC 差值及其配对 bootstrap 95% CI（推导队列）
"""
import numpy as np, pandas as pd, statsmodels.api as sm

OUT = os.environ.get("INSPIRE_OUT", "output")
d = pd.read_csv(f"{OUT}/p0_aki_inspire_harmonized.csv", low_memory=False)
d = d[(d.eskd == 0) & (d.ltx == 0) & (d.elig == True)].copy()

B0 = ["age", "male", "asa", "op_duration", "cr"]   # 无 open_/emop 列，用共同模型变量
d["male"] = (d.sex.astype(str).str.upper().str[0] == "M").astype(float)
d["cr"] = pd.to_numeric(d["cr"], errors="coerce")
d["pni"] = pd.to_numeric(d.pni_onodera, errors="coerce")
d["alb"] = pd.to_numeric(d.alb, errors="coerce")

need = B0 + ["alb", "pni", "aki_unified"]
D = d.dropna(subset=need).copy()
y = D.aki_unified.astype(int).values
print(f"分析集 n={len(D)}  events={int(y.sum())} ({100*y.mean():.2f}%)")

# 方向统一：z 越大 = 越差（alb 与 pni 均为“越高越好”，故取负）
for c in ["alb", "pni"]:
    D["z_" + c] = -(D[c] - D[c].mean()) / D[c].std()


def auc(yy, p):
    r = pd.Series(p).rank()
    n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pred(cols, idx):
    X = sm.add_constant(D.iloc[idx][cols].astype(float), has_constant="add")
    m = sm.Logit(y[idx], X).fit(disp=0)
    return m.predict(X)


idx0 = np.arange(len(D))
p0 = pred(B0, idx0)
pa = pred(B0 + ["z_alb"], idx0)
pp = pred(B0 + ["z_pni"], idx0)
a0, aa, ap = auc(y, p0), auc(y, pa), auc(y, pp)

d_alb, d_pni = aa - a0, ap - a0
print(f"\n基线 AUC        {a0:.4f}")
print(f"  + 白蛋白      {aa:.4f}   ΔAUC = {d_alb:+.4f}")
print(f"  + PNI         {ap:.4f}   ΔAUC = {d_pni:+.4f}")
print(f"\n差值 ΔAUC(白蛋白) − ΔAUC(PNI) = {d_alb - d_pni:+.4f}")

# 配对 bootstrap
rng = np.random.default_rng(20261024)
B = 500
diffs, da, dp = [], [], []
for b in range(B):
    idx = rng.integers(0, len(D), len(D))
    if y[idx].sum() < 20 or y[idx].sum() > len(idx) - 20:
        continue
    try:
        q0 = pred(B0, idx); qa = pred(B0 + ["z_alb"], idx); qp = pred(B0 + ["z_pni"], idx)
        yb = y[idx]
        b0 = auc(yb, q0); ba = auc(yb, qa); bp = auc(yb, qp)
        da.append(ba - b0); dp.append(bp - b0); diffs.append((ba - b0) - (bp - b0))
    except Exception:
        continue
    if (b + 1) % 100 == 0:
        print(f"  bootstrap {b+1}/{B} ...", flush=True)

def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.mean(x), np.percentile(x, 2.5), np.percentile(x, 97.5)

for lab, arr in [("ΔAUC 白蛋白", da), ("ΔAUC PNI", dp), ("差值(白- PNI)", diffs)]:
    m, lo, hi = ci(arr)
    print(f"  {lab:16s} {m:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")

m, lo, hi = ci(diffs)
print("\n结论：差值的 95% CI " + ("包含 0 —— 未发现明确增量优势" if lo < 0 < hi
                                else ("<0 —— 白蛋白增量更大" if hi < 0 else ">0 —— PNI 增量更大")))
