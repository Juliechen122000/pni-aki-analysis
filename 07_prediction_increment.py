import os
# -*- coding: utf-8 -*-
"""
07_prediction_increment.py
=====================================================================
术前 PNI 是否在基线 AKI 风险模型之上带来预测增量？
并且：PNI 的增量是否强于它自身的两个组分（白蛋白 / 淋巴细胞）？

这是论文能否讲"临床可用"的关键检验。若 PNI 的增量不及单用白蛋白，
则 PNI 只是一个包装，整篇的立论要重做。

基线模型 B0 = 年龄 + 性别(M) + ASA + 手术时长 + 术前Cr + 开放入路 + 急诊
（即主分析 Spec B 去掉 PNI）

对比模型：
  M1  B0 + PNI(per 1 SD, SD=6.4452)
  M2  B0 + 白蛋白(g/dL)
  M3  B0 + ALC(×10^3/uL, 标准化)
  M4  B0 + 白蛋白 + ALC

指标：AUROC、连续NRI(含分解)、分类NRI(10%/20%)、IDI、DCA净收益
不确定性：bootstrap B=500（分层重采样，保持事件数）
=====================================================================
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

OUT = os.environ.get("INSPIRE_OUT", "output")
SD_PNI = 6.4452
B = 500
SEED = 20261024
rng = np.random.default_rng(SEED)

# ---------------- 1. 数据 ----------------
d = pd.read_csv(f"{OUT}/p0_aki_inspire_harmonized.csv")
d = d[(d.eskd == 0) & (d.ltx == 0) & (d.elig == True)].copy()
d["open_"] = (d.pcs.astype(str).str[4] == "0").astype(int)
d["male"] = (d.sex == "M").astype(int)
d["alc"] = d.wbc * d.lym / 100.0            # 绝对淋巴细胞计数 ×10^3/uL
d["pni_sd"] = d.pni_onodera / SD_PNI
d["alb"] = d.alb
# ALC 标准化（便于解释，不改变 AUC/NRI）
d["alc_sd"] = (d.alc - d.alc.mean()) / d.alc.std()

B0 = ["age", "male", "asa", "op_duration", "cr", "open_", "emop"]
MODELS = {
    "M0_base": B0,
    "M1_+PNI": B0 + ["pni_sd"],
    "M2_+alb": B0 + ["alb"],
    "M3_+ALC": B0 + ["alc_sd"],
    "M4_+alb+ALC": B0 + ["alb", "alc_sd"],
}
need = sorted(set(sum(MODELS.values(), [])) | {"aki_unified"})
D = d.dropna(subset=need).copy()
y = D.aki_unified.astype(int).values
print(f"主队列 7,995  →  complete-case n={len(D)}  events={int(y.sum())} "
      f"({100*y.mean():.2f}%)")
print(f"ALC 中位 {D.alc.median():.2f}  白蛋白中位 {D.alb.median():.1f}  "
      f"PNI 中位 {D.pni_onodera.median():.1f}")


# ---------------- 2. 工具函数 ----------------
def fit_pred(cols, idx=None):
    """返回样本上的预测概率（在给定 idx 上拟合）"""
    X = D[cols].astype(float)
    X = sm.add_constant(X, has_constant="add")
    yy = y
    if idx is not None:
        X = X.iloc[idx]; yy = y[idx]
    m = sm.Logit(yy, X).fit(disp=0)
    return m.predict(X) if idx is None else m.predict(X)


def auc(yy, p):
    yy = np.asarray(yy); p = np.asarray(p)
    n1 = int(yy.sum()); n0 = len(yy) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(p)
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def nri_cont(yy, po, pn):
    ev = yy == 1; ne = yy == 0
    return ((pn[ev] > po[ev]).mean() - (pn[ev] < po[ev]).mean()) + \
           ((pn[ne] < po[ne]).mean() - (pn[ne] > po[ne]).mean())


def nri_cont_parts(yy, po, pn):
    ev = yy == 1; ne = yy == 0
    return dict(up_ev=(pn[ev] > po[ev]).mean(), down_ev=(pn[ev] < po[ev]).mean(),
                up_ne=(pn[ne] > po[ne]).mean(), down_ne=(pn[ne] < po[ne]).mean())


def cat3(p, t1, t2):
    return np.where(p < t1, 0, np.where(p < t2, 1, 2))


def nri_cat(yy, po, pn, t1, t2):
    co = cat3(po, t1, t2); cn = cat3(pn, t1, t2)
    ev = yy == 1; ne = yy == 0
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
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return (np.nan, np.nan, np.nan)
    return (np.nanmean(x), np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5))


# ---------------- 3. 点估计 ----------------
P = {k: fit_pred(v) for k, v in MODELS.items()}
a0 = auc(y, P["M0_base"])
print(f"\nM0 基线 AUC = {a0:.4f}")

PTS = []
for k in MODELS:
    if k == "M0_base":
        continue
    po, pn = P["M0_base"], P[k]
    PTS.append(dict(contrast=k, n=len(D), events=int(y.sum()),
                    auc=auc(y, pn), d_auc=auc(y, pn) - a0,
                    nri_cont=nri_cont(y, po, pn),
                    nri_cat10=nri_cat(y, po, pn, 0.05, 0.10),
                    nri_cat20=nri_cat(y, po, pn, 0.10, 0.20),
                    idi=idi(y, po, pn)))
    print(f"{k:13s} AUC={auc(y,pn):.5f}  ΔAUC={auc(y,pn)-a0:+.5f}  "
          f"NRIcont={nri_cont(y,po,pn):+.4f}  IDI={idi(y,po,pn):+.5f}")

# ---------------- 4. Bootstrap ----------------
ev_idx = np.where(y == 1)[0]
ne_idx = np.where(y == 0)[0]
boot = {k: [] for k in MODELS if k != "M0_base"}
boot_dca = {k: [] for k in MODELS if k != "M0_base"}
boot_pair = []  # paired ΔAUC(albumin) − ΔAUC(PNI), all within Model B
THR = [0.05, 0.10, 0.15, 0.20, 0.25]

for b in range(B):
    idx = np.concatenate([rng.choice(ev_idx, len(ev_idx), replace=True),
                          rng.choice(ne_idx, len(ne_idx), replace=True)])
    yb = y[idx]
    try:
        Pb = {k: fit_pred(v, idx) for k, v in MODELS.items()}
    except Exception:
        continue
    a0b = auc(yb, Pb["M0_base"])
    for k in boot:
        po, pn = Pb["M0_base"], Pb[k]
        boot[k].append(dict(d_auc=auc(yb, pn) - a0b,
                            nri_cont=nri_cont(yb, po, pn),
                            nri_cat10=nri_cat(yb, po, pn, 0.05, 0.10),
                            nri_cat20=nri_cat(yb, po, pn, 0.10, 0.20),
                            idi=idi(yb, po, pn)))
        boot_dca[k].append([nb(yb, pn, t) - nb(yb, po, t) for t in THR])
    boot_pair.append(auc(yb, Pb["M2_+alb"]) - auc(yb, Pb["M1_+PNI"]))
    if (b + 1) % 100 == 0:
        print(f"  bootstrap {b+1}/{B}")

# ---------------- 5. 汇总 ----------------
RES = []
for pt in PTS:
    k = pt["contrast"]
    bb = pd.DataFrame(boot[k])
    for met in ["d_auc", "nri_cont", "nri_cat10", "nri_cat20", "idi"]:
        m, lo, hi = catci(bb[met].values)
        # 双侧 bootstrap p（以 0 为原假设）
        pv = 2 * min((bb[met] <= 0).mean(), (bb[met] >= 0).mean())
        RES.append(dict(contrast=k, metric=met, est=pt[met], boot_mean=m,
                        lo=lo, hi=hi, p_boot=min(pv, 1.0)))
    for j, t in enumerate(THR):
        arr = np.array([r[j] for r in boot_dca[k]])
        m, lo, hi = catci(arr)
        RES.append(dict(contrast=k, metric=f"dNB@{int(t*100)}%",
                        est=nb(y, P[k], t) - nb(y, P["M0_base"], t),
                        boot_mean=m, lo=lo, hi=hi, p_boot=np.nan))

pair_est = auc(y, P["M2_+alb"]) - auc(y, P["M1_+PNI"])
pm, plo, phi = catci(np.asarray(boot_pair))
pp = 2 * min((np.asarray(boot_pair) <= 0).mean(),
             (np.asarray(boot_pair) >= 0).mean())
RES.append(dict(contrast="M2_alb_minus_M1_PNI", metric="paired_d_auc",
                est=pair_est, boot_mean=pm, lo=plo, hi=phi, p_boot=min(pp, 1.0)))
print(f"\nModel B paired ΔAUC(albumin) − ΔAUC(PNI) = {pair_est:+.5f} "
      f"(bootstrap mean {pm:+.5f}, 95% CI {plo:+.5f} to {phi:+.5f})")
R = pd.DataFrame(RES)
R.to_csv(f"{OUT}/prediction_increment.csv", index=False)

# 连续 NRI 分解（点估计）
parts = []
for k in MODELS:
    if k == "M0_base":
        continue
    p = nri_cont_parts(y, P["M0_base"], P[k])
    parts.append(dict(contrast=k, **p))
pd.DataFrame(parts).to_csv(f"{OUT}/prediction_nri_parts.csv", index=False)

print("\n" + "=" * 78)
print(R.pivot_table(index="metric", columns="contrast", values="est").round(5).to_string())
print("\n" + "=" * 78)
print(R[R.metric.isin(["d_auc", "idi", "nri_cont", "nri_cat20"])]
      .assign(sig=lambda x: np.where((x.lo > 0) | (x.hi < 0), "*", ""))
      [["contrast", "metric", "est", "lo", "hi", "sig"]]
      .to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
print("\n已写出 prediction_increment.csv / prediction_nri_parts.csv")
