import os
# -*- coding: utf-8 -*-
"""
10c_preop_density_analysis.py
=====================================================================
A1：术前 30 天检测密度作为「监测倾向」代理

为什么用它：整段窗口在手术之前，不可能受术后 AKI 反向影响。
它测的不是「术后检测偏倚」本身，而是「这个病人/这个团队本来就爱查」
——若它与 AKI 相关，说明存在监测倾向型患者；若调整后主关联稳如泰山，
说明术后检测密度的调整可能是一种过度校正。

分析顺序（按用户要求）：
  0. 主模型复现（校验 n / events / OR / CI）
  1. 同一样本上加入术前密度（区分样本变化与变量影响）
  2. 嵌套比较：看效应量变化，不只看 P
  3. 稳健性：分布、零检测、极端值、trim、是否由少数高频者驱动
  4. 与术后检测密度结果并列
=====================================================================
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT = os.environ.get("INSPIRE_OUT", "output")
SD = 6.4452

d = pd.read_csv(f"{OUT}/p0_aki_inspire_harmonized.csv", low_memory=False)
dens = pd.read_csv(f"{OUT}/p0_preop_density.csv", low_memory=False)
d = d.merge(dens, on="subject_id", how="left")

d = d[(d.eskd == 0) & (d.ltx == 0) & (d.elig == True)].copy()
d["open_"] = (d.pcs.astype(str).str[4] == "0").astype(int)
d["emale"] = (d.sex == "M").astype(int)
d["x"] = -d.pni_onodera / SD            # per -1 SD
for c in ["age", "asa", "op_duration", "cr"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")

for c in ["n_test_total", "n_uniq_time_total", "n_test_cr", "n_uniq_time_cr"]:
    d[c] = d[c].fillna(0).astype(float)
d["log_visits"] = np.log1p(d.n_uniq_time_total)   # 术前抽血次数（去重时间点）
d["log_items"] = np.log1p(d.n_test_total)         # 术前检验条数
d["log_cr"] = np.log1p(d.n_uniq_time_cr)          # 术前肌酐检测时点次数

B0 = ["x", "asa", "emale", "op_duration", "cr", "age", "open_", "emop"]
BASE = ["x", "asa", "emale", "op_duration", "cr", "age", "open_", "emop", "aki_unified"]

print("=" * 74)
print("0. 主模型复现")
print("=" * 74)
D0 = d.dropna(subset=BASE)
g0 = sm.Logit(D0.aki_unified.astype(int),
              sm.add_constant(D0[B0].astype(float))).fit(disp=0, cov_type="HC0")
b = g0.params["x"]; se = g0.bse["x"]
print(f"n={len(D0)}  events={int(D0.aki_unified.sum())}  "
      f"OR={np.exp(b):.4f} ({np.exp(b-1.96*se):.4f}–{np.exp(b+1.96*se):.4f})  "
      f"p={g0.pvalues['x']:.3g}")
OR0 = float(np.exp(b))
print("滚动口径锚点校验：Model B 应为 1.4659（±0.0005）→ ",
      "PASS" if abs(OR0 - 1.4659) < 0.0005 else "FAIL")

print("\n" + "=" * 74)
print("1. 术前检测密度的分布（分析样本内）")
print("=" * 74)
for c, lab in [("n_uniq_time_total", "术前抽血次数(去重)"),
               ("n_test_total", "术前检验条数"),
               ("n_uniq_time_cr", "术前肌酐次数")]:
    s = D0[c]
    print(f"  {lab:18s} 中位{s.median():5.0f}  IQR {s.quantile(.25):.0f}–{s.quantile(.75):.0f}"
          f"  P95 {s.quantile(.95):5.0f}  最大 {s.max():6.0f}"
          f"  =0 者 {int((s==0).sum()):4d} ({100*(s==0).mean():.2f}%)")

# 按术前抽血次数分组看 AKI 率
D0 = D0.copy()
D0["vis_q"] = pd.qcut(D0.n_uniq_time_total.rank(method="first"), 5,
                      labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
print("\n  术前抽血次数五分位与 AKI 发生率（是否也呈梯度？）:")
tmp = D0.groupby("vis_q", observed=True).agg(
    n=("aki_unified", "size"), aki=("aki_unified", "mean"),
    med=("n_uniq_time_total", "median"), pni=("pni_onodera", "mean"))
for q, r in tmp.iterrows():
    print(f"    {q}  n={int(r.n):5d}  AKI {100*r.aki:5.2f}%  中位抽血 {r.med:.0f}  平均PNI {r.pni:.1f}")

print("\n" + "=" * 74)
print("2. 嵌套模型比较（同一样本，避免样本变化与变量效应混淆）")
print("=" * 74)
# 共同样本：所有模型都用同一批人（含全部分母变量）
COMMON = BASE + ["log_visits", "log_items", "log_cr"]
DC = d.dropna(subset=COMMON)
print(f"共同样本 n={len(DC)}  events={int(DC.aki_unified.sum())}")
gC = sm.Logit(DC.aki_unified.astype(int),
              sm.add_constant(DC[B0].astype(float))).fit(disp=0, cov_type="HC0")
ORC = float(np.exp(gC.params["x"]))
print(f"共同样本上的主模型 OR = {ORC:.4f}")

rows = []
def fit_add(cols, tag):
    X = sm.add_constant(DC[cols].astype(float))
    g = sm.Logit(DC.aki_unified.astype(int), X).fit(disp=0, cov_type="HC0")
    bb, ss = g.params["x"], g.bse["x"]
    o = float(np.exp(bb))
    rows.append(dict(adjustment=tag, n=len(DC), events=int(DC.aki_unified.sum()),
                     OR=o, lo=float(np.exp(bb-1.96*ss)), hi=float(np.exp(bb+1.96*ss)),
                     p=float(g.pvalues["x"]), dOR_pct=100*(o-ORC)/ORC))
    return o

fit_add(B0, "none (reference)")
fit_add(B0 + ["log_visits"], "+ log(preop blood-draw visits)")
fit_add(B0 + ["log_items"], "+ log(preop lab items)")
fit_add(B0 + ["log_cr"], "+ log(preop creatinine-testing time points)")

# 五分位因子版本
DC2 = DC.copy()
DC2["vis_q"] = pd.qcut(DC2.n_uniq_time_total.rank(method="first"), 5, labels=False)
qcols = [f"vq{i}" for i in range(1, 5)]
for i in range(1, 5):
    DC2[f"vq{i}"] = (DC2.vis_q == i).astype(float)
X = sm.add_constant(DC2[B0 + qcols].astype(float))
gq = sm.Logit(DC2.aki_unified.astype(int), X).fit(disp=0, cov_type="HC0")
bb, ss = gq.params["x"], gq.bse["x"]
o = float(np.exp(bb))
rows.append(dict(adjustment="+ preop visit QUINTILE factor (4 df)", n=len(DC2),
                 events=int(DC2.aki_unified.sum()), OR=o,
                 lo=float(np.exp(bb-1.96*ss)), hi=float(np.exp(bb+1.96*ss)),
                 p=float(gq.pvalues["x"]), dOR_pct=100*(o-ORC)/ORC))

R = pd.DataFrame(rows)
print("\n" + R.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n" + "=" * 74)
print("3. 稳健性")
print("=" * 74)
# 3a 极端值：剔除术前抽血次数最高 1%
cut = DC.n_uniq_time_total.quantile(0.99)
D1 = DC[DC.n_uniq_time_total <= cut]
g1 = sm.Logit(D1.aki_unified.astype(int),
              sm.add_constant(D1[B0 + ["log_visits"]].astype(float))).fit(disp=0, cov_type="HC0")
print(f"  剔除最高1%(>{cut:.0f}次)后  n={len(D1)}  "
      f"OR={np.exp(g1.params['x']):.4f}  变化 {100*(np.exp(g1.params['x'])-ORC)/ORC:+.1f}%")

# 3b 只保留有术前肌酐者
D2 = DC[DC.n_uniq_time_cr > 0]
g2 = sm.Logit(D2.aki_unified.astype(int),
              sm.add_constant(D2[B0 + ["log_visits"]].astype(float))).fit(disp=0, cov_type="HC0")
print(f"  仅术前查过肌酐者     n={len(D2)}  "
      f"OR={np.exp(g2.params['x']):.4f}  变化 {100*(np.exp(g2.params['x'])-ORC)/ORC:+.1f}%")

# 3c 术前抽血次数本身的效应（它是否独立预测 AKI？）
g3 = sm.Logit(DC.aki_unified.astype(int),
              sm.add_constant(DC[B0 + ["log_visits"]].astype(float))).fit(disp=0, cov_type="HC0")
bv, sv = g3.params["log_visits"], g3.bse["log_visits"]
print(f"\n  术前抽血次数自身的 OR（每 log 单位）: {np.exp(bv):.3f} "
      f"({np.exp(bv-1.96*sv):.3f}–{np.exp(bv+1.96*sv):.3f})  p={g3.pvalues['log_visits']:.3g}")

# 3d 与术后检测密度并列
print("\n" + "=" * 74)
print("4. 与术后检测密度（已发表口径）并列")
print("=" * 74)
print("  术后检测密度五分位校正：见 quintile_stratified_analysis.py 同期输出")
print("    注：该校正存在反向因果风险，不作为偏倚校正值")
print("  术前检测密度校正（本次，无反向因果风险）:")
for _, r in R[R.adjustment != "none (reference)"].iterrows():
    print(f"    {r.adjustment:38s} {ORC:.4f} → {r.OR:.4f}  ({r.dOR_pct:+.1f}%)  p={r.p:.3g}")

R.to_csv(f"{OUT}/preop_density_results.csv", index=False)
print("\n已写出 preop_density_results.csv")
