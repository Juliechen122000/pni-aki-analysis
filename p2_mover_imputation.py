# -*- coding: utf-8 -*-
"""
P2：MOVER 术前 PNI 缺失的多重插补敏感性分析

问题
----
MOVER 重建队列 5,710 例中，术前 PNI 可得仅 1,432 例（25.1%）；
排除 ESKD 后分析样本 1,338 例。可得率远低于 INSPIRE 的 90.6%。
原因是美国 UCI 不常规检查术前白蛋白 —— 缺失与"是否被安排检查"有关，
属 MAR 而非 MCAR，必须用多重插补 + Rubin 合并做敏感性分析。

注意：MOVER 侧待插补的还包括 albumin/淋巴细胞%/ALC 三个成分，
但主暴露是 pni_onodera，直接插补复合指标即可（避免插补-变换偏差）。

方法
----
1. 完整病例分析（CCA）：现有 1,338 例
2. 多重插补（MI，m = 20）：
   插补模型 pni_onodera ~ ASA + 性别 + 手术时长 + 术前Cr + AKI + 中重度
   （插补模型必须含结局，否则会低估暴露-结局关联）
   随机回归插补：预测值 + 从残差分布抽样的扰动
3. Rubin 规则合并：Qbar, within/between variance, 自由度校正
4. 报告 CCA vs MI 的 OR 差异

局限（须写入 Limitations）
--------------------------
缺失率 74.9%，MAR 假设无法验证。MI 结果只能说明"在 MAR 假设下结论不变"，
不能证明缺失机制无害。

输出
----
  p2_mover_mi.csv     CCA vs MI 的效应量对照
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import OUT  # noqa: E402

M_IMP = 20
SEED = 20261023
SD_FROZEN = 6.4452          # INSPIRE 冻结的 pni_onodera SD


def logit_or(d, key, cov, ycol, sd_scale=None):
    """拟合 logistic，返回以 -sd_scale 为单位的 log-odds 系数与 SE"""
    dd = d[[key] + cov + [ycol]].dropna()
    if len(dd) < 100 or dd[ycol].sum() < 20:
        return None
    X = sm.add_constant(dd[[key] + cov])
    try:
        r = sm.Logit(dd[ycol].values.astype(float), X).fit(disp=0, maxiter=200, cov_type="HC0")
    except Exception:
        return None
    sd = dd[key].std() if sd_scale is None else sd_scale
    b = r.params[key] * -sd
    se = r.bse[key] * sd
    return dict(n=len(dd), events=int(dd[ycol].sum()),
                logodds=b, se=se, or_=np.exp(b))


def rubin(ests):
    """Rubin 规则合并 m 个插补数据集的估计"""
    q = np.array([e["logodds"] for e in ests])
    u = np.array([e["se"] ** 2 for e in ests])
    m = len(q)
    qbar = q.mean()
    ubar = u.mean()
    b = ((q - qbar) ** 2).sum() / (m - 1)          # between-imputation variance
    t = ubar + (1 + 1.0 / m) * b                    # total variance
    lam = ((1 + 1.0 / m) * b) / t                   # 因缺失损失的信息比例
    df_old = (m - 1) / (lam ** 2) if lam > 0 else np.inf
    df_adj = 1.0 / (lam ** 2 / (m - 1) + 1.0 / max(
        np.mean([e["n"] for e in ests]) - 8, 1)) if lam > 0 else np.inf
    return dict(or_=np.exp(qbar), lo=np.exp(qbar - 1.96 * np.sqrt(t)),
                hi=np.exp(qbar + 1.96 * np.sqrt(t)),
                se=np.sqrt(t), fmi=lam, m=m)


def main():
    print("=" * 78)
    print("P2  MOVER 术前 PNI 缺失的多重插补敏感性分析")
    print("=" * 78)

    M = pd.read_csv(os.path.join(OUT, "p0_mover_rebuilt.csv"), low_memory=False)
    M["asa_n"] = pd.to_numeric(M["ASA_RATING_C"], errors="coerce")
    M["sex_m"] = (M["SEX"].astype(str).str.upper().str.startswith("M")).astype(float)
    M["or_min_n"] = pd.to_numeric(M["or_min"], errors="coerce")
    M["cr_pre"] = pd.to_numeric(M["cr_base"], errors="coerce")
    M = M.rename(columns={"pni_onodera": "pni_pre"})

    # 分析池：AKI 可判定（有基线 Cr 与术后 Cr）且非 ESKD
    pool = M[M.st_unified.notna() & M.cr_pre.notna() & M.eskd.eq(0)].copy()
    print("MOVER 分析池（AKI 可判定 & 非 ESKD）N = %d" % len(pool))
    print("  术前 PNI 可得 %d (%.1f%%)；缺失 %d (%.1f%%)"
          % (pool.pni_pre.notna().sum(), 100 * pool.pni_pre.notna().mean(),
             pool.pni_pre.isna().sum(), 100 * pool.pni_pre.isna().mean()))

    COV = ["asa_n", "sex_m", "or_min_n", "cr_pre"]
    Y = "aki_unified"

    # ---------- 1. 完整病例 ----------
    cca = logit_or(pool, "pni_pre", COV, Y, sd_scale=SD_FROZEN)
    print()
    print("=== 完整病例分析（CCA）===")
    print("  N = %d  事件 = %d  OR 每 -1SD = %.3f (%.3f-%.3f)"
          % (cca["n"], cca["events"], cca["or_"],
             np.exp(cca["logodds"] - 1.96 * cca["se"]),
             np.exp(cca["logodds"] + 1.96 * cca["se"])))

    # ---------- 2. 多重插补 ----------
    print()
    print("=== 多重插补（m = %d）===" % M_IMP)
    # 插补模型：PNI ~ 协变量 + 结局 + 中重度（结局必须进入，否则衰减暴露-结局关联）
    IMP_PRED = COV + ["aki_unified", "sev_unified"]   # 插补模型的预测变量
    dd = pool[["pni_pre"] + IMP_PRED].copy()

    # 拟合插补模型要求预测变量完整；ASA 有约 1% 缺失，此处剔除
    obs = dd[dd.pni_pre.notna()].dropna(subset=IMP_PRED)
    mis_idx = dd.index[dd.pni_pre.isna()]
    print("  插补模型：pni_pre ~ %s" % " + ".join(IMP_PRED))
    print("  用于拟合插补模型的观测例数 = %d，待插补 %d 例" % (len(obs), len(mis_idx)))

    Xo = sm.add_constant(obs[IMP_PRED]).astype(float)
    fit = sm.OLS(obs["pni_pre"].astype(float), Xo).fit()
    resid = fit.resid
    print("  插补模型 R2 = %.3f  残差 SD = %.3f"
          % (fit.rsquared, resid.std()))

    # 预测时若有预测变量缺失，用观测集的中位数填充（否则无法给出插补值）
    Xm = dd.loc[mis_idx, IMP_PRED].astype(float)
    med = Xo.median()
    n_fill = int(Xm.isna().any(axis=1).sum())
    if n_fill:
        print("  待插补者中有 %d 例预测变量缺失，用中位数填充" % n_fill)
    Xm = Xm.fillna(med)
    rng = np.random.default_rng(SEED)
    pred = fit.predict(sm.add_constant(Xm).astype(float))

    ests = []
    for k in range(M_IMP):
        imp = pool.copy()
        draw = pred + rng.choice(resid, size=len(pred), replace=True)
        imp.loc[mis_idx, "pni_pre"] = draw.values
        e = logit_or(imp, "pni_pre", COV, Y, sd_scale=SD_FROZEN)
        if e is not None:
            ests.append(e)
    if len(ests) < 5:
        print("  插补后拟合失败次数过多，中止")
        return

    R = rubin(ests)
    print()
    print("=== 合并结果（Rubin 规则）===")
    print("  N = %d  事件 = %d（每份插补集）" % (ests[0]["n"], ests[0]["events"]))
    print("  OR 每 -1SD = %.3f (%.3f-%.3f)" % (R["or_"], R["lo"], R["hi"]))
    print("  缺失信息比例 FMI = %.3f（%.1f%% 的信息因缺失损失）"
          % (R["fmi"], 100 * R["fmi"]))

    # ---------- 3. 对照 ----------
    print()
    print("=== CCA vs MI 对照 ===")
    or_cca = cca["or_"]
    print("  CCA  OR = %.3f (%.3f-%.3f)" % (or_cca,
          np.exp(cca["logodds"] - 1.96 * cca["se"]),
          np.exp(cca["logodds"] + 1.96 * cca["se"])))
    print("  MI   OR = %.3f (%.3f-%.3f)" % (R["or_"], R["lo"], R["hi"]))
    delta = 100 * (R["or_"] - or_cca) / (or_cca - 1) if or_cca != 1 else np.nan
    print("  相对差异（以 OR-1 计）= %.1f%%" % delta)
    same = (R["lo"] <= or_cca <= R["hi"])
    print("  CCA 点估计是否落在 MI 的 95%%CI 内：%s" % ("是" if same else "否"))

    rows = [dict(analysis="CCA", n=cca["n"], events=cca["events"],
                 or_=or_cca, lo=np.exp(cca["logodds"] - 1.96 * cca["se"]),
                 hi=np.exp(cca["logodds"] + 1.96 * cca["se"]),
                 fmi=np.nan, note="complete case"),
            dict(analysis="MI", n=ests[0]["n"], events=ests[0]["events"],
                 or_=R["or_"], lo=R["lo"], hi=R["hi"], fmi=R["fmi"],
                 note="m=%d, Rubin" % M_IMP)]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "p2_mover_mi.csv"), index=False,
                              encoding="ascii", errors="ignore")
    print()
    print("输出：p2_mover_mi.csv")


if __name__ == "__main__":
    main()
