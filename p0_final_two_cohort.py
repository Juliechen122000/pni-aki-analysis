# -*- coding: utf-8 -*-
"""INSPIRE / MOVER 两国最终对照（统一口径）

在 AKI 判定函数完全相同、排除项完全对称的前提下，
产出投稿级的 Table 1 / 纳排漏斗 / 效应量对照。
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

PD = os.environ.get("INSPIRE_OUT", "output")
SD_FROZEN = {"pni_onodera": 6.445246, "pni_simple": 0.773313}

# ------------------------------------------------------------------ 加载
I = pd.read_csv(os.path.join(PD, "p0_aki_inspire_harmonized.csv"), low_memory=False)
M = pd.read_csv(os.path.join(PD, "p0_mover_rebuilt.csv"), low_memory=False)

I["eskd"] = (I.cr >= 3.0).astype(int)
I["ltx"] = I.pcs.astype(str).str.startswith("0FY").astype(int)
I["op_duration"] = I.opend_time - I.opstart_time
I["open_"] = (I.pcs.astype(str).str[4] == "0").astype(int)
I["emop_"] = (pd.to_numeric(I.emop, errors="coerce") == 1).astype(int)
I["asa_n"] = pd.to_numeric(I.asa, errors="coerce")
I["sex_m"] = (I.sex == "M").astype(int)
I["cr_pre"] = I.cr
FIN_I = (I.pni_onodera.notna() & I.cr.notna() & I.st_unified.notna()
         & I.eskd.eq(0) & I.ltx.eq(0))

M["asa_n"] = pd.to_numeric(M.ASA_RATING_C, errors="coerce")
M["sex_m"] = (M.SEX.astype(str).str.upper().str.startswith("M")).astype(int)
M["or_min_n"] = pd.to_numeric(M.or_min, errors="coerce")
M["cr_pre"] = M.cr_base
FIN_M = (M.pni_onodera.notna() & M.cr_base.notna() & M.st_unified.notna()
         & M.eskd.eq(0))

di = I[FIN_I].copy()
dm = M[FIN_M].copy()

print("=" * 84)
print("表 1：两国队列对照（统一 KDIGO 口径）")
print("=" * 84)
rows = [
    ("数据平台", "INSPIRE v1.4.2（韩国 SNUH）", "MOVER / EPIC（美国 UCI）"),
    ("定位", "发现队列", "外部验证队列"),
    ("原始就诊", "130,960 台手术", "64,354 例就诊 / 39,683 位患者"),
    ("① 腹部（ICD-10-PCS 0D/0F）", "17,796", "15,503（就诊级）"),
    ("② 手术时长 ≥120 min", "10,442 台 / 10,051 人", "9,022 例 / 5,710 位"),
    ("③ 每例仅保留索引手术", "10,051", "5,710"),
    ("④ 术前PNI ∩ 术前Cr ∩ 术后Cr", "%d" % int((I.pni_onodera.notna() & I.cr.notna() & I.st_unified.notna()).sum()),
     "%d" % int((M.pni_onodera.notna() & M.cr_base.notna() & M.st_unified.notna()).sum())),
    ("⑤ 排除 ESKD（术前 Cr≥3.0）", "%d" % int(FIN_I.sum()), "%d" % int(FIN_M.sum())),
    ("⑥ 排除肝移植（PCS 0FY）", "含于上一行", "数据不可识别"),
    ("", "", ""),
    ("【分析样本】", "N = %s" % f"{len(di):,}", "N = %s" % f"{len(dm):,}"),
    ("年龄（岁，中位）", "%.0f" % di.age.median(), "去标识化不可用"),
    ("男性（%%）", "%.1f" % (100 * di.sex_m.mean()), "%.1f" % (100 * dm.sex_m.mean())),
    ("ASA ≥3（%%）", "%.1f" % (100 * (di.asa_n >= 3).mean()),
     "%.1f" % (100 * (dm.asa_n >= 3).mean())),
    ("手术时长（min，中位）", "%.0f" % di.op_duration.median(), "%.0f" % dm.or_min_n.median()),
    ("开放入路（%%）", "%.1f" % (100 * di.open_.mean()), "数据不可识别"),
    ("急诊（%%）", "%.1f" % (100 * di.emop_.mean()), "数据不可识别"),
    ("术前肌酐（mg/dL，中位）", "%.2f" % di.cr_pre.median(), "%.2f" % dm.cr_pre.median()),
    ("", "", ""),
    ("【暴露】术前 PNI（传统 Onodera）", "中位 %.1f (SD %.2f)" % (di.pni_onodera.median(), di.pni_onodera.std()),
     "中位 %.1f (SD %.2f)" % (dm.pni_onodera.median(), dm.pni_onodera.std())),
    ("【结局】AKI 全部期", "%d (%.2f%%)" % (di.aki_unified.sum(), 100 * di.aki_unified.mean()),
     "%d (%.2f%%)" % (dm.aki_unified.sum(), 100 * dm.aki_unified.mean())),
    ("【结局】中重度（≥2 期）", "%d (%.2f%%)" % (di.sev_unified.sum(), 100 * di.sev_unified.mean()),
     "%d (%.2f%%)" % (dm.sev_unified.sum(), 100 * dm.sev_unified.mean())),
]
w = max(len(r[0]) for r in rows)
print("%-*s  %-34s  %-34s" % (w, "项目", "INSPIRE（发现）", "MOVER（验证）"))
print("-" * (w + 72))
for a, b, c in rows:
    print("%-*s  %-34s  %-34s" % (w, a, b, c))

# ------------------------------------------------------------ 效应量
print()
print("=" * 84)
print("表 2：主暴露效应量（术前 PNI，每 −1 个 INSPIRE-SD；统一校正变量集）")
print("=" * 84)
COV_SPEC_A = ["asa_n", "sex_m", "op_duration", "cr_pre"]     # 两国对称集
COV_M = ["asa_n", "sex_m", "or_min_n", "cr_pre"]
# INSPIRE 完整集（Spec B，仅发现队列可得）
COV_SPEC_B = ["age", "sex_m", "asa_n", "op_duration", "cr_pre",
              "open_", "emop_"]


def ors(dd, key, cov, ycol):
    X = sm.add_constant(dd[[key] + cov])
    r = sm.Logit(dd[ycol].values, X).fit(disp=0, maxiter=200, cov_type="HC0")
    b, se = r.params[key], r.bse[key]
    sdF = SD_FROZEN[key]
    sdO = dd[key].std()
    # beta<0（PNI 高 → AKI 少），故 -1SD 的 OR = exp(b*-sd)
    lo = np.exp((b + 1.96 * se) * -sdF)
    hi = np.exp((b - 1.96 * se) * -sdF)
    return (np.exp(b * -sdF), lo, hi, np.exp(b * -sdO), r.pvalues[key])


for key, lab_ in [("pni_onodera", "传统 Onodera PNI"),
                  ("pni_simple", "简化 PNI")]:
    print("\n【%s】INSPIRE SD = %.4f（冻结）" % (lab_, SD_FROZEN[key]))
    print("%-26s %8s %8s %7s   %-24s   %-9s" % (
        "队列 / 协变量集", "N", "事件", "EPV", "OR 每 −1SD (95%CI)", "自身SD OR"))
    print("-" * 92)
    specs = [("INSPIRE  Spec A 对称集", di, COV_SPEC_A),
             ("INSPIRE  Spec B 完整集", di, COV_SPEC_B),
             ("MOVER    Spec A 对称集", dm, COV_M)]
    for nm, d, cov in specs:
        dd = d[[key] + cov + ["aki_unified", "sev_unified"]].dropna()
        o, lo, hi, oo, p = ors(dd, key, cov, "aki_unified")
        print("%-26s %8s %8d %7.1f   %.3f (%.3f–%.3f)      %.3f   p=%.2e" % (
            nm, f"{len(dd):,}", int(dd.aki_unified.sum()),
            dd.aki_unified.sum() / (len(cov) + 1), o, lo, hi, oo, p))

    print("  中重度 AKI（≥2 期）:")
    for nm, d, cov in specs:
        dd = d[[key] + cov + ["sev_unified"]].dropna()
        o, lo, hi, oo, p = ors(dd, key, cov, "sev_unified")
        print("    %-22s n=%-7s 事件=%-4d OR=%.3f (%.3f–%.3f)  p=%.2e" % (
            nm, f"{len(dd):,}", int(dd.sev_unified.sum()), o, lo, hi, p))

# ------------------------------------------------------------ 梯度
print()
print("=" * 84)
print("表 3：PNI 四分位梯度（各自四分位切点）")
print("=" * 84)
for key, lab_ in [("pni_onodera", "传统 Onodera PNI"),
                  ("pni_simple", "简化 PNI")]:
    print("\n【%s】" % lab_)
    out = {}
    for nm, d in [("INSPIRE", di), ("MOVER", dm)]:
        dd = d[[key, "aki_unified", "sev_unified"]].dropna().copy()
        dd["qq"] = pd.qcut(dd[key], 4, labels=["Q1(最低)", "Q2", "Q3", "Q4(最高)"])
        g = dd.groupby("qq", observed=True).agg(
            n=("aki_unified", "size"), ev=("aki_unified", "sum"),
            sev=("sev_unified", "sum"))
        g["rate%"] = (100 * g.ev / g.n).round(2)
        out[nm] = g
    t = pd.concat(out, axis=1)
    print(t.to_string())
    for nm in ["INSPIRE", "MOVER"]:
        g = out[nm]
        print("  %-8s 梯度 Q1/Q4 = %.2f 倍   最小层事件=%d 最小层severe=%d" % (
            nm, g["rate%"].iloc[0] / max(g["rate%"].iloc[-1], 1e-9),
            g.ev.min(), g.sev.min()))

# ------------------------------------------------- 纳入 vs 排除（MOVER）
print()
print("=" * 84)
print("表 4：MOVER 入组者 vs 排除者（可得率仅 %.1f%%，必须报告）"
      % (100 * M.pni_onodera.notna().mean()))
print("=" * 84)
M["elig"] = FIN_M
for k in [1, 0]:
    s = M[M.elig == k]
    j = s[s.st_unified.notna()]
    print("  %s n=%-5d 男=%.1f%% ASA>=3=%.1f%% 时长中位=%.0fmin | 可判定AKI=%-5d AKI=%.2f%% 中重度=%.2f%%" % (
        "入组" if k else "排除", len(s), 100 * s.sex_m.mean(),
        100 * (pd.to_numeric(s.ASA_RATING_C, errors="coerce") >= 3).mean(),
        pd.to_numeric(s.or_min, errors="coerce").median(), len(j),
        100 * j.aki_unified.mean() if len(j) else np.nan,
        100 * j.sev_unified.mean() if len(j) else np.nan))

print()
print("=" * 84)
print("表 5：ASA 分布对照（两国差异巨大，需解释）")
print("=" * 84)
print("  INSPIRE ASA 原始分布:")
print(di.asa_n.value_counts(dropna=False).sort_index().to_string())
print("  MOVER ASA 原始分布:")
print(dm.asa_n.value_counts(dropna=False).sort_index().to_string())

print()
print("=" * 84)
print("INSPIRE 侧对照（入组率 %.1f%%）" % (100 * FIN_I.mean()))
print("=" * 84)
for k in [1, 0]:
    s = I[FIN_I == k] if k else I[~FIN_I]
    j = s[s.st_unified.notna()]
    print("  %s n=%-5d 年龄中位=%.0f 男=%.1f%% 急诊=%.1f%% | 可判定AKI=%-5d AKI=%.2f%% 中重度=%.2f%%" % (
        "入组" if k else "排除", len(s), s.age.median(), 100 * s.sex_m.mean(),
        100 * s.emop_.mean(), len(j),
        100 * j.aki_unified.mean() if len(j) else np.nan,
        100 * j.sev_unified.mean() if len(j) else np.nan))
