# -*- coding: utf-8 -*-
"""AKI 口径统一 —— INSPIRE 侧（发现队列）

目标：INSPIRE 与 MOVER 使用同一套 KDIGO 肌酐判定函数。
2026-10-25 起主口径严格执行时间窗：0.3 mg/dL 升高只承认术后 48h
（相对基线），1.5/2/3 倍标准保留术后 7 天窗口。旧的无 48h 版本以
st_unified_no48 留痕，禁止再用于稿件主分析。

统一口径（unified）：
  基线 base = 术前 30 天内最近一次肌酐（相对 opend_time，day ∈ (-30, -0.02)）
  峰值 peak = 术后 0-7 天最大肌酐；p48 = 术后 0-2 天最大肌酐
  acute = peak >= 1.5*base 或 (p48 - base) >= 0.3 mg/dL
  3 期 = peak >= 3*base 或 (peak >= 4.0 mg/dL 且 acute)
  2 期 = peak >= 2*base
  1 期 = acute

输出：
  p0_aki_inspire_harmonized.csv   每人一行，含四种口径分期
  控制台口径对照表 + 排除链 + 最终队列效应量
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import OUT  # noqa: E402
from aki_rolling import judge_aki  # noqa: E402

PD = os.environ.get("INSPIRE_OUT", "output")

# ------------------------------------------------------------------ 数据
base = pd.read_csv(os.path.join(PD, "p0_pool_base.csv"), low_memory=False)
lab = pd.read_csv(os.path.join(PD, "p0_lab_pool.csv"), low_memory=False)
pt = pd.read_csv(os.path.join(PD, "p0_pni_traditional.csv"), low_memory=False)

print("候选池 %d 台手术 / %d 位患者" % (len(base), base.subject_id.nunique()))
print("lab_pool %d 条" % len(lab))

lab["value"] = pd.to_numeric(lab.value, errors="coerce")
lab["chart_time"] = pd.to_numeric(lab.chart_time, errors="coerce")
lab = lab.merge(base[["subject_id", "opend_time"]].drop_duplicates("subject_id"),
                on="subject_id", how="inner")
lab["day"] = (lab.chart_time - lab.opend_time) / 1440.0

# ------------------------------------------------- 术后 0-7 天肌酐（按人）
cr = lab[(lab.item_name == "creatinine") & (lab.day >= -0.02) & (lab.day <= 7)]
cr = cr.dropna(subset=["value"])
cr = cr[cr.value > 0]
print("术后 0-7 天肌酐: %d 条 / %d 人" % (len(cr), cr.subject_id.nunique()))

# 滚动判定需要覆盖每个术后候选时点之前的 7 天（含术前），故取 [-7.02, 7]
cr_hist = lab[(lab.item_name == "creatinine") & (lab.day >= -7.02)
              & (lab.day <= 7)].dropna(subset=["value"])
cr_hist = cr_hist[cr_hist.value > 0]
print("滚动序列（术前7天+术后7天）: %d 条 / %d 人"
      % (len(cr_hist), cr_hist.subject_id.nunique()))

# 敏感性 B 的固定术前基线：术前 30 天内最近一次（与协变量口径一致）
_pre = lab[(lab.item_name == "creatinine") & (lab.day < 0)
           & (lab.day >= -30)].dropna(subset=["value"])
_pre = _pre[_pre.value > 0].sort_values("day")
pb = _pre.groupby("subject_id").value.last()


def stage_harmonized(b, pk):
    """统一口径（KDIGO 2012 官方，0-7 天窗口）"""
    if not (np.isfinite(b) and np.isfinite(pk)) or b <= 0:
        return np.nan
    if pk >= 3.0 * b or pk >= 4.0:
        return 3.0
    if pk >= 2.0 * b:
        return 2.0
    if pk >= 1.5 * b or (pk - b) >= 0.3:
        return 1.0
    return 0.0


def stage_no_abs(b, pk):
    """去掉绝对值 4.0（MOVER 现状等价）"""
    if not (np.isfinite(b) and np.isfinite(pk)) or b <= 0:
        return np.nan
    if pk >= 3.0 * b:
        return 3.0
    if pk >= 2.0 * b:
        return 2.0
    if pk >= 1.5 * b or (pk - b) >= 0.3:
        return 1.0
    return 0.0


def stage_unified_no48(b, pk):
    """旧口径留痕：0.3 mg/dL 错误地应用于整个术后 0–7 天窗口。

    绝对值 4.0 必须附加急性条件：KDIGO stage 3 原文为 "Increase in SCr to
    >=4.0 mg/dL"，关键词是 increase。若不加此条件，慢性终末期肾病
    （基线 Cr 中位 5.25 mg/dL、peak/base 中位 1.17）会被误判为 3 期，
    实测使 MOVER 中重度 AKI 从 7.05% 虚高到 14.28%。
    """
    if not (np.isfinite(b) and np.isfinite(pk)) or b <= 0:
        return np.nan
    if pk >= 3.0 * b:
        return 3.0
    if pk >= 4.0 and (pk >= 1.5 * b or (pk - b) >= 0.3):
        return 3.0
    if pk >= 2.0 * b:
        return 2.0
    if pk >= 1.5 * b or (pk - b) >= 0.3:
        return 1.0
    return 0.0


def stage_unified(b, pk, p48):
    """【最终统一口径】KDIGO：0.3 升高限术后 48h，倍数标准限 7 天。

    绝对值 4.0 必须附加急性条件：KDIGO stage 3 原文为 "Increase in SCr to
    >=4.0 mg/dL"，关键词是 increase。若不加此条件，慢性终末期肾病
    （基线 Cr 中位 5.25 mg/dL、peak/base 中位 1.17）会被误判为 3 期，
    实测使 MOVER 中重度 AKI 从 7.05% 虚高到 14.28%。
    """
    if not (np.isfinite(b) and np.isfinite(pk)) or b <= 0:
        return np.nan
    acute = pk >= 1.5 * b or (np.isfinite(p48) and (p48 - b) >= 0.3)
    if pk >= 3.0 * b:
        return 3.0
    if pk >= 4.0 and acute:
        return 3.0
    if pk >= 2.0 * b:
        return 2.0
    if acute:
        return 1.0
    return 0.0


rows = []
for sid, g in cr_hist.groupby("subject_id", sort=False):
    t = g.day.values
    v = g.value.values
    r_min = judge_aki(t, v, "min")
    r_last = judge_aki(t, v, "last")
    r_base = judge_aki(t, v, "preop_base", preop_base=pb.get(sid, np.nan))
    post = g[(g.day >= -0.02) & (g.day <= 7)]
    vv = post.value.values
    dd = post.day.values
    rows.append(dict(
        subject_id=sid,
        cr_peak=float(np.nanmax(vv)) if len(vv) else np.nan,
        cr_peak48=(float(np.nanmax(vv[dd <= 2]))
                   if len(vv[dd <= 2]) else np.nan),
        cr_n=len(vv),
        st_roll_min=r_min["stage"],
        first_aki_day=r_min["first_time"],
        path_first=r_min["path"],
        basis_first=r_min["basis"],
        n_abs_eval=r_min["n_abs_eval"],
        n_rel_eval=r_min["n_rel_eval"],
        aki_evaluable=r_min["evaluable"],
        st_roll_last=r_last["stage"],
        first_aki_day_last=r_last["first_time"],
        st_preopbase=r_base["stage"],
        first_aki_day_b=r_base["first_time"],
    ))
crt = pd.DataFrame(rows)

# ------------------------------------------------------------ 合并暴露
d = base.drop_duplicates("subject_id").copy()
d = d.merge(crt, on="subject_id", how="left")
d = d.merge(pt[["subject_id", "alb", "lym", "wbc", "cr", "al_c",
                "pni_onodera", "pni_simple"]], on="subject_id", how="left")

# 四种口径
def stage_current(b, pk, p48):
    """INSPIRE 现状（带 48h 窗口 + 4.0），与 1.3 桥接口径"""
    if not (np.isfinite(b) and np.isfinite(pk)) or b <= 0:
        return np.nan
    if pk >= 4.0 or pk >= 3 * b or (np.isfinite(p48) and p48 >= 3 * b):
        return 3.0
    if pk >= 2 * b:
        return 2.0
    if ((pk - b) >= 0.3 or (np.isfinite(p48) and (p48 - b) >= 0.3)
            or pk >= 1.5 * b):
        return 1.0
    return 0.0


d["st_current"] = [stage_current(b, p, q) for b, p, q in
                   zip(d.cr, d.cr_peak, d.cr_peak48)]
d["st_harmonized"] = [stage_harmonized(b, p) for b, p in
                      zip(d.cr, d.cr_peak)]
d["st_no_abs"] = [stage_no_abs(b, p) for b, p in zip(d.cr, d.cr_peak)]
# 主分析：滚动 7 天最低值参照（aki_rolling.judge_aki）
d["st_unified"] = d["st_roll_min"]
d["st_unified_no48"] = [stage_unified_no48(b, p) for b, p in
                        zip(d.cr, d.cr_peak)]

for c, s in [("current", "st_current"), ("harmonized", "st_harmonized"),
             ("no_abs", "st_no_abs"), ("unified", "st_unified"),
             ("unified_no48", "st_unified_no48"),
             ("roll_last", "st_roll_last"),
             ("preopbase", "st_preopbase")]:
    d["aki_" + c] = (d[s] >= 1).astype(float)
    d.loc[d[s].isna(), "aki_" + c] = np.nan
    d["sev_" + c] = (d[s] >= 2).astype(float)
    d.loc[d[s].isna(), "sev_" + c] = np.nan

print()
print("=" * 78)
print("口径对照：候选池 %d 例中可判定者" % len(d))
print("=" * 78)
print("%-14s %8s %10s %10s   %s" % ("口径", "可判定", "AKI", "中重度", "分期分布"))
for c in ["unified", "roll_last", "preopbase", "unified_no48"]:
    j = d[d["st_" + c].notna()]
    vc = j["st_" + c].value_counts().sort_index()
    dist = " ".join("%d期=%d" % (int(k), v) for k, v in vc.items())
    print("%-14s %8d %9.2f%% %9.2f%%   %s" % (
        c, len(j), 100 * j["aki_" + c].mean(),
        100 * j["sev_" + c].mean(), dist))

print()
print("参照规则对照（主分析 min vs 敏感性 last vs 术前基线 B）：")
for a, b in [("unified", "roll_last"), ("unified", "preopbase")]:
    cc = d[d["st_" + a].notna() & d["st_" + b].notna()]
    chg = cc[cc["st_" + a] != cc["st_" + b]]
    print("  %s vs %s: 分期改变 %d 例 (%.2f%%)，降级 %d / 升级 %d"
          % (a, b, len(chg), 100 * len(chg) / max(len(cc), 1),
             (chg["st_" + a] > chg["st_" + b]).sum(),
             (chg["st_" + a] < chg["st_" + b]).sum()))
print("  可判定路径覆盖：无任何可判定窗口者 %d 例"
      % int((d.aki_evaluable == 0).sum()))

# ------------------------------------------------------------ 排除链
d["eskd"] = (d.cr >= 3.0).astype(int)
d["ltx"] = d.pcs.astype(str).str.startswith("0FY").astype(int)
d = d.sort_values("opstart_time")
d["rn"] = d.groupby("subject_id").cumcount()

d["elig"] = d.pni_onodera.notna() & d.cr.notna() & d.st_unified.notna()

print()
print("=" * 78)
print("排除链（统一口径 unified）")
print("=" * 78)
steps = [("起点：新入组标准（术前PNI+术前Cr+术后Cr）", d.elig)]
s1 = d.elig & d.eskd.eq(0)
steps.append(("  + 排除 ESKD 候选（术前 Cr>=3.0）", s1))
s2 = s1 & d.ltx.eq(0)
steps.append(("  + 排除肝移植（PCS 0FY）", s2))
s3 = s2 & d.rn.eq(0)
steps.append(("  + 每患者仅保留首次手术", s3))
for nm, s in steps:
    j = d[s]
    print("  %-38s N=%-5d AKI=%-4d (%5.2f%%) 中重度=%-4d (%5.2f%%)" % (
        nm, len(j), j.aki_unified.sum(), 100 * j.aki_unified.mean(),
        j.sev_unified.sum(), 100 * j.sev_unified.mean()))

FINAL = s3
d.to_csv(os.path.join(PD, "p0_aki_inspire_harmonized.csv"), index=False)
print("\nsaved p0_aki_inspire_harmonized.csv")

# ------------------------------------------------------------ 效应量
f = d[FINAL].copy()
f["op_duration"] = f.opend_time - f.opstart_time
f["open_"] = (f.pcs.astype(str).str[4] == "0").astype(int)
f["emop_"] = (pd.to_numeric(f.emop, errors="coerce") == 1).astype(int)
f["asa_n"] = pd.to_numeric(f.asa, errors="coerce")
f["sex_m"] = (f.sex == "M").astype(int)
f["cr_pre"] = f.cr
COVS = ["age", "sex_m", "asa_n", "op_duration", "cr_pre", "open_", "emop_"]

print()
print("=" * 78)
print("最终分析队列效应量（统一口径 harmonized）")
print("=" * 78)
SD_INSPIRE = {}
for key, lab_ in [("pni_onodera", "传统 Onodera PNI"),
                  ("pni_simple", "简化 PNI(alb+0.05*lym%)")]:
    dd = f[[key] + COVS + ["aki_unified", "sev_unified"]].dropna()
    sd = dd[key].std()
    SD_INSPIRE[key] = sd
    print("\n【%s】N=%d 事件=%d (%.2f%%) 中重度=%d (%.2f%%) EPV=%.1f  SD=%.3f" % (
        lab_, len(dd), dd.aki_unified.sum(),
        100 * dd.aki_unified.mean(), dd.sev_unified.sum(),
        100 * dd.sev_unified.mean(), dd.aki_unified.sum() / 8, sd))
    for yv, yl in [("aki_unified", "主要 AKI"), ("sev_unified", "中重度 AKI")]:
        X = sm.add_constant(dd[[key] + COVS])
        r = sm.Logit(dd[yv].values, X).fit(disp=0, maxiter=200)
        b, se = r.params[key] * (-sd), r.bse[key] * sd
        print("   %-10s 每 -1SD OR=%.3f (%.3f-%.3f)  p=%.2e" % (
            yl, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se),
            r.pvalues[key]))
    dd2 = dd.copy()
    dd2["qq"] = pd.qcut(dd2[key], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    g = dd2.groupby("qq", observed=True).agg(
        n=("aki_unified", "size"), ev=("aki_unified", "sum"),
        sev=("sev_unified", "sum"))
    g["rate%"] = (100 * g.ev / g.n).round(2)
    print(g.to_string())
    print("   最小层事件=%d  最小层 severe=%d" % (g.ev.min(), g.sev.min()))

print()
print("=" * 78)
print("供 MOVER 冻结的 INSPIRE SD：")
for k, v in SD_INSPIRE.items():
    print("  %-14s SD = %.6f" % (k, v))
