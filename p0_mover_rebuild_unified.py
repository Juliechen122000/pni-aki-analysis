# -*- coding: utf-8 -*-
"""MOVER 队列重建 + AKI 口径统一（外部验证队列，美国 UCI/EPIC）

三件必须同时解决的事：

【1】队列重建
  现役 mover_c2_traj.csv (2,150) 是按"轨迹合格(≥3 时段术后 PNI)"建的，
  而 INSPIRE 新入组标准已改为"术前 PNI"，两国入组条件不对称。
  重建为与 INSPIRE 对称的口径：
    腹部 PCS 0D/0F ∩ 手术时长 ≥120 min  ∩  每 MRN 仅首次手术
  （MOVER 的 age_deid 与 BIRTH_DATE 均已去标识化，SD 仅 1.2 年，
    无法施加 18-90 岁限制，也无法用于年龄校正 —— 见局限性）

【2】AKI 口径统一
  统一判定函数（与 INSPIRE 完全相同；2026-10-25 起执行 48h）：
    基线 base = 术前 30 天内最近一次肌酐
    峰值 pk = 术后 0-7 天最大肌酐；p48 = 术后 0-2 天最大肌酐
    acute = pk ≥ 1.5×base 或 p48−base ≥ 0.3
    3 期 = pk ≥ 3×base 或 (pk ≥ 4.0 且 acute)
    2 期 = pk ≥ 2×base
    1 期 = acute；旧 7 天绝对增量口径以 st_unified_no48 留痕
  【关键】绝对值 4.0 必须附加急性条件：否则慢性终末期肾病
    （基线 Cr 中位 5.25、ratio 中位 1.17）会被误判为 3 期，
    实测使 MOVER 中重度 AKI 从 7.05% 虚高到 14.28%。

【3】ESKD 排除（两国同式）
  术前肌酐 ≥3.0 mg/dL 者排除（INSPIRE 侧已执行，MOVER 侧此前缺失）

输出：
  p0_mover_rebuilt.csv            重建队列 + 暴露 + 结局
  控制台：队列漏斗、口径对照、效应量（冻结 INSPIRE SD）
"""
import os
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

from aki_rolling import judge_aki

PD = os.environ.get("INSPIRE_OUT", "output")
LAB = os.environ.get("MOVER_LABS", "./MOVER/patient_labs.csv")
CACHE = os.path.join(PD, "p0_mover_lab_rebuilt.csv")
OUT = os.path.join(PD, "p0_mover_rebuilt.csv")

t0 = time.time()

# INSPIRE 侧冻结的 SD（由 p0_aki_harmonize_inspire.py 在最终队列上算出）
SD_FROZEN = {"pni_onodera": 6.445246, "pni_simple": 0.773313}

PCT_CODE = {"736-9", "19644-3"}          # 淋巴细胞百分比
ABS_CODE = {"731-0", "26474-7"}          # 淋巴细胞绝对值 (ALC)
ALB_CODE = {"1751-7"}                    # 白蛋白 g/dL
CR_CODE = {"2160-0"}                     # 肌酐 mg/dL
ALL_CODE = PCT_CODE | ABS_CODE | ALB_CODE | CR_CODE

# ------------------------------------------------------------------ 队列
info = pd.read_pickle(os.path.join(PD, "mover_info_cohort.pkl"))
print("MOVER 原始就诊 %d 例 / %d 位患者" % (len(info), info.MRN.nunique()))

or_min = pd.to_numeric(info.or_min, errors="coerce")
m = info.abd_pcs & (or_min >= 120)
coh = info[m].copy()
print("① 腹部 PCS 0D/0F ∩ 时长 ≥120 min : %d 例 / %d 位患者" % (
    len(coh), coh.MRN.nunique()))

coh["surg_in"] = pd.to_datetime(coh["IN_OR_DTTM"], errors="coerce")
coh["surg_out"] = pd.to_datetime(coh["OUT_OR_DTTM"], errors="coerce")
coh = coh[coh.surg_in.notna()].copy()
print("② 手术时间可解析 : %d 例" % len(coh))

coh = coh.sort_values("surg_in")
coh["rn"] = coh.groupby("MRN").cumcount()
coh = coh[coh.rn == 0].copy()
print("③ 每 MRN 仅保留首次手术 : %d 例" % len(coh))

keep_log = set(coh.LOG_ID)

# ------------------------------------------------------- 扫描 labs
if os.path.exists(CACHE):
    print("复用缓存", CACHE)
    lab = pd.read_csv(CACHE, low_memory=False)
else:
    print("扫描 %s（取 %d 个 Lab Code）..." % (LAB, len(ALL_CODE)))
    rows = []
    total = 0
    for ch in pd.read_csv(LAB, chunksize=500_000, low_memory=False):
        total += len(ch)
        msk = ch["Lab Code"].astype(str).str.strip().isin(ALL_CODE)
        if msk.any():
            s = ch.loc[msk, ["LOG_ID", "Lab Code", "Observation Value",
                             "Measurement Units", "Collection Datetime"]]
            s = s[s["LOG_ID"].isin(keep_log)]
            if len(s):
                rows.append(s)
    print("  扫描 %s 行，保留 %d 条" % (f"{total:,}", sum(len(r) for r in rows)))
    lab = pd.concat(rows, ignore_index=True)
    lab.to_csv(CACHE, index=False)
    print("  缓存已写入", CACHE)

lab["val"] = pd.to_numeric(lab["Observation Value"], errors="coerce")
lab["dt"] = pd.to_datetime(lab["Collection Datetime"], errors="coerce")
lab["code"] = lab["Lab Code"].astype(str).str.strip()
lab = lab.dropna(subset=["val", "dt", "code"])

print()
print("=== 各 Lab Code 原始取回情况 ===")
for cd in sorted(ALL_CODE):
    s = lab.loc[lab.code == cd, "val"]
    if len(s):
        print("  %-10s n=%-7d 中位=%-8.2f 四分位=%.2f/%.2f/%.2f 范围=%.2f~%.2f" % (
            cd, len(s), s.median(), s.quantile(.25), s.median(),
            s.quantile(.75), s.min(), s.max()))

# 清洗
lab = lab[lab.val > 0]
lab.loc[lab.code.isin(ALB_CODE), "val"] = np.where(
    lab.loc[lab.code.isin(ALB_CODE), "val"].between(1.0, 6.0),
    lab.loc[lab.code.isin(ALB_CODE), "val"], np.nan)
lab.loc[lab.code.isin(PCT_CODE), "val"] = np.where(
    lab.loc[lab.code.isin(PCT_CODE), "val"].between(0.5, 90.0),
    lab.loc[lab.code.isin(PCT_CODE), "val"], np.nan)
lab.loc[lab.code.isin(ABS_CODE), "val"] = np.where(
    lab.loc[lab.code.isin(ABS_CODE), "val"].between(0.05, 20.0),
    lab.loc[lab.code.isin(ABS_CODE), "val"], np.nan)
lab.loc[lab.code.isin(CR_CODE), "val"] = np.where(
    lab.loc[lab.code.isin(CR_CODE), "val"].between(0.1, 15.0),
    lab.loc[lab.code.isin(CR_CODE), "val"], np.nan)
lab = lab.dropna(subset=["val"])

lab = lab.merge(coh[["LOG_ID", "surg_out", "surg_in"]], on="LOG_ID", how="inner")
lab["surg"] = lab.surg_out
lab.loc[lab.surg.isna(), "surg"] = lab["surg_in"].loc[lab.surg.isna()]
lab["day"] = (lab.dt - lab.surg).dt.total_seconds() / 86400.0


def pre_last(codes, lo=-30.0):
    s = lab[lab.code.isin(codes) & (lab.day < 0) & (lab.day > lo)]
    if not len(s):
        return pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]")
    idx = s.groupby("LOG_ID").day.idxmax()
    r = s.loc[idx]
    return (r.set_index("LOG_ID").val.rename("v"),
            r.set_index("LOG_ID").dt.rename("t"))


alb_v, alb_t = pre_last(ALB_CODE)
lym_v, lym_t = pre_last(PCT_CODE)
alc_v, alc_t = pre_last(ABS_CODE)
crb_v, crb_t = pre_last(CR_CODE)

pre = pd.concat([alb_v.rename("alb"), lym_v.rename("lym_pct"),
                 alc_v.rename("alc"), crb_v.rename("cr_base"),
                 alb_t.rename("t_alb"), lym_t.rename("t_lym")], axis=1)

# 术后 0-7 天肌酐峰值
post = lab[lab.code.isin(CR_CODE) & (lab.day >= -0.02) & (lab.day <= 7)]
pk = post.groupby("LOG_ID").val.max().rename("cr_peak")
p48 = post[post.day <= 2].groupby("LOG_ID").val.max().rename("cr_peak48")
pn = post.groupby("LOG_ID").val.count().rename("cr_n")

# 滚动判定序列：覆盖每个术后候选之前 7 天（含术前）
cr_hist = lab[lab.code.isin(CR_CODE) & (lab.day >= -7.02) & (lab.day <= 7)]
_pb = (lab[lab.code.isin(CR_CODE) & (lab.day < 0) & (lab.day >= -30)]
       .sort_values("day").groupby("LOG_ID").val.last())
_rows = []
for lid, g in cr_hist.groupby("LOG_ID", sort=False):
    r_min = judge_aki(g.day.values, g.val.values, "min")
    r_last = judge_aki(g.day.values, g.val.values, "last")
    r_base = judge_aki(g.day.values, g.val.values, "preop_base",
                       preop_base=_pb.get(lid, np.nan))
    _rows.append(dict(
        LOG_ID=lid,
        st_roll_min=r_min["stage"], first_aki_day=r_min["first_time"],
        path_first=r_min["path"], basis_first=r_min["basis"],
        n_abs_eval=r_min["n_abs_eval"], n_rel_eval=r_min["n_rel_eval"],
        aki_evaluable=r_min["evaluable"],
        st_roll_last=r_last["stage"], first_aki_day_last=r_last["first_time"],
        st_preopbase=r_base["stage"],
        first_aki_day_b=r_base["first_time"]))
roll = pd.DataFrame(_rows)

coh = coh.merge(pre, left_on="LOG_ID", right_index=True, how="left")
coh = coh.merge(pk, left_on="LOG_ID", right_index=True, how="left")
coh = coh.merge(p48, left_on="LOG_ID", right_index=True, how="left")
coh = coh.merge(pn, left_on="LOG_ID", right_index=True, how="left")
coh = coh.merge(roll, on="LOG_ID", how="left")

# PNI
coh["pni_simple"] = coh.alb + 0.05 * coh.lym_pct                 # 简化
coh["pni_onodera"] = 10 * coh.alb + 5 * coh.alc                  # 传统（ALC 直接可得）

print()
print("=== 术前暴露可得率（术前 30 天内最近一次）===")
for c in ["alb", "lym_pct", "alc", "cr_base"]:
    print("  %-10s %5d (%.1f%%)" % (c, coh[c].notna().sum(),
                                    100 * coh[c].notna().mean()))
print("  %-10s %5d (%.1f%%)" % ("pni_simple", coh.pni_simple.notna().sum(),
                                100 * coh.pni_simple.notna().mean()))
print("  %-10s %5d (%.1f%%)" % ("pni_onodera", coh.pni_onodera.notna().sum(),
                                100 * coh.pni_onodera.notna().mean()))
# 清洗后分布
print()
print("=== 清洗后术前值分布 ===")
for c in ["alb", "lym_pct", "alc", "cr_base"]:
    s = coh[c].dropna()
    if len(s):
        print("  %-10s n=%-5d 中位=%-7.2f 四分位=%.2f/%.2f/%.2f 范围=%.2f~%.2f" % (
            c, len(s), s.median(), s.quantile(.25), s.median(),
            s.quantile(.75), s.min(), s.max()))
for c in ["pni_simple", "pni_onodera"]:
    s = coh[c].dropna()
    if len(s):
        print("  %-12s n=%-5d 中位=%-7.2f 范围=%.2f~%.2f" % (
            c, len(s), s.median(), s.min(), s.max()))

# 配对时间差
ok = coh[["t_alb", "t_lym"]].notna().all(axis=1)
if ok.any():
    dt_ = (coh.loc[ok, "t_alb"] - coh.loc[ok, "t_lym"]).dt.total_seconds().abs() / 3600
    print("\n  albumin 与淋巴细胞百分比采样时间差(小时): 中位=%.1f 四分位=%.1f/%.1f" % (
        dt_.median(), dt_.quantile(.25), dt_.quantile(.75)))
    print("  其中时间差 <=24h 者占 %.1f%%" % (100 * (dt_ <= 24).mean()))
print("\n  对照：INSPIRE 侧术前 PNI 可得率 90.6%%，MOVER 侧 %.1f%%"
      % (100 * coh.pni_onodera.notna().mean()))


# ------------------------------------------------------------ AKI 判定
def stage_unified_no48(b, pk_):
    """旧口径留痕：0.3 mg/dL 应用于整个术后 0–7 天窗口。"""
    if not (np.isfinite(b) and np.isfinite(pk_)) or b <= 0:
        return np.nan
    if pk_ >= 3.0 * b:
        return 3.0
    if pk_ >= 4.0 and (pk_ >= 1.5 * b or (pk_ - b) >= 0.3):
        return 3.0
    if pk_ >= 2.0 * b:
        return 2.0
    if pk_ >= 1.5 * b or (pk_ - b) >= 0.3:
        return 1.0
    return 0.0


def stage_unified(b, pk_, p48):
    """统一口径：0.3 升高限术后 48h，倍数标准限 7 天。"""
    if not (np.isfinite(b) and np.isfinite(pk_)) or b <= 0:
        return np.nan
    acute = pk_ >= 1.5 * b or (np.isfinite(p48) and (p48 - b) >= 0.3)
    if pk_ >= 3.0 * b:
        return 3.0
    if pk_ >= 4.0 and acute:
        return 3.0
    if pk_ >= 2.0 * b:
        return 2.0
    if acute:
        return 1.0
    return 0.0


def stage_naive(b, pk_):
    """MOVER 现状：无 4.0"""
    if not (np.isfinite(b) and np.isfinite(pk_)) or b <= 0:
        return np.nan
    if pk_ >= 3.0 * b:
        return 3.0
    if pk_ >= 2.0 * b:
        return 2.0
    if pk_ >= 1.5 * b or (pk_ - b) >= 0.3:
        return 1.0
    return 0.0


def stage_abs_naive(b, pk_):
    """4.0 无急性条件（错误做法，仅作对照）"""
    if not (np.isfinite(b) and np.isfinite(pk_)) or b <= 0:
        return np.nan
    if pk_ >= 3.0 * b or pk_ >= 4.0:
        return 3.0
    if pk_ >= 2.0 * b:
        return 2.0
    if pk_ >= 1.5 * b or (pk_ - b) >= 0.3:
        return 1.0
    return 0.0


for nm in ["st_naive", "st_abs_naive"]:
    fn = stage_naive if nm == "st_naive" else stage_abs_naive
    coh[nm] = [fn(b, p) for b, p in zip(coh.cr_base, coh.cr_peak)]
    coh["aki_" + nm[3:]] = (coh[nm] >= 1).astype(float)
    coh.loc[coh[nm].isna(), "aki_" + nm[3:]] = np.nan
    coh["sev_" + nm[3:]] = (coh[nm] >= 2).astype(float)
    coh.loc[coh[nm].isna(), "sev_" + nm[3:]] = np.nan

coh["st_unified"] = coh["st_roll_min"]
coh["st_unified_no48"] = [stage_unified_no48(b, p) for b, p in
                          zip(coh.cr_base, coh.cr_peak)]
for nm in ["st_unified", "st_unified_no48", "st_roll_last", "st_preopbase"]:
    coh["aki_" + nm[3:]] = (coh[nm] >= 1).astype(float)
    coh.loc[coh[nm].isna(), "aki_" + nm[3:]] = np.nan
    coh["sev_" + nm[3:]] = (coh[nm] >= 2).astype(float)
    coh.loc[coh[nm].isna(), "sev_" + nm[3:]] = np.nan

print()
print("=" * 78)
print("口径对照（重建队列 %d 例中可判定者）" % len(coh))
print("=" * 78)
print("%-16s %8s %9s %9s   %s" % ("口径", "可判定", "AKI", "中重度", "分期分布"))
for nm in ["st_naive", "st_abs_naive", "st_unified",
           "st_roll_last", "st_preopbase"]:
    j = coh[coh[nm].notna()]
    vc = j[nm].value_counts().sort_index()
    dist = " ".join("%d期=%d" % (int(k), v) for k, v in vc.items())
    print("%-16s %8d %8.2f%% %8.2f%%   %s" % (
        nm, len(j), 100 * j["aki_" + nm[3:]].mean(),
        100 * j["sev_" + nm[3:]].mean(), dist))

# ------------------------------------------------------------ 排除链
coh["elig"] = coh.pni_onodera.notna() & coh.cr_base.notna() & coh.st_unified.notna()
coh["eskd"] = (coh.cr_base >= 3.0).astype(int)

print()
print("=" * 78)
print("MOVER 入组漏斗（对齐 INSPIRE 新标准）")
print("=" * 78)
steps = [("① 腹部 0D/0F ∩ ≥120min ∩ 每MRN首次", pd.Series(True, index=coh.index))]
s1 = coh.elig
steps.append(("② + 术前PNI ∩ 术前Cr ∩ 术后Cr 可得", s1))
s2 = s1 & coh.eskd.eq(0)
steps.append(("③ + 排除 ESKD 候选（术前 Cr≥3.0）", s2))
for nm, s in steps:
    j = coh[s]
    print("  %-38s N=%-5d AKI=%-4d (%5.2f%%) 中重度=%-4d (%5.2f%%)" % (
        nm, len(j), j.aki_unified.sum(), 100 * j.aki_unified.mean(),
        j.sev_unified.sum(), 100 * j.sev_unified.mean()))

FINAL = s2
coh.to_csv(OUT, index=False)
print("\nsaved", OUT)

# ------------------------------------------------------------ 效应量
f = coh[FINAL].copy()
f["asa_n"] = pd.to_numeric(f.ASA_RATING_C, errors="coerce")
f["sex_m"] = (f.SEX.astype(str).str.upper().str.startswith("M")).astype(int)
f["or_min_n"] = pd.to_numeric(f.or_min, errors="coerce")
f["cr_pre"] = f.cr_base
COVS = ["asa_n", "sex_m", "or_min_n", "cr_pre"]

print()
print("=" * 78)
print("MOVER 外部验证效应量（统一口径；INSPIRE SD 冻结）")
print("=" * 78)
for key, lab_ in [("pni_onodera", "传统 Onodera PNI"),
                  ("pni_simple", "简化 PNI")]:
    dd = f[[key] + COVS + ["aki_unified", "sev_unified"]].dropna()
    if len(dd) < 100:
        print("\n【%s】样本不足 n=%d，跳过" % (lab_, len(dd)))
        continue
    sd_own = dd[key].std()
    print("\n【%s】N=%d 事件=%d (%.2f%%) 中重度=%d (%.2f%%)  MOVER SD=%.3f" % (
        lab_, len(dd), dd.aki_unified.sum(), 100 * dd.aki_unified.mean(),
        dd.sev_unified.sum(), 100 * dd.sev_unified.mean(), sd_own))
    for yv, yl in [("aki_unified", "主要 AKI"), ("sev_unified", "中重度 AKI")]:
        X = sm.add_constant(dd[[key] + COVS])
        r = sm.Logit(dd[yv].values, X).fit(disp=0, maxiter=200)
        beta, se_b = r.params[key], r.bse[key]
        # 冻结 INSPIRE SD
        sdF = SD_FROZEN[key]
        bF, sF = beta * (-sdF), se_b * sdF
        # 自身 SD
        bO, sO = beta * (-sd_own), se_b * sd_own
        print("   %-10s 冻结INSPIRE-SD OR=%.3f (%.3f-%.3f) p=%.2e | 自身SD OR=%.3f (%.3f-%.3f)" % (
            yl, np.exp(bF), np.exp(bF - 1.96 * sF), np.exp(bF + 1.96 * sF),
            r.pvalues[key], np.exp(bO), np.exp(bO - 1.96 * sO),
            np.exp(bO + 1.96 * sO)))
    dd2 = dd.copy()
    dd2["qq"] = pd.qcut(dd2[key], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    g = dd2.groupby("qq", observed=True).agg(
        n=("aki_unified", "size"), ev=("aki_unified", "sum"),
        sev=("sev_unified", "sum"))
    g["rate%"] = (100 * g.ev / g.n).round(2)
    print(g.to_string())
    print("   最小层事件=%d  最小层 severe=%d  梯度倍数=%.2f" % (
        g.ev.min(), g.sev.min(), g["rate%"].iloc[0] / max(g["rate%"].iloc[-1], 1e-9)))

print("\nelapsed %.1fs" % (time.time() - t0))
