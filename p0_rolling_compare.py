# -*- coding: utf-8 -*-
"""三种结局定义对照（主分析 min / 敏感性 last / 敏感性 preop_base）

输出：output//rolling_definition_comparison.csv
同时打印：事件数、分期转移、事件重叠、模型 OR、检测密度梯度下的判定差异、
          首次达标时间与 PNI 时序比例。
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT = os.environ.get("INSPIRE_OUT", "output")
SD = 6.445246
DEFS = [("min", "st_unified"), ("last", "st_roll_last"),
        ("preop_base", "st_preopbase"), ("old48", "st_unified_no48")]

I = pd.read_csv(os.path.join(OUT, "p0_aki_inspire_harmonized.csv"),
                low_memory=False)
M = pd.read_csv(os.path.join(OUT, "p0_mover_rebuilt.csv"), low_memory=False)

I = I[(I.eskd == 0) & (I.ltx == 0) & (I.elig == True)].copy()
M = M[(M.eskd == 0) & (M.elig == True)].copy()

I["male"] = (I.sex == "M").astype(int)
I["open_"] = (I.pcs.astype(str).str.zfill(5).str[4] == "0").astype(int)
I["emop_"] = (pd.to_numeric(I.emop, errors="coerce") == 1).astype(int)
I["asa_n"] = pd.to_numeric(I.asa, errors="coerce")
I["opdur"] = pd.to_numeric(I.op_duration, errors="coerce")
I["crb"] = pd.to_numeric(I.cr, errors="coerce")
I["x"] = -I.pni_onodera / SD

M["male"] = (M.SEX.astype(str).str.upper().str.startswith("M")).astype(int)
M["asa_n"] = pd.to_numeric(M.ASA_RATING_C, errors="coerce")
M["opdur"] = pd.to_numeric(M.or_min, errors="coerce")
M["crb"] = pd.to_numeric(M.cr_base, errors="coerce")
M["x"] = -M.pni_onodera / SD

COV_I = ["x", "age", "male", "asa_n", "opdur", "crb", "open_", "emop_"]
COV_M = ["x", "asa_n", "male", "opdur", "crb"]
rows = []


def add(cohort, definition, metric, value, note=""):
    rows.append(dict(cohort=cohort, definition=definition, metric=metric,
                     value=value, note=note))


def or_hc0(d, cov, ycol):
    dd = d[cov + [ycol]].dropna()
    if len(dd) < 50 or dd[ycol].sum() < 10:
        return np.nan, np.nan, np.nan, len(dd), int(dd[ycol].sum())
    X = sm.add_constant(dd[cov].astype(float))
    r = sm.Logit(dd[ycol].astype(float), X).fit(disp=0, maxiter=200,
                                                cov_type="HC0")
    b, se = r.params["x"], r.bse["x"]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), len(dd), int(dd[ycol].sum())


for coh, d, cov in [("INSPIRE", I, COV_I), ("MOVER", M, COV_M)]:
    print("\n" + "=" * 78)
    print(coh, "n=%d" % len(d))
    print("=" * 78)
    d = d.copy()
    d["q"] = pd.qcut(d.cr_n.rank(method="first"), 5,
                     labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    for name, col in DEFS:
        if col not in d:
            continue
        aki = d[col] >= 1
        sev = d[col] >= 2
        n_ev, n_sev = int(aki.sum()), int(sev.sum())
        print("  %-11s AKI %4d (%.2f%%)  中重度 %3d (%.2f%%)"
              % (name, n_ev, 100 * aki.mean(), n_sev, 100 * sev.mean()))
        add(coh, name, "n", len(d))
        add(coh, name, "aki_n", n_ev)
        add(coh, name, "aki_pct", round(100 * aki.mean(), 2))
        add(coh, name, "sev_n", n_sev)
        add(coh, name, "sev_pct", round(100 * sev.mean(), 2))
        ycol = "aki_" + {"min": "unified", "last": "roll_last",
                         "preop_base": "preopbase",
                         "old48": "unified_no48"}[name]
        if ycol in d:
            o, lo, hi, nn, ee = or_hc0(d, cov, ycol)
            add(coh, name, "model_or", round(float(o), 4) if np.isfinite(o) else np.nan,
                "HC0; %d/%d" % (ee, nn))
            add(coh, name, "model_or_lo", round(float(lo), 4) if np.isfinite(lo) else np.nan)
            add(coh, name, "model_or_hi", round(float(hi), 4) if np.isfinite(hi) else np.nan)
            print("      model OR %.3f (%.3f-%.3f), n=%d events=%d"
                  % (o, lo, hi, nn, ee))

    # 事件重叠与分期转移：min vs last
    a, b = d["st_unified"], d["st_roll_last"]
    both = int(((a >= 1) & (b >= 1)).sum())
    only_min = int(((a >= 1) & (b < 1)).sum())
    only_last = int(((a < 1) & (b >= 1)).sum())
    neither = int(((a < 1) & (b < 1)).sum())
    print("\n  [min vs last] both=%d only_min=%d only_last=%d neither=%d"
          % (both, only_min, only_last, neither))
    add(coh, "min_vs_last", "both_aki", both)
    add(coh, "min_vs_last", "only_min", only_min)
    add(coh, "min_vs_last", "only_last", only_last)
    add(coh, "min_vs_last", "neither", neither)
    ct = pd.crosstab(a, b)
    print("  分期转移（行=min, 列=last）:\n", ct.to_string())
    for i in ct.index:
        for j in ct.columns:
            add(coh, "min_vs_last", "stage_%d_to_%d" % (int(i), int(j)),
                int(ct.loc[i, j]))

    # 检测密度各五分位下的判定差异
    print("\n  检测密度五分位内的定义差异：")
    for q, g in d.groupby("q", observed=True):
        r_min = 100 * (g.st_unified >= 1).mean()
        r_last = 100 * (g.st_roll_last >= 1).mean()
        diff = 100 * ((g.st_unified >= 1) != (g.st_roll_last >= 1)).mean()
        print("    %s n=%-4d min=%.2f%% last=%.2f%% 不一致=%.1f%%"
              % (q, len(g), r_min, r_last, diff))
        add(coh, "min_vs_last", "q_%s_min_pct" % q, round(r_min, 2))
        add(coh, "min_vs_last", "q_%s_last_pct" % q, round(r_last, 2))
        add(coh, "min_vs_last", "q_%s_disagree_pct" % q, round(diff, 2))

# ---------------- 首次达标时间与 PNI 时序比例（INSPIRE） ----------------
print("\n" + "=" * 78)
print("首次达标时间与 PNI 时序（INSPIRE 轨迹观测）")
print("=" * 78)
T = pd.read_csv(os.path.join(OUT, "pni_aki_timing_v2.csv"))
T = T.drop(columns=[c for c in ["first_aki_day", "first_aki_day_last",
                                "st_unified", "st_roll_last"] if c in T.columns])
T = T.merge(I[["subject_id", "first_aki_day", "first_aki_day_last",
               "st_unified", "st_roll_last"]], on="subject_id", how="left")
for name, daycol, stcol in [("min", "first_aki_day", "st_unified"),
                            ("last", "first_aki_day_last", "st_roll_last")]:
    z = T[(T[stcol] >= 1) & T[daycol].notna() & T.bin_med_day.notna()]
    after = int((z.bin_med_day > z[daycol]).sum())
    print("  %-5s AKI 患者 %d 人，观测 %d 条，晚于达标 %d 条 (%.1f%%)，"
          "达标中位 %.2f 天" % (name, z.subject_id.nunique(), len(z), after,
                                100 * after / len(z), z[daycol].median()))
    add("INSPIRE", name, "timing_after_pct", round(100 * after / len(z), 2))
    add("INSPIRE", name, "timing_n_obs", len(z))
    add("INSPIRE", name, "first_aki_median_days", round(float(z[daycol].median()), 3))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "rolling_definition_comparison.csv"),
                          index=False)
print("\n已写出 rolling_definition_comparison.csv")
