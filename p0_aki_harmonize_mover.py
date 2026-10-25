# -*- coding: utf-8 -*-
"""AKI 口径统一 —— MOVER 侧（外部验证队列，美国 UCI/EPIC）

目的：把 MOVER 的 KDIGO 实现从"纯 ratio/diff、术前基线无时间窗"统一到
      与 INSPIRE 完全相同的判定函数与窗口定义。

MOVER 现状（mover_09_aki.py）：
  - 基线 = 术前【任意时间】最近一次肌酐（无 30 天限制）
  - 峰值 = surg(IN_OR_DTTM) ~ +7 天
  - 分期 = ratio>=3 / >=2 / >=1.5 或 diff>=0.3     ← 无绝对值 4.0
  - 清洗 = 0.1 < Cr <= 15 且单位含 MG/DL

INSPIRE 现状（aki_kdigo_uo.py，已证明 p48 恒冗余）：
  - 基线 = 术前 30 天内最近一次（相对 opend_time）
  - 峰值 = day in [-0.02, 7]（相对 opend_time = 手术结束）
  - 分期 = ratio>=3 或 pk>=4.0 / >=2 / >=1.5 或 diff>=0.3

统一口径（2026-10-25 起）：
  基线 = 术前 30 天内最近一次；peak = 术后 0-7 天；p48 = 术后 0-2 天
  0.3 mg/dL 升高仅限 p48 相对基线；倍数标准限 7 天；3 期含急性 4.0

本脚本输出多口径对照，量化每一处差异的贡献。
"""
import os
import time

import numpy as np
import pandas as pd

PD = os.environ.get("INSPIRE_OUT", "output")
LAB = os.environ.get("MOVER_LABS", "./MOVER/patient_labs.csv")
COH = os.path.join(PD, "mover_c2_traj.csv")
CACHE = os.path.join(PD, "p0_mover_creatinine_raw.csv")
OUT = os.path.join(PD, "p0_aki_mover_harmonized.csv")

t0 = time.time()

# ------------------------------------------------------------------ 队列
coh = pd.read_csv(COH, low_memory=False)
print("MOVER 队列 %d 例" % len(coh))
coh["surg_in"] = pd.to_datetime(coh["IN_OR_DTTM"], errors="coerce")
coh["surg_out"] = pd.to_datetime(coh["OUT_OR_DTTM"], errors="coerce")
coh["surg_dt"] = pd.to_datetime(coh["SURGERY_DATE"], errors="coerce")
print("IN_OR_DTTM 可得 %d (%.1f%%)  OUT_OR_DTTM 可得 %d (%.1f%%)" % (
    coh.surg_in.notna().sum(), 100 * coh.surg_in.notna().mean(),
    coh.surg_out.notna().sum(), 100 * coh.surg_out.notna().mean()))
log_ids = set(coh["LOG_ID"])

# ------------------------------------------------- 扫描 labs（带缓存）
if os.path.exists(CACHE):
    print("复用缓存", CACHE)
    cr = pd.read_csv(CACHE, low_memory=False)
else:
    print("扫描 %s ..." % LAB)
    rows = []
    total = 0
    for ch in pd.read_csv(LAB, chunksize=500_000, low_memory=False):
        total += len(ch)
        sub = ch[ch["Lab Code"].astype(str).str.strip() == "2160-0"]
        if len(sub):
            rows.append(sub[["LOG_ID", "Observation Value",
                             "Measurement Units", "Collection Datetime"]])
    print("  扫描 %s 行，保留肌酐 %d 条" % (f"{total:,}", sum(len(r) for r in rows)))
    cr = pd.concat(rows, ignore_index=True)
    cr.to_csv(CACHE, index=False)
    print("  缓存已写入", CACHE)

cr["Observation Value"] = pd.to_numeric(cr["Observation Value"], errors="coerce")
cr["Collection Datetime"] = pd.to_datetime(cr["Collection Datetime"], errors="coerce")
cr = cr.dropna(subset=["Observation Value", "Collection Datetime"])

# 与 INSPIRE 对齐的清洗：0.1 < Cr <= 15，单位 mg/dL
cr["units_u"] = cr["Measurement Units"].astype(str).str.upper()
print("清洗前 %d 条；单位分布 TOP5:" % len(cr))
print(cr.units_u.value_counts().head(5).to_string())
cr = cr[cr.units_u.str.contains("MG/DL", na=False)]
cr = cr[(cr["Observation Value"] > 0.1) & (cr["Observation Value"] <= 15)]
print("清洗后 %d 条" % len(cr))

cr = cr.merge(coh[["LOG_ID", "surg_in", "surg_out", "surg_dt"]],
              on="LOG_ID", how="inner")
print("队列内肌酐 %d 条 / %d 例" % (len(cr), cr.LOG_ID.nunique()))

# 手术时间原点：优先 OUT_OR_DTTM（对齐 INSPIRE 的 opend_time = 手术结束），
# 缺失时回退 IN_OR_DTTM
cr["surg"] = cr["surg_out"]
cr.loc[cr.surg.isna(), "surg"] = cr["surg_in"].loc[cr.surg.isna()]
cr["day"] = (cr["Collection Datetime"] - cr["surg"]).dt.total_seconds() / 86400.0


# ------------------------------------------------------------ 判定函数
def stage_harm(b, pk, p48):
    """统一口径：0.3 升高限 48h，倍数标准限 7 天，含急性 4.0。"""
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


def stage_harm_no48(b, pk):
    """旧口径留痕：0.3 升高应用于整个 7 天窗口。"""
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


def stage_no_abs(b, pk):
    """MOVER 现状：无绝对值 4.0"""
    if not (np.isfinite(b) and np.isfinite(pk)) or b <= 0:
        return np.nan
    if pk >= 3.0 * b:
        return 3.0
    if pk >= 2.0 * b:
        return 2.0
    if pk >= 1.5 * b or (pk - b) >= 0.3:
        return 1.0
    return 0.0


def build(base_win_days, post_hi, use_abs):
    """base_win_days: 术前窗口天数(None=不限)；post_hi: 术后上限天数"""
    out = {}
    for lid, g in cr.groupby("LOG_ID", sort=False):
        if base_win_days is None:
            pre = g[g.day < 0]
        else:
            pre = g[(g.day < 0) & (g.day > -base_win_days)]
        post = g[(g.day >= -0.02) & (g.day <= post_hi)]
        if not len(pre) or not len(post):
            out[lid] = (np.nan, np.nan, np.nan, np.nan, np.nan)
            continue
        g2 = pre.sort_values("day")
        b = float(g2["Observation Value"].iloc[-1])
        pk = float(post["Observation Value"].max())
        p48v = post.loc[post.day <= 2, "Observation Value"]
        p48 = float(p48v.max()) if len(p48v) else np.nan
        st = stage_harm(b, pk, p48) if use_abs else stage_no_abs(b, pk)
        st_no48 = stage_harm_no48(b, pk) if use_abs else np.nan
        out[lid] = (b, pk, p48, st, st_no48)
    r = pd.DataFrame(out).T
    r.columns = ["cr_base", "cr_peak", "cr_peak48", "stage", "stage_no48"]
    r.index.name = "LOG_ID"
    return r.reset_index()


SPECS = [
    ("现状(基线不限/无4.0)", None, 7, False),
    ("+基线限30天", 30, 7, False),
    ("+含绝对值4.0", 30, 7, True),
    ("=统一口径 harmonized", 30, 7, True),
]
print()
print("=" * 78)
print("MOVER 口径对照")
print("=" * 78)
print("%-24s %8s %9s %9s   %s" % ("口径", "可判定", "AKI", "中重度", "分期分布"))
res = {}
for nm, bw, ph, ua in SPECS[:3]:
    r = build(bw, ph, ua)
    res[nm] = r
    j = r[r.stage.notna()]
    vc = j.stage.value_counts().sort_index()
    dist = " ".join("%d期=%d" % (int(k), v) for k, v in vc.items())
    print("%-24s %8d %8.2f%% %8.2f%%   %s" % (
        nm, len(j), 100 * (j.stage >= 1).mean(),
        100 * (j.stage >= 2).mean(), dist))

HARM = res["+含绝对值4.0"]
HARM.to_csv(OUT, index=False)
print("\nsaved", OUT)

# ------------------------------------------------------------ 变化拆解
a = res["现状(基线不限/无4.0)"].set_index("LOG_ID").stage
b = res["+基线限30天"].set_index("LOG_ID").stage
c = res["+含绝对值4.0"].set_index("LOG_ID").stage
j = pd.concat([a, b, c], axis=1, keys=["s0", "s1", "s2"]).dropna()
print()
print("变化来源拆解（可判定且三种口径均可比者 n=%d）：" % len(j))
print("  基线限 30 天导致分期改变: %d 例 (%.2f%%)" % (
    (j.s0 != j.s1).sum(), 100 * (j.s0 != j.s1).mean()))
print("  加绝对值 4.0 导致分期改变: %d 例 (%.2f%%)" % (
    (j.s1 != j.s2).sum(), 100 * (j.s1 != j.s2).mean()))
print("  两者合计改变: %d 例 (%.2f%%)" % ((j.s0 != j.s2).sum(),
                                        100 * (j.s0 != j.s2).mean()))
print("\nelapsed %.1fs" % (time.time() - t0))
