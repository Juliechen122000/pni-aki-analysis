# -*- coding: utf-8 -*-
"""
11b_pni_aki_timing_v2.py
=====================================================================
时间顺序核查 v2 —— 用实际测量时间，而非 bin 起点

v1 的缺陷：以 bin 起点（0/2/4/6 天）与首次达标时间比较，
  bin 0 的起点恒为 0，永不会晚于首次达标，严重低估。
  但 bin 0 覆盖术后 0–2 天，而 84% 的 AKI 在 2 天内即达标（中位 0.75 天），
  所以 bin 0 内的测量同样可能落在 AKI 之后。

v2 方法：
  1. 从 lab pool 取每条 albumin 记录的精确 chart_time（相对 opend_time 的天数）
  2. 按 (subject, bin) 对应到轨迹长表的每个观测
  3. 用该观测在 bin 内所有 albumin 记录的时间中位数，与首次达标时间比较
  4. 分层统计：bin × 是否晚于 AKI
输出：pni_aki_timing_v2.csv
=====================================================================
"""
import os
import numpy as np
import pandas as pd

OUT = os.environ.get("INSPIRE_OUT", "output")

# ---------- 首次达标时间（复用 v1 输出）----------
T = pd.read_csv(os.path.join(OUT, "aki_first_day.csv"))
print("AKI 首次达标时间：可定者 %d 人，达标 %d 人"
      % (len(T), T.first_aki_day.notna().sum()))

# ---------- albumin 精确时间 ----------
lab = pd.read_csv(os.path.join(OUT, "p0_lab_pool.csv"), low_memory=False)
lab["value"] = pd.to_numeric(lab.value, errors="coerce")
lab["chart_time"] = pd.to_numeric(lab.chart_time, errors="coerce")

I = pd.read_csv(os.path.join(OUT, "p0_aki_inspire_harmonized.csv"), low_memory=False)
I = I[(I.eskd == 0) & (I.ltx == 0) & (I.elig == True)]
op = I[["subject_id", "opend_time"]].dropna()

alb = lab[lab.item_name == "albumin"].dropna(subset=["value", "chart_time"])
alb = alb.merge(op, on="subject_id", how="inner")
alb["day"] = (alb.chart_time - alb.opend_time) / 1440.0
alb = alb[(alb.day >= -0.02) & (alb.day <= 8)]
# 与建长表一致的分箱
alb["bin"] = np.floor((alb.day + 0.02) / 2) * 2
alb = alb[alb.bin.isin([0.0, 2.0, 4.0, 6.0])]

# 每个 (subject, bin) 的实测时间中位数
bt = (alb.groupby(["subject_id", "bin"])["day"]
        .agg(["median", "min", "max", "size"])
        .reset_index()
        .rename(columns={"median": "bin_med_day", "min": "bin_min_day",
                         "max": "bin_max_day", "size": "n_alb"}))

# ---------- 轨迹长表 ----------
L = pd.read_csv(os.path.join(OUT, "lcmm_long_onodera.csv"))
L = L.rename(columns={"time": "bin"}).merge(bt, on=["subject_id", "bin"], how="left")
L = L.merge(T, on="subject_id", how="left")
print("轨迹观测 %d 条，其中可匹配到精确时间者 %d 条 (%.1f%%)"
      % (len(L), L.bin_med_day.notna().sum(),
         100 * L.bin_med_day.notna().mean()))

L = L.dropna(subset=["bin_med_day"])

print("\n" + "=" * 74)
print("按实际测量时间判断：该次 PNI 测量是否发生在 AKI 首次达标之后")
print("=" * 74)

aki = L.dropna(subset=["first_aki_day"])
print("AKI 患者的轨迹观测：%d 条（覆盖 %d 人）"
      % (len(aki), aki.subject_id.nunique()))

n_after = int((aki.bin_med_day > aki.first_aki_day).sum())
print("  测量时间晚于首次达标：%d 条 (%.1f%%)"
      % (n_after, 100 * n_after / len(aki)))

print("\n按 bin 分层（AKI 患者内）：")
print("  bin   观测数   晚于达标数   占比    测量时间中位(天)   AKI达标中位(天)")
for b in sorted(aki.bin.unique()):
    z = aki[aki.bin == b]
    a = int((z.bin_med_day > z.first_aki_day).sum())
    print("  %.0f   %6d   %8d   %5.1f%%      %6.2f            %6.2f"
          % (b, len(z), a, 100 * a / len(z), z.bin_med_day.median(),
             z.first_aki_day.median()))

# 患者层面
print("\n患者层面（仅 AKI 患者，n=%d）：" % aki.subject_id.nunique())
per = aki.groupby("subject_id").agg(
    n_obs=("bin", "size"),
    n_after=("bin_med_day", lambda s: 0),   # 占位，下面重算
)
tmp = aki.assign(after=(aki.bin_med_day > aki.first_aki_day).astype(int))
per = tmp.groupby("subject_id").agg(n_obs=("after", "size"), n_after=("after", "sum"))
per["share"] = per.n_after / per.n_obs
print("  全部测量均在 AKI 之后者：%d 人 (%.1f%%)"
      % (int((per.share == 1).sum()), 100 * (per.share == 1).mean()))
print("  部分测量在 AKI 之后者：  %d 人 (%.1f%%)"
      % (int(((per.share > 0) & (per.share < 1)).sum()),
         100 * ((per.share > 0) & (per.share < 1)).mean()))
print("  全部测量均在 AKI 之前者：%d 人 (%.1f%%)"
      % (int((per.share == 0).sum()), 100 * (per.share == 0).mean()))
print("  每例被‘污染’测量的中位比例：%.1f%%" % (100 * per.share.median()))

# 敏感性：仅用 bin 0（首测）做 landmark
f0 = aki[aki.bin == 0]
a0 = int((f0.bin_med_day > f0.first_aki_day).sum())
print("\n首测（bin 0）中晚于达标者：%d / %d (%.1f%%)"
      % (a0, len(f0), 100 * a0 / len(f0)))
print("  bin 0 测量时间中位 %.2f 天，四分位 %.2f–%.2f"
      % (f0.bin_med_day.median(), f0.bin_med_day.quantile(.25),
         f0.bin_med_day.quantile(.75)))

L[["subject_id", "bin", "bin_med_day", "first_aki_day"]].to_csv(
    os.path.join(OUT, "pni_aki_timing_v2.csv"), index=False)
print("\n已写出 pni_aki_timing_v2.csv")
