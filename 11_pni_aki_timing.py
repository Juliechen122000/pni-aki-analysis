# -*- coding: utf-8 -*-
"""
11_pni_aki_timing.py
=====================================================================
核实轨迹分析的时间顺序问题
  AKI 判定窗：术后 0–7 天
  PNI 轨迹 bin：0/2/4/6（每 2 天一段，bin 起点 = 0/2/4/6 天）

问题：bin 2/4/6 的 PNI 测量，有多少发生在该患者首次满足 AKI 标准之后？
若比例很高，轨迹反映的可能是 AKI 发生后的生理/治疗变化，而非病前风险状态。

方法：
  1. 用逐条肌酐记录确定每个患者「首次满足 KDIGO」的时间（day, 相对 opend_time）
  2. 对轨迹长表中的每条记录，判断其 bin 起点是否晚于首次达标时间
  3. 统计测量条数与患者数的占比
输出：pni_aki_timing.csv + 控制台报告
=====================================================================
"""
import os
import numpy as np
import pandas as pd

OUT = os.environ.get("INSPIRE_OUT", "output")

# ---------- 1. 首次达标时间：直接取同一 AKI 判定函数的输出 ----------
# 2026-10-25 起不再独立重算（旧逻辑与新滚动定义不同源，会污染时序比例）
I = pd.read_csv(os.path.join(OUT, "p0_aki_inspire_harmonized.csv"),
                low_memory=False)
I = I[(I.eskd == 0) & (I.ltx == 0) & (I.elig == True)]
T = I[["subject_id", "first_aki_day"]].copy()
print("可定首次达标时间者 %d 人；其中达标的 %d 人 (%.1f%%)"
      % (len(T), T.first_aki_day.notna().sum(),
         100 * T.first_aki_day.notna().mean()))

# ---------- 2. 轨迹长表 → bin 起点 ----------
L = pd.read_csv(os.path.join(OUT, "lcmm_long_onodera.csv"))
L = L.merge(T, on="subject_id", how="left")
# bin 起点即该 bin 的起始天数（0/2/4/6）
L["bin_start"] = L.time.astype(float)

print("\n" + "=" * 72)
print("轨迹测量相对于 AKI 首次达标的时间关系")
print("=" * 72)

# 全体轨迹记录
n_all = len(L)
n_after = int((L.bin_start > L.first_aki_day).sum())
print("轨迹记录总数 %d 条" % n_all)
print("  其中 bin 起点晚于首次达标时间：%d 条 (%.1f%%)"
      % (n_after, 100 * n_after / n_all))
print("  未发生 AKI 者（无首次达标时间）：%d 条 (%.1f%%)"
      % (int(L.first_aki_day.isna().sum()), 100 * L.first_aki_day.isna().mean()))

# 按 bin 分层
print("\n按 bin 分层：")
for b in sorted(L.time.unique()):
    z = L[L.time == b]
    a = int((z.bin_start > z.first_aki_day).sum())
    print("  bin %.0f：%5d 条，其中晚于 AKI 首次达标 %5d 条 (%.1f%%)"
          % (b, len(z), a, 100 * a / len(z) if len(z) else 0))

# 患者层面：首个测量是否已在 AKI 之后
first = L[L.time == 0]
n1 = int((first.bin_start > first.first_aki_day).sum())
print("\n首个 bin（bin 0）的测量中，晚于 AKI 首次达标者：%d / %d (%.1f%%)"
      % (n1, len(first), 100 * n1 / len(first)))

# AKI 患者中：有多少人的 AKI 在 bin 0 结束前就已发生
aki_pts = L[L.time == 0].dropna(subset=["first_aki_day"])
early = int((aki_pts.first_aki_day < 2).sum())
print("在轨迹队列的 AKI 患者中，首次达标发生在术后 <2 天者：%d / %d (%.1f%%)"
      % (early, len(aki_pts), 100 * early / len(aki_pts)))
print("  上述患者的中位首次达标时间：%.2f 天"
      % aki_pts.first_aki_day.median())
print("  四分位：%.2f–%.2f 天"
      % (aki_pts.first_aki_day.quantile(.25), aki_pts.first_aki_day.quantile(.75)))

L[["subject_id", "time", "bin_start", "first_aki_day"]].to_csv(
    os.path.join(OUT, "pni_aki_timing.csv"), index=False)
T.to_csv(os.path.join(OUT, "aki_first_day.csv"), index=False)
print("\n已写出 pni_aki_timing.csv / aki_first_day.csv")
