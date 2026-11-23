# -*- coding: utf-8 -*-
"""
18_low_bp_sensitivity.py —— INSPIRE 术中低血压敏感性分析（v2）
用与主模型完全一致的构建：
  数据: output/p0_aki_inspire_harmonized.csv
  结局: aki_unified（滚动KDIGO，min定义，与主稿一致）
  暴露: pni_onodera，SD冻结 6.445246，每-1SD
  Model B 协变量: age, sex_m, asa_n, op_duration, cr_pre, open_(pcs[4]=='0'), emop_
  HC0 稳健SE
复现主模型（应≈1.466），再加术中低血压暴露看移动。
"""
import csv, math, os
import numpy as np
import pandas as pd
import statsmodels.api as sm

PD = os.environ.get("INSPIRE_OUT", "output")
COHORT = os.path.join(PD, "p0_aki_inspire_harmonized.csv")
LOWBP = os.path.join(PD, "low_bp_exposure.csv")
SD_FROZEN = 6.445246

# 1) 加载 + 构建（与 p0_final_two_cohort.py 一致）
I = pd.read_csv(COHORT, low_memory=False)
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
di = I[FIN_I].copy()
print("INSPIRE 主分析样本:", len(di), "事件:", int(di.aki_unified.sum()))

# 2) 复现主模型 Model B
def run_model(df, extra_covs=None, extra_scale=None):
    cov = ["age", "sex_m", "asa_n", "op_duration", "cr_pre", "open_", "emop_"]
    cols = ["pni_onodera"] + cov
    Xcols = ["pni_onodera"] + cov
    if extra_covs:
        Xcols += extra_covs
    dd = df[Xcols + ["aki_unified"]].dropna()
    X = sm.add_constant(dd[Xcols])
    r = sm.Logit(dd["aki_unified"].values, X).fit(disp=0, maxiter=300, cov_type="HC0")
    b, se = r.params["pni_onodera"], r.bse["pni_onodera"]
    orr = math.exp(b * -SD_FROZEN)
    lo = math.exp((b + 1.96*se) * -SD_FROZEN)
    hi = math.exp((b - 1.96*se) * -SD_FROZEN)
    return dd, r, orr, lo, hi

# 主模型（无低血压）
dd0, r0, or0, lo0, hi0 = run_model(di)
print("\n=== 主模型 Model B（复现） ===")
print("N=%d 事件=%d  OR=%.4f (%.4f-%.4f)" % (len(dd0), int(dd0.aki_unified.sum()), or0, lo0, hi0))
print("（冻结值 1.466, 1.349-1.593）")

# 3) 合并低血压
lb = pd.read_csv(LOWBP)
di = di.merge(lb[["op_id","min_map","total_under_min","auc_under65","n_under65"]], on="op_id", how="left")
print("合并后低血压数据:", di["min_map"].notna().sum(), "/", len(di))

# 变量
di["under_min"] = di["total_under_min"].fillna(0)
di["has_under"] = (di["under_min"] > 0).astype(int)
di["low_map"] = di["min_map"]

# 3a + 低血压是否发生
print("\n=== +术中低血压(任何MAP<65) ===")
dd, r, o, lo, hi = run_model(di, extra_covs=["has_under"])
print("PNI OR=%.4f (%.4f-%.4f)  变化=%.1f%%" % (o, lo, hi, (o-or0)/or0*100))
print("低血压本身 OR=%.3f" % math.exp(r.params["has_under"]))

# 3b + 低血压持续分钟（每10min）
print("\n=== +低血压持续分钟(每10min) ===")
di["under_10"] = di["under_min"] / 10
dd, r, o, lo, hi = run_model(di, extra_covs=["under_10"])
print("PNI OR=%.4f (%.4f-%.4f)  变化=%.1f%%" % (o, lo, hi, (o-or0)/or0*100))
print("低血压分钟本身 OR=%.3f" % math.exp(r.params["under_10"]))

# 3c + 最低MAP（每5mmHg降）
print("\n=== +最低MAP(每5mmHg) ===")
di["low_map_5"] = di["low_map"] / 5
dd, r, o, lo, hi = run_model(di, extra_covs=["low_map_5"])
print("PNI OR=%.4f (%.4f-%.4f)  变化=%.1f%%" % (o, lo, hi, (o-or0)/or0*100))
print("最低MAP本身 OR=%.3f" % math.exp(-r.params["low_map_5"]))

# 3d 同时加
print("\n=== +低血压分钟+最低MAP ===")
di["low_map_5"] = di["low_map"] / 5
dd, r, o, lo, hi = run_model(di, extra_covs=["under_10","low_map_5"])
print("PNI OR=%.4f (%.4f-%.4f)  变化=%.1f%%" % (o, lo, hi, (o-or0)/or0*100))
