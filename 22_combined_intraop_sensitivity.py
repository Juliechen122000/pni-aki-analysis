# -*- coding: utf-8 -*-
"""
22_combined_intraop_sensitivity.py —— 术中变量综合敏感性分析
在主模型 Model B 基础上，逐一 + 组合加入：
  低血压（任何MAP<65、持续分钟、最低MAP）
  液体（晶体、胶体）
  输血、失血、尿量
  血管活性药（任一使用、麻黄碱、去氧肾上腺素）
输出: $INSPIRE_OUT/combined_intraop_sensitivity.csv
"""
import csv, math, os
import pandas as pd
import statsmodels.api as sm

PD = os.environ.get("INSPIRE_OUT", "output")
os.makedirs(PD, exist_ok=True)
COHORT = os.path.join(PD, "p0_aki_inspire_harmonized.csv")
SD_FROZEN = 6.445246

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
print("主样本:", len(di), "事件:", int(di.aki_unified.sum()))

# 合并低血压、液体、血管活性药
for f in ["low_bp_exposure.csv", "fluid_transfuse_exposure.csv", "vasopressor_exposure.csv"]:
    d = pd.read_csv(os.path.join(PD, f))
    di = di.merge(d, on="op_id", how="left")
print("合并后:", len(di))

# 变量
di["has_under"] = (di["total_under_min"].fillna(0) > 0).astype(int)
di["under_10"] = di["total_under_min"].fillna(0) / 10
di["low_map_5"] = di["min_map"] / 5
di["crystal_500"] = di["crystal_ml"].fillna(0) / 500
di["colloid_500"] = di["colloid_ml"].fillna(0) / 500
di["transfuse_any"] = (di["transfuse_units"].fillna(0) > 0).astype(int)
di["ebl_500"] = di["ebl_ml"].fillna(0) / 500
di["urine_500"] = di["urine_ml"].fillna(0) / 500
di["any_vaso"] = di["any_vaso"].fillna(0).astype(int)
di["eph_mg"] = di["eph_mg"].fillna(0)

def run(df, extra=None):
    cov = ["age", "sex_m", "asa_n", "op_duration", "cr_pre", "open_", "emop_"]
    Xcols = ["pni_onodera"] + cov + (extra or [])
    dd = df[Xcols + ["aki_unified"]].dropna()
    X = sm.add_constant(dd[Xcols])
    r = sm.Logit(dd["aki_unified"].values, X).fit(disp=0, maxiter=300, cov_type="HC0")
    b, se = r.params["pni_onodera"], r.bse["pni_onodera"]
    orr = math.exp(b * -SD_FROZEN)
    lo = math.exp((b + 1.96*se) * -SD_FROZEN)
    hi = math.exp((b - 1.96*se) * -SD_FROZEN)
    return dd, r, orr, lo, hi

dd0, r0, or0, lo0, hi0 = run(di)
print("主模型 OR=%.4f (%.4f-%.4f)" % (or0, lo0, hi0))

results = []
def add(name, extra, note=""):
    dd, r, o, lo, hi = run(di, extra)
    pct = (o - or0) / or0 * 100
    print("  %-34s OR=%.4f (%.4f-%.4f)  %+.1f%%" % (name, o, lo, hi, pct))
    results.append({"analysis": name, "or_pni": round(o,4), "ci": "%.4f-%.4f"%(lo,hi),
                    "pct_change": round(pct,1), "note": note})
    return o

# 低血压
add("+ 任何低血压(MAP<65)", ["has_under"])
add("+ 低血压持续分钟", ["under_10"])
add("+ 最低MAP", ["low_map_5"])
# 液体
add("+ 晶体量", ["crystal_500"])
add("+ 胶体量", ["colloid_500"])
add("+ 输血", ["transfuse_any"])
add("+ 失血量", ["ebl_500"])
add("+ 尿量", ["urine_500"])
# 血管活性药
add("+ 任一血管活性药", ["any_vaso"])
add("+ 麻黄碱", ["eph_mg"])
# 组合
add("+ 低血压+液体+输血+血管活性药", ["has_under","under_10","low_map_5","crystal_500","colloid_500","transfuse_any","ebl_500","urine_500","any_vaso"])
add("+ 全部(低血压+液体+输血+血管活性药+麻黄碱)", ["has_under","under_10","low_map_5","crystal_500","colloid_500","transfuse_any","ebl_500","urine_500","any_vaso","eph_mg"])

with open(os.path.join(PD, "combined_intraop_sensitivity.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["analysis","or_pni","ci","pct_change","note"])
    w.writeheader()
    w.writerows(results)
print("写出 -> combined_intraop_sensitivity.csv")
