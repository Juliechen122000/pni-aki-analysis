# -*- coding: utf-8 -*-
"""
20_fluid_sensitivity.py —— 术中液体/输血/失血/尿量敏感性分析
在主模型 Model B（N=7,889, 739事件, OR=1.4647）基础上，逐一加入：
  术中晶体量、胶体量、输血量、失血量、尿量，以及组合。
输出: $INSPIRE_OUT/fluid_sensitivity_results.csv
"""
import csv, math, os
import pandas as pd
import statsmodels.api as sm

PD = os.environ.get("INSPIRE_OUT", "output")
os.makedirs(PD, exist_ok=True)
COHORT = os.path.join(PD, "p0_aki_inspire_harmonized.csv")
FLUID = os.path.join(PD, "fluid_transfuse_exposure.csv")
SD_FROZEN = 6.445246

# 1) 构建主样本
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

# 2) 合并液体数据
fl = pd.read_csv(FLUID)
di = di.merge(fl, on="op_id", how="left")
print("主样本:", len(di), "事件:", int(di.aki_unified.sum()))
print("液体合并:", di["crystal_ml"].notna().sum(), "/", len(di))

# 3) 模型函数
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

# 主模型
dd0, r0, or0, lo0, hi0 = run(di)
print("\n主模型 OR=%.4f (%.4f-%.4f)" % (or0, lo0, hi0))

# 4) 逐一加入
# 晶体量每500ml、胶体量每500ml、输血（是否）、失血每500ml、尿量每500ml
di["crystal_500"] = di["crystal_ml"] / 500
di["colloid_500"] = di["colloid_ml"] / 500
di["transfuse_any"] = (di["transfuse_units"] > 0).astype(int)
di["ebl_500"] = di["ebl_ml"] / 500
di["urine_500"] = di["urine_ml"] / 500

results = []
def add(name, extra, note=""):
    dd, r, o, lo, hi = run(di, extra)
    pct = (o - or0) / or0 * 100
    print("  %-28s OR=%.4f (%.4f-%.4f)  %+.1f%%" % (name, o, lo, hi, pct))
    results.append({"analysis": name, "or_pni": round(o,4), "ci": "%.4f-%.4f"%(lo,hi),
                    "pct_change": round(pct,1), "note": note})
    return o

add("+晶体量(每500ml)", ["crystal_500"])
add("+胶体量(每500ml)", ["colloid_500"])
add("+输血(是否)", ["transfuse_any"])
add("+失血量(每500ml)", ["ebl_500"])
add("+尿量(每500ml)", ["urine_500"])
# 组合：全部液体/输血
add("+晶体+胶体", ["crystal_500","colloid_500"])
add("+晶体+胶体+输血", ["crystal_500","colloid_500","transfuse_any"])
add("+晶体+胶体+输血+失血", ["crystal_500","colloid_500","transfuse_any","ebl_500"])
add("+全部(晶体+胶体+输血+失血+尿量)", ["crystal_500","colloid_500","transfuse_any","ebl_500","urine_500"])

# 写结果
with open(os.path.join(PD, "fluid_sensitivity_results.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["analysis","or_pni","ci","pct_change","note"])
    w.writeheader()
    w.writerows(results)
print("\n写出 -> fluid_sensitivity_results.csv")
