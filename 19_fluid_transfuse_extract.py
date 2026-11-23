# -*- coding: utf-8 -*-
"""
19_fluid_transfuse_extract.py —— 提取 INSPIRE 腹部手术队列术中液体/输血/失血/尿量
术中窗口 = opstart_time <= chart_time <= opend_time（入院后分钟）。
从 vitals.csv.gz 汇总各字段总量（mL / Unit）：
  晶体: ns, hs, hns, psa, ds, d5w, d10w, d50w
  胶体: hes, alb20, alb5
  输血: rbc, ffp, pc, pheresis, cryo
  失血: ebl
  尿量: uo
输出: $INSPIRE_OUT/fluid_transfuse_exposure.csv
"""
import csv, gzip, collections, statistics, os

DATA_DIR = os.environ.get("INSPIRE_DATA", "./INSPIRE-1.4.2")
OUT_DIR = os.environ.get("INSPIRE_OUT", "output")
os.makedirs(OUT_DIR, exist_ok=True)
VITALS = os.path.join(DATA_DIR, "vitals.csv.gz")
OPS = os.path.join(DATA_DIR, "operations.csv.gz")
COHORT = os.path.join(OUT_DIR, "p0_aki_inspire_harmonized.csv")
OUT = os.path.join(OUT_DIR, "fluid_transfuse_exposure.csv")

CRYSTAL = ["ns", "hs", "hns", "psa", "ds", "d5w", "d10w", "d50w"]
COLLOID = ["hes", "alb20", "alb5"]
TRANSFUSE = ["rbc", "ffp", "pc", "pheresis", "cryo"]

# 1) 队列 op_id
ops_info = {}
with open(COHORT, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ops_info[row["op_id"]] = True
print("队列 op_id:", len(ops_info))

# 2) 术中窗口
ops_window = {}
with gzip.open(OPS, "rt", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        oid = row.get("op_id", "")
        if oid in ops_info:
            try:
                os_ = float(row["opstart_time"]) if row.get("opstart_time") else None
                oe_ = float(row["opend_time"]) if row.get("opend_time") else None
            except (ValueError, TypeError):
                os_, oe_ = None, None
            ops_window[oid] = (os_, oe_)
print("有术中窗口:", len(ops_window))

# 3) 汇总各字段
sums = collections.defaultdict(lambda: collections.defaultdict(float))
counts = collections.defaultdict(lambda: collections.defaultdict(int))
n_read = 0
with gzip.open(VITALS, "rt", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        oid = row.get("op_id", "")
        if oid not in ops_window:
            continue
        os_, oe_ = ops_window[oid]
        if os_ is None or oe_ is None:
            continue
        it = row.get("item_name", "")
        if it not in (CRYSTAL + COLLOID + TRANSFUSE + ["ebl", "uo"]):
            continue
        try:
            val = float(row.get("value", ""))
            t = float(row.get("chart_time", ""))
        except (ValueError, TypeError):
            continue
        if os_ <= t <= oe_:
            sums[oid][it] += val
            counts[oid][it] += 1
        n_read += 1
print("扫描行数:", n_read)

# 4) 写出
fieldnames = ["op_id"] + ["crystal_ml", "colloid_ml", "transfuse_units",
                          "ebl_ml", "urine_ml", "n_fluid_records"]
rows = []
for oid in ops_window:
    s = sums[oid]
    rows.append({
        "op_id": oid,
        "crystal_ml": round(s.get("ns", 0) + s.get("hs", 0) + s.get("hns", 0) + s.get("psa", 0)
                           + s.get("ds", 0) + s.get("d5w", 0) + s.get("d10w", 0) + s.get("d50w", 0), 0),
        "colloid_ml": round(s.get("hes", 0) + s.get("alb20", 0) + s.get("alb5", 0), 0),
        "transfuse_units": round(s.get("rbc", 0) + s.get("ffp", 0) + s.get("pc", 0)
                                 + s.get("pheresis", 0) + s.get("cryo", 0), 1),
        "ebl_ml": round(s.get("ebl", 0), 0),
        "urine_ml": round(s.get("uo", 0), 0),
        "n_fluid_records": sum(counts[oid].values()),
    })
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print("写出:", len(rows), "->", OUT)

# 5) 汇总
if rows:
    tot = len(rows)
    for col in ["crystal_ml", "colloid_ml", "transfuse_units", "ebl_ml", "urine_ml"]:
        vals = [r[col] for r in rows]
        nz = sum(1 for v in vals if v > 0)
        print(f"  {col}: 中位 {statistics.median(vals):.0f}, 有>0 {nz}/{tot} ({nz/tot*100:.1f}%)")
