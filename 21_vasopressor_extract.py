# -*- coding: utf-8 -*-
"""
21_vasopressor_extract.py —— 提取 INSPIRE 腹部手术队列术中血管活性药
术中窗口 = opstart_time <= chart_time <= opend_time（入院后分钟）。
字段: nepi(去甲肾上腺素 ug/kg/min), epi(肾上腺素 ug), epii(肾上腺素输注),
      dobui(多巴酚丁胺), dopai(多巴胺), mlni(米力农), ntgi(硝酸甘油),
      eph(麻黄碱 mg), phe(去氧肾上腺素 ug), vaso(血管活性药)
输出: $INSPIRE_OUT/vasopressor_exposure.csv
"""
import csv, gzip, collections, statistics, os

DATA_DIR = os.environ.get("INSPIRE_DATA", "./INSPIRE-1.4.2")
OUT_DIR = os.environ.get("INSPIRE_OUT", "output")
os.makedirs(OUT_DIR, exist_ok=True)
VITALS = os.path.join(DATA_DIR, "vitals.csv.gz")
OPS = os.path.join(DATA_DIR, "operations.csv.gz")
COHORT = os.path.join(OUT_DIR, "p0_aki_inspire_harmonized.csv")
OUT = os.path.join(OUT_DIR, "vasopressor_exposure.csv")

VASO = ["nepi", "epi", "epii", "dobui", "dopai", "mlni", "ntgi", "eph", "phe", "vaso"]

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

# 3) 提取（是否使用 + 总记录数 + 累计剂量）
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
        if it not in VASO:
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

# 4) 写出（是否使用任一血管活性药）
rows = []
for oid in ops_window:
    s = sums[oid]
    c = counts[oid]
    any_vaso = any(c.get(v, 0) > 0 for v in VASO)
    rows.append({
        "op_id": oid,
        "any_vaso": int(any_vaso),
        "n_vaso_records": sum(c.get(v, 0) for v in VASO),
        "nepi_rec": c.get("nepi", 0),
        "epi_rec": c.get("epi", 0),
        "eph_mg": round(s.get("eph", 0), 1),
        "phe_ug": round(s.get("phe", 0), 0),
    })
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("写出:", len(rows), "->", OUT)

# 5) 汇总
if rows:
    tot = len(rows)
    anyv = sum(1 for r in rows if r["any_vaso"])
    print(f"  任一血管活性药使用: {anyv}/{tot} ({anyv/tot*100:.1f}%)")
    for col in ["eph_mg", "phe_ug"]:
        vals = [r[col] for r in rows]
        nz = sum(1 for v in vals if v > 0)
        print(f"  {col}: 中位 {statistics.median(vals):.1f}, 有>0 {nz}/{tot} ({nz/tot*100:.1f}%)")
