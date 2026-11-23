# -*- coding: utf-8 -*-
"""
17_low_bp_extract.py —— 提取 INSPIRE 腹部手术队列的术中低血压暴露
v2：确认时间轴为「入院后分钟」（admission_time=0 锚点），
    术中窗口 = opstart_time <= chart_time <= opend_time（均为入院后分钟）。
从 vitals.csv.gz 提取 MAP（nibp_mbp 无创 + art_mbp 动脉），
计算：MAP<65 mmHg 累计持续分钟、低血压 AUC、最低 MAP、MAP<65 占比。
输出：$INSPIRE_OUT/low_bp_exposure.csv（op_id 级）
"""
import csv, gzip, collections, statistics, os

DATA_DIR = os.environ.get("INSPIRE_DATA", "./INSPIRE-1.4.2")
OUT_DIR = os.environ.get("INSPIRE_OUT", "output")
os.makedirs(OUT_DIR, exist_ok=True)
VITALS = os.path.join(DATA_DIR, "vitals.csv.gz")
OPS = os.path.join(DATA_DIR, "operations.csv.gz")
COHORT = os.path.join(OUT_DIR, "p0_aki_inspire_harmonized.csv")
OUT = os.path.join(OUT_DIR, "low_bp_exposure.csv")

# 1) 队列 op_id
ops_info = {}
with open(COHORT, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ops_info[row["op_id"]] = True
print(f"队列 op_id: {len(ops_info)}")

# 2) 术中窗口（入院后分钟）
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
print(f"有术中窗口: {len(ops_window)}")

# 3) 提取 MAP（只取术中窗口内）
map_series = collections.defaultdict(list)
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
        if it not in ("nibp_mbp", "art_mbp"):
            continue
        try:
            val = float(row.get("value", ""))
            t = float(row.get("chart_time", ""))
        except (ValueError, TypeError):
            continue
        if os_ <= t <= oe_:  # 术中窗口内
            map_series[oid].append((t, val, it))
        n_read += 1
print(f"扫描 {n_read} 行，术中窗口内有 MAP 的 op: {len(map_series)}")

# 4) 计算低血压暴露
results = []
for oid, series in map_series.items():
    series.sort(key=lambda x: x[0])
    times = [x[0] for x in series]
    maps = [x[1] for x in series]
    total_under = 0.0
    auc = 0.0
    n_under = sum(1 for m in maps if m < 65)
    min_map = min(maps)
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        if 0 < dt <= 15:  # 间隔<=15min（分钟）
            avg_map = (maps[i] + maps[i + 1]) / 2
            if avg_map < 65:
                total_under += dt
            auc += max(0, 65 - avg_map) * dt
    results.append({
        "op_id": oid,
        "n_map": len(maps),
        "min_map": min_map,
        "n_under65": n_under,
        "total_under_min": round(total_under, 1),
        "auc_under65": round(auc, 1),
        "pct_under65": round(n_under / len(maps) * 100, 1),
    })
    if len(results) % 500 == 0:
        print(f"  ...{len(results)}")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["op_id"])
    w.writeheader()
    w.writerows(results)
print(f"写出 {len(results)} 例 -> {OUT}")

if results:
    tot = len(results)
    have_min = sum(1 for r in results if r["min_map"] < 65)
    have_under = sum(1 for r in results if r["total_under_min"] > 0)
    print(f"有 MAP 患者: {tot}")
    print(f"最低 MAP<65: {have_min} ({have_min/tot*100:.1f}%)")
    print(f"低血压时间>0: {have_under} ({have_under/tot*100:.1f}%)")
    print(f"最低 MAP 中位: {statistics.median(r['min_map'] for r in results)}")
    print(f"<65 累计分钟中位: {statistics.median(r['total_under_min'] for r in results)}")
    print(f"AUC 中位: {statistics.median(r['auc_under65'] for r in results)}")
