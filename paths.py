"""
统一路径与读取封装 —— INSPIRE 1.4.2 升级版
=========================================
用途：消灭散落在各脚本中的硬编码 DATA_DIR / OUT，
      并统一 .csv.gz + UTF-8-BOM + case_id 浮点化的读取方式。

用法：
    from paths import DATA_DIR, OUT, rd, load_ops, PBW
"""

import os

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ 路径
# 通过环境变量指定，避免把个人机器路径硬编码进公开仓库。
# 运行前请设置（示例）：
#   INSPIRE_DATA = /path/to/INSPIRE-1.4.2        （INSPIRE 数据目录，从 PhysioNet 下载）
#   INSPIRE_OUT  = /path/to/output               （结果输出目录）
#   MOVER_RAW    = /path/to/MOVER/EPIC_EMR       （MOVER 原始抽取目录，签 UCI DUA 后下载）
DATA_DIR = os.environ.get("INSPIRE_DATA", "./INSPIRE-1.4.2")
OUT = os.environ.get("INSPIRE_OUT", "output")
WORK = os.environ.get("INSPIRE_WORK", ".")

os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ 常量
MIN_OP_DURATION = 120      # 腹部大手术操作性定义：手术时长 >=120 min
TRAJ_WINDOW = (-0.02, 7)   # 术后 0-7 天暴露窗口
MIN_PERIODS = 3            # 至少 3 个 2 天时段
PERIOD_LEN = 2             # 每时段 2 天

# labs.csv.gz 中本研究使用的 8 项指标
LAB_ITEMS = [
    "albumin", "lymphocyte", "creatinine", "bun",
    "hba1c", "hct", "glucose", "crp",
]

# ATC 前缀：肾毒性药物类别
NEPHROTOXIN_ATC = {
    "NSAID": ["M01A"],
    "Diuretic": ["C03"],
    "ACEI": ["C09A", "C09B"],
    "ARB": ["C09C", "C09D"],
}


def rd(table, **kw):
    """读取 compressed csv（不带扩展名）。

    INSPIRE 1.4.2 全部为 .csv.gz 且带 UTF-8 BOM，务必走此封装，
    禁止裸 pd.read_csv(DATA_DIR/...)。
    """
    kw.setdefault("compression", "gzip")
    kw.setdefault("encoding", "utf-8-sig")
    kw.setdefault("low_memory", False)
    return pd.read_csv(os.path.join(DATA_DIR, table + ".csv.gz"), **kw)


def load_ops():
    """读取 operations 表并补齐派生列。

    1.4.2 的 operations 表：
      - 无 approach 列 -> 从 icd10_pcs 第 5 位（0-based index 4）解析
      - 无 op_duration 列 -> opend_time - opstart_time（单位：分钟）
      - case_id 被写成 6392.0 -> 须 Int64
    """
    ops = rd("operations", dtype={"case_id": "Int64"})
    code = ops.icd10_pcs.astype(str)
    ops["approach"] = code.str[4]
    ops["op_duration"] = ops.opend_time - ops.opstart_time
    return ops


def clean_hw_ops(ops, verbose=True):
    """清洗身高体重。

    1.4.2 CHANGELOG 声称已清洗，实测未生效：
      height <100cm          185 行
      height 1400-1900        4 行（疑似单位 mm / 或录入错位）
      weight <10kg          135 行
    """
    ops = ops.copy()
    h = pd.to_numeric(ops["height"], errors="coerce")
    w = pd.to_numeric(ops["weight"], errors="coerce")

    if verbose:
        print(f"  [清洗前] height 中位 {h.median():.1f} 范围 {h.min():.1f}-{h.max():.1f} ; "
              f"weight 中位 {w.median():.1f} 范围 {w.min():.1f}-{w.max():.1f}")

    # 1400-1900 段视为单位错误（mm -> cm）
    wrong_unit = h.between(1400, 1900)
    if verbose and wrong_unit.sum():
        print(f"  修正 height 疑似单位错误(/10): {wrong_unit.sum()} 行")
    h = pd.Series(np.where(wrong_unit, h / 10.0, h), index=ops.index)

    # 生理范围外置 NaN
    n_bad = ((~h.between(100, 200)) & h.notna()).sum()
    if verbose and n_bad:
        print(f"  height 超出 [100,200] 置缺失: {n_bad} 行")
    h = h.where(h.between(100, 200))

    n_badw = ((~w.between(10, 200)) & w.notna()).sum()
    if verbose and n_badw:
        print(f"  weight 超出 [10,200] 置缺失: {n_badw} 行")
    w = w.where(w.between(10, 200))

    ops["height_c"] = h
    ops["weight_c"] = w

    if verbose:
        print(f"  [清洗后] height 中位 {h.median():.1f} 缺失 {h.isna().sum()} ; "
              f"weight 中位 {w.median():.1f} 缺失 {w.isna().sum()}")
    return ops


def devine_pbw(height_cm, is_male):
    """Devine 理想体重（kg）。身高单位 cm。

    男性 50 + 0.91*(h-152.4)；女性 45.5 + 0.91*(h-152.4)
    注意：不要使用 operations 表自带的 pbw 列 —— 它已被百分位截尾
    （75/95/99 分位同为 60.92），无法用于 mL/kg 换算。
    """
    h = pd.to_numeric(height_cm, errors="coerce")
    return np.where(is_male, 50.0, 45.5) + 0.91 * (h - 152.4)


def save(df, name, **kw):
    """统一保存到 OUT 目录。"""
    kw.setdefault("index", False)
    kw.setdefault("encoding", "utf-8-sig")
    p = os.path.join(OUT, name)
    df.to_csv(p, **kw)
    print(f"  已保存: {p}  ({len(df):,} 行 x {df.shape[1]} 列)")
    return p
