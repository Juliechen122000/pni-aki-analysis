# -*- coding: utf-8 -*-
"""KDIGO 滚动时间窗 AKI 判定 —— 单一来源实现（2026-10-25 冻结版）

冻结口径（见 plans/quantum-aurora-babbage-kx44CwMP.md）：
  绝对升高路径：对每个术后 0-7 天候选值 C(t)，存在更早测量 C(s) 且
                0 < t-s <= 48h，C(t)-C(s) >= 0.3 mg/dL。
  相对升高路径：
      ref_rule="min"  -> B_min(t) = min{C(s): t-7d <= s < t}     （主分析）
      ref_rule="last" -> B_last(t)= 最近一次严格早于 t 的值       （敏感性）
      ref_rule="preop_base" -> 固定术前最近一次肌酐为参照         （敏感性 B）
      C(t) >= 1.5 * ref 即达标；分期 2.0x / 3.0x 同参照。
  共同规则：只用候选时点之前的真实测量；窗口内无参照 -> 该时点不触发相对路径，
            但患者不剔除（按其他时点判定，并标记是否曾可判定）。
  首次 AKI 时间、是否 AKI、分期均由本函数同一逻辑输出。
"""
import numpy as np

POST_LO = 0.0       # 术后候选必须严格在手术结束之后（负值属术前，不得作为候选）
POST_HI = 7.0
ABS_WIN_D = 2.0     # 48 h
ABS_DELTA = 0.3
REL_WIN_D = 7.0


def judge_aki(t, c, ref_rule="min", preop_base=None):
    """返回 AKI 判定结果字典。

    Parameters
    ----------
    t, c : array-like
        同一患者的肌酐测量时间（相对手术结束的天数）与数值，顺序任意。
    ref_rule : {"min", "last", "preop_base"}
    preop_base : float, optional
        ref_rule="preop_base" 时使用的术前基线肌酐。

    Returns
    -------
    dict: aki (0/1/NaN), stage (0-3/NaN), first_time, path, basis,
          n_cand, n_abs_eval, n_rel_eval, evaluable
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    ok = np.isfinite(t) & np.isfinite(c) & (c > 0)
    t, c = t[ok], c[ok]
    order = np.argsort(t, kind="stable")
    t, c = t[order], c[order]

    out = dict(aki=np.nan, stage=np.nan, first_time=np.nan, path="",
               basis="", n_cand=0, n_abs_eval=0, n_rel_eval=0, evaluable=0)
    if len(t) == 0:
        return out

    post = np.where((t >= POST_LO) & (t <= POST_HI))[0]
    out["n_cand"] = int(len(post))
    if len(post) == 0:
        return out

    best_stage = 0.0
    first_time = np.nan
    first_path = ""
    first_basis = ""
    any_eval = False

    pb = np.nan
    if ref_rule == "preop_base":
        try:
            pb = float(preop_base)
        except (TypeError, ValueError):
            pb = np.nan

    for i in post:
        ti, ci = t[i], c[i]

        # ---------------- 绝对路径：真实 48h 测量对 ----------------
        prior_abs = np.where((t < ti) & ((ti - t) <= ABS_WIN_D))[0]
        abs_hit = False
        abs_detail = ""
        if len(prior_abs):
            out["n_abs_eval"] += 1
            any_eval = True
            diffs = ci - c[prior_abs]
            j = int(np.argmax(diffs))
            if diffs[j] >= ABS_DELTA:
                abs_hit = True
                abs_detail = "abs +%.2f/%.2fd" % (
                    diffs[j], ti - t[prior_abs[j]])

        # ---------------- 相对路径 ----------------
        ref = np.nan
        rel_detail = ""
        if ref_rule == "preop_base":
            if np.isfinite(pb) and pb > 0:
                ref = pb
                out["n_rel_eval"] += 1
                any_eval = True
        else:
            prior_rel = np.where((t < ti) & ((ti - t) <= REL_WIN_D))[0]
            if len(prior_rel):
                out["n_rel_eval"] += 1
                any_eval = True
                if ref_rule == "min":
                    ref = float(np.min(c[prior_rel]))
                else:                       # last
                    ref = float(c[prior_rel[-1]])

        ratio = ci / ref if (np.isfinite(ref) and ref > 0) else np.nan
        rel_hit = bool(np.isfinite(ratio) and ratio >= 1.5)
        if rel_hit:
            rel_detail = "rel %.2fx" % ratio

        # ---------------- 分期 ----------------
        s = 0
        if abs_hit or rel_hit:
            s = 1
        if np.isfinite(ratio) and ratio >= 2.0:
            s = 2
        if np.isfinite(ratio) and ratio >= 3.0:
            s = 3
        if ci >= 4.0 and (abs_hit or rel_hit):
            s = 3

        if s >= 1 and not np.isfinite(first_time):
            first_time = float(ti)
            if abs_hit and rel_hit:
                first_path = "abs+rel"
            elif abs_hit:
                first_path = "abs"
            else:
                first_path = "rel"
            first_basis = "; ".join(x for x in (abs_detail, rel_detail) if x)

        if s > best_stage:
            best_stage = float(s)

    out["aki"] = 1.0 if best_stage >= 1 else 0.0
    out["stage"] = best_stage
    out["first_time"] = first_time
    out["path"] = first_path
    out["basis"] = first_basis
    out["evaluable"] = 1 if any_eval else 0
    return out
