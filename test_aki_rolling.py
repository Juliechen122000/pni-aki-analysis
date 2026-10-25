# -*- coding: utf-8 -*-
"""aki_rolling.judge_aki 边界案例测试（冻结前必须全部通过）"""
from aki_rolling import judge_aki

CASES = []


def case(name, t, c, expect_aki, expect_stage, ref_rule="min",
         preop_base=None, expect_first=None, note=""):
    r = judge_aki(t, c, ref_rule=ref_rule, preop_base=preop_base)
    ok = (r["aki"] == expect_aki) and (r["stage"] == expect_stage)
    if expect_first is not None:
        ok = ok and abs(r["first_time"] - expect_first) < 1e-9
    CASES.append((name, ok, r, note))


# 1 术前 5 天 0.8 -> 术后 1 天 1.1：无 48h 内前值，0.3 路径不应触发；
#   但相对路径参照 0.8 -> 1.1/0.8=1.375 <1.5，也不触发 => 非 AKI
case("late_preop_no_48h_pair", [-5, 1.0], [0.8, 1.1], 0.0, 0.0,
     expect_first=None, note="相隔6天，不能算48h内升高")

# 2 术后 0.2 天 0.8 -> 1.1（间隔0.8天）：0.3 路径触发
case("abs_within_48h", [0.2, 1.0], [0.8, 1.1], 1.0, 1.0, expect_first=1.0)

# 3 48h 内只升 0.25，但 7 天内比值 1.6：相对路径触发 1 期
case("rel_via_min_ref", [0.5, 3.0], [1.0, 1.6], 1.0, 1.0, expect_first=3.0)

# 4 最低值参照 vs 最近值参照的差异：
#    0.5d=1.0, 2d=1.05（最近），4d=1.55
#    min 参照=1.0 -> 1.55x 触发；last 参照=1.05 -> 1.476x 不触发
case("min_vs_last_primary", [0.5, 2.0, 4.5], [1.0, 1.05, 1.55],
     1.0, 1.0, ref_rule="min", expect_first=4.5)
case("min_vs_last_sensitivity", [0.5, 2.0, 4.5], [1.0, 1.05, 1.55],
     0.0, 0.0, ref_rule="last")

# 5 无任何 7 天内参照（只有术前 10 天一个值 + 术后一个值）：
#    min/last 都不能触发相对路径，且无 48h 对 => 不判 AKI，但保留患者
case("no_reference_window", [-10, 3.0], [0.9, 1.6], 0.0, 0.0,
     note="窗口内无参照，不触发但患者保留")

# 6 绝对路径 0.3 + 相对 2.2x => 2 期
case("stage2", [0.5, 1.0], [0.8, 1.76], 1.0, 2.0, expect_first=1.0)

# 7 3.2x => 3 期
case("stage3_ratio", [0.5, 1.0], [0.5, 1.6], 1.0, 3.0, expect_first=1.0)

# 8 峰值 >=4.0 且有急性升高 => 3 期
case("stage3_abs4", [0.5, 1.0], [1.2, 4.1], 1.0, 3.0, expect_first=1.0)

# 9 峰值 >=4.0 但无急性升高（长期稳定高值）=> 不应 3 期
case("stage3_abs4_no_acute", [0.5, 1.0], [4.0, 4.05], 0.0, 0.0,
     note="稳定高值不得误判3期")

# 10 术前基线参照（B 敏感性）：术前 0.8，术后峰值 1.3 => 1.6x 触发
case("preop_base_rel", [-3, 2.0], [0.8, 1.3], 1.0, 1.0,
     ref_rule="preop_base", preop_base=0.8, expect_first=2.0)

# 11 未来值不得回看：0.5d=1.6, 1d=1.0（后续下降）不应因未来低值改判
case("no_future_lookahead", [-0.5, 0.5, 1.0], [0.6, 0.8, 0.5], 0.0, 0.0,
     note="若误用未来的0.5，0.8/0.5=1.6会假阳性；正确应为0")

# 12 同一时间戳重复值：间隔0，不得作为48h对
case("same_timestamp", [1.0, 1.0], [0.8, 1.2], 0.0, 0.0)

fails = 0
for name, ok, r, note in CASES:
    print("%-30s %s  aki=%s stage=%s first=%s path=%s %s"
          % (name, "PASS" if ok else "FAIL", r["aki"], r["stage"],
             r["first_time"], r["path"], ("| " + note) if note else ""))
    if not ok:
        fails += 1

print("\n%d/%d passed" % (len(CASES) - fails, len(CASES)))
raise SystemExit(1 if fails else 0)
