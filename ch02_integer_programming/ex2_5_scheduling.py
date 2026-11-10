"""例 2.5  生产线工人排班问题（第 2 章 · 整数规划）

题目
----
一条生产线 24h 不间断运转，6 个 4 小时班次各自需要的最少工人数已知
（35/40/50/45/55/30）。每名工人连续工作 8h，只能在某个班次开始时报到。
问至少需要多少名工人？

模型
----
设 x_i = 在第 i 个班次报到的工人数。

    min z = sum_i x_i
    s.t. x1+x6 >= 35   (0:00~4:00 由第1、6班次的人覆盖)
         x1+x2 >= 40
         x2+x3 >= 50
         x3+x4 >= 45
         x4+x5 >= 55
         x5+x6 >= 30
         x_i >= 0 且为整数

建模要点
--------
1. 8h 工作制正好跨两个班次，所以每个班次的人来自「本班次 + 上一班次」。
2. 变量必须是整数（人不能半个），这正是整数规划的意义：
   连续松弛解会给出小数人。
3. 注意 LP 松弛解与整数解的差别，这里可以算出来对比。

书中答案：x = (35, 5, 45, 0, 55, 0)，z = 140 人。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import solve_lp, solve_milp, report, check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

rows = []
with open(os.path.join(DATA, "ch2_demand.csv"), encoding="utf-8-sig") as f:
    for line in f.read().splitlines()[1:]:
        p = line.split(",")
        rows.append((int(p[0]), p[1], int(p[2])))
need = [r[2] for r in rows]
n = len(need)
print("各班次需求:", [(r[1], r[2]) for r in rows])

# 约束：第 i 个班次由「本班次报到的人 + 上一班次报到的人」覆盖
#      （每人连续工作 8h = 2 个班次，跨过午夜时 6 号班接 1 号班）
A_ub = np.zeros((n, n))
for i in range(n):
    A_ub[i, i] = -1.0
    A_ub[i, (i - 1) % n] = -1.0
b_ub = -np.array(need, dtype=float)          # -(x_i + x_{i-1}) <= -need_i

c = np.ones(n)

# ------------------------------------------------------- 整数解（真问题）
res = solve_milp(c, A_ub=A_ub, b_ub=b_ub, name="例2.5 排班(整数)")
x_int = res["x"]

# --------------------------------------- LP 松弛解（对比：会有小数人）
res_lp = solve_lp(c, A_ub=A_ub, b_ub=b_ub, name="例2.5 排班(LP松弛)")

report("例 2.5  工人排班最优解", [
    ("最少工人数 z", res["obj"]),
    ("各班次人数 x", np.round(x_int, 3).tolist()),
    ("求解状态", res["status"]),
])

print("\n与书中答案对照：")
check("最少工人数 z", res["obj"], 140, tol=1e-9)
for k, expected in enumerate([35, 5, 45, 0, 55, 0]):
    check(f"x{k+1}", x_int[k], expected, tol=1e-6)

print("\nLP 松弛解（如果允许小数人）：", np.round(res_lp["x"], 3).tolist(),
      f"  z = {res_lp['obj']:.4f}")
print(f"   整数性缺口 = {res_lp['obj'] - res['obj']:.4f} 人")
print("   -> 本例的 LP 松弛解恰好也是整数，所以缺口为 0；")
print("      但一般情况下松弛解是小数（比如 34.7 个人），必须靠整数规划保证可行性。")

print("\n覆盖情况复核：")
for i in range(n):
    cov = x_int[i] + x_int[(i - 1) % n]
    print(f"   {rows[i][1]:<12} 覆盖 {cov:5.0f} 人 >= 需求 {need[i]:3d}  "
          f"{'满足' if cov >= need[i] else '不满足!'}")
