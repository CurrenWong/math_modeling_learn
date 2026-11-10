"""例 1.5  6 产地 8 销地的最小费用运输问题（第 1 章 · 线性规划）

模型
----
设 x_ij = 产地 A_i 运往销地 B_j 的数量。

    min sum_i sum_j c_ij x_ij
    s.t. sum_i x_ij = d_j      (每个销地的需求必须满足)
         sum_j x_ij <= e_i     (产地的产量上限)
         x_ij >= 0

注意总产量 302 > 总需求 280，这是「满足供应」的运输问题：约束是
「= 需求」+「<= 产量」，两者不是同时取等的标准运输问题。

书中答案：最小运费 664（最优解不唯一）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import solve_lp, report, check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

raw = []
with open(os.path.join(DATA, "ch1_transport.csv"), encoding="utf-8-sig") as f:
    for line in f.read().splitlines()[1:]:
        raw.append([t.strip() for t in line.split(",")])

m, n = 6, 8
cost = np.array([[float(v) for v in r[1:1 + n]] for r in raw[:m]])      # 运价
supply = np.array([float(r[1 + n]) for r in raw[:m]])                   # 产量
demand = np.array([float(v) for v in raw[m][1:1 + n]])                  # 需求量

print("运价矩阵 (6x8):\n", cost.astype(int))
print("产量:", supply.astype(int), " 合计", supply.sum())
print("需求:", demand.astype(int), " 合计", demand.sum())
print("总产量 - 总需求 =", supply.sum() - demand.sum(), "-> 满足供应的运输问题")


def idx(i, j):
    return i * n + j


# 约束：需求等式 sum_i x_ij = d_j
A_eq = np.zeros((n, m * n))
for j in range(n):
    for i in range(m):
        A_eq[j, idx(i, j)] = 1.0
b_eq = demand

# 约束：产量不等式 sum_j x_ij <= e_i
A_ub = np.zeros((m, m * n))
for i in range(m):
    for j in range(n):
        A_ub[i, idx(i, j)] = 1.0
b_ub = supply

res = solve_lp(cost.ravel(), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
               name="例1.5 运输问题")
x = res["x"].reshape(m, n)

report("例 1.5  运输问题最优解", [
    ("最小总运费", res["obj"]),
    ("求解状态", res["status"]),
])
print("\n最优运量矩阵（行=产地，列=销地）：")
for i in range(m):
    print(f"   A{i+1}: " + " ".join(f"{x[i, j]:4.0f}" for j in range(n)))
print("   合计: " + " ".join(f"{x[:, j].sum():4.0f}" for j in range(n)) + "   <- 各销地收到的量")

print("\n与书中答案对照：")
check("最小运费", res["obj"], 664, tol=1e-6, rel=True)
for j in range(n):
    assert abs(x[:, j].sum() - demand[j]) < 1e-6, f"销地 B{j+1} 需求未满足"
for i in range(m):
    assert x[i, :].sum() <= supply[i] + 1e-6, f"产地 A{i+1} 超产"
print("  [一致] 需求全部满足、产量全部不超 -> 约束正确")

print("\n提示：最优解不唯一。书中给的表 1.3 也是其中一个最优解，"
      "\n      逐元素与本代码结果比对没有意义，比对目标值才有意义。")
