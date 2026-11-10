"""例 2.6  装修任务分配（0-1 指派问题，第 2 章 · 整数规划）

题目
----
5 个新门店要装修，4 家装饰公司各自报价。为保证质量，每家公司最多承担 2 个门店；
每个门店必须由且仅由一家装修公司完成。如何分配使总费用最小？

模型
----
设 x_ij = 1 表示第 i 家公司承担第 j 个门店，否则 0。

    min sum_i sum_j c_ij x_ij
    s.t. sum_i x_ij = 1        (每个门店恰好一家公司)
         sum_j x_ij <= 2       (每家公司最多两个门店)
         x_ij ∈ {0,1}

注意这不是标准指派问题：4 家公司、5 个门店，人数与任务数不等，
所以不能直接套 Hungarian 算法，要用 0-1 整数规划。

书中答案：A 负责门店 3，B 负责门店 4，C 负责门店 1、2，D 负责门店 5，总费用 63.8。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import solve_milp, report, check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

rows = []
with open(os.path.join(DATA, "ch2_renovation.csv"), encoding="utf-8-sig") as f:
    for line in f.read().splitlines()[1:]:
        p = line.split(",")
        rows.append((p[0], [float(v) for v in p[1:]]))
companies = [r[0] for r in rows]
cost = np.array([r[1] for r in rows])        # 4 x 5
m, k = cost.shape
print("装修公司:", companies)
print("报价矩阵 (4x5):\n", cost)

# 变量 x_ij 展平成 m*k
def idx(i, j):
    return i * k + j


c = cost.ravel()

# 每个门店恰好一家公司：sum_i x_ij = 1
A_eq = np.zeros((k, m * k))
for j in range(k):
    for i in range(m):
        A_eq[j, idx(i, j)] = 1.0
b_eq = np.ones(k)

# 每家公司最多 2 个门店：sum_j x_ij <= 2
A_ub = np.zeros((m, m * k))
for i in range(m):
    for j in range(k):
        A_ub[i, idx(i, j)] = 1.0
b_ub = np.full(m, 2.0)

res = solve_milp(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                 bounds=(0, 1), name="例2.6 装修分配")
x = res["x"].reshape(m, k)

report("例 2.6  装修任务分配", [
    ("最小总费用 (万元)", res["obj"]),
    ("求解状态", res["status"]),
])

print("\n分配矩阵（1=承担）:")
for i in range(m):
    assigned = [f"门店{j+1}" for j in range(k) if x[i, j] > 0.5]
    print(f"   {companies[i]}: {assigned if assigned else '（未承担）'}")

print("\n与书中答案对照：")
check("最小总费用", res["obj"], 63.8, tol=1e-6, rel=True)
expected = {"A": ["门店3"], "B": ["门店4"], "C": ["门店1", "门店2"], "D": ["门店5"]}
got = {companies[i]: [f"门店{j+1}" for j in range(k) if x[i, j] > 0.5]
       for i in range(m)}
for comp in companies:
    same = set(got[comp]) == set(expected[comp])
    print(f"  [{'一致' if same else '不一致'}] {comp}: 代码={got[comp]}  书={expected[comp]}")
    assert same, f"{comp} 的分配与书中不一致"

print("\n提示：最优解可能不唯一（费用相同则有替代方案），"
      "检验时看总费用和可行性，而不是死抠某一张分配表。")
