"""例 2.10  比赛项目排序（第 2 章 · 0-1 规划 + 旅行商问题）

题目（2005 年电工杯数学建模竞赛 B 题）
------------------------------------
某小型运动会有 14 个比赛项目、40 名运动员，报名表已知。
要求：安排比赛项目的先后顺序，使得「同一名运动员连续参加两项比赛」的人次尽可能少。

建模思路（书里的做法，非常漂亮）
--------------------------------
1. 把「项目」当图的顶点，把「同时参加这两个项目的人数」当边权：
       w_ij = sum_m a_mi * a_mj
   注意 w_ij = w_ji，且 w_ii 无意义（设为无穷大）。
2. 目标就变成：找一个顶点排列，使相邻顶点之间的边权和最小 —— 这就是 TSP。
3. 因为起点和终点之间没有连接，引入一个虚拟项目 15，它与所有项目的边权都是 0，
   把问题变成 15 个点的 TSP，求最短回路后去掉虚拟点即可。

TSP 的 0-1 整数规划模型（MTZ 割）
-------------------------------
    min sum_{i,j} w_ij x_ij
    s.t. sum_j x_ij = 1     (每个点恰好离开一次)
         sum_i x_ij = 1     (每个点恰好进入一次)
         u_i - u_j + n x_ij <= n-1   (MTZ 割，消除子回路)
         1 <= u_i <= n-1
         x_ij ∈ {0,1}

书中答案：回路总长 2，即只有 2 名运动员连续参加比赛；
          顺序之一为 4→9→8→1→5→11→7→3→6→2→14→12→10→13。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import solve_milp, report, check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# ------------------------------------------------------------ 读报名表
raw = []
with open(os.path.join(DATA, "ch2_tournament.csv"), encoding="utf-8-sig") as f:
    for line in f.read().splitlines()[1:]:
        raw.append([int(v) for v in line.split(",")])
A = np.array([r[1:] for r in raw], dtype=float)      # 40 x 14
m, ne = A.shape
print(f"报名表: {m} 名运动员 x {ne} 个比赛项目")
print("每名运动员参加的项目数:", A.sum(axis=1).astype(int).tolist())

# -------------------------------------------------- 构造赋权图 w_ij
W = A.T @ A            # 14x14，同时参加两个项目的人数
print("\n边权矩阵 w_ij（同时参加两项目的人数，对角线无意义记为 inf）:")
W_show = W.astype(int).copy()
np.fill_diagonal(W_show, -1)
print(W_show)

# 加入虚拟项目 15（与所有点边权为 0），共 n=15 个点
n = ne + 1
anchor = ne                    # 以虚拟点 15 作为 MTZ 锚点，回路从它出发
Wt = np.zeros((n, n))
Wt[:ne, :ne] = W               # 虚拟点与所有真实项目的边权为 0
# 自环 x_ii 用等式约束直接禁掉，不必用大数（用大数会让 HiGHS 数值不稳）

# ------------------------------------------------------------- 变量
nvar = n * n + n       # x_ij (n*n) + u_i (n)


def xidx(i, j):
    return i * n + j


def uidx(i):
    return n * n + i


c = np.zeros(nvar)
for i in range(n):
    for j in range(n):
        c[xidx(i, j)] = Wt[i, j]

A_eq, b_eq = [], []
# 每个点恰好离开一次 / 到达一次
for i in range(n):
    row = np.zeros(nvar)
    for j in range(n):
        row[xidx(i, j)] = 1.0
    A_eq.append(row)
    b_eq.append(1.0)
for j in range(n):
    row = np.zeros(nvar)
    for i in range(n):
        row[xidx(i, j)] = 1.0
    A_eq.append(row)
    b_eq.append(1.0)
# 禁掉自环 x_ii = 0
for i in range(n):
    row = np.zeros(nvar)
    row[xidx(i, i)] = 1.0
    A_eq.append(row)
    b_eq.append(0.0)

A_ub, b_ub = [], []
# MTZ 割（锚点 = 虚拟点 anchor）：u_i - u_j + n*x_ij <= n-1
for i in range(n):
    if i == anchor:
        continue
    for j in range(n):
        if j == anchor or j == i:
            continue
        row = np.zeros(nvar)
        row[uidx(i)] = 1.0
        row[uidx(j)] = -1.0
        row[xidx(i, j)] = n
        A_ub.append(row)
        b_ub.append(n - 1)
# 1 <= u_i <= n-1（非锚点）
for i in range(n):
    if i == anchor:
        continue
    row = np.zeros(nvar)
    row[uidx(i)] = -1.0          # -u_i <= -1      ->  u_i >= 1
    A_ub.append(row)
    b_ub.append(-1.0)
    row = np.zeros(nvar)
    row[uidx(i)] = 1.0           #  u_i <= n-1
    A_ub.append(row)
    b_ub.append(n - 1)
# 锚点 u_anchor = 0
row = np.zeros(nvar)
row[uidx(anchor)] = 1.0
A_eq.append(row)
b_eq.append(0.0)

integrality = np.ones(nvar)
integrality[n * n:] = 0            # u_i 是连续的顺序号，不用取整
# 逐变量边界：x_ij ∈ [0,1]；u_i ∈ [0, n-1]
bounds = [(0, 1)] * (n * n) + [(0, n - 1)] * n

res = solve_milp(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                 A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                 bounds=bounds, integrality=integrality,
                 name="例2.10 比赛项目排序")
if not res["success"]:
    raise RuntimeError(f"TSP 求解失败：{res['status']}")

x = res["x"][:n * n].reshape(n, n)

# --------------------------------------------------------------- 还原回路
start = int(np.argmax(x[anchor, :]))       # 虚拟点之后的第一个真实项目
order_nodes, cur = [], start
for _ in range(ne):                        # 走 ne 步应当正好回到锚点
    order_nodes.append(cur)
    cur = int(np.argmax(x[cur, :]))
assert cur == anchor, "回路未闭合"
order = [t + 1 for t in order_nodes]       # 转成 1 起的项目编号（已去掉虚拟点）
assert sorted(order) == list(range(1, ne + 1)), "回路未覆盖所有项目"

report("例 2.10  比赛项目排序", [
    ("回路总长（连续参赛人次）", res["obj"]),
    ("最优顺序", " -> ".join(map(str, order))),
    ("求解状态", res["status"]),
])

print("\n与书中答案对照：")
check("回路总长", res["obj"], 2, tol=1e-6)
book_order = [4, 9, 8, 1, 5, 11, 7, 3, 6, 2, 14, 12, 10, 13]
print(f"   代码顺序: {order}")
print(f"   书中顺序: {book_order}")
print("   -> 最优回路往往不唯一，两条顺序可能不同，但总长必须相同")

# -------------------------------------------- 独立复核：数一数连续参赛的人次
print("\n独立复核（直接用报名表检查）：")
consecutive = 0
detail = []
for k in range(len(order) - 1):
    e1, e2 = order[k] - 1, order[k + 1] - 1
    cnt = int((A[:, e1] * A[:, e2]).sum())
    consecutive += cnt
    if cnt:
        detail.append((order[k], order[k + 1], cnt))
check("连续参赛人次（按顺序重算）", consecutive, 2, tol=1e-9)
print("   发生连续参赛的位置:", detail if detail else "无")
print("   全部 14 个项目各出现一次:",
      "是" if sorted(order) == list(range(1, 15)) else "否")
