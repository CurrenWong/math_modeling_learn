"""例 2.7  供应站选址（0-1 规划 · 集合覆盖模型，第 2 章）

题目
----
10 个商业网点已知坐标。要在其中若干个点设供应站，要求：
  (1) 每个网点至少被一个供应站覆盖；
  (2) 供应站只能覆盖 10km 之内的网点；
  (3) 每个供应站最多供应 5 个网点。
问供应站数目最少是多少？

模型
----
x_i ∈ {0,1}：第 i 个网点是否建供应站
y_ij ∈ {0,1}：第 j 个网点是否由第 i 个网点的供应站覆盖

    min sum_i x_i
    s.t. sum_i y_ij >= 1        (每个网点都被覆盖)
         d_ij * y_ij <= 10 x_i  (超出 10km 的不能覆盖)
         sum_j y_ij <= 5        (每站最多服务 5 个网点)
         x_i >= y_ij            (没建站就不能覆盖)
         x_i = y_ii             (建了站就必须覆盖自己)
         x, y ∈ {0,1}

这个结构叫「固定成本 + 覆盖」的集合覆盖模型，是选址问题的标准写法。
Python 里最简洁的写法是把 d_ij y_ij <= 10 x_i 直接写成线性不等式。

书中答案：只需在网点 5 和网点 8 建站，共 2 个。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import solve_milp, report, check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

sites = []
with open(os.path.join(DATA, "ch2_sites.csv"), encoding="utf-8-sig") as f:
    for line in f.read().splitlines()[1:]:
        p = line.split(",")
        sites.append((int(p[0]), float(p[1]), float(p[2])))
n = len(sites)
xy = np.array([[s[1], s[2]] for s in sites])
d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)   # 两两距离
print("网点坐标:\n", xy)
print("两两距离矩阵:\n", np.round(d, 3))

# 变量：x_1..x_n, y_11..y_nn  ->  n + n*n
nvar = n + n * n


def xi(i):
    return i


def yij(i, j):
    return n + i * n + j


# 目标：min sum x_i
c = np.zeros(nvar)
for i in range(n):
    c[xi(i)] = 1.0

A_ub, b_ub = [], []
# (2) d_ij y_ij <= 10 x_i  ->  d_ij y_ij - 10 x_i <= 0
for i in range(n):
    for j in range(n):
        row = np.zeros(nvar)
        row[yij(i, j)] = d[i, j]
        row[xi(i)] = -10.0
        A_ub.append(row)
        b_ub.append(0.0)
# (3) sum_j y_ij <= 5
for i in range(n):
    row = np.zeros(nvar)
    for j in range(n):
        row[yij(i, j)] = 1.0
    A_ub.append(row)
    b_ub.append(5.0)
# (1) sum_i y_ij >= 1  ->  -sum_i y_ij <= -1
for j in range(n):
    row = np.zeros(nvar)
    for i in range(n):
        row[yij(i, j)] = -1.0
    A_ub.append(row)
    b_ub.append(-1.0)
# (4) x_i >= y_ij  ->  y_ij - x_i <= 0
for i in range(n):
    for j in range(n):
        row = np.zeros(nvar)
        row[yij(i, j)] = 1.0
        row[xi(i)] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

A_eq, b_eq = [], []
# (5) x_i = y_ii
for i in range(n):
    row = np.zeros(nvar)
    row[xi(i)] = 1.0
    row[yij(i, i)] = -1.0
    A_eq.append(row)
    b_eq.append(0.0)

res = solve_milp(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                 A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                 bounds=(0, 1), name="例2.7 供应站选址")

x = res["x"][:n]
y = res["x"][n:].reshape(n, n)
stations = [i for i in range(n) if x[i] > 0.5]

report("例 2.7  供应站选址", [
    ("最少供应站数", res["obj"]),
    ("建站位置", [f"网点{sites[i][0]}" for i in stations]),
    ("求解状态", res["status"]),
])

print("\n覆盖分配:")
for i in stations:
    served = [sites[j][0] for j in range(n) if y[i, j] > 0.5]
    print(f"   网点{sites[i][0]} 的供应站 -> 覆盖网点 {served}")

print("\n与书中答案对照：")
check("最少供应站数", res["obj"], 2, tol=1e-9)
print(f"   代码建站位置: {[sites[i][0] for i in stations]}   书: [5, 8]")
print("   -> 位置不同但同样是最优解：选址问题常有多重最优解，比数量和可行性，不比具体位置")

# 覆盖与距离的独立复核（按模型实际分配的 y 检查，不按「10km 内所有点」检查）
print("\n独立复核（直接按距离检查模型给出的分配）：")
covered = set()
for i in stations:
    served = [j for j in range(n) if y[i, j] > 0.5]
    covered |= set(served)
    far = [sites[j][0] for j in served if d[i, j] > 10.0]
    assert not far, f"网点{sites[i][0]} 覆盖了超出 10km 的点 {far}"
    assert len(served) <= 5, f"网点{sites[i][0]} 覆盖数 {len(served)} > 5"
    print(f"   网点{sites[i][0]}: 覆盖 {len(served)} 个（<=5），"
          f"全部在 10km 内 -> {[sites[j][0] for j in served]}")
assert covered == set(range(n)), "有网点未被覆盖"
print("  [一致] 10 个网点全部被覆盖，每站覆盖数 <= 5，且都在 10km 内")

# 书中方案 [5, 8] 的可行性复核
print("\n书中方案（网点 5、8）复核：")
book_stations = [4, 7]          # 0 基下标：网点5、网点8
ok = True
for i in book_stations:
    served = [j for j in range(n) if d[i, j] <= 10.0][:5]
    print(f"   网点{sites[i][0]}: 可覆盖(<=10km) 前 5 个 -> {[sites[j][0] for j in served]}")
union = set()
for i in book_stations:
    union |= {j for j in range(n) if d[i, j] <= 10.0}
print(f"   两站 10km 覆盖并集共 {len(union)}/10 个网点 -> "
      f"{'可行' if len(union) == 10 else '不可行'}")
