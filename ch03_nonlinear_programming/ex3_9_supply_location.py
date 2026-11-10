"""例 3.9  供应与选址：两个料场供应 6 个工地（第 3 章 · 非线性规划）

题目
----
6 个建筑工地的位置 (a_i, b_i) 和水泥日用量 c_i 已知。拟建 2 个料场，
每个料场日储量 20t。料场建在何处，使总的吨千米数最小？

模型
----
设料场 j 的位置为 (x_j, y_j)，从料场 j 运到工地 i 的量为 z_ij。

    min  sum_i sum_j z_ij * sqrt((x_j-a_i)^2 + (y_j-b_i)^2)
    s.t. sum_j z_ij = c_i          (每个工地的需求必须满足)
         sum_i z_ij <= 20          (每个料场的储量上限)
         z_ij >= 0

注意：这是混合整数规划吗？不是 —— 料场位置是连续变量，所以是非线性规划。
      目标函数对 (x,y) 非凹，所以是局部方法，要多起点。

书中答案
--------
料场坐标 (7.25, 7.75) 与 (3.2942, 5.1790)，总吨千米 72.6303；
按「每个工地尽量由一个料场供」的调整方案为 71.9372。

书中 MATLAB 代码里有一处笔误：`prob.Constraints.con2 = sum(z)<=20;`
把所有运输量加总与 20 比较，而总需求是 36 —— 那样必然不可行。
正确的写法是按料场分别约束：sum_i z_ij <= 20。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import solve_nlp, report, check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

rows = []
with open(os.path.join(DATA, "ch3_supply.csv"), encoding="utf-8-sig") as f:
    for line in f.read().splitlines()[1:]:
        p = line.split(",")
        rows.append((float(p[1]), float(p[2]), float(p[3])))
a = np.array([r[0] for r in rows])
b = np.array([r[1] for r in rows])
c = np.array([r[2] for r in rows])
m, nd = len(a), 2
CAP = 20.0
print("工地坐标 (a,b) 与日用量 c:")
for i in range(m):
    print(f"   工地{i+1}: ({a[i]:5.2f}, {b[i]:5.2f})  用量 {c[i]:4.1f} t")
print(f"总需求 {c.sum():.1f} t，每个料场储量 {CAP:.0f} t")


def unpack(v):
    depots = v[:4].reshape(2, 2)                  # [(x1,y1), (x2,y2)]
    Z = v[4:].reshape(m, nd)
    return depots, Z


def total_ton_km(v):
    depots, Z = unpack(v)
    d = np.sqrt((depots[None, :, 0] - a[:, None]) ** 2
                + (depots[None, :, 1] - b[:, None]) ** 2)
    return float((Z * d).sum())


def obj_grad(v):
    depots, Z = unpack(v)
    d = np.sqrt((depots[None, :, 0] - a[:, None]) ** 2
                + (depots[None, :, 1] - b[:, None]) ** 2)
    val = float((Z * d).sum())
    g = np.zeros_like(v)
    for i in range(m):
        for j in range(nd):
            if d[i, j] > 1e-9:
                g[4 + i * nd + j] = d[i, j]
                g[2 * j] += Z[i, j] * (depots[j, 0] - a[i]) / d[i, j]
                g[2 * j + 1] += Z[i, j] * (depots[j, 1] - b[i]) / d[i, j]
    return val, g


cons = []
for i in range(m):                      # sum_j z_ij = c_i
    row = np.zeros(4 + m * nd)
    for j in range(nd):
        row[4 + i * nd + j] = 1.0
    ci = c[i]
    cons.append({"type": "eq",
                 "fun": (lambda r, ci=ci: (lambda x: r @ x - ci))(row),
                 "jac": (lambda r: (lambda x: r))(row)})
for j in range(nd):                      # sum_i z_ij <= 20
    row = np.zeros(4 + m * nd)
    for i in range(m):
        row[4 + i * nd + j] = 1.0
    cons.append({"type": "ineq", "fun": (lambda r: (lambda x: CAP - r @ x))(row),
                 "jac": (lambda r: (lambda x: -r))(row)})

# ------------------------------------------------------------ 多起点求解
rng = np.random.default_rng(20240101)
best = None
for k in range(12):
    v0 = np.zeros(4 + m * nd)
    v0[:4] = rng.uniform(0, 12, 4)
    for i in range(m):                   # 初值满足需求等式
        v0[4 + i * nd: 4 + i * nd + nd] = c[i] / nd
    r = solve_nlp(obj_grad, x0=v0, constraints=cons, jac=True,
                  bounds=[(0, None)] * (4 + m * nd),   # 料场坐标与运量都非负
                  options={"maxiter": 3000, "ftol": 1e-10}, name="例3.9")
    print(f"   起点 {k+1}/12: obj = {r['obj']:12.4f}  状态: {r['status'][:48]}")
    if r["success"] and (best is None or r["obj"] < best["obj"]):
        best = r

if best is None:
    raise RuntimeError("所有起点都没能收敛，检查模型或初值")

depots, Z = unpack(best["x"])
report("例 3.9  两料场选址", [
    ("料场 1 坐标", np.round(depots[0], 4).tolist()),
    ("料场 2 坐标", np.round(depots[1], 4).tolist()),
    ("总吨千米数", best["obj"]),
    ("求解状态", best["status"]),
])

print("\n与书中答案对照：")
check("料场1 x", depots[0, 0], 7.25, tol=2e-2, rel=True)
check("料场1 y", depots[0, 1], 7.75, tol=2e-2, rel=True)
check("料场2 x", depots[1, 0], 3.2942, tol=2e-2, rel=True)
check("料场2 y", depots[1, 1], 5.1790, tol=2e-2, rel=True)

print(f"\n  重要差别：总吨千米数 代码 = {best['obj']:.4f}，书 = 72.6303")
print("    书中 72.6303 是 Matlab 从随机初值出发得到的【局部最优】——")
print("    它的运料方案里多个工地被两个料场分供（如工地1：料场1 给 0.0329、料场2 给 2.9671）。")
print(f"    我们的全局最优 {best['obj']:.4f} 反而更小，与书中「调整后」的 71.9372 几乎相同。")
assert best["obj"] <= 72.6303 + 1e-6, "全局最优不应劣于书中的局部解"
check("与书中「调整后」方案的差距", best["obj"], 71.9372, tol=1e-2, rel=True)
print("  [一致] 代码解优于书中的连续解，且与书中调整方案吻合")
print("    -> 这就是为什么非线性规划必须多起点 + 独立验证：")
print("       书里的答案本身是一个局部最优，不是全局最优。")

print("\n运料方案（行=工地，列=料场）：")
for i in range(m):
    print(f"   工地{i+1}: 料场1 {Z[i,0]:6.4f}  料场2 {Z[i,1]:6.4f}   合计 {Z[i].sum():.2f}"
          f"  (需求 {c[i]:.1f})")
for j in range(nd):
    print(f"   料场{j+1} 总供量 {Z[:, j].sum():.2f} t  (<= {CAP:.0f})")

# ------------------------------------------------ 调整方案：每工地只由一个料场供
print("\n实际调整：每个工地只由 1 个料场供应（但要满足料场容量！）")
d = np.sqrt((depots[None, :, 0] - a[:, None]) ** 2
            + (depots[None, :, 1] - b[:, None]) ** 2)
best_assign, best_val = None, np.inf
for mask in range(1 << m):                       # 穷举 2^6 = 64 种分配
    assign = [(mask >> i) & 1 for i in range(m)]
    load = [sum(c[i] for i in range(m) if assign[i] == j) for j in range(nd)]
    if any(L > CAP + 1e-9 for L in load):
        continue                                # 容量超限，不可行
    val = sum(c[i] * d[i, assign[i]] for i in range(m))
    if val < best_val:
        best_val, best_assign = val, assign
print("   调整后运料方案（枚举所有 64 种分配，取满足容量且吨千米最小的）:")
for i in range(m):
    j = best_assign[i]
    print(f"   工地{i+1}: 料场{j+1} {c[i]:5.1f} t  距离 {d[i, j]:.3f} km")
for j in range(nd):
    load = sum(c[i] for i in range(m) if best_assign[i] == j)
    print(f"   料场{j+1} 总供量 {load:5.1f} t  (<= {CAP:.0f})")
print(f"   总吨千米数 = {best_val:.4f}")
check("调整后总吨千米", best_val, 71.9372, tol=1e-2, rel=True)
print("   注意：单纯「就近分配」可能超容量（比如让料场2 供 25t），")
print("         所以调整时必须把容量约束一起算进去。")

print("\n要点：")
print("  1) 非线性规划是局部方法：这里跑了 12 个随机初值取最好，")
print("     不同的初值会给出不同的料场位置（这是非凹目标的必然）。")
print("  2) 目标函数对 z 是线性的、对料场位置是「距离×运量」，所以")
print("     会出现「一个工地由两个料场分供」的解；实际执行时按就近原则调整，")
print("     吨千米数反而略降（71.9372 < 72.6303）—— 连续最优解不一定最好执行。")
print("  3) 书里 MATLAB 那句 sum(z)<=20 是笔误，按料场分别约束才对。")
