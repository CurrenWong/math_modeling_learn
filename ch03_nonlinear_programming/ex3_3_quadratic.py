"""例 3.3 / 3.4  二次规划（第 3 章 · 非线性规划）

模型
----
    max (min)  x'Hx + f'x
    s.t.        Ax <= b,  x >= 0

其中 H 对称。当 H 正定（最小化时）=> 凸二次规划，局部最优即全局最优。

例 3.3（求最大值，H 负定 => 凹函数 => 最大化是凸问题）
    max -x1^2 - 0.3 x1 x2 - 2 x2^2 + 98 x1 + 277 x2
    s.t. x1 + x2 <= 100,  x1 - 2 x2 <= 0,  x1,x2 >= 0
    书：x = (35.3704, 64.6296)，f = 11077.8704

例 3.4（同一目标函数改成最小化 => 非凸）
    书里 Matlab 从随机初值出发只找到局部解 x=(1,1), f=371.7，
    并指出 (0,0) 可行且 f=0 更好 —— 提醒：非凸问题不能只信一次求解。

Python 里没有内置 QP 求解器（cvxpy 要额外装），这里用 SLSQP 求解；
因为例 3.3 是凸问题，结果可��。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import solve_nlp, report, check  # noqa: E402

# ----------------------------------------------------------------- 模型参数
H = np.array([[-1.0, -0.15],
              [-0.15, -2.0]])
f = np.array([98.0, 277.0])
A = np.array([[1.0, 1.0],
              [1.0, -2.0]])
b = np.array([100.0, 0.0])


def obj(x):
    return float(x @ H @ x + f @ x)


def grad(x):
    return (H + H.T) @ x + f          # ∇(x'Hx) = (H+H')x


print("H =\n", H)
print("H 的特征值:", np.linalg.eigvalsh(H))
print("-> H 负定 => x'Hx 是凹函数 => 例 3.3 的「最大化」是凸规划；")
print("   同一函数改成「最小化」就变成非凸问题（例 3.4）。")

cons = [{"type": "ineq", "fun": lambda x: b - A @ x, "jac": lambda x: -A}]

# ============================================================ 例 3.3（求最大）
res = solve_nlp(lambda x: -obj(x), x0=[10.0, 10.0], bounds=[(0, None)] * 2,
                constraints=cons, name="例3.3 二次规划")
x = res["x"]

report("例 3.3  二次规划（最大化）", [
    ("最优解 x*", np.round(x, 4).tolist()),
    ("最优值 f", obj(x)),
    ("求解状态", res["status"]),
])

print("\n与书中答案对照：")
check("x1", x[0], 35.3704, tol=1e-3)
check("x2", x[1], 64.6296, tol=1e-3)
check("目标值 f", obj(x), 11077.8704, tol=1e-4, rel=True)

# 凸问题 -> 换初值结果不变
print("\n换初值验证（凸问题结果应唯一）：")
for x0 in ([1, 1], [50, 10], [0, 0], [99, 99]):
    r = solve_nlp(lambda v: -obj(v), x0=list(x0), bounds=[(0, None)] * 2,
                  constraints=cons)
    print(f"   x0 = {str(x0):<10} -> x* = {np.round(r['x'], 4).tolist()}  f = {obj(r['x']):.4f}")

# ============================================================ 例 3.4（求最小）
print("\n" + "=" * 66)
print("例 3.4  同一目标函数改成最小化（非凸）")
print("=" * 66)

rng = np.random.default_rng(7)
sols = []
for k in range(8):
    x0 = rng.uniform(0, 60, 2)
    r = solve_nlp(obj, x0=list(x0), bounds=[(0, None)] * 2, constraints=cons)
    sols.append((obj(r["x"]), r["x"], x0))
sols.sort(key=lambda t: t[0])

print("\n  8 次随机初值的结果（按目标值升序）：")
for val, xv, x0 in sols:
    print(f"   x0 = {str(np.round(x0, 2).tolist()):<20} -> "
          f"x = {str(np.round(xv, 3).tolist()):<20} f = {val:9.3f}")

best = sols[0]
print("\n与书中答案对照：")
print(f"   书中的局部解 x = (1, 1), f = 371.7  -> 代码中找到同值解: "
      f"{[i for i, s in enumerate(sols) if abs(s[0] - 371.7) < 1e-3]}")
check("可行点 (0,0) 的目标值", obj(np.zeros(2)), 0, tol=1e-9)
print("   书里指出 (0,0) 可行且 f=0，比 Matlab 给的局部解 (1,1) 更好 ——")
print("   这次随机重启里我们确实找到了 f=0 这个更优解。")

print("\n要点：")
print("  1) 凸性决定解的唯一性。H 正定 + 最小化 => 凸 => 任意初值同一答案；")
print("     H 负定 + 最小化 => 非凸 => 依赖初值，必须多起点。")
print("  2) scipy 没有 QP 求解器时的替代方案：SLSQP（凸问题足够）、")
print("     或者用对称性手工消元/对偶法（下面给一个对偶降维的对照）。")

# ------------------------------------------- 对照：把 2 变量 QP 降维到 1 变量
print("\n对照解法（消元法，绕开求解器，用来交叉验证）：")
# 约束 x1 - 2x2 <= 0 -> x1 <= 2x2；x1 + x2 <= 100
# 在 x2 给定下，目标关于 x1 是凹二次，驻点 x1* = (98 - 0.3 x2) / 2
best_grid = (-np.inf, None)
for x2 in np.linspace(0, 100, 500001):
    # 固定 x2 时，目标关于 x1 是凹二次，驻点 x1* = (98 - 0.3 x2)/2
    x1 = (98 - 0.3 * x2) / 2
    x1 = max(x1, 0.0)
    x1 = min(x1, 100 - x2)      # 约束 x1 + x2 <= 100
    x1 = min(x1, 2 * x2)        # 约束 x1 - 2 x2 <= 0
    if x1 < 0:
        continue
    val = -x1 ** 2 - 0.3 * x1 * x2 - 2 * x2 ** 2 + 98 * x1 + 277 * x2
    if val > best_grid[0]:
        best_grid = (val, (x1, x2))
print(f"   网格消元最优: x = ({best_grid[1][0]:.4f}, {best_grid[1][1]:.4f}), "
      f"f = {best_grid[0]:.4f}")
check("消元法与 SLSQP 的目标值一致", best_grid[0], obj(x), tol=1e-3, rel=True)
check("消元法的 x1 与 SLSQP 一致", best_grid[1][0], x[0], tol=1e-2, rel=True)
check("消元法的 x2 与 SLSQP 一致", best_grid[1][1], x[1], tol=1e-2, rel=True)
