"""例 1.6 / 1.7 / 1.8  把非线性问题线性化（第 1 章 · 线性规划）

这一节的思想非常重要：**很多看起来不是 LP 的问题，都能通过变量替换变回 LP。**
变回 LP 之后，交给求解器的是更少、更好的模型。

三个例子
--------
例 1.6  通用形式   min |x1|+...+|xn|   s.t. Ax <= b
例 1.7  具体算例   min |x1|+2|x2|+3|x3|+4|x4|  s.t. 三条线性约束
例 1.8  min max_i |eps_i|，其中 eps_i = x_i - y_i

线性化技巧
----------
1. 绝对值：引入 u_i, v_i >= 0，令 x_i = u_i - v_i，则 |x_i| = u_i + v_i。
   于是 min |x| 变成 min sum(u+v)，约束 Ax <= b 变成 A(u-v) <= b。
2. max |eps|：引入 v，令 -v <= eps_i <= v，目标变成 min v（一个变量、线性约束）。
   这个结构后面第 4 章「目标规划」、第 14 章「理想解法」还会反复出现。

书中答案（例 1.7）：x = (-2, 0, 0, 0)，最优值 z = 2。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import solve_lp, report, check  # noqa: E402

# ------------------------------------------------------------------ 例 1.7
c = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([-2.0, -1.0, -0.5])
A = np.array([
    [1, -1, -1, 1],
    [1, -1, 1, -3],
    [1, -1, -2, 3],
])

n = 4
# 变量顺序：[u1..u4, v1..v4]
# 目标 min c'(u+v)
c_full = np.concatenate([c, c])
# 约束 A(u-v) <= b  ->  [A, -A] [u;v] <= b
A_ub = np.hstack([A, -A])

res = solve_lp(c_full, A_ub=A_ub, b_ub=b, name="例1.7 绝对值线性化")
uv = res["x"]
u, v = uv[:n], uv[n:]
x = u - v

report("例 1.7  绝对值问题线性化求解", [
    ("最优值 z", res["obj"]),
    ("u = max(x,0)", np.round(u, 4).tolist()),
    ("v = max(-x,0)", np.round(v, 4).tolist()),
    ("还原 x = u - v", np.round(x, 4).tolist()),
])

print("\n与书中答案对照：")
check("最优值 z", res["obj"], 2, tol=1e-6, rel=True)
check("x1", x[0], -2, tol=1e-6, rel=True)
for k in range(1, 4):
    check(f"x{k+1}", x[k], 0, tol=1e-6)
assert np.allclose(A @ x, b, atol=1e-9) or np.all(A @ x <= b + 1e-9), "约束未满足"
print("  [一致] 还原后的 x 满足原约束 A x <= b")

# ------------------------------------------------------------------ 例 1.8
print("\n" + "=" * 66)
print("例 1.8  min max_i |eps_i|  的线性化")
print("=" * 66)
rng = np.random.default_rng(0)
y = rng.normal(size=5)
x0 = rng.normal(size=5)
eps = x0 - y
# 原问题：min_v  max_i |eps_i|，其中 eps_i = x_i - y_i 是已给定的量
# 线性化：min v  s.t.  eps_i <= v,  -eps_i <= v     （只对 v 这一个变量求解）
#   eps_i <= v   ⟺  -v <= -eps_i
#  -eps_i <= v   ⟺  -v <=  eps_i
A_ub8 = -np.ones((10, 1))
b_ub8 = np.concatenate([-eps, eps])
c8 = np.array([1.0])
res8 = solve_lp(c8, A_ub=A_ub8, b_ub=b_ub8, name="例1.8 min-max 线性化")
v_star = res8["x"][0]
report("例 1.8  min-max 问题", [
    ("最优值 v*", v_star),
    ("原问题 max|eps| 的精确值", np.max(np.abs(eps))),
    ("各 eps_i", np.round(eps, 4).tolist()),
])
assert abs(v_star - np.max(np.abs(eps))) < 1e-9
print("  [一致] 线性化模型解出的 v* 与直接算 max|eps| 完全相同")

print("\n小结：线性化不是「技巧」，而是建模的基本功。")
print("      遇到 |x|、max、分段函数，先想「能不能拆成 LP」。")
