"""例 3.1  一个凸非线性规划问题（第 3 章 · 非线性规划）

题目
----
    min f(x) = x1^2 + x2^2 - 4x1 + 4
    s.t. g1(x) = -x1 + x2 - 2 <= 0
         g2(x) = x1^2 - x2 + 1 <= 0
         x1, x2 >= 0

书中答案：x* = (0.5536, 1.3064)，f(x*) = 3.7989。

三件事一起做
------------
1. 判定是不是凸问题：Hessian 半正定 + 约束凸 => 凸规划 => 局部最优即全局最优；
2. 用 SLSQP 求解（初值很重要，SLSQP 是局部方法）；
3. 验证 K-T 条件：算出力学意义上「一阶条件」是否成立。

K-T 条件的实质
--------------
在最优点 x* 处，存在乘子 λ >= 0，使得
    ∇f(x*) + Σ λ_i ∇g_i(x*) = 0,   λ_i g_i(x*) = 0
第二式叫「互补松弛」：约束不起作用时乘子必须为 0。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import solve_nlp, report, check  # noqa: E402


def f(x):
    return x[0] ** 2 + x[1] ** 2 - 4 * x[0] + 4


def grad_f(x):
    return np.array([2 * x[0] - 4, 2 * x[1]])


def g1(x):
    return -x[0] + x[1] - 2


def g2(x):
    return x[0] ** 2 - x[1] + 1


def grad_g1(x):
    return np.array([-1.0, 1.0])


def grad_g2(x):
    return np.array([2 * x[0], -1.0])


# ---------------------------------------------------------- 1) 凸性判定
def hessian(x):
    return np.array([[2.0, 0.0], [0.0, 2.0]])


H = hessian([0, 0])
eig = np.linalg.eigvalsh(H)
print("目标函数 Hessian:\n", H, " 特征值:", eig)
print("-> 特征值全为正 => f 严格凸；g2 的 Hessian 半正定 => g2 凸；g1 线性。")
print("-> 整个问题是凸规划：K-T 条件既是必要条件也是充分条件。")

# ------------------------------------------------------------ 2) 求解
constraints = [
    {"type": "ineq", "fun": lambda x: -g1(x), "jac": lambda x: -grad_g1(x)},
    {"type": "ineq", "fun": lambda x: -g2(x), "jac": lambda x: -grad_g2(x)},
]
res = solve_nlp(f, x0=[1.0, 1.0], bounds=[(0, None), (0, None)],
                constraints=constraints, name="例3.1 凸NLP")

report("例 3.1  凸非线性规划", [
    ("最优解 x*", np.round(res["x"], 4).tolist()),
    ("最优值 f(x*)", res["obj"]),
    ("g1(x*)", g1(res["x"])),
    ("g2(x*)", g2(res["x"])),
    ("求解状态", res["status"]),
])

print("\n与书中答案对照：")
check("x1*", res["x"][0], 0.5536, tol=1e-3)
check("x2*", res["x"][1], 1.3064, tol=1e-3)
check("f(x*)", res["obj"], 3.7989, tol=1e-3, rel=True)

# -------------------------------------------------------- 3) K-T 条件验证
x_star = res["x"]
active = []
if abs(g1(x_star)) < 1e-8:
    active.append((grad_g1(x_star), "g1"))
if abs(g2(x_star)) < 1e-8:
    active.append((grad_g2(x_star), "g2"))
print(f"\n起作用的约束: {[n for _, n in active] or '无'}")

if active:
    J = np.column_stack([g for g, _ in active])
    lam, *_ = np.linalg.lstsq(J, -grad_f(x_star), rcond=None)
    print("解出的乘子 λ =", np.round(lam, 6), " (应 >= 0)")
    stat = grad_f(x_star) + J @ lam
    print("平稳性残差 |∇f + Σλ∇g| =", float(np.linalg.norm(stat)))
    assert lam.min() >= -1e-6, f"乘子应为非负，实际 {lam}"
    print("  [一致] 乘子非负（λ >= 0），符合 K-T 必要条件")
    check("平稳性残差", float(np.linalg.norm(stat)), 0, tol=1e-6)
    for (g, name), lam_i in zip(active, lam):
        print(f"   互补松弛 λ_{name} * g_{name}(x*) = {lam_i * eval(name + '(x_star)'):.2e}")

# ------------------------------------------ 4) 初值敏感性（局部方法的坑）
print("\n不同初值下的求解结果（凸问题下应该都一样）：")
for x0 in ([0.1, 0.1], [3, 3], [5, 0.5], [0.01, 5]):
    r = solve_nlp(f, x0=list(x0), bounds=[(0, None), (0, None)],
                  constraints=constraints)
    print(f"   x0 = {str(x0):<12} -> x* = {np.round(r['x'], 4).tolist()}  f = {r['obj']:.4f}")

print("\n要点：非线性规划是局部方法。凸问题时初值不敏感（结果唯一）；")
print("      非凸问题（例 3.4、3.9、3.10）就必须多起点 + 独立验证。")
