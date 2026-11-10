"""例 3.2  彩电定价问题（第 3 章 · 非线性规划 · 建模五步法）

题目
----
两种彩电（19 英寸、21 英寸），零售价 339 / 399 美元，成本 195 / 225 美元，
固定成本 400000 美元。每多卖一台，同型号均价下降 1 美分；
交叉影响：每卖一台 21 英寸，19 英寸均价降 0.3 美分；每卖一台 19 英寸，
21 英寸均价降 0.4 美分。问各生产多少台？

建模五步法（书里强调的通用流程）
------------------------------
(1) 提出问题：定义变量、写假设，列成表 3.1
(2) 选择建模方法：本题是无约束多变量最优化
(3) 推导数学表达式：把售价、成本、收入写成 x1,x2 的函数
(4) 求解模型
(5) 回答问题：用普通话解释结果，并做灵敏度分析

推导
----
    p1 = 339 - 0.01 x1 - 0.003 x2
    p2 = 399 - 0.004 x1 - 0.01  x2
    c  = 400000 + 195 x1 + 225 x2
    r  = p1 x1 + p2 x2
    f  = r - c = -0.01 x1^2 - 0.007 x1 x2 - 0.01 x2^2 + 144 x1 + 174 x2 - 400000

f 是凹二次函数（负定 Hessian），无约束最优点解 ∇f = 0 即可。

书中答案：x1 = 4735, x2 = 7043，最大利润 f = 553641.025 美元。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import solve_nlp, report, check  # noqa: E402

A_PRICE, A_COST = 339.0, 399.0
C19, C21 = 195.0, 225.0
FIXED = 400000.0
SELF1, SELF2 = 0.01, 0.01          # 每多卖一台本型号，均价下降的幅度
CROSS12, CROSS21 = 0.003, 0.004    # 交叉影响


def prices(x1, x2):
    return (A_PRICE - SELF1 * x1 - CROSS12 * x2,
            A_PRICE * 0 + 399.0 - CROSS21 * x1 - SELF2 * x2)


def profit(x):
    x1, x2 = x
    p1 = A_PRICE - SELF1 * x1 - CROSS12 * x2
    p2 = 399.0 - CROSS21 * x1 - SELF2 * x2
    cost = FIXED + C19 * x1 + C21 * x2
    return p1 * x1 + p2 * x2 - cost


def profit_grad(x):
    x1, x2 = x
    df_dx1 = A_PRICE - 2 * SELF1 * x1 - (CROSS12 + CROSS21) * x2 - C19
    df_dx2 = 399.0 - (CROSS12 + CROSS21) * x1 - 2 * SELF2 * x2 - C21
    return np.array([df_dx1, df_dx2])


# ------------------------------------------------- 解析解（∇f = 0）
A = np.array([[2 * SELF1, CROSS12 + CROSS21],
              [CROSS12 + CROSS21, 2 * SELF2]])
b = np.array([A_PRICE - C19, 399.0 - C21])
x_analytic = np.linalg.solve(A, b)
print("解析解（令 ∇f = 0）: x1 =", round(x_analytic[0], 4),
      " x2 =", round(x_analytic[1], 4))

# ------------------------------------------------------- 数值求解（更通用）
res = solve_nlp(lambda x: -profit(x), x0=[4000.0, 7000.0],
                bounds=[(0, None), (0, None)], name="例3.2 彩电定价")
x = res["x"]
p1, p2 = prices(x[0], x[1])

report("例 3.2  彩电定价最优解", [
    ("19 英寸产量 x1", x[0]),
    ("21 英寸产量 x2", x[1]),
    ("年利润 f", profit(x)),
    ("19 英寸均价 p1", p1),
    ("21 英寸均价 p2", p2),
    ("总成本", FIXED + C19 * x[0] + C21 * x[1]),
    ("利润率", profit(x) / (FIXED + C19 * x[0] + C21 * x[1])),
])

print("\n与书中答案对照：")
check("x1", x[0], 4735, tol=1e-3, rel=True)
check("x2", x[1], 7043, tol=1e-3, rel=True)
check("年利润 f", profit(x), 553641.025, tol=1e-4, rel=True)
check("19 英寸均价 p1", p1, 270.52, tol=1e-4, rel=True)
check("21 英寸均价 p2", p2, 309.63, tol=1e-4, rel=True)

# ------------------------------------------- 第五步：灵敏度分析（价格弹性 a）
print("\n灵敏度分析：19 英寸的价格弹性系数 a（每多卖一台降价 a 美元）")
print("      a        x1         x2        年利润      利润率")
for a in [0.006, 0.008, 0.009, 0.010, 0.011, 0.012, 0.015, 0.020]:
    def pf(v, a=a):
        x1, x2 = v
        p1 = A_PRICE - a * x1 - CROSS12 * x2
        p2 = 399.0 - CROSS21 * x1 - SELF2 * x2
        return p1 * x1 + p2 * x2 - (FIXED + C19 * x1 + C21 * x2)

    r = solve_nlp(lambda v: -pf(v), x0=[4000.0, 7000.0],
                  bounds=[(0, None), (0, None)])
    xv = r["x"]
    cost = FIXED + C19 * xv[0] + C21 * xv[1]
    print(f"   {a:.3f}   {xv[0]:9.0f}   {xv[1]:9.0f}   {pf(xv):11.0f}   {pf(xv)/cost:8.1%}")

print("\n结论：a 越大（卖得越多降价越狠），最优产量越小、利润越薄。")
print("      这正是「薄利多销 vs 厚利少销」在模型里的样子 ——")
print("      灵敏度分析就是把模型的假设变成可以讨论的决策依据。")

# ------------------------------------------- 三维曲面图（书里的图 3.1/3.2）
try:
    from common.plotting import plt
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    g1_, g2_ = np.meshgrid(np.linspace(0, 12000, 60), np.linspace(0, 12000, 60))
    Z = (A_PRICE - SELF1 * g1_ - CROSS12 * g2_) * g1_ \
        + (399.0 - CROSS21 * g1_ - SELF2 * g2_) * g2_ \
        - (FIXED + C19 * g1_ + C21 * g2_)
    ax1.plot_surface(g1_ / 1000, g2_ / 1000, Z / 1000, cmap="viridis",
                     linewidth=0, antialiased=True)
    ax1.set_xlabel("x1 (千台)")
    ax1.set_ylabel("x2 (千台)")
    ax1.set_zlabel("利润 (千美元)")
    ax1.set_title("图 3.1 利润曲面")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contour(g1_ / 1000, g2_ / 1000, Z / 1000, levels=30, cmap="viridis")
    ax2.plot(x[0] / 1000, x[1] / 1000, "r*", ms=14, label="最优解")
    ax2.set_xlabel("x1 (千台)")
    ax2.set_ylabel("x2 (千台)")
    ax2.legend()
    ax2.set_title("图 3.2 水平集")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "figures", "ch3_tv_pricing.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\n图已保存: {out}")
except Exception as e:            # 绘图失败不该影响建模部分
    print("（绘图跳过：", e, "）")
