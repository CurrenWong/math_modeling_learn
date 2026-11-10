"""例 3.8  Markowitz 投资组合（第 3 章 · 非线性规划 · 二次规划）

题目
----
A、B、C 三只股票过去 12 年的年收益率已知。
 (1) 希望组合收益的方差最小（风险最低），且期望年收益 >= 15%；
 (2) 希望在方差不超过 0.09 的前提下收益最大。
问两种情况下 A、B、C 各投多少比例。

模型
----
记 mu = 各股票平均收益率，F = 协方差矩阵，x = 投资比例。

 (1)  min  x'F x
      s.t. sum x = 1,  mu'x >= 0.15,  x >= 0

 (2)  max  mu'x
      s.t. sum x = 1,  x'F x <= 0.09,  x >= 0

两个问题都是凸问题：
 (1) 目标 x'Fx 在 F 半正定时是凸函数，最小化它是凸规划；
 (2) 目标线性，约束 x'Fx <= 0.09 定义的是凸集。

书中答案
--------
 (1) x = (0.5269, 0.3578, 0.1153)，最小方差 0.0228
 (2) x = (0, 0.0562, 0.9438)，最大收益 0.2334

本脚本还会用表 3.2 的原始数据反算 mu 和 F，核对表 3.3 的口径
（样本协方差用 n-1 作分母）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import solve_nlp, report, check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

raw = []
with open(os.path.join(DATA, "ch3_stocks.csv"), encoding="utf-8-sig") as f:
    for line in f.read().splitlines()[1:]:
        p = line.split(",")
        raw.append([float(v) for v in p[1:]])
R = np.array(raw)                       # 12 x 3
print("12 年收益率数据:\n", R)

mu_calc = R.mean(axis=0)
F_calc = np.cov(R, rowvar=False, ddof=1)      # 样本协方差（n-1）
F_pop = np.cov(R, rowvar=False, ddof=0)

# 书里表 3.3 的数值
mu_book = np.array([0.0882, 0.2137, 0.2346])
F_book = np.array([
    [0.0111, 0.0128, 0.0134],
    [0.0128, 0.0584, 0.0554],
    [0.0134, 0.0554, 0.0942],
])

print("\n用表 3.2 原始数据反算：")
print("   均值（代码）", np.round(mu_calc, 6).tolist(), " 书:", mu_book.tolist())
print("   协方差（代码，ddof=1）:\n", np.round(F_calc, 6))
print("   协方差（代码，ddof=0）:\n", np.round(F_pop, 6))
print("   协方差（书 表 3.3）:\n", F_book)
check("均值 A", mu_calc[0], mu_book[0], tol=1e-3, rel=True)
check("均值 B", mu_calc[1], mu_book[1], tol=1e-3, rel=True)
check("均值 C", mu_calc[2], mu_book[2], tol=1e-3, rel=True)
for i in range(3):
    for j in range(3):
        # 书里只给到小数点后 4 位，用绝对容差 5e-5 比较
        check(f"F[{i+1},{j+1}]", F_calc[i, j], F_book[i, j], tol=5e-5)
print("  [一致] 书里的 mu 与 F 就是这 12 年数据的均值和样本协方差（分母 n-1）")

# =============================================================== 问题 (1)
def var_fun(x):
    return float(x @ F_calc @ x)


cons1 = [
    {"type": "eq", "fun": lambda x: np.sum(x) - 1.0,
     "jac": lambda x: np.ones(3)},
    {"type": "ineq", "fun": lambda x: mu_calc @ x - 0.15,
     "jac": lambda x: mu_calc},
]
res1 = solve_nlp(var_fun, x0=np.array([1 / 3, 1 / 3, 1 / 3]),
                 bounds=[(0, None)] * 3, constraints=cons1, name="例3.8(1) 最小方差")

report("例 3.8 (1)  期望收益 >= 15% 时的最小风险组合", [
    ("投资比例 x", np.round(res1["x"], 4).tolist()),
    ("组合方差（风险）", res1["obj"]),
    ("组合期望收益", float(mu_calc @ res1["x"])),
    ("求解状态", res1["status"]),
])
print("\n与书中答案对照：")
for k, expected in enumerate([0.5269, 0.3578, 0.1153]):
    check(f"x{k+1}", res1["x"][k], expected, tol=1e-3)
check("最小方差", res1["obj"], 0.0228, tol=5e-5)

# =============================================================== 问题 (2)
def neg_ret(x):
    return -float(mu_calc @ x)


cons2 = [
    {"type": "eq", "fun": lambda x: np.sum(x) - 1.0,
     "jac": lambda x: np.ones(3)},
    {"type": "ineq", "fun": lambda x: 0.09 - var_fun(x),
     "jac": lambda x: -2 * F_calc @ x},
]
res2 = solve_nlp(neg_ret, x0=np.array([1 / 3, 1 / 3, 1 / 3]),
                 bounds=[(0, None)] * 3, constraints=cons2, name="例3.8(2) 最大收益")

report("例 3.8 (2)  方差 <= 0.09 时的最大收益组合", [
    ("投资比例 x", np.round(res2["x"], 4).tolist()),
    ("组合期望收益", float(mu_calc @ res2["x"])),
    ("组合方差", var_fun(res2["x"])),
    ("求解状态", res2["status"]),
])
print("\n与书中答案对照：")
for k, expected in enumerate([0.0, 0.0562, 0.9438]):
    check(f"x{k+1}", res2["x"][k], expected, tol=1e-3)
check("最大收益", float(mu_calc @ res2["x"]), 0.2334, tol=5e-5)

# ------------------------------------------------------- 有效前沿（顺带画）
print("\n最小方差前沿（沿收益率目标扫描）：")
frontier = []
for target in np.linspace(0.10, 0.235, 40):
    c = [
        {"type": "eq", "fun": lambda x, t=target: np.sum(x) - 1.0,
         "jac": lambda x: np.ones(3)},
        {"type": "ineq", "fun": lambda x, t=target: mu_calc @ x - t,
         "jac": lambda x: mu_calc},
    ]
    r = solve_nlp(var_fun, x0=np.array([1 / 3, 1 / 3, 1 / 3]),
                  bounds=[(0, None)] * 3, constraints=c)
    if r["success"]:
        frontier.append((r["obj"], float(mu_calc @ r["x"]), r["x"]))
for var, ret, xv in frontier[::8]:
    print(f"   方差 {var:.4f}  收益 {ret:.4f}  比例 {np.round(xv, 3).tolist()}")

try:
    from common.plotting import plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    v = [f[0] for f in frontier]
    rr = [f[1] for f in frontier]
    ax[0].plot(rr, v, "-o", ms=4, color="#2b6cb0")
    ax[0].scatter([rr[0]], [v[0]], color="red", zorder=5, label="全局最小方差")
    ax[0].set_xlabel("期望收益")
    ax[0].set_ylabel("方差（风险）")
    ax[0].set_title("Markowitz 有效前沿")
    ax[0].legend()

    names = ["A", "B", "C"]
    for xv, lbl, mk in ((res1["x"], "(1) 收益>=15%, 最小风险", "o"),
                        (res2["x"], "(2) 方差<=0.09, 最大收益", "s")):
        ax[1].scatter(range(3), xv, marker=mk, s=120, label=lbl)
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(names)
    ax[1].set_ylabel("投资比例")
    ax[1].set_title("两个最优解的资产配置对比")
    ax[1].legend()
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "ch3_markowitz.png")
    fig.savefig(out, dpi=130)
    print(f"\n图已保存: {out}")
except Exception as e:
    print("（绘图跳过：", e, "）")

print("\n要点：")
print("  1) 协方差矩阵的样本口径很关键：用 n-1 还是 n，结果差一个系数。")
print("  2) 两个问题都是凸规划，SLSQP 给的是全局最优；")
print("     但 (1) 的约束 mu'x >= 0.15 会把解「推」到可行域边界上。")
print("  3) (2) 里 A 的权重是 0：A 的方差虽小但收益也低，")
print("     在 0.09 的风险额度下不如全押 C —— 这正是分散化的代价。")
