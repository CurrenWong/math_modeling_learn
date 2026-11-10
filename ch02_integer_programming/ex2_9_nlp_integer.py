"""例 2.9  非线性整数规划：蒙特卡洛法 + 局部搜索（第 2 章）

题目
----
max z = x1^2 + x2^2 + 3x3^2 + 4x4^2 + 2x5^2 - 8x1 - 2x2 - 3x3 - x4 - 2x5
s.t.
    0 <= x_i <= 99,  x_i 为整数
    x1+x2+x3+x4+x5 <= 400
    x1+2x2+2x3+x4+6x5 <= 800
    2x1+x2+6x3 <= 200
    x3+x4+5x5 <= 200

为什么要用蒙特卡洛
------------------
枚举要算 100^5 = 10^10 个点，不现实。但整数解只有有限个，
用随机采样撞上高值区的概率可以估算：
    P(至少命中一次) = 1 - (1-p)^N
    p=0.01,  N=10^6  ->  1-0.99^10^6 ≈ 1
    p=0.00001, N=10^6 -> 1-0.99999^10^6 ≈ 0.99995

书中答案
--------
蒙特卡洛（10^6 个点）：x=(46, 98, 1, 99),  z = 50273
Lingo 精确全局最优：  x=(50, 99, 0, 99, 20), z = 51568

本脚本做两件事
--------------
1. 复现书里的蒙特卡洛法（种子固定，结果可复现）；
2. 在蒙特卡洛解上做一次「整数邻域爬山」，看看能不能跳到真正的最优解
   —— 这正是实际工作中用 MC 时的标准做法：MC 给初值，局部搜索给答案。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import report, check  # noqa: E402


def objective(x):
    return (x[0] ** 2 + x[1] ** 2 + 3 * x[2] ** 2 + 4 * x[3] ** 2 + 2 * x[4] ** 2
            - 8 * x[0] - 2 * x[1] - 3 * x[2] - x[3] - 2 * x[4])


def violations(x):
    """返回所有违反的约束量（<=0 表示满足）。"""
    return np.array([
        np.sum(x) - 400,
        x[0] + 2 * x[1] + 2 * x[2] + x[3] + 6 * x[4] - 800,
        2 * x[0] + x[1] + 6 * x[2] - 200,
        x[2] + x[3] + 5 * x[4] - 200,
    ])


def feasible(x):
    return np.all(violations(x) <= 0) and np.all(x >= 0) and np.all(x <= 99)


# ======================================================= 1) 蒙特卡洛（书中做法）
N = 1_000_000
rng = np.random.default_rng(0)          # 书里也是 rng(0)，便于对比
best_x, best_f = None, -np.inf
# 向量化采样 + 逐点可行性判定（比 Python 循环快几个数量级）
X = rng.integers(0, 100, size=(N, 5))
G = np.column_stack([
    X.sum(axis=1) - 400,
    X[:, 0] + 2 * X[:, 1] + 2 * X[:, 2] + X[:, 3] + 6 * X[:, 4] - 800,
    2 * X[:, 0] + X[:, 1] + 6 * X[:, 2] - 200,
    X[:, 2] + X[:, 3] + 5 * X[:, 4] - 200,
])
ok = np.all(G <= 0, axis=1)
F = (X[:, 0] ** 2 + X[:, 1] ** 2 + 3 * X[:, 2] ** 2 + 4 * X[:, 3] ** 2
     + 2 * X[:, 4] ** 2 - 8 * X[:, 0] - 2 * X[:, 1] - 3 * X[:, 2] - X[:, 3] - 2 * X[:, 4])
feasible_f = F[ok]
if feasible_f.size:
    i = int(np.argmax(feasible_f))
    best_x, best_f = X[ok][i], float(feasible_f.max())

report("蒙特卡洛（10^6 个随机点）", [
    ("可行点比例", f"{ok.mean() * 100:.3f}%"),
    ("最优解 x", best_x.tolist() if best_x is not None else None),
    ("目标值 z", best_f),
])
print("   书中蒙特卡洛解: x = [46, 98, 1, 99]      z = 50273")
print("   书中 Lingo 精确最优: x = [50, 99, 0, 99, 20]  z = 51568")

# ============================================ 2) 整数邻域爬山（把 MC 解磨到最优）
def hill_climb(x0, max_rounds=20000):
    """整数邻域爬山。

    邻域不能只有「单变量 ±1」：第三个约束 2x1+x2+6x3<=200 把 x1,x2,x3 耦合在一起，
    从 [49,96,1,...] 想走到 [50,99,0,...] 必须同时动两个变量。
    所以邻域里加入两变量联动移动 (+1,-1)/(-1,+1)/(+1,+1)/(+1,-2) 等。
    """
    x = x0.astype(int).copy()
    f = objective(x)

    def moves():
        for i in range(5):
            for d in (-3, -2, -1, 1, 2, 3):
                yield (i, d)
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                for di, dj in ((1, -1), (-1, 1), (1, 1), (-1, -1), (2, -1), (-1, 2)):
                    yield (i, j, di, dj)

    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for mv in moves():
            cand = x.copy()
            if len(mv) == 2:
                i, d = mv
                cand[i] += d
            else:
                i, j, di, dj = mv
                cand[i] += di
                cand[j] += dj
            if np.any(cand < 0) or np.any(cand > 99):
                continue
            if not feasible(cand):
                continue
            fc = objective(cand)
            if fc > f + 1e-9:
                x, f, improved = cand, fc, True
    return x, f, rounds


x_hc, f_hc, rounds = hill_climb(best_x)
report("蒙特卡洛 + 整数邻域爬山", [
    ("爬山轮数", rounds),
    ("最优解 x", x_hc.tolist()),
    ("目标值 z", f_hc),
])

print("\n与书中答案对照：")
check("Lingo 精确最优 z", f_hc, 51568, tol=1e-6, rel=True)
for k, expected in enumerate([50, 99, 0, 99, 20]):
    check(f"x{k+1}", x_hc[k], expected, tol=1e-9)

# 多次随机重启，验证解的稳健性
print("\n10 次随机重启（每次 10^5 点 + 爬山）：")
z_list = []
for s in range(10):
    r = np.random.default_rng(100 + s)
    Xs = r.integers(0, 100, size=(100_000, 5))
    Gs = np.column_stack([
        Xs.sum(axis=1) - 400,
        Xs[:, 0] + 2 * Xs[:, 1] + 2 * Xs[:, 2] + Xs[:, 3] + 6 * Xs[:, 4] - 800,
        2 * Xs[:, 0] + Xs[:, 1] + 6 * Xs[:, 2] - 200,
        Xs[:, 2] + Xs[:, 3] + 5 * Xs[:, 4] - 200,
    ])
    oks = np.all(Gs <= 0, axis=1)
    if not oks.any():
        continue
    Fs = (Xs[:, 0] ** 2 + Xs[:, 1] ** 2 + 3 * Xs[:, 2] ** 2 + 4 * Xs[:, 3] ** 2
          + 2 * Xs[:, 4] ** 2 - 8 * Xs[:, 0] - 2 * Xs[:, 1] - 3 * Xs[:, 2]
          - Xs[:, 3] - 2 * Xs[:, 4])
    xs, fs = Xs[oks][int(np.argmax(Fs[oks]))], float(Fs[oks].max())
    xh, fh, _ = hill_climb(xs)
    z_list.append(fh)
print("   爬山后的 z:", "  ".join(f"{z:.0f}" for z in z_list))
print(f"   最好 {max(z_list):.0f} / 最差 {min(z_list):.0f} / 平均 {np.mean(z_list):.1f}")
print("   -> 蒙特卡洛只保证「大概率撞到好解」，要拿到最优解必须再加局部搜索。")

print("\n为什么这里能做局部搜索？目标函数是凸二次型，但我们做的是**最大化**，")
print("属于非凸优化，局部搜索只能保证局部最优 —— 所以必须靠多起点 + 联动移动，")
print("这也是「蒙特卡洛给初值、局部搜索给答案」这条经验法则的意义。")
print("一般非线性整数规划连这种结构都没有，这正是书里说「尚未有成熟算法」的原因。")
