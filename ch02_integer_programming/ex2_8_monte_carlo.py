"""例 2.8  蒙特卡洛法求曲边三角形面积（第 2 章 · 随机模拟）

题目
----
y = x^2 与 y = 12 - x 与 x 轴在第一象限围成一个曲边三角形，
用蒙特卡洛法求其面积的近似值。

方法
----
在矩形 [0,12] x [0,9] 上撒 N 个均匀随机点，统计落在曲边三角形内的比例，
面积 ≈ 矩形面积 x 比例。

判定「在三角形内」的条件（分界点是两曲线的交点 x=3）：
    x <= 3  时，  y < x^2
    x >= 3  时，  y < 12 - x

精确值（可用来验证模拟精度）：
    ∫0^3 x^2 dx + ∫3^12 (12-x) dx = 9 + 31.5 = 49.5

书里用 10^7 个点，Matlab 单点循环跑了很久；Python 里用 numpy 向量化，
10^7 点只要 1 秒不到。

这也解释了第 2.2.2 节后面讲的蒙特卡洛可信度问题：
    P(至少命中一次) = 1 - (1-p)^N
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import report  # noqa: E402

# ---------------------------------------------------------------- 精确值
exact = 3 ** 3 / 3 + (12 * 12 - 12 ** 2 / 2) - (12 * 3 - 3 ** 2 / 2)
report("曲边三角形面积", [
    ("精确面积（解析积分）", exact),
    ("矩形面积 12x9", 108.0),
])

# ------------------------------------------------------- 蒙特卡洛（分块向量化）
def monte_carlo(n_total, block=1_000_000, seed=2024):
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    while done < n_total:
        m = min(block, n_total - done)
        x = rng.uniform(0, 12, m)
        y = rng.uniform(0, 9, m)
        inside = np.where(x <= 3, y < x ** 2, y < 12 - x)
        hits += int(inside.sum())
        done += m
    return hits / n_total * 108.0, hits, n_total


print("\n蒙特卡洛模拟（同一随机种子，样本量递增）：")
print("      N        面积估计        绝对误差")
for n_total in [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]:
    area, hits, n = monte_carlo(n_total)
    print(f"  {n:>9,}   {area:10.4f}      {abs(area - exact):8.4f}")

print("\n收敛速度：误差 ~ 1/sqrt(N)。样本量放大 100 倍，误差只缩小 10 倍，")
print("这就是蒙特卡洛法「便宜但慢收敛」的本质。")

# 多次重复，看估计量的分布（蒙特卡洛给出的是估计，不是精确答案）
print("\n用 N=10^6 重复 10 次，观察估计值的波动：")
ests = [monte_carlo(10 ** 6, seed=s)[0] for s in range(10)]
print("   " + "  ".join(f"{e:.3f}" for e in ests))
print(f"   均值 {np.mean(ests):.4f}   标准差 {np.std(ests):.4f}   精确值 {exact}")
