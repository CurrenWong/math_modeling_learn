"""例 1.4  捷运公司仓库租赁问题（第 1 章 · 线性规划）

题目
----
捷运公司要在 1~4 月租用仓库堆放物资。各月所需面积已知，仓库租借费用随合同期
变长而打折。合同每月初可签若干份（面积与期限可不同），5 月起不再需要仓库。
求使总租借费用最小的签约方案。

模型
----
设 x_ij = 第 i 个月初签订的、租期 j 个月的仓库面积（单位 100m²）。

    min z = 2800(x11+x21+x31+x41) + 4500(x12+x22+x32)
             + 6000(x13+x23) + 7300 x14
    s.t. 第 k 个月的在租面积 >= 第 k 个月的需求量
         x_ij >= 0

「在租面积」= 当月新签的 + 之前签的尚未到期部分，例如
    第 1 月：x11+x12+x13+x14 >= 15
    第 2 月：x12+x13+x14+x21+x22+x23 >= 10
    第 3 月：x13+x14+x22+x23+x31+x32 >= 20
    第 4 月：x14+x23+x32+x41 >= 12

注意 x33/x34/x42/x43/x44 为 0：这些合同会跨到 5 月，而 5 月不需要仓库。

书中答案：x11=3, x31=8, x14=12，其余为 0，最优值 z=118400 元。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import solve_lp, report, check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# ----------------------------------------------------------------- 读数据
rows = np.genfromtxt(os.path.join(DATA, "ch1_warehouse.csv"), delimiter=",",
                      skip_header=1, dtype=None, encoding="utf-8-sig")
need = np.array([r[1] for r in rows], dtype=float)      # 各月需求面积
fee = {j: float(rows[0][2 + j - 1]) for j in range(1, 5)}  # 各期限合同的总费用
print("各月所需面积(100m²):", need.tolist())
print("各合同期总费用(元):", fee)

# ------------------------------------------------- 变量编号：x(i,j) -> 4*i+j
NM, NJ = 4, 4


def idx(i, j):
    """i, j 均为 1 起的下标。"""
    return (i - 1) * NJ + (j - 1)


# 目标函数：x_ij 只有在「合同期不会跨出 4 个月」时才有意义
BIG = 1e6
c = np.full(NM * NJ, BIG)
for i in range(1, NM + 1):
    for j in range(1, NJ + 1):
        if i + j - 1 <= NM:          # 合同覆盖到第 i+j-1 月，不越过第 4 月
            c[idx(i, j)] = fee[j]

# 约束：每月在租面积 >= 需求（转成 linprog 需要的 <= 形式）
A_ub, b_ub = [], []
for k in range(1, NM + 1):
    row = np.zeros(NM * NJ)
    for i in range(1, NM + 1):
        for j in range(1, NJ + 1):
            # 合同 (i,j) 在第 k 月仍生效 <=> i <= k <= i+j-1
            if i <= k <= i + j - 1:
                row[idx(i, j)] = 1.0
    A_ub.append(-row)                 # -row <= -need  等价于 row >= need
    b_ub.append(-need[k - 1])

# ----------------------------------------------------------------- 求解
res = solve_lp(c, A_ub=A_ub, b_ub=b_ub, name="例1.4 仓库租赁")
x = res["x"].reshape(NM, NJ)

report("例 1.4  仓库租赁最优解", [
    ("最优总费用 z (元)", res["obj"]),
    ("求解状态", res["status"]),
])
print("\n签合同矩阵（行=起始月份，列=合同期限，空白=0）：")
for i in range(1, NM + 1):
    print("   第%d月初: %s" % (i, "  ".join(f"{x[i-1, j-1]:6.1f}" for j in range(1, NJ + 1))))

print("\n与书中答案对照：")
check("最优值 z", res["obj"], 118400, tol=1e-6, rel=True)

# ------------------------------------------------------------ 书中方案核对
# 书里给的解是 x11=3, x31=8, x14=12。它和 HiGHS 找到的解不一样，
# 但两者目标值都是 118400 —— 这是退化线性规划（存在多个最优解）。
# 建模时要知道：LP 的「最优解」通常不唯一，别把某个具体解当成唯一答案。
book_x = np.zeros((NM, NJ))
book_x[0, 0], book_x[2, 0], book_x[0, 3] = 3, 8, 12
book_cost = sum(fee[j + 1] * book_x[i, j] for i in range(NM) for j in range(NJ))
book_cov = []
for k in range(1, NM + 1):
    book_cov.append(sum(book_x[i - 1, j - 1] for i in range(1, NM + 1)
                        for j in range(1, NJ + 1) if i <= k <= i + j - 1))
print("\n书中给出的最优解（另一组最优解）：")
print(f"   x11=3, x31=8, x14=12，其余为 0  ->  费用 {book_cost:,.0f} 元")
print(f"   逐月在租面积 {book_cov}  需求 {need.tolist()}")
check("书中方案的费用", book_cost, 118400, tol=1e-9, rel=True)
assert all(cov >= req - 1e-9 for cov, req in zip(book_cov, need)), "书中方案应满足约束"
print("  [一致] 书中方案同样可行且同价 -> 本问题存在多个最优解")
print("         HiGHS 返回的是另一个顶点，不代表模型写错了。")

# 代码解的可行性复核：逐月约束必须全部满足
print("\n代码解的可行性复核：")
for k in range(1, NM + 1):
    cov = sum(x[i - 1, j - 1] for i in range(1, NM + 1) for j in range(1, NJ + 1)
              if i <= k <= i + j - 1)
    assert cov >= need[k - 1] - 1e-9, f"第{k}月不满足需求"
    print(f"   第{k}月: 在租 {cov:5.1f} >= 需求 {need[k-1]:5.1f}  满足")

# --------------------------------------------------- 逐月核对约束是否满足
print("\n逐月核对（在租面积 vs 需求）：")
for k in range(1, NM + 1):
    cov = sum(x[i - 1, j - 1] for i in range(1, NM + 1) for j in range(1, NJ + 1)
              if i <= k <= i + j - 1)
    print(f"   第{k}月: 在租 {cov:5.1f}  需求 {need[k-1]:5.1f}  "
          f"{'满足' if cov >= need[k-1] - 1e-9 else '不满足!'}")
