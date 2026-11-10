"""例 1.9  投资的收益与风险（第 1 章 · 线性规划 · 1998 全国数学建模竞赛 A 题）

题目
----
市场上 n 种风险资产 + 银行存款可用。已知各资产的平均收益率 r_i、风险损失率 q_i、
交易费率 p_i 和「按 u_i 计价」的阈值 u_i。资金 M 相当大，要求：
**净收益尽可能大，总体风险尽可能小**。

建模关键
--------
1. 总体风险用「所投资产中最大的风险」度量：max_i q_i x_i。
   这是个 max，不是线性的 —— 但可以引入变量线性化。
2. 交易费是分段函数；因 u_i 相对 M 很小、p_i u_i 更小，净收益可近似为
   (r_i - p_i) x_i，约束变成 sum (1 + p_i) x_i = M。
3. 这是**双目标**问题，书里给了三个模型：
   模型一：固定风险水平 a，最大化收益   -> 单目标 LP
   模型二：固定盈利水平 k，最小化风险   -> 单目标 LP（书里留作习题）
   模型三：给风险和收益加权 w 和 (1-w) 求和 -> 仍要线性化

模型三的线性化：引入 x_{n+1} = max_i q_i x_i，则
    min  w * x_{n+1} - (1-w) * sum (r_i - p_i) x_i
    s.t. q_i x_i <= x_{n+1},  sum (1+p_i) x_i = M,  x >= 0

书中答案：模型一在 a=0.006 附近出现转折，Q ≈ 2019；
          模型三 w<=0.7 时收益 2673.27、风险 247.52，w=1.0 时全存银行（收益 500、风险 0）。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")          # 无图形界面的服务器上也能出图
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import solve_lp, check  # noqa: E402
from common.plotting import plt as _plt  # noqa: E402,F401  (统一中文字体配置)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

M = 10000.0

rows = []
with open(os.path.join(DATA, "ch1_assets.csv"), encoding="utf-8-sig") as f:
    for line in f.read().splitlines()[1:]:
        p = line.split(",")
        rows.append((p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4])))

names = [r[0] for r in rows]
r = np.array([x[1] for x in rows])      # 收益率（含银行）
q = np.array([x[2] for x in rows])      # 风险损失率（银行为 0）
p = np.array([x[3] for x in rows])      # 交易费率（银行为 0）
net = r - p                            # 净收益率
ones = 1 + p                           # 占用资金系数

print("资产:", names)
print("净收益率:", net.round(4).tolist())
print("风险损失率:", q.round(4).tolist())


def model1(a):
    """模型一：固定风险水平 a，最大化净收益。"""
    n = len(r)
    c = net                                  # 目标：最大化净收益（方向由 solve_lp 处理）
    A_ub = np.zeros((n - 1, n))
    for i in range(1, n):                      # 银行没有风险约束
        A_ub[i - 1, i] = q[i]
    b_ub = np.full(n - 1, a * M)
    A_eq = ones.reshape(1, -1)
    b_eq = [M]
    res = solve_lp(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   maximize=True, name=f"模型一 a={a}")
    return res["x"], res["obj"]


def model3(w):
    """模型三：加权求和（已线性化）。变量 [x0..x4, x5(风险)]。"""
    n = len(r)
    c = np.zeros(n + 1)
    c[:n] = -(1 - w) * net                    # -(1-w)*净收益
    c[n] = w                                  # + w*风险
    A_ub = np.zeros((n - 1, n + 1))
    for i in range(1, n):
        A_ub[i - 1, i] = q[i]
        A_ub[i - 1, n] = -1.0                 # q_i x_i - x5 <= 0
    b_ub = np.zeros(n - 1)
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = ones
    b_eq = [M]
    res = solve_lp(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   name=f"模型三 w={w}")
    x = res["x"]
    risk = float(np.max(q * x[:n]))
    ret = float(net @ x[:n])
    return x, risk, ret


# ================================================================ 模型一
print("\n" + "=" * 66)
print("模型一：固定风险水平 a，最大化净收益（a 从 0 到 0.05，步长 0.001）")
print("=" * 66)

a_grid = np.arange(0, 0.0501, 0.001)
Qs, Xs = [], []
for a in a_grid:
    x, Q = model1(round(float(a), 4))
    Qs.append(Q)
    Xs.append(x)
Qs = np.array(Qs)

print("\n a(风险度)   Q(净收益/元)   投资分配 (x0..x4)")
for i, a in enumerate(a_grid):
    if abs(a - 0.006) < 1e-9 or i % 10 == 0:
        alloc = "  ".join(f"{v:7.1f}" for v in Xs[i])
        print(f"   {a:.3f}     {Qs[i]:10.2f}    {alloc}")

k = int(np.argmin(np.abs(a_grid - 0.006)))
print("\n与书中答案对照（a = 0.006）：")
check("净收益 Q", Qs[k], 2019, tol=2e-3, rel=True)
print(f"   代码解: x = {np.round(Xs[k], 2).tolist()}")
print("   书中解: x = [0, 2400, 4000, 1091, 2212]")

# 曲线分析：分段边际收益 + 拐点
seg_slope = np.diff(Qs) / np.diff(a_grid)          # 每段的风险-收益边际
knots = [(a_grid[i], seg_slope[i - 1], seg_slope[i])
         for i in range(1, len(seg_slope))
         if abs(seg_slope[i] - seg_slope[i - 1]) > 1e-6]
knots.sort(key=lambda t: t[1] - t[2], reverse=True)   # 斜率下降最多的排前面
print("\n曲线拐点（资产组合切换处，边际收益下降）：")
for a, s1, s2 in knots[:6]:
    print(f"   a = {a:.3f}  边际收益 {s1:>10,.0f} -> {s2:>10,.0f} 元/单位风险")
turn = knots[0]
print("\n   书中结论：a ≈ 0.006 时「风险增加很少但利润增长很快」，")
print(f"   即该点落在边际收益最高的那一段（{seg_slope.max():,.0f} 元/单位风险）里；")
print("   越过各个拐点后，边际收益逐段衰减，这就是曲线「先陡后平」的原因。")

# ================================================================ 模型三
print("\n" + "=" * 66)
print("模型三：加权求和 w*风险 - (1-w)*收益")
print("=" * 66)

ws = np.round(np.arange(0, 1.01, 0.1), 3)
print("\n   w     风险(元)     收益(元)    投资分配")
table = []
for w in ws:
    x, risk, ret = model3(float(w))
    table.append((w, risk, ret))
    alloc = "  ".join(f"{v:7.1f}" for v in x[:5])
    print(f"  {w:.1f}   {risk:8.2f}   {ret:9.2f}   {alloc}")

print("\n与书中答案对照：")
w05 = [t for t in table if abs(t[0] - 0.5) < 1e-9][0]
w10 = [t for t in table if abs(t[0] - 1.0) < 1e-9][0]
check("w=0.5 的风险", w05[1], 247.52, tol=1e-4, rel=True)
check("w=0.5 的收益", w05[2], 2673.27, tol=1e-4, rel=True)
check("w=1.0 的风险", w10[1], 0, tol=1e-6)
check("w=1.0 的收益", w10[2], 500, tol=1e-6, rel=True)

# 细化 w，得到有效前沿
w_fine = np.round(np.arange(0.70, 1.001, 0.005), 4)
front = [model3(float(w)) for w in w_fine]
front = sorted(front, key=lambda t: t[1])
print(f"\n有效前沿（w 从 0.70 到 1.000 细化）：共 {len(front)} 个点")
print("   最小风险点 -> 最大收益点 的若干样本：")
for x, risk, ret in front[::max(1, len(front) // 8)]:
    print(f"      风险 {risk:8.2f}   收益 {ret:9.2f}")

# ================================================================== 作图
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

ax[0].plot(a_grid, Qs, "-o", ms=3.5, color="#2b6cb0")
ax[0].axvline(0.006, ls="--", color="#c05621", lw=1.2)
ax[0].annotate("转折点 a=0.006", xy=(0.006, Qs[k]),
               xytext=(0.012, Qs[k] * 0.72), fontsize=10, color="#c05621",
               arrowprops=dict(arrowstyle="->", color="#c05621"))
ax[0].set_xlabel("风险水平 a")
ax[0].set_ylabel("净收益 Q (元)")
ax[0].set_title("模型一：风险 a 与收益 Q 的关系")
ax[0].grid(alpha=0.3)

risks = [t[1] for t in front]
rets = [t[2] for t in front]
ax[1].plot(risks, rets, "-o", ms=4, color="#2f855a")
ax[1].set_xlabel("风险 (元)")
ax[1].set_ylabel("收益 (元)")
ax[1].set_title("模型三：投资的有效前沿")
ax[1].grid(alpha=0.3)

fig.suptitle("例 1.9  投资的收益与风险（1998 国赛 A 题）", fontsize=13)
fig.tight_layout()
out = os.path.join(FIG, "ch1_portfolio.png")
fig.savefig(out, dpi=130)
print(f"\n图已保存: {out}")

print("\n要点回顾：")
print("  1) 双目标问题有两条路：把一个目标固定成约束（模型一/二），")
print("     或者给两个目标加权（模型三）。加权法能画出有效前沿。")
print("  2) max_i q_i x_i 这类 max 一定要先线性化，否则求解器用不上。")
print("  3) 风险越大收益越大，这是权衡而不是 bug；真正的「最优」取决于偏好。")
