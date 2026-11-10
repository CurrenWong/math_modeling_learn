"""生成教学报告用的可视化图（全部用真实数据算出来，不是手绘示意）。

输出：report/figs/*.png
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.plotting import plt  # noqa: E402

FIG = os.path.join(ROOT, "report", "figs")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREEN, RED, PURPLE = "#2563eb", "#ea580c", "#059669", "#dc2626", "#7c3aed"


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", name, f"{os.path.getsize(p)/1024:.0f} KB")


# ============================================ 1. LP 可行域 + 目标等高线
def fig_lp_region():
    fig, ax = plt.subplots(1, 1, figsize=(9, 6.5))
    # max z = x + 2y ; x + y <= 4 ; x + 3y <= 6 ; x,y >= 0
    V = [(0, 0), (4, 0), (2, 2), (0, 2)]
    poly = plt.Polygon(V, facecolor="#dbeafe", edgecolor=BLUE, lw=2.2, alpha=.75,
                       zorder=1, label="可行域（所有约束的交集）")
    ax.add_patch(poly)
    x = np.linspace(0, 4.6, 400)
    ax.plot(x, (4 - x), "k--", lw=1.2, alpha=.55, zorder=2)
    ax.plot(x, (6 - x) / 3, "k--", lw=1.2, alpha=.55, zorder=2)
    ax.text(4.15, -0.16, "x + y = 4", fontsize=13, color="#334155")
    ax.text(3.62, 0.82, "x + 3y = 6", fontsize=13, color="#334155")

    for c in [2, 4, 6]:
        yy = np.linspace(0, 3, 300)
        xx = np.clip(c - 2 * yy, 0, None)
        ax.plot(xx, yy, color=ORANGE, lw=1.7, alpha=.85, zorder=3)
        lx = min(c * 0.42, 3.3)
        ly = (c - lx) / 2
        ax.text(lx + .08, ly + .1, f"z = {c}", fontsize=13.5, color=ORANGE,
                fontweight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.8))
    ax.plot([2], [2], "*", ms=26, color=RED, zorder=6)
    ax.annotate("最优解 (2, 2)\nz = 6",
                xy=(2, 2), xytext=(2.28, 1.02), fontsize=14.5, color=RED, zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=.85),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    for (px, py), t, (dx, dy) in zip(V, ["(0,0)", "(4,0)", "(2,2)", "(0,2)"],
                                   [(.12, -.34), (.12, -.36), (.14, .12), (-.66, .1)]):
        ax.plot([px], [py], "o", color="#0f172a", ms=8, zorder=5)
        ax.text(px + dx, py + dy, t, fontsize=13.5, zorder=8,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=.8))
    ax.set_xlim(-0.3, 4.9)
    ax.set_ylim(-0.45, 3.1)
    ax.set_xlabel("x1（产品一产量）", fontsize=15)
    ax.set_ylabel("x2（产品二产量）", fontsize=15)
    ax.set_title("线性规划的几何图像：往目标线的「最高一条」推，直到撞上约束",
                 fontsize=17, pad=14)
    ax.legend(fontsize=13, loc="lower center", bbox_to_anchor=(.5, .015),
              framealpha=.93)
    ax.grid(alpha=.18)
    save(fig, "fig1_lp_region.png")


# ============================================ 2. 绝对值 → 线性化
def fig_abs_linearize():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    # 左：原问题 min |x|  约束 |x| <= 3
    ax = axes[0]
    x = np.linspace(-3.4, 3.4, 500)
    ax.plot(x, np.abs(x), color=RED, lw=2.6, label="目标函数 y = |x|（V 形）")
    ax.fill_between(x, 0, 4, where=np.abs(x) <= 3, color="#fee2e2", alpha=.8,
                    label="可行域 |x| ≤ 3")
    for c in [1, 2, 3]:
        ax.plot([-c, c], [c, c], color=ORANGE, lw=1.6)
        ax.text(c + .08, c - .18, f"|x| = {c}", fontsize=12, color=ORANGE)
    ax.plot([0], [0], "*", ms=24, color="#0f172a", zorder=5)
    ax.text(.12, .28, "最优 x = 0", fontsize=14)
    ax.set_title("原问题：目标是 V 形曲线，约束是竖条", fontsize=16)
    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("目标值", fontsize=15)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(alpha=.18)

    # 右：线性化后 (u,v) 平面  x = u - v >= 0 ; min u+v
    ax = axes[1]
    u = np.linspace(0, 6.2, 400)
    ax.fill_between(u, 0, 6 - u, color="#dcfce7", alpha=.9,
                    label="可行域（u,v ≥ 0，u+v ≤ 3）")
    for c in [1, 2, 3]:
        m = u <= c
        ax.plot(u[m], (c - u)[m], color=ORANGE, lw=1.7)
        ax.text(c + .08, c - .22, f"u+v = {c}", fontsize=12, color=ORANGE)
    ax.plot([0], [0], "*", ms=24, color="#0f172a", zorder=5)
    ax.text(.14, .3, "最优 u=v=0", fontsize=14)
    ax.plot([0, 3], [0, 0], "k-", lw=1, alpha=.4)
    ax.plot([0, 0], [0, 3], "k-", lw=1, alpha=.4)
    ax.annotate("x = u − v", xy=(2.4, .2), fontsize=14, color="#334155")
    ax.set_xlim(-.3, 6.2)
    ax.set_ylim(-.4, 3.4)
    ax.set_title("线性化后：目标变成一组平行直线（线性！）", fontsize=16)
    ax.set_xlabel("u ≥ 0", fontsize=15)
    ax.set_ylabel("v ≥ 0", fontsize=15)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(alpha=.18)
    fig.suptitle("把 |x| 拆成 u + v：V 形目标 → 线性目标（求解器最擅长的形状）",
                 fontsize=17, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_abs_linearize.png")


# ============================================ 3. 整数规划 vs LP 松弛
def fig_integer_relax():
    fig, ax = plt.subplots(1, 1, figsize=(9, 6.5))
    # max 3x + 5y ; x + 2y <= 12 ; 3x + y <= 12 ; x,y >= 0 整数
    V = [(0, 0), (4, 0), (2.4, 4.8), (0, 6)]
    ax.add_patch(plt.Polygon(V, facecolor="#e0e7ff", edgecolor=PURPLE, lw=2,
                             alpha=.7, zorder=1, label="LP 松弛后的可行域"))
    gx, gy = np.meshgrid(np.arange(0, 6.05, .16), np.arange(0, 6.05, .16))
    ok = (gx + 2 * gy <= 12) & (3 * gx + gy <= 12)
    ax.scatter(gx[ok], gy[ok], s=4, color="#c7d2fe", zorder=2)
    ip = [(x, y) for x in range(7) for y in range(7)
          if x + 2 * y <= 12 and 3 * x + y <= 12]
    ax.scatter([p[0] for p in ip], [p[1] for p in ip], s=95, facecolor="none",
               edgecolor=BLUE, lw=2, zorder=4, label="整数可行点")
    ax.plot([2.4], [4.8], "x", ms=17, mew=4, color=RED, zorder=6)
    ax.annotate("LP 松弛解 (2.4, 4.8)\nz = 31.2 —— 不可实现！",
                xy=(2.4, 4.8), xytext=(3.1, 5.55), fontsize=14, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    ax.plot([2], [5], "*", ms=27, color=GREEN, zorder=7)
    ax.annotate("整数最优 (2, 5)\nz = 31",
                xy=(2, 5), xytext=(.15, 2.6), fontsize=14, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8))
    ax.plot([4, 2.4, 0], [0, 4.8, 6], "k--", lw=1.1, alpha=.5)
    ax.set_xlim(-.4, 6.4)
    ax.set_ylim(-.4, 6.6)
    ax.set_xlabel("x（整数）", fontsize=15)
    ax.set_ylabel("y（整数）", fontsize=15)
    ax.set_title("整数规划：LP 最优点常常落在网格的缝隙里，必须往回挪",
                 fontsize=17, pad=13)
    ax.legend(fontsize=13, loc="lower right")
    ax.grid(alpha=.16)
    save(fig, "fig3_integer_relax.png")


# ============================================ 4. 分支定界树
def fig_branch_bound():
    fig, ax = plt.subplots(1, 1, figsize=(12, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def node(x, y, text, color="#334155", fc="white", w=2.6, h=1.0):
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc,
                                   edgecolor=color, lw=1.9, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=13.5,
                color=color, zorder=4, linespacing=1.5)

    def link(x1, y1, x2, y2, color="#64748b", style="-"):
        ax.plot([x1, x2], [y1, y2], color=color, lw=1.6, ls=style, zorder=2)

    node(6, 6.1, "松弛 LP 最优解\nz = 31.2 (不可实现)", color=RED, fc="#fef2f2")
    link(6, 5.6, 3.2, 4.6)
    link(6, 5.6, 8.8, 4.6)
    node(3.2, 4.1, "分支 x ≤ 2\nz = 31  ← 找到整数解!", color=GREEN, fc="#ecfdf5")
    node(8.8, 4.1, "分支 x ≥ 3\n下界 z = 28", color="#334155", fc="#f8fafc")
    link(8.8, 3.6, 7.2, 2.6)
    link(8.8, 3.6, 10.4, 2.6)
    node(7.2, 2.1, "分支 y ≤ 4\n下界 z = 26", fc="#f8fafc")
    node(10.4, 2.1, "分支 y ≥ 5\n下界 z = 25", fc="#f8fafc")
    link(7.2, 1.6, 7.2, .8, style="--")
    link(10.4, 1.6, 10.4, .8, style="--")
    ax.text(7.2, .45, "下界 26 < 31 → 剪枝", ha="center", fontsize=13, color="#94a3b8")
    ax.text(10.4, .45, "下界 25 < 31 → 剪枝", ha="center", fontsize=13, color="#94a3b8")
    ax.text(3.2, 3.0, "剪枝：下界已不优于\n当前最优 31 → 整棵子树砍掉",
            fontsize=13, color=GREEN, ha="center")
    ax.text(6, 7.05, "分支定界：把「算不出整数解」的难题，拆成一堆好解的小问题",
            ha="center", fontsize=17, color="#0f172a")
    save(fig, "fig4_branch_bound.png")


# ============================================ 5. 蒙特卡洛散点
def fig_monte_carlo():
    """例 2.8：曲线 y=x^2、y=12-x 与 x 轴围成的面积（精确值 49.5）。

    采样域取整个包围矩形 [0,12] x [0,12]（面积 144），
    落在区域内的点：x<=3 且 y<=x^2，或 x>=3 且 y<=12-x。
    估计面积 = 144 * 命中比例。
    """
    fig, ax = plt.subplots(1, 1, figsize=(9.4, 6.4))
    rng = np.random.default_rng(42)
    N = 6000
    px = rng.uniform(0, 12, N)
    py = rng.uniform(0, 12, N)
    inside = ((px <= 3) & (py <= px ** 2)) | ((px >= 3) & (py <= 12 - px))
    ax.scatter(px[~inside], py[~inside], s=6, color="#cbd5e1", alpha=.6,
               label=f"落在区域外（{int(np.sum(~inside))} 点）")
    ax.scatter(px[inside], py[inside], s=9, color=GREEN, alpha=.8,
               label=f"落在区域内（{int(np.sum(inside))} 点）")
    xc = np.linspace(0, 3, 300)
    ax.plot(xc, xc ** 2, color=RED, lw=2.6, label="y = x²")
    xd = np.linspace(3, 12, 300)
    ax.plot(xd, 12 - xd, color=BLUE, lw=2.6, label="y = 12 − x")
    ax.fill_between(xc, 0, xc ** 2, color="#d1fae5", alpha=.75)
    ax.fill_between(xd, 0, 12 - xd, color="#d1fae5", alpha=.75)
    ax.plot([0, 3, 12], [0, 9, 0], "k-", lw=1.4)
    ax.plot([3], [9], "o", color="#0f172a", ms=8, zorder=6)
    ax.text(3.15, 9.3, "交点 (3, 9)", fontsize=13)
    ax.set_xlim(-0.4, 12.4)
    ax.set_ylim(-0.4, 12.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("y", fontsize=15)
    est = 144 * inside.mean()
    ax.set_title(f"蒙特卡洛：随机撒 {N} 个点，数落在区域内的比例\n"
                 f"估计面积 ≈ 144 × {inside.mean():.4f} = {est:.3f}"
                 f"（精确值 49.5，误差 {abs(est - 49.5):.3f}）",
                 fontsize=15.5, pad=13)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(alpha=.16)
    save(fig, "fig5_monte_carlo.png")


# ============================================ 6. 凸 vs 非凸
def fig_convex_nonconvex():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    xs = np.linspace(-3, 3, 300)
    for ax, kind in zip(axes, ["convex", "nonconvex"]):
        if kind == "convex":
            X, Y = np.meshgrid(xs, xs)
            Z = X ** 2 + Y ** 2
            ax.contour(X, Y, Z, levels=14, cmap="Blues", linewidths=1.6)
            ax.plot([0], [0], "*", ms=24, color=RED, zorder=6)
            ax.set_title("凸（碗形）\n局部最优 = 全局最优，换初值结果一样", fontsize=15)
            ax.text(0.15, .2, "唯一最优", fontsize=14, color=RED)
        else:
            X, Y = np.meshgrid(xs, xs)
            Z = (X ** 2 - 1) ** 2 + (Y ** 2 - 1) ** 2
            ax.contour(X, Y, Z, levels=14, cmap="Reds", linewidths=1.6)
            for sx, sy in [(-1, -1), (1, 1), (-1, 1), (1, -1)]:
                ax.plot([sx], [sy], "o", ms=11, color=GREEN, zorder=6)
                ax.plot([sx + .42], [sy + .42], "x", ms=9, mew=2.4, color="#64748b")
            ax.plot([0], [0], "*", ms=25, color=RED, zorder=7)
            ax.set_title("非凸（山地）\n局部最优很多，随初值跳来跳去", fontsize=15)
            ax.text(.15, .12, "全局最优", fontsize=13, color=RED)
            ax.text(-2.6, -2.6, "局部最优（也可能是答案）", fontsize=12, color=GREEN)
        ax.set_xlabel("x1", fontsize=14)
        ax.set_ylabel("x2", fontsize=14)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(alpha=.14)
    fig.suptitle("凸性是第 3 章的全部关键：它决定「一个解」还是「一堆解」",
                 fontsize=17, y=1.02)
    fig.tight_layout()
    save(fig, "fig6_convex_nonconvex.png")


# ============================================ 7. K-T 条件几何
def fig_kt():
    fig, ax = plt.subplots(1, 1, figsize=(9, 6.4))
    # 例3.1: min f = x1^2+x2^2-4x1+4 ; g1=-x1+x2-2<=0 ; g2=x1^2-x2+1<=0 ; x>=0
    xs = np.linspace(-0.2, 2.6, 400)
    g1 = (xs - 2)                       # g1 = 0 -> x2 = x1-2
    g2 = xs ** 2 + 1                    # g2 = 0 -> x2 = x1^2+1
    ax.plot(xs, g1, color=BLUE, lw=2.3, label="g1 = −x1 + x2 − 2 = 0")
    ax.plot(xs, g2, color=PURPLE, lw=2.3, label="g2 = x1² − x2 + 1 = 0")
    # 可行域：x2 <= x1-2 (g1<=0) 且 x2 >= x1^2+1 (g2<=0)
    xx = np.linspace(1.35, 2.3, 300)
    ax.fill_between(xx, xx ** 2 + 1, xx - 2, color="#dcfce7", alpha=.9,
                    label="可行域")
    xstar, ystar = 0.5536, 1.3064
    ax.plot([xstar], [ystar], "*", ms=27, color=RED, zorder=8)
    ax.annotate("最优解 x* = (0.554, 1.306)\nf* = 3.7989",
                xy=(xstar, ystar), xytext=(1.42, .52), fontsize=14.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.9))
    # 目标梯度
    gr = np.array([2 * xstar - 4, 2 * ystar])
    gr = gr / np.linalg.norm(gr) * .62
    ax.arrow(xstar, ystar, gr[0], gr[1], head_width=.09, color="#0f172a",
             length_includes_head=True, zorder=9, lw=2.2)
    ax.text(xstar + gr[0] + .06, ystar + gr[1] + .1, "目标梯度", fontsize=13, color="#0f172a")
    # g2 在最优点的切线
    slope = 2 * xstar
    t = np.linspace(-.45, .75, 50)
    ax.plot(xstar + t, ystar + slope * t, color=RED, lw=1.7, ls="--", zorder=7,
            label="g2 在最优点处的切线")
    ax.set_xlim(-0.2, 2.7)
    ax.set_ylim(-0.2, 2.7)
    ax.set_xlabel("x1", fontsize=15)
    ax.set_ylabel("x2", fontsize=15)
    ax.set_title("K-T 条件的几何含义：目标函数的梯度方向与起作用的约束的切线一致",
                 fontsize=16, pad=13)
    ax.legend(fontsize=12.5, loc="upper left")
    ax.grid(alpha=.16)
    save(fig, "fig7_kt.png")


# ============================================ 8. 灵敏度：薄利多销
def fig_sensitivity_tv():
    """与 ex3_2_tv_pricing.py 完全同一套模型：
       p1 = 339 - a*x1 - 0.003*x2 ; p2 = 399 - 0.004*x1 - 0.01*x2
       cost = 400000 + 195*x1 + 225*x2
    """
    A_PRICE, FIXED, C19, C21 = 339.0, 400000.0, 195.0, 225.0
    CROSS12, CROSS21, SELF2 = 0.003, 0.004, 0.01

    def solve_a(a):
        # 令 grad f = 0 解 2x2 线性方程组（凹二次，解析解即全局最优）
        Amat = np.array([[2 * a, CROSS12 + CROSS21],
                         [CROSS12 + CROSS21, 2 * SELF2]])
        bvec = np.array([A_PRICE - C19, 399.0 - C21])
        x1, x2 = np.linalg.solve(Amat, bvec)
        p1 = A_PRICE - a * x1 - CROSS12 * x2
        p2 = 399.0 - CROSS21 * x1 - SELF2 * x2
        cost = FIXED + C19 * x1 + C21 * x2
        prof = p1 * x1 + p2 * x2 - cost
        return x1, x2, prof, prof / cost

    As = np.linspace(0.004, 0.024, 60)
    q1, q2, pf, rate = [], [], [], []
    for a in As:
        x1, x2, p, r = solve_a(a)
        q1.append(x1)
        q2.append(x2)
        pf.append(p)
        rate.append(r)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    ax.plot(As, q1, "o-", ms=4.5, color=BLUE, label="19 英寸最优产量")
    ax.plot(As, q2, "s-", ms=4.5, color=ORANGE, label="21 英寸最优产量")
    ax.axvline(0.010, color="#94a3b8", ls="--", lw=1.4)
    ax.text(0.0103, 8400, "基准 a = 0.010", fontsize=12.5, color="#64748b")
    ax.set_xlabel("19 英寸降价幅度 a（美元/台）", fontsize=14)
    ax.set_ylabel("最优产量（台）", fontsize=14)
    ax.set_title("降价越狠 → 19 英寸减产、21 英寸增产（资金挪去赚另一种钱）",
                 fontsize=14.5)
    ax.legend(fontsize=13)
    ax.grid(alpha=.18)

    ax = axes[1]
    ax.plot(As, pf, "o-", ms=4.5, color=GREEN, label="年利润（美元）")
    ax2 = ax.twinx()
    ax2.plot(As, np.array(rate) * 100, "s--", ms=4, color=RED, label="利润率")
    ax.set_xlabel("19 英寸降价幅度 a（美元/台）", fontsize=14)
    ax.set_ylabel("年利润（美元）", fontsize=14)
    ax2.set_ylabel("利润率 (%)", fontsize=14, color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax.set_title("利润与利润率一起走低 —— 这就是「薄利多销」的代价", fontsize=14.5)
    ax.grid(alpha=.18)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=12.5, loc="upper right")
    fig.suptitle("灵敏度分析：把「要不要多销」变成一个可以算的决策",
                 fontsize=17, y=1.02)
    fig.tight_layout()
    save(fig, "fig8_sensitivity_tv.png")


# ============================================ 9. 飞行管理轨迹
def fig_flight():
    rows = [(150, 140, 243), (85, 85, 236), (150, 155, 220.5),
            (145, 50, 159), (130, 150, 230), (0, 0, 52)]
    p0 = np.array([[r[0], r[1]] for r in rows], float)
    th0 = np.array([r[2] for r in rows], float)
    SPEED, SIDE = 800.0, 160.0

    def sim(dtheta, dt=1 / 3600, horizon=1.2):
        th = np.radians(th0 + np.asarray(dtheta))
        p = p0.copy()
        traj = [p.copy()]
        min_sep, worst = np.inf, None
        t = 0.0
        while t <= horizon:
            p[:, 0] += SPEED * dt * np.cos(th)
            p[:, 1] += SPEED * dt * np.sin(th)
            traj.append(p.copy())
            t += dt
            for i in range(6):
                for j in range(i + 1, 6):
                    if any(0 <= p[k, 0] <= SIDE and 0 <= p[k, 1] <= SIDE for k in (i, j)):
                        s = float(np.linalg.norm(p[i] - p[j]))
                        if s < min_sep:
                            min_sep, worst = s, (i, j, p[i].copy(), p[j].copy())
        return np.array(traj), min_sep, worst

    book = np.array([0, 0, 0, 0, 2.7979, 0.9330])
    ours = np.array([0.0, 0.0, 2.5579, 0.0, 0.0, 1.0716])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
    for ax, d, title in [(axes[0], book, "按书中的答案调整：最小间距 2.927 km（危险）"),
                         (axes[1], ours, "本工程的解：最小间距 8.000 km（安全）")]:
        traj, sep, worst = sim(d)
        for k in range(6):
            ax.plot(traj[:, k, 0], traj[:, k, 1], lw=1.5, alpha=.85,
                    label=f"飞机 {k+1}" if k < 5 else "新进入")
            ax.plot(traj[0, k, 0], traj[0, k, 1], "o", ms=7, color="black")
        if worst:
            i, j, pi, pj = worst
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], "r-", lw=2.6)
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], "r*", ms=17)
            ax.text(0.03, .93, f"最近距离 {sep:.3f} km", transform=ax.transAxes,
                    fontsize=14, color=RED, fontweight="bold")
        ax.add_patch(plt.Rectangle((0, 0), SIDE, SIDE, fill=False, ec="#334155",
                                   lw=1.6, ls="--"))
        ax.set_xlim(-5, 168)
        ax.set_ylim(-5, 168)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)", fontsize=13)
        ax.set_ylabel("y (km)", fontsize=13)
        ax.set_title(title, fontsize=14.5)
        ax.legend(fontsize=10, ncol=3, loc="lower left")
        ax.grid(alpha=.14)
    fig.suptitle("飞行管理：模型解出来 ≠ 事情解决了 —— 必须把轨迹真跑一遍",
                 fontsize=17, y=1.02)
    fig.tight_layout()
    save(fig, "fig9_flight.png")


# ============================================ 10. 严格凹 vs 半凹：唯一性
def fig_concave_unique():
    """『凹 → 唯一』要成立需要『严格凹』。
    左：例 3.2 的利润函数（严格凹，Hessian 负定）-> 唯一最优点
    右：f = -(x-y)^2（半负定，有平坦方向）-> 整条对角线都是最优解
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))

    # 左：彩电定价利润函数（归一化后画等高线）
    A_PRICE, FIXED, C19, C21 = 339.0, 400000.0, 195.0, 225.0
    CROSS12, CROSS21, SELF2 = 0.003, 0.004, 0.01

    def profit(x1, x2):
        p1 = A_PRICE - 0.01 * x1 - CROSS12 * x2
        p2 = 399.0 - CROSS21 * x1 - SELF2 * x2
        return p1 * x1 + p2 * x2 - (FIXED + C19 * x1 + C21 * x2)

    g1, g2 = np.meshgrid(np.linspace(0, 12000, 220), np.linspace(0, 12000, 220))
    Z = profit(g1, g2)
    ax = axes[0]
    ax.contour(g1 / 1000, g2 / 1000, Z / 1000, levels=16, cmap="Blues", linewidths=1.5)
    xs = np.linalg.solve(np.array([[0.02, 0.007], [0.007, 0.02]]),
                         np.array([A_PRICE - C19, 399.0 - C21]))
    ax.plot(xs[0] / 1000, xs[1] / 1000, "*", ms=26, color=RED, zorder=6)
    ax.annotate("唯一最优点 (4735, 7043)\nHessian 特征值 -0.027 / -0.013\n-> 负定 = 严格凹",
                xy=(xs[0] / 1000, xs[1] / 1000), xytext=(7.2, 1.6),
                fontsize=13.5, color=RED, ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=RED, alpha=.92),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.9))
    ax.set_title("严格凹（负定）-> 最大点唯一", fontsize=15.5)
    ax.set_xlabel("x1 (千台)", fontsize=14)
    ax.set_ylabel("x2 (千台)", fontsize=14)

    # 右：f = -(x-y)^2
    ax = axes[1]
    xx, yy = np.meshgrid(np.linspace(-3, 3, 240), np.linspace(-3, 3, 240))
    Z2 = -(xx - yy) ** 2
    ax.contour(xx, yy, Z2, levels=14, cmap="Reds", linewidths=1.5)
    t = np.linspace(-3, 3, 10)
    ax.plot(t, t, "-", color=GREEN, lw=3.2, zorder=6)
    for v in t[::2]:
        ax.plot([v], [v], "o", color=GREEN, ms=7, zorder=7)
    ax.text(0.1, -2.65, "这一整条线上的点\n全都是最优解\n(Hessian 特征值 -4, 0 -> 半负定)",
            fontsize=13.5, color=GREEN, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GREEN, alpha=.92))
    ax.set_title("半凹（半负定，有平坦方向）-> 解不唯一", fontsize=15.5)
    ax.set_xlabel("x1", fontsize=14)
    ax.set_ylabel("x2", fontsize=14)

    for ax in axes:
        ax.grid(alpha=.15)
        ax.set_xlim(-3, 12) if ax is axes[0] else ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 12) if ax is axes[0] else ax.set_ylim(-3, 3)
    fig.suptitle("『凹 → 解唯一』只在【严格凹】时成立：多一个 0 特征值，解就变成一条线",
                 fontsize=17, y=1.02)
    fig.tight_layout()
    save(fig, "fig10_concave_unique.png")


if __name__ == "__main__":
    print("生成教学图：")
    fig_lp_region()
    fig_abs_linearize()
    fig_integer_relax()
    fig_branch_bound()
    fig_monte_carlo()
    fig_convex_nonconvex()
    fig_kt()
    fig_sensitivity_tv()
    fig_flight()
    fig_concave_unique()
    print("完成 ->", FIG)
