"""例 3.10  飞行管理避免碰撞（1995 全国数学建模竞赛 A 题 · 第 3 章）

题目
----
边长 160km 的正方形空域内若干架飞机水平飞行，速度均为 800km/h。
一架飞机进入区域边缘时，需要立即判断是否会与区域内飞机碰撞；
若会碰撞，要计算各机（含新进入的）方向角的调整量，使调整幅度尽量小。

条件
----
(1) 两机距离始终大于 8km；(2) 方向角调整不超过 30°；
(3) 速度 800km/h；(4) 新进入飞机进入时距区域内飞机 60km 以上；
(5) 最多 6 架；(6) 不考虑飞出区域后。

模型一（相对运动 + 几何判据）
---------------------------
把飞机 i 看作不动，j 以相对速度 v = v_j - v_i 相对运动。
相对速度方向角 beta_ij 与两机连线夹角为 beta_ij，
调整方向角后 beta 变为 beta_ij + (dtheta_i + dtheta_j)/2。
危险角 alpha_ij = arcsin(8 / r_ij)，不碰撞条件：

    alpha_ij < beta_ij < 2pi - alpha_ij

书中把它写成平方形式
    (beta_ij + (dtheta_i + dtheta_j)/2)^2 >= alpha_ij^2
两者等价（危险扇区就是 0 附近的 [-alpha, alpha]），平方形式更���求解器用。

目标：min sum_i |dtheta_i|

模型二（对所有时刻成立）
-----------------------
两机距离平方 >= 64 对所有 t 成立，等价于二次不等式
    Delta_ij = btilde^2 - 4 atilde ctilde <= 0
目标改成 min sum_i (dtheta_i)^2

本脚本做四件事
--------------
1. 复现书中的 alpha 表 3.8 与 beta 表 3.9；
2. 求解模型一（书中平方形式 + 严格区间形式）；
3. 【核对书中的答案】逐对检查书中给出的调整角是否满足它自己的约束；
4. 【独立验证】按时间步长仿真，检查飞行过程中的最小机间距是否真的 >= 8km。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import solve_nlp, report, check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

SPEED = 800.0          # km/h
SAFE = 8.0             # km
MAX_TURN = 30.0        # 度
SIDE = 160.0           # 正方形边长 km

rows = []
with open(os.path.join(DATA, "ch3_flights.csv"), encoding="utf-8-sig") as f:
    for line in f.read().splitlines()[1:]:
        p = line.split(",")
        rows.append((int(p[0]), float(p[1]), float(p[2]), float(p[3])))
n = len(rows)
x0 = np.array([r[1] for r in rows])
y0 = np.array([r[2] for r in rows])
theta0 = np.array([r[3] for r in rows])
labels = [str(r[0]) if r[0] <= 5 else "新进入" for r in rows]
PAIRS = [(i, j) for i in range(n) for j in range(i + 1, n)]

print("飞行记录:")
for i in range(n):
    print(f"   {labels[i]:<5} 位置({x0[i]:6.1f},{y0[i]:6.1f})  方向角 {theta0[i]:6.1f}°")

# ------------------------------------------------------------ 表 3.8：alpha
xy = np.column_stack([x0, y0])
d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
alpha = np.degrees(np.arcsin(np.clip(SAFE / np.where(d > 0, d, np.inf), -1, 1)))
np.fill_diagonal(alpha, 0.0)

# ------------------------------------------------------------ 表 3.9：beta
complex_xy = x0 + 1j * y0
complex_v = np.exp(1j * np.radians(theta0))
beta = np.zeros((n, n))
for m in range(n):
    for nn in range(n):
        if m != nn:
            beta[m, nn] = np.angle((complex_v[nn] - complex_v[m])
                                   / (complex_xy[m] - complex_xy[nn]))
beta = np.degrees(beta)

print("\n表 3.8  alpha_ij (度):")
print("      " + "".join(f"{labels[j]:>9}" for j in range(n)))
for i in range(n):
    print(f"   {labels[i]:<4}" + "".join(f"{alpha[i, j]:9.4f}" for j in range(n)))
print("\n表 3.9  beta_ij (度):")
print("      " + "".join(f"{labels[j]:>11}" for j in range(n)))
for i in range(n):
    print(f"   {labels[i]:<4}" + "".join(f"{beta[i, j]:11.4f}" for j in range(n)))

print("\n与书中表格对照（抽样）：")
check("alpha(1,2)", alpha[0, 1], 5.3912, tol=1e-3)
check("alpha(1,3)", alpha[0, 2], 32.2310, tol=1e-3)
check("alpha(3,6)", alpha[2, 5], 2.1255, tol=1e-3)
check("beta(1,2)", beta[0, 1], 109.2636, tol=1e-3)
check("beta(1,3)", beta[0, 2], -128.2500, tol=1e-3)
check("beta(2,6)", beta[1, 5], 9.0000, tol=1e-3)
check("beta(4,6)", beta[3, 5], -3.5256, tol=1e-3)


# ============================================================== 模型一
def build_cons_sq():
    """变量 v = [dtheta(6), t(6)]；t 把 |dtheta| 线性化；碰撞约束用平方形式。

    约束 g_ij = alpha^2 - (beta + (d_i+d_j)/2)^2 >= 0，梯度解析给出。
    """
    cons = []
    for i in range(n):                                  # t_i >= |dtheta_i|
        for sgn in (1.0, -1.0):
            row = np.zeros(2 * n)
            row[n + i] = 1.0
            row[i] = -sgn
            cons.append({"type": "ineq",
                         "fun": (lambda r: (lambda v: r @ v))(row),
                         "jac": (lambda r: (lambda v: r))(row)})
    for i, j in PAIRS:
        a_ = alpha[i, j]
        b_ = beta[i, j]

        def mk(i=i, j=j, a_=a_, b_=b_):
            # 约束：|beta + s| >= alpha  <=>  (s/alpha)^2 - 1 >= 0
            def fun(v):
                s = b_ + (v[i] + v[j]) / 2
                return (s / a_) ** 2 - 1.0

            def jac(v):
                s = b_ + (v[i] + v[j]) / 2
                g = np.zeros(2 * n)
                # dg/dv_i = 2s/a^2 * (ds/dv_i),  ds/dv_i = 1/2
                g[i] = s / a_ ** 2
                g[j] = s / a_ ** 2
                return g
            return {"type": "ineq", "fun": fun, "jac": jac}
        cons.append(mk())
    return cons


def solve_model1(starts):
    """多起点求解模型一，返回最好的可行解。"""
    def obj(v):
        return float(np.sum(v[n:]))

    def obj_jac(v):
        g = np.zeros(2 * n)
        g[n:] = 1.0
        return g

    cons = build_cons_sq()
    best = None
    for d_start in starts:
        v0 = np.zeros(2 * n)
        v0[:n] = d_start
        v0[n:] = np.abs(d_start)
        r = solve_nlp(obj, x0=v0, bounds=[(-MAX_TURN, MAX_TURN)] * n + [(0, None)] * n,
                      constraints=cons, jac=obj_jac,
                      options={"maxiter": 5000, "ftol": 1e-12}, name="模型一")
        if r["success"]:
            viol = worst_violation(r["x"][:n])     # 变量本身就是「度」，不要再换算
            if viol["value"] >= -1e-6 and (best is None or r["obj"] < best["obj"]):
                best = r
    return best


def worst_violation(dtheta_deg):
    """检查所有机对的碰撞约束，返回最紧的一对的余量（>=0 表示全部满足）。"""
    worst = {"value": np.inf, "pair": None}
    for i, j in PAIRS:
        s = beta[i, j] + (dtheta_deg[i] + dtheta_deg[j]) / 2
        margin = abs(s) - alpha[i, j]        # >= 0 表示满足
        if margin < worst["value"]:
            worst = {"value": margin, "pair": (i, j)}
    return worst


# 热启动点：先手工构造一个可行的调整方案（下面会验证）
warm = np.array([0.0, 0.0, 4.0, -2.0, 1.0, 0.0])
starts = [warm,
          np.array([0, 0, 3.7, -1.5, 0.9, 0]),
          np.array([0, 0, 6, -3, 1.5, 0]),
          np.array([0, 0, 2, -1, 0.5, 0]),
          np.array([0, 0, 0, 0, 0, 4]),
          np.array([0, 0, 3, 0, 0.8, 0])]

res1 = solve_model1(starts)
if res1 is None:
    raise RuntimeError("模型一没有找到可行解")
dtheta1 = res1["x"][:n]      # 单位：度

report("模型一（平方形式，min sum|dtheta|）", [
    ("各机调整角 (度)", np.round(dtheta1, 4).tolist()),
    ("总调整角 sum|dtheta| (度)", float(np.sum(np.abs(dtheta1)))),
    ("最紧的一对的安全余量 (度)", worst_violation(dtheta1)["value"]),
    ("求解状态", res1["status"]),
])

print("\n与书中答案对照：")
print("   书: dtheta5 = 2.7979, dtheta6 = 0.9330, 其余 0，总调整角 3.7310")
print(f"   代码: {np.round(dtheta1, 4).tolist()}，总调整角 {np.sum(np.abs(dtheta1)):.4f}")
book_d = np.array([0.0, 0.0, 0.0, 0.0, 2.7979, 0.9330])
check("书中方案的总调整角", float(np.sum(np.abs(book_d))), 3.7310, tol=1e-3, rel=True)

# --------------------------------------------- 核对：书的解满足自己的约束吗？
print("\n" + "=" * 70)
print("核对：书中给出的调整角，逐对检查碰撞约束（|beta + s| >= alpha）")
print("=" * 70)
print(f"   {'机对':<12}{'alpha':>10}{'beta+s':>12}{'|beta+s|':>12}   判定")
for i, j in PAIRS:
    s = np.radians(beta[i, j]) + np.radians((book_d[i] + book_d[j]) / 2)
    a_ = np.radians(alpha[i, j])
    ok = abs(s) >= a_ - 1e-9
    print(f"   {labels[i]+'-'+labels[j]:<12}{alpha[i, j]:10.4f}"
          f"{np.degrees(s):12.4f}{abs(np.degrees(s)):12.4f}   "
          f"{'满足' if ok else '★ 不满足'}")
bad = [(i, j) for i, j in PAIRS
       if abs(np.radians(beta[i, j])
              + np.radians((book_d[i] + book_d[j]) / 2))
       < np.radians(alpha[i, j]) - 1e-9]
print(f"\n   书中解不满足约束的机对: "
      f"{[(labels[i], labels[j]) for i, j in bad] if bad else '无'}")
if bad:
    print("   -> 书中那个解在自己的模型里就不是可行解（很可能是 Matlab 提前终止，")
    print("      或者结果转录有误）。下面用仿真给出硬证据。")

# ---------------------------------------------------------- 仿真验证（关键）
def simulate(dtheta_deg, dt=1.0 / 3600.0, horizon=1.2):
    """按时间步长推进所有飞机，返回过程中的最小机间距。"""
    th = np.radians(theta0 + np.asarray(dtheta_deg))
    p = xy.copy()
    min_sep, worst, t = np.inf, None, 0.0
    while t <= horizon:
        p[:, 0] += SPEED * dt * np.cos(th)
        p[:, 1] += SPEED * dt * np.sin(th)
        t += dt
        for i, j in PAIRS:
            inside = any(0 <= p[k, 0] <= SIDE and 0 <= p[k, 1] <= SIDE for k in (i, j))
            if inside:
                sep = float(np.linalg.norm(p[i] - p[j]))
                if sep < min_sep:
                    min_sep, worst = sep, (i, j, t)
    return min_sep, worst


print("\n" + "=" * 70)
print("独立验证：沿飞行过程仿真，检查最小机间距（安全标准 > 8 km）")
print("=" * 70)
sep0, w0 = simulate(np.zeros(n))
sep_book, wb = simulate(book_d)
sep1, w1 = simulate(dtheta1)
print(f"   不做任何调整 : 最小间距 {sep0:8.3f} km   "
      f"（最近 {labels[w0[0]]}-{labels[w0[1]]}，t={w0[2]*60:.1f} 分钟）"
      f"   {'安全' if sep0 > SAFE else '★ 危险'}")
print(f"   书中的调整角 : 最小间距 {sep_book:8.3f} km   "
      f"（最近 {labels[wb[0]]}-{labels[wb[1]]}）"
      f"   {'安全' if sep_book > SAFE else '★ 危险'}")
print(f"   代码的调整角 : 最小间距 {sep1:8.3f} km   "
      f"（最近 {labels[w1[0]]}-{labels[w1[1]]}）"
      f"   {'安全' if sep1 > SAFE else '★ 危险'}")
assert sep1 > SAFE, f"代码解未通过安全标准：{sep1:.3f} km"
print(f"\n   [一致] 代码方案在整段飞行中最小间距 {sep1:.3f} km > {SAFE} km -> 安全")

# ============================================================== 模型二
def model2_constraints():
    """要求「两条飞行轨迹之间的最小距离 >= 8km」，对所有时刻成立。

    两机相对运动：r(t) = (p_i - p_j) + (v_i - v_j) t
    两条直线的最小距离平方 = |r|^2 - (r.w)^2/|w|^2（垂距），
    若最近点在 t <= 0（还没进入相对运动段）则取 |r|^2。

    这与书中 Delta_ij = btilde^2 - 4 atilde ctilde <= 0 是同一件事，
    但数量级好得多（前者 ~1e4，后者 ~1e10），数值上稳定得多。
    """
    cons = []
    for i, j in PAIRS:
        def mk(i=i, j=j):
            def fun(v):
                th_i = np.radians(theta0[i] + v[i])
                th_j = np.radians(theta0[j] + v[j])
                vi = SPEED * np.array([np.cos(th_i), np.sin(th_i)])
                vj = SPEED * np.array([np.cos(th_j), np.sin(th_j)])
                r = np.array([x0[i] - x0[j], y0[i] - y0[j]])
                w = vi - vj
                ww = float(w @ w)
                if ww < 1e-12:
                    d2 = float(r @ r)
                else:
                    t_star = -float(r @ w) / ww
                    d2 = float(r @ r)
                    if t_star > 0:
                        d2 -= float(r @ w) ** 2 / ww
                return d2 - SAFE ** 2
            return {"type": "ineq", "fun": fun}
        cons.append(mk())
    return cons


res2 = solve_nlp(lambda v: float(np.sum(v ** 2)), x0=dtheta1,   # 从模型一的解热启动
                 bounds=[(-30, 30)] * n, constraints=model2_constraints(),
                 options={"maxiter": 5000, "ftol": 1e-12}, name="模型二")
dtheta2 = res2["x"]          # 变量本身就是「度」
sep2, w2 = simulate(dtheta2)
report("模型二（对所有时刻成立，目标为平方和）", [
    ("各机调整角 (度)", np.round(dtheta2, 4).tolist()),
    ("目标 sum(dtheta^2)", res2["obj"]),
    ("仿真最小间距 (km)", sep2),
    ("求解状态", res2["status"]),
])
print(f"\n   仿真最小间距 {sep2:.3f} km   （最近 {labels[w2[0]]}-{labels[w2[1]]}）"
      f"   {'安全' if sep2 > SAFE else '★ 危险'}")
assert sep2 > SAFE - 1e-6, f"模型二解未满足安全标准：{sep2:.3f} km"

print("\n要点：")
print("  1) 模型一把安全压缩成几何角度判据，快；模型二直接要求「任意时刻距离 >= 8km」，")
print("     更严谨但约束是二次的、且必须用初值技巧才能求解。")
print("  2) 无论模型多漂亮，一定要仿真独立复核：这次复核直接发现书里的答案")
print("     在自己的约束下就不可行 —— 只有跑过一遍仿真才看得出来。")
print("  3) 目标选「绝对值和」还是「平方和」会给出不同的调整策略：")
print("     绝对值和倾向于少动几架飞机，平方和倾向于把角度摊开。")
