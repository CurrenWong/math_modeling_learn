"""scipy.optimize 的薄封装，让三章代码共用一套调用与打印风格。

设计原则
--------
1. 书里全是「最大化」，而 scipy.optimize.linprog / milp 只做「最小化」。
   封装层统一处理这个方向问题：solve_lp(maximize=True) 内部取负，
   返回的 obj 已经是原问题的目标值。
2. 每个求解器都返回 dict，含 x / obj / status，方便打印和断言。
3. check() 用来把「书上的答案」和「代码算出来的答案」摆在一起对照，
   这是学建模最有效的习惯：模型对不对，要能对上已知答案。
"""
from __future__ import annotations

import numpy as np
from scipy import optimize

__all__ = [
    "solve_lp", "solve_milp", "solve_nlp", "report", "check",
    "LINEAR_CONSTRAINTS", "NONLINEAR_CONSTRAINTS",
]

LINEAR_CONSTRAINTS = "linear"
NONLINEAR_CONSTRAINTS = "nonlinear"


# ---------------------------------------------------------------- 线性规划
def solve_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
              bounds=None, maximize=False, name="LP"):
    """求解线性规划。

    c        : 目标系数（一维数组）
    A_ub/b_ub: A_ub x <= b_ub
    A_eq/b_eq: A_eq x == b_eq
    bounds   : 变量上下界，默认 (0, None)
    maximize : 是否最大化（内部取负，obj 已还原为原目标值）
    """
    c = np.asarray(c, dtype=float)
    sign = -1.0 if maximize else 1.0
    res = optimize.linprog(
        sign * c,
        A_ub=None if A_ub is None else np.atleast_2d(A_ub),
        b_ub=None if b_ub is None else np.asarray(b_ub, dtype=float),
        A_eq=None if A_eq is None else np.atleast_2d(A_eq),
        b_eq=None if b_eq is None else np.asarray(b_eq, dtype=float),
        bounds=(0, None) if bounds is None else bounds,
        method="highs",
    )
    obj = float(res.fun) * sign if res.success else float("nan")
    return {
        "name": name, "x": res.x, "obj": obj, "success": bool(res.success),
        "status": res.message, "raw": res,
    }


# ------------------------------------------------------------ 混合整数规划
def solve_milp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
               bounds=None, integrality=None, maximize=False, name="MILP"):
    """求解（混合）整数规划，底层是 HiGHS 分支定界。

    integrality : 0=连续, 1=整数（默认全整数）
    """
    c = np.asarray(c, dtype=float)
    n = c.size
    if integrality is None:
        integrality = np.ones(n)
    integrality = np.asarray(integrality, dtype=int)

    cons = []
    if A_ub is not None:
        cons.append(optimize.LinearConstraint(
            np.atleast_2d(A_ub), -np.inf, np.asarray(b_ub, dtype=float)))
    if A_eq is not None:
        cons.append(optimize.LinearConstraint(
            np.atleast_2d(A_eq), np.asarray(b_eq, dtype=float),
            np.asarray(b_eq, dtype=float)))
    if bounds is None:
        bounds = [(0, None)] * n
    elif (isinstance(bounds, tuple) and len(bounds) == 2
          and np.isscalar(bounds[0]) and np.isscalar(bounds[1])):
        bounds = [tuple(bounds)] * n          # (0, 1) 这种标量形式 -> 展开到所有变量
    lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds], dtype=float)
    hi = np.array([b[1] if b[1] is not None else np.inf for b in bounds], dtype=float)
    bounds = optimize.Bounds(lo, hi)
    sign = -1.0 if maximize else 1.0
    res = optimize.milp(
        c=sign * c, constraints=cons, integrality=integrality,
        bounds=bounds, options={"disp": False},
    )
    obj = float(res.fun) * sign if res.success else float("nan")
    return {
        "name": name, "x": res.x, "obj": obj, "success": bool(res.success),
        "status": res.message, "raw": res,
    }


# -------------------------------------------------------------- 非线性规划
def solve_nlp(objective, x0, bounds=None, constraints=None, method="SLSQP",
              options=None, jac=None, name="NLP"):
    """求解非线性规划（默认 SLSQP）。

    objective  : f(x) -> float；若 jac=True 则返回 (f, grad)
    x0         : 初始点
    bounds     : [(lo, hi), ...]
    constraints: list of dict，键为 type/fun/jac：
                 {"type": "eq",    "fun": lambda x: ..., "jac": ...}
                 {"type": "ineq",  "fun": lambda x: ...}   # 要求 >= 0
    jac        : 传给 scipy 的 jac 参数；True 表示 objective 自带梯度
    """
    opts = {"maxiter": 2000, "ftol": 1e-10}
    if options:
        opts.update(options)
    res = optimize.minimize(
        objective, np.asarray(x0, dtype=float), method=method,
        bounds=bounds, constraints=constraints or [], options=opts, jac=jac,
    )
    return {
        "name": name, "x": res.x, "obj": float(res.fun), "success": bool(res.success),
        "status": res.message, "raw": res,
    }


# ------------------------------------------------------------------ 打印
def report(title, rows):
    """打印一张对照表：rows = [(项目, 值), ...]"""
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")
    for k, v in rows:
        if isinstance(v, float):
            v = f"{v:,.4f}"
        print(f"  {k:<28} {v}")


def check(label, computed, expected, tol=1e-2, rel=False):
    """把代码结果与书中答案比对，不一致就抛异常。

    相对误差用于「答案本身很大」的场景（如 11 万的运费）。
    """
    computed = float(computed)
    expected = float(expected)
    if rel and expected != 0:
        err = abs(computed - expected) / abs(expected)
    else:
        err = abs(computed - expected)      # 期望为 0 时只能比绝对误差
    ok = err <= tol
    flag = "一致" if ok else "不一致"
    print(f"  [{flag}] {label:<34} 代码={computed:>14,.4f}  书={expected:>14,.4f}"
          f"  相对误差={err:.2e}")
    if not ok:
        raise AssertionError(f"{label}: 代码 {computed} 与书中 {expected} 不一致")
    return ok
