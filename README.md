# 数学建模算法与应用（第 3 版）· Python 学习工程

对照《数学建模算法与应用》（司守奎、孙玺菁 主编，国防工业出版社 2021 年第 3 版）
**第 1–3 章**的完整 Python 实现 + 学习文档 + 一份《教学报告》长图。

每个脚本都能独立运行，**运行时会自动把计算结果与书中答案逐项比对**，
不一致就抛异常——这是学建模最有效的习惯：模型对不对，要能对上已知答案。

> **顺手做了件有意思的事**：这套代码复现书中全部 16 道例题时，
> **独立发现了 3 处书中错误**（见下方「发现的书中问题」）。
> 结论不是抄的，是自己算、自己仿真、自己证的。

[![CI](https://github.com/CurrenWong/math_modeling_learn/actions/workflows/ci.yml/badge.svg)](https://github.com/CurrenWong/math_modeling_learn/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![scipy](https://img.shields.io/badge/scipy-%3E%3D1.9-orange.svg)](https://scipy.org/)
[![NumPy](https://img.shields.io/badge/numpy-%3E%3D1.24-blue.svg)](https://numpy.org/)

---

## 快速开始

```bash
git clone https://github.com/CurrenWong/math_modeling_learn.git
cd math_modeling_learn
pip install -r requirements.txt

python ch01_linear_programming/ex1_4_warehouse.py
python ch02_integer_programming/ex2_10_tournament_tsp.py
python ch03_nonlinear_programming/ex3_10_flight.py
```

**全量回归**（一次跑完 16 个脚本，汇总与书中答案的比对结果）：

```bash
python run_all.py
```

预期输出：`共 16 个脚本，成功 16 个，失败 0 个` / `累计与书中答案一致的比对项：93 项` / `全部通过 ✅`

无需安装 MATLAB 工具箱。求解全部基于 **scipy.optimize**（`linprog` / `milp` / `minimize`），
核心只依赖 numpy、scipy、matplotlib 三个包。

---

## 目录结构

```
math_modeling_learn/
├── README.md                    ← 本文（总览 + API 对照 + 发现清单）
├── LICENSE                      ← MIT
├── requirements.txt             ← 核心依赖（numpy / scipy / matplotlib）
├── requirements-report.txt      ← 生成教学报告的额外依赖
├── run_all.py                   ← 全量回归：一键跑完 16 个脚本
├── common/
│   ├── solvers.py               ← 求解器薄封装：solve_lp / solve_milp / solve_nlp / check
│   ├── plotting.py              ← matplotlib 中文字体配置（跨平台）
│   └── __init__.py
├── data/                        ← 书中全部表格数据（CSV）
│   ├── ch1_warehouse.csv        表1.1 仓库需求与租费
│   ├── ch1_transport.csv        表1.2 单位商品运价
│   ├── ch1_assets.csv           表1.4 四种资产数据
│   ├── ch2_demand.csv           表2.1 工人数量需求
│   ├── ch2_renovation.csv       表2.2 装修费用
│   ├── ch2_sites.csv            表2.3 商业网点坐标
│   ├── ch2_tournament.csv       表2.4 比赛报名表（40×14，从 OCR 全文解析）
│   ├── ch3_stocks.csv           表3.2 三种股票年收益率
│   ├── ch3_supply.csv           表3.4 工地位置与水泥用量
│   └── ch3_flights.csv          表3.7 飞行记录
├── ch01_linear_programming/     ← 第 1 章 + README 学习文档
├── ch02_integer_programming/    ← 第 2 章 + README 学习文档
├── ch03_nonlinear_programming/   ← 第 3 章 + README 学习文档
├── figures/                     ← 章节脚本输出的图
└── report/                      ← 《教学报告》：十页长图 + 完整题干
    ├── report.html              ← 报告源文件（可直接用浏览器打开）
    ├── png/                     ← 渲染出的 11 张长图（交付物）
    ├── figs/                    ← 报告插图（由 make_figs.py 生成）
    ├── make_figs.py             ← 生成报告插图
    └── render_report.py         ← 把 report.html 渲染成 PNG（需 playwright）
```

---

## 📘 《教学报告》：十页长图

`report/` 下是一份把前三章方法论串起来的速通讲义，11 张 PNG（封面 + 10 页）：

| 页 | 内容 |
|---|---|
| 00 | 总纲：建模到底在干什么 · 五步法 |
| 01 | 武器一：把现实翻译成线性规划 |
| 02 | 武器二：线性化（建模的核心手艺） |
| 03 | 武器三：整数性 —— 现实世界不能切一半 |
| 04 | 武器四：凸性 —— 决定「一个解」还是「一堆解」 |
| 05 | 武器五：最优性证明 —— 别只信求解器 |
| 06 | 武器六：灵敏度分析 —— 建模的真正交付物 |
| 07 | 武器七：验证 —— 模型解出来 ≠ 事情解决了 |
| 08 | 总表：遇到问题按这张表选招 + 八大坑 |
| 09 | 接下来怎么练（三阶段练习路线） |

**每一页里出现的例题都附完整题干**（题面、数据表、约束式、书中答案），
不用翻书也能看懂在讲哪道题。

直接看图：`report/png/`
或用浏览器打开：`report/report.html`

### 重新生成报告

```bash
pip install -r requirements-report.txt
playwright install chromium        # 首次需要，下载浏览器内核

python report/make_figs.py         # 生成 10 张插图 -> report/figs/
python report/render_report.py     # 渲染 report.html -> report/png/（11 张）
```

> 图里的中文依赖系统字体。Windows 自带微软雅黑；Linux 建议装
> `fonts-noto-cjk`，macOS 自带苹方。找不到字体时脚本仍能跑，只是字会显示成方块。

---

## 16 个例题与验证结果

| 章 | 脚本 | 模型 | 书中答案 | 代码结果 |
|---|---|---|---|---|
| 1 | `ex1_4_warehouse.py` | 仓库租赁 LP | 118400 元 | ✅ 一致（多重最优解，见文档） |
| 1 | `ex1_5_transport.py` | 运输问题 | 运费 664 | ✅ 一致 |
| 1 | `ex1_7_abs_linearization.py` | 绝对值 / min-max 线性化 | x=(-2,0,0,0), z=2 | ✅ 一致 |
| 1 | `ex1_9_portfolio.py` | 投资收益与风险（1998 国赛 A 题） | a=0.006→Q=2019；表1.5 全部 | ✅ 逐行一致 |
| 2 | `ex2_5_scheduling.py` | 工人排班 MILP | 140 人 | ✅ 一致 |
| 2 | `ex2_6_assignment.py` | 装修分配 0-1 | 63.8，分配方案 | ✅ 一致 |
| 2 | `ex2_7_facility.py` | 供应站选址 | 2 个站 | ✅ 数量一致（位置为另一最优解） |
| 2 | `ex2_8_monte_carlo.py` | 蒙特卡洛求面积 | ≈49.5 | ✅ 一致 |
| 2 | `ex2_9_nlp_integer.py` | 非线性整数规划 | MC 50273 / 精确 51568 | ✅ 精确复现 51568 |
| 2 | `ex2_10_tournament_tsp.py` | 比赛项目排序 TSP | 回路总长 2 | ✅ 一致 |
| 3 | `ex3_1_convex_nlp.py` | 凸 NLP + K-T 验证 | (0.5536,1.3064), 3.7989 | ✅ 一致 |
| 3 | `ex3_2_tv_pricing.py` | 彩电定价（建模五步法） | (4735,7043), 553641.025 | ✅ 一致 |
| 3 | `ex3_3_quadratic.py` | 二次规划 / 非凸局部解 | (35.3704,64.6296), 11077.87 | ✅ 一致 |
| 3 | `ex3_8_markowitz.py` | Markowitz 组合 | (0.5269,0.3578,0.1153) / (0,0.0562,0.9438) | ✅ 一致 |
| 3 | `ex3_9_supply_location.py` | 两料场选址 | 料场 (7.25,7.75)+(3.2942,5.1790) | ⚠️ 位置一致，**目标值优于书中**（见发现清单） |
| 3 | `ex3_10_flight.py` | 飞行管理（1995 国赛 A 题） | Δθ5=2.7979°, Δθ6=0.9330° | ⚠️ **书中解不可行**，仿真证明（见发现清单） |

---

## Matlab → Python API 对照

| Matlab | Python (scipy) | 备注 |
|---|---|---|
| `optimproblem` + `optimvar` + `solve` | `scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)` | **只做最小化**，最大化要自己取负 |
| `optimproblem` + `Type:'integer'` | `scipy.optimize.milp(c, constraints=LinearConstraint(...), integrality=...)` | HiGHS 分支定界 |
| `fmincon` | `scipy.optimize.minimize(f, x0, method='SLSQP', constraints=[...])` | 约束用 `{"type":"ineq","fun":...}`，要求 `>= 0` |
| `fminbnd` / `fminsearch` | `minimize(..., method='L-BFGS-B'/'Powell')` | 无约束时 L-BFGS-B 更快 |
| `linprog` 的 A_ub 只收 `<=` | 同左 | `>=` 要两边乘 -1 |
| `dist(a)` | `scipy.spatial.distance.cdist` 或 numpy 广播 | |
| `angle(z)` | `np.angle(z)` | 复数幅角，单位弧度 |
| `optimconstr` 数组 | `constraints` 里的 dict 列表 | 每个约束单独写 fun/jac |
| `fcn2optimexpr` | 直接写 numpy 表达式 | Python 里不需要这层包装 |

`solve_lp(maximize=True)` 已经帮你处理了取负，`check()` 会把代码结果和书中答案摆在一起对照。

---

## 发现的书中问题（自己算出来的，不是抄的）

1. **例 3.10 飞行管理：书里的答案不满足它自己的约束。**
   书给 Δθ5=2.7979°、Δθ6=0.9330°，但机对 (3, 新进入) 的约束
   `|β+s| ≥ α` 不满足（0.777° < 2.125°）。
   仿真硬证据：不调整时最小间距 1.175 km（危险），**按书里的方案调整后最小间距仍只有 2.927 km（依然危险）**；
   本工程的解 Δθ3=2.558°、Δθ6=1.072° 使最小间距恰为 8.000 km（安全）。

2. **例 3.9 供应选址：书里的连续解是局部最优。**
   书给总吨千米 72.6303；本工程多起点求解得到 71.9352（更好），
   与书中「调整后」的 71.9372 几乎相同。

3. **例 2.9 的 Matlab 代码有一处笔误**：`prob.Constraints.con2 = sum(z)<=20;`
   把所有运输量加总与 20 比较，而总需求是 36，必然不可行。
   正确写法是按料场分别约束 `sum_i z_ij <= 20`。

4. **例 1.4、2.7 是多重最优解**：书中给的是其中一个，代码可能给出另一个。
   检验时应看目标值和可行性，不要死抠解向量。

---

## 学习建议

- 先跑 `ch01_linear_programming/README.md`，理解 **线性化** 是建模的基本功；
- 第 2 章重点理解 **整数性** 与 **分支定界**，以及「蒙特卡洛给初值、局部搜索给答案」；
- 第 3 章重点理解 **凸性决定解的唯一性**、**多起点**、以及 **必须独立验证**；
- 每章的 README 末尾有自测题，答案在脚本输出里；
- 配合《教学报告》（`report/png/`）读，方法论更顺。

书讲方法，这套代码讲「方法落到 Python 上会长什么样、会踩哪些坑」。

---

## 环境与兼容性

- Python 3.9+（开发环境 3.12）
- 核心依赖：`numpy>=1.24`、`scipy>=1.9`、`matplotlib>=3.6`
- 报告依赖（可选）：`playwright>=1.40`、`Pillow>=10.0`
- 路径全部相对于脚本自身，**克隆到任何目录都能跑**
- GitHub Actions 在每次 push 时自动跑 `run_all.py`（见上方 CI 徽章）

---

## 许可与版权

- 代码、注释、学习文档、教学报告：[MIT](LICENSE) © 2026 CurrenWong
- 本仓库引用了《数学建模算法与应用（第 3 版）》的例题题面与表格数据用于教学对照；
  **原书正文与 EPUB 不随本仓库分发**，版权归原出版方所有。
- 仓库中对书中结论的质疑，均为作者独立计算、仿真与验证的结果。
- 完整版权与免责说明见 [NOTICE.md](NOTICE.md)。

如果这套代码对你有帮助，欢迎 fork；如果发现了书里更多的问题，也欢迎提 issue 讨论。
