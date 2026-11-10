"""全量回归：依次运行 16 个例题脚本，汇总成功/失败。

用法：python run_all.py
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# stdout 被重定向到管道/文件时，Windows 会用本地代码页（cp936/cp1252）编码，
# 打印中文直接 UnicodeEncodeError —— CI 里跑的正是这种情况。
# 终端直连时保留原编码，否则 cmd 里会显示成乱码。
if hasattr(sys.stdout, "reconfigure"):
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS = [
    "ch01_linear_programming/ex1_4_warehouse.py",
    "ch01_linear_programming/ex1_5_transport.py",
    "ch01_linear_programming/ex1_7_abs_linearization.py",
    "ch01_linear_programming/ex1_9_portfolio.py",
    "ch02_integer_programming/ex2_5_scheduling.py",
    "ch02_integer_programming/ex2_6_assignment.py",
    "ch02_integer_programming/ex2_7_facility.py",
    "ch02_integer_programming/ex2_8_monte_carlo.py",
    "ch02_integer_programming/ex2_9_nlp_integer.py",
    "ch02_integer_programming/ex2_10_tournament_tsp.py",
    "ch03_nonlinear_programming/ex3_1_convex_nlp.py",
    "ch03_nonlinear_programming/ex3_2_tv_pricing.py",
    "ch03_nonlinear_programming/ex3_3_quadratic.py",
    "ch03_nonlinear_programming/ex3_8_markowitz.py",
    "ch03_nonlinear_programming/ex3_9_supply_location.py",
    "ch03_nonlinear_programming/ex3_10_flight.py",
]

results = []
for s in SCRIPTS:
    path = os.path.join(ROOT, s.replace("/", os.sep))
    t0 = time.time()
    # 子进程强制 UTF-8 输出，否则 Windows 控制台默认 cp936，
    # 捕获的中文会变成乱码（Linux/macOS 本身就是 UTF-8）
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.run([PY, path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=ROOT, env=env)
    dt = time.time() - t0
    out = p.stdout or ""
    ok = p.returncode == 0
    n_check = out.count("[一致]")
    n_bad = out.count("[不一致]")
    results.append((s, ok, dt, n_check, n_bad, (p.stderr or "").strip().splitlines()[-1:] ))
    flag = "OK  " if ok else "FAIL"
    print(f"{flag} {s:<52} {dt:6.2f}s  一致 {n_check:>2} 项 / 不一致 {n_bad} 项")
    if not ok:
        for line in results[-1][5]:
            print("       ", line)

print("\n" + "=" * 78)
failed = [r for r in results if not r[1]]
print(f"共 {len(results)} 个脚本，成功 {len(results) - len(failed)} 个，失败 {len(failed)} 个")
print(f"累计与书中答案一致的比对项：{sum(r[3] for r in results)} 项")
if failed:
    print("失败清单：")
    for r in failed:
        print("   ", r[0], r[5])
    sys.exit(1)
print("全部通过 ✅")
