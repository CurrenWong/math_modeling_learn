"""项目级绘图配置：让 matplotlib 正常显示中文，避免豆腐块。

Windows 上优先用微软雅黑 / 黑体；找不到就退回默认字体（脚本仍可运行）。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402,F401  (对外重导出：脚本统一 `from common.plotting import plt`)

_CJK = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
        "WenQuanYi Micro Hei", "PingFang SC", "Heiti SC"]

# 保留全部可用中文字体：matplotlib >= 3.6 支持逐字形回退，
# 某个字体缺的符号（如 ⇒）会自动用列表里的下一个字体补上。
_installed = {f.name for f in font_manager.fontManager.ttflist}
_available = [n for n in _CJK if n in _installed]
if _available:
    matplotlib.rcParams["font.sans-serif"] = (
        _available + matplotlib.rcParams["font.sans-serif"])
    print(f"[字体] 中文字体: {', '.join(_available)}")
else:
    print("[警告] 未找到中文字体，图中中文可能显示为方块。"
          "Windows 自带微软雅黑；Linux 建议装 fonts-noto-cjk；"
          "或在 common/plotting.py 的 _CJK 列表里补充字体名。")
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110
matplotlib.rcParams["savefig.dpi"] = 130
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.alpha"] = 0.3
