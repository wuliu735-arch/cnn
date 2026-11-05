"""绘图统一配置：优先使用中文字体。"""
from __future__ import annotations

import matplotlib.pyplot as plt


def use_chinese_font() -> None:
    """设置中文字体；找不到时退回 DejaVu Sans，避免直接报错。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["figure.facecolor"] = "white"
