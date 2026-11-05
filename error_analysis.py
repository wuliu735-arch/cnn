"""根据测试集逐样本预测画混淆矩阵和各类别准确率。"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import CLASS_NAMES_ZH
from src.models import DISPLAY_NAMES
from src.plotting import use_chinese_font


def load_predictions(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["test"]["labels"]), np.asarray(data["test"]["predictions"])


def main() -> None:
    parser = argparse.ArgumentParser(description="混淆矩阵与各类别准确率")
    parser.add_argument("--results", default="results")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--model", default="gelu", help="用于画混淆矩阵的模型")
    parser.add_argument("--compare", default="baseline", help="各类别准确率的对照模型")
    args = parser.parse_args()

    results_dir = Path(args.results)
    figures_dir = Path(args.figures)
    figures_dir.mkdir(exist_ok=True)
    use_chinese_font()

    labels, preds = load_predictions(results_dir / f"{args.model}.json")
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (labels, preds), 1)
    support = cm.sum(axis=1)
    recall = np.diag(cm) / np.maximum(support, 1)

    with (results_dir / "per_class.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["类别", "测试样本数", "正确数", "召回率"])
        for i, name in enumerate(CLASS_NAMES_ZH):
            writer.writerow([name, int(support[i]), int(cm[i, i]), f"{recall[i]:.4f}"])

    fig, ax = plt.subplots(figsize=(7.6, 6.4), layout="constrained")
    im = ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if i == j or cm[i, j] >= 30:
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "#203040",
                )
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES_ZH, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES_ZH)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"混淆矩阵：{DISPLAY_NAMES[args.model]}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="样本数")
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    labels_b, preds_b = load_predictions(results_dir / f"{args.compare}.json")
    recall_b = np.zeros(10)
    for i in range(10):
        mask = labels_b == i
        recall_b[i] = (preds_b[mask] == i).mean() if mask.any() else 0.0

    x = np.arange(10)
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 4.4), layout="constrained")
    ax.bar(x - width / 2, 100 * recall_b, width, label=DISPLAY_NAMES[args.compare], color="#2463a6")
    ax.bar(x + width / 2, 100 * recall, width, label=DISPLAY_NAMES[args.model], color="#b54a4a")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES_ZH)
    ax.set_ylabel("类别准确率（%）")
    ax.set_title("各类别测试准确率对比")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    for i in range(10):
        ax.text(i - width / 2, 100 * recall_b[i] + 0.5, f"{100 * recall_b[i]:.1f}", ha="center", fontsize=6)
        ax.text(i + width / 2, 100 * recall[i] + 0.5, f"{100 * recall[i]:.1f}", ha="center", fontsize=6)
    fig.savefig(figures_dir / "per_class_accuracy.png", dpi=200)
    plt.close(fig)

    print(f"已写入 {results_dir / 'per_class.csv'}")
    print("各类别召回率：")
    for name, value in zip(CLASS_NAMES_ZH, recall):
        print(f"  {name}: {100 * value:.2f}%")


if __name__ == "__main__":
    main()
