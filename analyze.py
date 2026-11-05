"""汇总实验 JSON，并画出中文对比图。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plotting import use_chinese_font


COLORS = {
    "baseline": "#2463a6",
    "depthwise": "#e1843c",
    "se": "#328f79",
    "depthwise_se": "#8156a5",
    "gelu": "#b54a4a",
}


def load_results(results_dir: Path) -> list[dict]:
    """读取目录下所有单次实验 JSON。"""
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name in {"summary.json", "robustness.json"}:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "test" not in data or "parameters" not in data:
            continue
        rows.append(data)
    return rows


def write_summary(rows: list[dict], results_dir: Path) -> None:
    """输出一份中文表头的汇总 CSV/JSON。"""
    fields = [
        "模型", "变体", "参数量", "MACs", "GFLOPs",
        "最佳验证准确率", "最佳轮次", "测试准确率",
        "测试正确数", "测试总数", "CPU延迟毫秒", "GPU延迟毫秒",
        "训练秒数",
    ]
    flat = []
    for row in rows:
        flat.append({
            "模型": row["name"],
            "变体": row.get("variant", row["name"]),
            "参数量": row["parameters"],
            "MACs": row["macs"],
            "GFLOPs": row["gflops"],
            "最佳验证准确率": row["best_val_accuracy"],
            "最佳轮次": row["best_epoch"],
            "测试准确率": row["test"]["accuracy"],
            "测试正确数": row["test"]["correct"],
            "测试总数": row["test"]["total"],
            "CPU延迟毫秒": row.get("cpu_latency_ms"),
            "GPU延迟毫秒": row.get("gpu_latency_ms"),
            "训练秒数": row.get("train_seconds"),
        })
    with (results_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    (results_dir / "summary.json").write_text(
        json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写入 {results_dir / 'summary.csv'}（{len(flat)} 次实验）")


def plot_all(rows: list[dict], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    use_chinese_font()

    # 图 1：验证准确率曲线
    fig, ax = plt.subplots(figsize=(8.2, 4.6), layout="constrained")
    for row in rows:
        history = row.get("history", [])
        if not history:
            continue
        color = COLORS.get(row.get("variant", ""), "#555555")
        ax.plot(
            [h["epoch"] for h in history],
            [100 * h["val_accuracy"] for h in history],
            label=row.get("display", row["name"]), color=color, lw=2,
        )
    ax.set(xlabel="训练轮次", ylabel="验证准确率（%）", title="验证准确率曲线")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(figures_dir / "accuracy_curves.png", dpi=200)
    plt.close(fig)

    # 图 2：准确率 vs 参数量 / 计算量
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    for ax, key, xlabel, title in [
        (axes[0], "parameters", "参数量（百万）", "准确率与参数量"),
        (axes[1], "macs", "单张图片 MACs（百万）", "准确率与计算量"),
    ]:
        for row in rows:
            x = row[key] / 1e6
            y = 100 * row["test"]["accuracy"]
            color = COLORS.get(row.get("variant", ""), "#555555")
            ax.scatter(x, y, s=90, color=color, zorder=3)
            ax.annotate(
                row.get("display", row["name"]), (x, y),
                xytext=(6, 6), textcoords="offset points", fontsize=8,
            )
        ax.set(xlabel=xlabel, ylabel="测试准确率（%）", title=title)
        ax.grid(alpha=0.18)
    fig.savefig(figures_dir / "accuracy_vs_efficiency.png", dpi=200)
    plt.close(fig)

    # 图 3：准确率和参数量横向对比
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    names = [row.get("display", row["name"]) for row in rows]
    acc = [100 * row["test"]["accuracy"] for row in rows]
    color = [COLORS.get(row.get("variant", ""), "#555555") for row in rows]
    axes[0].barh(range(len(rows)), acc, color=color, height=0.62)
    axes[0].set(yticks=range(len(rows)), yticklabels=names, xlabel="测试准确率（%）")
    axes[0].invert_yaxis()
    axes[0].set_xlim(max(0, min(acc) - 5), 100)
    axes[0].grid(axis="x", alpha=0.18)
    for i, value in enumerate(acc):
        axes[0].text(value + 0.5, i, f"{value:.2f}", va="center", fontsize=8)

    params = [row["parameters"] / 1e6 for row in rows]
    axes[1].barh(range(len(rows)), params, color=color, height=0.62)
    axes[1].set(yticks=range(len(rows)), yticklabels=names, xlabel="参数量（百万）")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.18)
    for i, value in enumerate(params):
        axes[1].text(value + max(params) * 0.015, i, f"{value:.2f}", va="center", fontsize=8)
    fig.savefig(figures_dir / "model_comparison.png", dpi=200)
    plt.close(fig)
    print(f"已生成对比图：{figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总实验结果并画图")
    parser.add_argument("--results", default="results", help="结果 JSON 所在目录")
    parser.add_argument("--figures", default="figures", help="图片输出目录")
    args = parser.parse_args()
    results_dir = Path(args.results)
    rows = load_results(results_dir)
    if not rows:
        raise SystemExit(f"{results_dir} 下没有找到结果 JSON，请先训练模型")
    rows.sort(key=lambda r: r["parameters"])
    write_summary(rows, results_dir)
    plot_all(rows, Path(args.figures))


if __name__ == "__main__":
    main()
