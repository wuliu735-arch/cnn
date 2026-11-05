"""低照度与传感器噪声场景下的鲁棒性评估。

对 5 个已训练模型，在 clean / 低照度 / 高斯噪声 / 低照度+噪声 四种条件下
重新评估测试集；并对最佳模型尝试推理阶段的亮度补偿和水平翻转 TTA。
结果写入 results/robustness.json，图写入 figures/。
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.data import CIFAR_MEAN, CIFAR_STD, CLASS_NAMES_ZH
from src.models import VARIANTS, make_model
from src.plotting import use_chinese_font

CONDITION_ZH = {
    "clean": "清晰",
    "low_light": "低照度",
    "noise": "高斯噪声",
    "low_light_noise": "低照度+噪声",
}
MODEL_ORDER = ["baseline", "depthwise", "se", "depthwise_se", "gelu"]
MODEL_ZH = {
    "baseline": "ResNet-18 基准",
    "depthwise": "深度可分离",
    "se": "SE",
    "depthwise_se": "深度可分离+SE",
    "gelu": "GELU",
}
COLORS = ["#2463a6", "#e1843c", "#328f79", "#8156a5", "#b54a4a"]


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR_MEAN, device=x.device)[None, :, None, None]
    std = torch.tensor(CIFAR_STD, device=x.device)[None, :, None, None]
    return (x - mean) / std


def corrupt(x: torch.Tensor, condition: str, seed: int = 42) -> torch.Tensor:
    """在 [0,1] 像素空间施加退化。"""
    if condition == "clean":
        return x
    if condition == "low_light":
        return (x * 0.35).clamp(0, 1)
    if condition == "noise":
        generator = torch.Generator(device=x.device).manual_seed(seed)
        noise = torch.randn(x.shape, generator=generator, device=x.device) * 0.08
        return (x + noise).clamp(0, 1)
    if condition == "low_light_noise":
        generator = torch.Generator(device=x.device).manual_seed(seed)
        noise = torch.randn(x.shape, generator=generator, device=x.device) * 0.08
        return (x * 0.35 + noise).clamp(0, 1)
    raise ValueError(condition)


def adapt(x: torch.Tensor, method: str) -> torch.Tensor:
    """在 [0,1] 像素空间做推理阶段补偿。"""
    if method == "none":
        return x
    if method == "gamma":
        # 低照度图像做 gamma<1 的提亮
        return x.clamp(0, 1).pow(0.5)
    if method == "brightness":
        # 把每张图的平均亮度拉到 0.45
        mean = x.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
        return (x * (0.45 / mean)).clamp(0, 1)
    raise ValueError(method)


@torch.inference_mode()
def evaluate_condition(model, loader, device, condition: str, method: str = "none", tta: bool = False):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = adapt(corrupt(x, condition), method)
        logits = model(normalize(x))
        if tta:
            logits = (logits + model(normalize(torch.flip(x, dims=[3])))) / 2
        correct += int((logits.argmax(1) == y).sum().item())
        total += y.numel()
    return correct / max(total, 1)


def plot_results(results: dict, adaptation: dict, figures: Path) -> None:
    use_chinese_font()
    conditions = list(CONDITION_ZH)
    fig, ax = plt.subplots(figsize=(10.5, 4.8), layout="constrained")
    width = 0.16
    x = torch.arange(len(conditions)).numpy()
    for i, name in enumerate(MODEL_ORDER):
        values = [100 * results[name][c] for c in conditions]
        ax.bar(x + (i - 2) * width, values, width, label=MODEL_ZH[name], color=COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_ZH[c] for c in conditions])
    ax.set_ylabel("测试准确率（%）")
    ax.set_title("不同成像条件下的测试准确率")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(figures / "robustness.png", dpi=200)
    plt.close(fig)

    methods = ["none", "gamma", "brightness", "gamma_tta"]
    labels = ["不补偿", "Gamma 提亮", "亮度归一化", "Gamma + TTA"]
    values = [100 * adaptation[m] for m in methods]
    fig, ax = plt.subplots(figsize=(7.6, 4.2), layout="constrained")
    bars = ax.bar(labels, values, color=["#8a8a8a", "#e1843c", "#328f79", "#2463a6"])
    ax.set_ylabel("低照度下测试准确率（%）")
    ax.set_title("最佳模型的低照度补偿效果")
    ax.set_ylim(max(0, min(values) - 5), min(100, max(values) + 5))
    ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.2f}", ha="center", fontsize=9)
    fig.savefig(figures / "adaptation.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="低照度/噪声鲁棒性评估")
    parser.add_argument("--results", default="results")
    parser.add_argument("--data", default="data")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--batch", default=128, type=int)
    args = parser.parse_args()

    results_dir = Path(args.results)
    figures_dir = Path(args.figures)
    figures_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set = datasets.CIFAR10(args.data, train=False, download=False, transform=transforms.ToTensor())
    loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=0)

    raw = {}
    for name in MODEL_ORDER:
        meta = json.loads((results_dir / f"{name}.json").read_text(encoding="utf-8"))
        model = make_model(meta["variant"], width=meta["width"])
        model.load_state_dict(torch.load(
            results_dir / "checkpoints" / f"{name}.pt", map_location=device, weights_only=True
        ))
        model.to(device)
        raw[name] = model
        print(f"已加载 {name}", flush=True)

    results = {}
    for name, model in raw.items():
        results[name] = {
            condition: evaluate_condition(model, loader, device, condition)
            for condition in CONDITION_ZH
        }
        print(name, results[name], flush=True)

    # 以清晰条件下测试准确率最高的模型做补偿实验
    best_name = max(MODEL_ORDER, key=lambda n: results[n]["clean"])
    best_model = raw[best_name]
    adaptation = {}
    for method in ("none", "gamma", "brightness"):
        adaptation[method] = evaluate_condition(
            best_model, loader, device, "low_light", method=method
        )
    adaptation["gamma_tta"] = evaluate_condition(
        best_model, loader, device, "low_light", method="gamma", tta=True
    )

    payload = {
        "conditions": CONDITION_ZH,
        "models": results,
        "best_model": best_name,
        "adaptation": adaptation,
    }
    (results_dir / "robustness.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_results(results, adaptation, figures_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
