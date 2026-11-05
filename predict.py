"""用训练好的 CIFAR-10 模型对单张图片做预测。"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.data import CIFAR_MEAN, CIFAR_STD, CLASS_NAMES, CLASS_NAMES_ZH
from src.models import VARIANTS, make_model


def main() -> None:
    p = argparse.ArgumentParser(description="单张图片分类预测")
    p.add_argument("image", type=Path, help="输入图片路径")
    p.add_argument("--model", choices=sorted(VARIANTS), default="baseline", help="模型名称")
    p.add_argument("--checkpoint", type=Path, default=None, help="权重路径")
    p.add_argument("--width", type=int, default=64, help="训练时的基础通道数")
    p.add_argument("--topk", type=int, default=3, help="显示前 k 个类别")
    args = p.parse_args()

    checkpoint = args.checkpoint or Path("results/checkpoints") / f"{args.model}.pt"
    if not checkpoint.exists():
        raise SystemExit(f"找不到权重文件：{checkpoint}")
    model = make_model(args.model, width=args.width)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    image = Image.open(args.image).convert("RGB")
    with torch.inference_mode():
        probs = torch.softmax(model(tf(image).unsqueeze(0)), dim=1)[0]
    values, indices = probs.topk(args.topk)
    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        print(f"{rank}. {CLASS_NAMES_ZH[index]}（{CLASS_NAMES[index]}）：{value:.4f}")


if __name__ == "__main__":
    main()
