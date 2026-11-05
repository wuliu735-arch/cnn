"""训练单个 CIFAR-10 结构对照实验。

所有结果写入 ``results/<name>.json``。测试集只在验证集选出最佳
checkpoint 之后评估一次，避免用测试集调参。
"""
from __future__ import annotations

import os
# 避免 PyTorch 与 Anaconda 的 OpenMP 运行库冲突导致 CUDA 初始化崩溃
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data import CLASS_NAMES_ZH, build_loaders
from src.engine import JsonlLogger, cpu_latency, evaluate, gpu_latency, seed_all
from src.models import DISPLAY_NAMES, VARIANTS, count_macs, count_parameters, make_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR-10 ResNet-18 结构对照实验")
    p.add_argument("--model", required=True, choices=sorted(VARIANTS), help="要训练的模型")
    p.add_argument("--data", default="data", type=str, help="CIFAR-10 数据目录")
    p.add_argument("--out", default="results", type=str, help="结果输出目录")
    p.add_argument("--epochs", default=30, type=int, help="训练轮数")
    p.add_argument("--batch", default=128, type=int, help="批大小")
    p.add_argument("--workers", default=0, type=int, help="DataLoader 进程数；Windows 建议 0")
    p.add_argument("--seed", default=42, type=int, help="随机种子")
    p.add_argument("--width", default=64, type=int, help="ResNet 基础通道数")
    p.add_argument("--lr", default=0.1, type=float, help="初始学习率")
    p.add_argument("--weight-decay", default=5e-4, type=float, help="权重衰减")
    p.add_argument("--label-smoothing", default=0.1, type=float, help="标签平滑系数")
    p.add_argument("--download", action="store_true", help="数据缺失时让 torchvision 自动下载")
    p.add_argument("--tag", default="", type=str, help="多次运行时给结果文件名加后缀")
    p.add_argument("--smoke", action="store_true", help="只跑几个 batch，用于检查代码")
    p.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    run_name = f"{args.model}{args.tag}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_batches = 3 if args.smoke else None

    train_loader, val_loader, test_loader = build_loaders(
        args.data, batch_size=args.batch, workers=args.workers,
        seed=args.seed, download=args.download,
    )
    model = make_model(args.model, width=args.width)
    parameters = count_parameters(model)
    macs = count_macs(model)
    model.to(device)

    criterion_train = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_eval = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    logger = JsonlLogger(out_dir / "logs" / f"{run_name}.jsonl")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history: list[dict] = []
    best_val = -1.0
    best_epoch = -1
    start = time.perf_counter()

    print(
        "开始训练 " + json.dumps({
            "运行名": run_name, "模型": DISPLAY_NAMES[args.model],
            "设备": str(device), "参数量": parameters, "MACs": macs,
            "GFLOPs": round(macs / 1e9, 4), "轮数": args.epochs,
            "批大小": args.batch, "宽度": args.width, "随机种子": args.seed,
        }, ensure_ascii=False),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = correct = total = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion_train(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * y.numel()
            correct += int((logits.argmax(1) == y).sum().item())
            total += y.numel()

        validation = evaluate(
            model, val_loader, device, criterion=criterion_eval,
            max_batches=max_batches,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(total, 1),
            "train_accuracy": correct / max(total, 1),
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.perf_counter() - epoch_start, 3),
        }
        history.append(record)
        logger.write(record)
        print(
            f"第 {epoch:02d} 轮 | 训练损失 {record['train_loss']:.4f} | "
            f"训练准确率 {100 * record['train_accuracy']:.2f}% | "
            f"验证损失 {record['val_loss']:.4f} | "
            f"验证准确率 {100 * record['val_accuracy']:.2f}% | "
            f"学习率 {record['lr']:.5f} | 用时 {record['epoch_seconds']:.1f}s",
            flush=True,
        )

        if validation["accuracy"] > best_val:
            best_val = validation["accuracy"]
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "checkpoints" / f"{run_name}.pt")

    train_seconds = time.perf_counter() - start
    checkpoint_path = out_dir / "checkpoints" / f"{run_name}.pt"
    if not checkpoint_path.exists():
        raise RuntimeError("没有保存任何 checkpoint")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    test_result = evaluate(
        model, test_loader, device, criterion=criterion_eval,
        collect_predictions=True, max_batches=max_batches,
    )
    gpu_ms = gpu_latency(model, device) if device.type == "cuda" else None
    cpu_ms = cpu_latency(model)
    peak_memory = (
        torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else None
    )

    result = {
        "name": run_name,
        "display": DISPLAY_NAMES[args.model],
        "variant": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch": args.batch,
        "width": args.width,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "parameters": parameters,
        "macs": macs,
        "gflops": macs / 1e9,
        "model_size_mb": parameters * 4 / 1e6,
        "best_val_accuracy": best_val,
        "best_epoch": best_epoch,
        "test": test_result,
        "cpu_latency_ms": cpu_ms,
        "gpu_latency_ms": gpu_ms,
        "peak_gpu_memory_mb": peak_memory,
        "train_seconds": train_seconds,
        "history": history,
        "class_names": list(CLASS_NAMES_ZH),
        "config": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "optimizer": "SGD(nesterov)",
            "scheduler": "CosineAnnealingLR",
            "amp": use_amp,
            "train_size": 45000,
            "val_size": 5000,
            "test_size": 10000,
        },
    }
    result_path = out_dir / f"{run_name}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "最终结果 " + json.dumps({
            "运行名": run_name,
            "最佳验证准确率": best_val,
            "最佳轮次": best_epoch,
            "测试准确率": test_result["accuracy"],
            "测试正确数": test_result["correct"],
            "参数量": parameters,
            "MACs": macs,
            "GFLOPs": round(macs / 1e9, 4),
            "训练秒数": round(train_seconds, 2),
            "CPU延迟毫秒": round(cpu_ms, 3),
            "GPU延迟毫秒": round(gpu_ms, 3) if gpu_ms is not None else None,
        }, ensure_ascii=False),
        flush=True,
    )
    print(f"结果文件：{result_path}", flush=True)


if __name__ == "__main__":
    main()
