"""训练和评估共用的工具函数。"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


def seed_all(seed: int) -> None:
    """固定 Python、NumPy、PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> int:
    """统计一个 batch 中预测正确的样本数。"""
    return int((logits.argmax(dim=1) == targets).sum().item())


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    criterion: nn.Module | None = None,
    collect_predictions: bool = False,
    max_batches: int | None = None,
) -> dict:
    """在给定数据集上评估模型，可选返回逐样本预测。"""
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    preds: list[int] = []
    labels: list[int] = []
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * y.numel()
        correct += accuracy_from_logits(logits, y)
        total += y.numel()
        if collect_predictions:
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(y.cpu().tolist())
    out = {
        "loss": loss_sum / max(total, 1) if criterion is not None else None,
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
    }
    if collect_predictions:
        out["predictions"] = preds
        out["labels"] = labels
    return out


@torch.inference_mode()
def cpu_latency(model: nn.Module, repeats: int = 50) -> float:
    """单线程 CPU 单张 32x32 图片的推理延迟（毫秒）。"""
    model = model.cpu().eval()
    x = torch.randn(1, 3, 32, 32)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(10):
            model(x)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        elapsed = (time.perf_counter() - start) * 1000.0 / repeats
    finally:
        torch.set_num_threads(old_threads)
    return elapsed


@torch.inference_mode()
def gpu_latency(model: nn.Module, device: torch.device, repeats: int = 50) -> float:
    """GPU 单张图片推理延迟（毫秒），先预热再取中位数。"""
    model = model.to(device).eval()
    x = torch.randn(1, 3, 32, 32, device=device)
    for _ in range(20):
        model(x)
    torch.cuda.synchronize(device)
    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start_event.record()
        model(x)
        end_event.record()
        torch.cuda.synchronize(device)
        times.append(start_event.elapsed_time(end_event))
    return float(np.median(times))


class JsonlLogger:
    """按轮追加 JSONL 日志，方便事后检查训练过程。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
