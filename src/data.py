"""CIFAR-10 数据加载和固定划分。

训练集 50000 张，按固定随机种子分成 45000 张训练 + 5000 张验证；
官方测试集 10000 张只在最后评估一次。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# 文件夹名和类别键保持英文，避免跨平台路径编码问题；显示时用 CLASS_NAMES_ZH
CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
CLASS_NAMES_ZH = (
    "飞机", "汽车", "鸟", "猫", "鹿",
    "狗", "青蛙", "马", "船", "卡车",
)


def build_transforms(train: bool) -> transforms.Compose:
    """训练集做随机裁剪和水平翻转，验证/测试只做归一化。"""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def build_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    workers: int = 0,
    seed: int = 42,
    val_size: int = 5000,
    download: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """返回训练、验证、测试三个 DataLoader，划分方式对所有模型完全一致。"""
    root = str(data_dir)
    train_set = datasets.CIFAR10(root, train=True, download=download, transform=build_transforms(True))
    val_set = datasets.CIFAR10(root, train=True, download=download, transform=build_transforms(False))
    test_set = datasets.CIFAR10(root, train=False, download=download, transform=build_transforms(False))

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_set))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    common = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    if workers > 0:
        common["prefetch_factor"] = 2
    train_loader = DataLoader(Subset(train_set, train_indices), shuffle=True, **common)
    val_loader = DataLoader(Subset(val_set, val_indices), shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def raw_test_dataset(data_dir: str | Path, download: bool = False) -> datasets.CIFAR10:
    """只做 ToTensor 的测试集，供后续鲁棒性/可视化脚本使用。"""
    return datasets.CIFAR10(str(data_dir), train=False, download=download, transform=transforms.ToTensor())
