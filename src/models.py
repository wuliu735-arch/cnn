"""CIFAR-10 上的 ResNet-18 及受控结构变体。

主干采用适配 CIFAR 的 ResNet-18：3x3 卷积 stem、不做最大池化、四个
BasicBlock 阶段、全局平均池化和线性分类器。所有变体保持相同拓扑，
只改变被研究的那一个组件。

变体说明
--------
baseline      标准 3x3 卷积 + ReLU + BatchNorm
depthwise     每个 3x3 卷积换成“深度卷积 + 逐点卷积”
se            残差分支后加入 SE 通道注意力
depthwise_se  深度可分离卷积 + SE
gelu          把 ReLU 换成 GELU，参数量不变
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    """Mish 激活函数：x * tanh(softplus(x))。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


def get_activation(name: str) -> nn.Module:
    """按名称返回激活函数。"""
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return Mish()
    raise ValueError(f"未知激活函数：{name}")


def get_norm(name: str, channels: int) -> nn.Module:
    """按名称返回归一化层。"""
    name = name.lower()
    if name == "bn":
        return nn.BatchNorm2d(channels)
    if name == "gn":
        groups = 8 if channels % 8 == 0 else 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"未知归一化层：{name}")


class SEBlock(nn.Module):
    """SE 通道注意力：先全局平均池化，再学习每个通道的权重。"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class DepthwiseSeparableConv(nn.Module):
    """3x3 深度卷积 + 1x1 逐点卷积。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
            groups=in_ch, bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


def make_conv(in_ch: int, out_ch: int, stride: int, depthwise: bool) -> nn.Module:
    """按开关返回标准卷积或深度可分离卷积。"""
    if depthwise:
        return DepthwiseSeparableConv(in_ch, out_ch, stride)
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """ResNet 基础残差块，可切换深度可分离卷积和 SE 注意力。"""

    expansion = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        depthwise: bool = False,
        attention: str = "none",
        activation: str = "relu",
        norm: str = "bn",
    ) -> None:
        super().__init__()
        self.conv1 = make_conv(in_ch, out_ch, stride, depthwise)
        self.norm1 = get_norm(norm, out_ch)
        self.conv2 = make_conv(out_ch, out_ch, 1, depthwise)
        self.norm2 = get_norm(norm, out_ch)
        self.act = get_activation(activation)
        self.attention = SEBlock(out_ch) if attention.lower() == "se" else nn.Identity()
        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            # 下采样或通道数变化时，用 1x1 卷积把 shortcut 对齐
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                get_norm(norm, out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.attention(out)
        out = self.act(out + self.shortcut(x))
        return out


class ResNet18CIFAR(nn.Module):
    """适配 CIFAR-10 的 ResNet-18，可配置残差块组件。"""

    def __init__(
        self,
        num_classes: int = 10,
        width: int = 64,
        depthwise: bool = False,
        attention: str = "none",
        activation: str = "relu",
        norm: str = "bn",
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1, bias=False),
            get_norm(norm, width),
            get_activation(activation),
        )
        blocks: list[nn.Module] = []
        in_ch = width
        for stage, count in enumerate((2, 2, 2, 2)):
            channels = width * (2 ** stage)
            for i in range(count):
                stride = 2 if stage > 0 and i == 0 else 1
                blocks.append(
                    BasicBlock(
                        in_ch, channels, stride,
                        depthwise=depthwise,
                        attention=attention,
                        activation=activation,
                        norm=norm,
                    )
                )
                in_ch = channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """按 ResNet 常用方式初始化权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# 模型名 -> (是否深度可分离, 注意力类型, 激活函数, 归一化层)
VARIANTS: dict[str, tuple[bool, str, str, str]] = {
    "baseline": (False, "none", "relu", "bn"),
    "depthwise": (True, "none", "relu", "bn"),
    "se": (False, "se", "relu", "bn"),
    "depthwise_se": (True, "se", "relu", "bn"),
    "gelu": (False, "none", "gelu", "bn"),
}

DISPLAY_NAMES = {
    "baseline": "ResNet-18 基准",
    "depthwise": "深度可分离 ResNet-18",
    "se": "SE-ResNet-18",
    "depthwise_se": "深度可分离+SE ResNet-18",
    "gelu": "GELU-ResNet-18",
}


def make_model(name: str, width: int = 64, num_classes: int = 10) -> ResNet18CIFAR:
    """根据名称创建模型。"""
    if name not in VARIANTS:
        raise ValueError(f"未知模型：{name}，可选：{sorted(VARIANTS)}")
    depthwise, attention, activation, norm = VARIANTS[name]
    return ResNet18CIFAR(
        num_classes=num_classes,
        width=width,
        depthwise=depthwise,
        attention=attention,
        activation=activation,
        norm=norm,
    )


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.inference_mode()
def count_macs(model: nn.Module, input_size: tuple[int, int, int] = (3, 32, 32)) -> int:
    """统计卷积和全连接层的乘加次数（MACs），不含 BN、激活、池化。"""
    total = [0]
    handles = []

    def hook(module: nn.Module, inputs, output) -> None:
        if isinstance(module, nn.Conv2d):
            total[0] += output.numel() * (module.in_channels // module.groups) * (
                module.kernel_size[0] * module.kernel_size[1]
            )
        elif isinstance(module, nn.Linear):
            total[0] += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    model(torch.zeros(1, *input_size))
    if was_training:
        model.train()
    for handle in handles:
        handle.remove()
    return total[0]
