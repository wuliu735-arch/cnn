# CIFAR-10 ResNet-18 结构对比实验

这是一个 **图像分类** 代码仓库：输入一张 32×32 的彩色图片，输出它属于 10 个类别中的哪一类。所有说明、注释、训练日志和图表标题均为中文。

实验比较经典 ResNet-18 与四种结构修改：

| 模型名 | 修改内容 | 目的 |
|---|---|---|
| `baseline` | 标准 3×3 卷积 + ReLU + BatchNorm | 基准 |
| `depthwise` | 3×3 卷积换成深度可分离卷积（DW 3×3 + PW 1×1） | 降参数量、降计算量 |
| `se` | 残差分支后加入 SE 通道注意力 | 观察注意力是否涨点 |
| `depthwise_se` | 深度可分离卷积 + SE | 轻量化组合 |
| `gelu` | ReLU 换成 GELU | 激活函数对照 |

所有变体保持相同的 ResNet-18 拓扑、数据划分、训练轮数、优化器和随机种子，只改变被研究的组件。

## 实验结果

30 epochs、width=32、seed=42，官方测试集 10,000 张：

| 模型 | 测试准确率 | 参数量 | MACs |
|---|---:|---:|---:|
| ResNet-18 基准 | 92.21% | 2.80M | 0.1393G |
| 深度可分离 | 90.49% | 0.37M | 0.0197G |
| SE | 92.11% | 2.82M | 0.1393G |
| 深度可分离+SE | 90.82% | 0.39M | 0.0198G |
| GELU | 92.36% | 2.80M | 0.1393G |

完整报告随课程作业单独提交，不放在本仓库。

> 本仓库只保留代码、运行说明、`results/` 和 `figures/`；不提交报告、5 个模型的 `.pt` 权重和 500MB 的 CIFAR-10 数据。按下方命令重新训练即可复现。

## 环境

已在以下环境验证：

- Windows 11
- Python 3.12
- PyTorch 2.12.1 + CUDA 12.6
- torchvision 0.27.1
- NVIDIA RTX 4060 Laptop GPU（8 GB）

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

如果使用 NVIDIA GPU，建议按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版本，不要直接装 CPU 版。

## 数据

数据来源：[CIFAR-10 官方页面](https://www.cs.toronto.edu/~kriz/cifar.html)  
直接下载：[cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)  
MD5：`c58f30108f718f92721af3b95e74349a`

代码默认读取 `data/cifar-10-batches-py/`。首次运行可让 torchvision 自动下载：

```powershell
python train.py --model baseline --data data --epochs 30 --download
```

如果官方源较慢，可使用镜像下载脚本：

```powershell
python download_data.py
```

脚本会下载 `cifar-10-python.tar.gz`，校验官方 MD5 `c58f30108f718f92721af3b95e74349a`，并自动解压到 `data/`。

数据划分固定为：

- 训练集：50,000 张中的 45,000 张
- 验证集：50,000 张中的 5,000 张（用验证集选最佳 checkpoint）
- 测试集：官方 10,000 张，只在最后评估一次

### 为什么资源管理器里看不到图片

CIFAR-10 官方发布格式不是 jpg 文件夹，而是 pickle 打包的二进制批次：

```text
data/cifar-10-batches-py/
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
└── test_batch
```

图片内容确实在里面，只是 Windows 资源管理器无法预览。把二进制批次解码成真正的 jpg 照片：

```powershell
python export_images.py --data data --out data/images --per-class 20 --size 128
```

导出后的结构：

```text
data/images/
├── train/
│   ├── airplane/00001.jpg
│   ├── automobile/00002.jpg
│   └── ...
└── test/
    └── ...
```

现在可以直接双击 `.jpg` 看照片。CIFAR-10 原图只有 32×32，脚本用 Lanczos 放大到 128×128，方便查看；放大不会增加真实细节。

生成一张 32 张图的预览：

```powershell
python preview_data.py --images data/images/train --out figures/cifar10_samples.png
```

如果还没导出 jpg，脚本会自动回退到读取原始 batch。

## 运行说明

训练单个模型：

```powershell
python train.py --model baseline --epochs 30 --width 32 --batch 128 --workers 0 --data data --out results
```

依次跑完五组：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
```

或手动执行：

```powershell
python train.py --model baseline      --epochs 30 --width 32 --workers 0 --data data --out results
python train.py --model depthwise     --epochs 30 --width 32 --workers 0 --data data --out results
python train.py --model se            --epochs 30 --width 32 --workers 0 --data data --out results
python train.py --model depthwise_se  --epochs 30 --width 32 --workers 0 --data data --out results
python train.py --model gelu          --epochs 30 --width 32 --workers 0 --data data --out results
python analyze.py --results results --figures figures
python robustness.py --results results --data data --figures figures
python error_analysis.py --results results --figures figures --model gelu --compare baseline
```

在本机 RTX 4060 上，`width=32`、30 轮的 baseline 约 11.6 分钟，五组共约 68 分钟；CPU 或更小的显卡会更慢。

快速检查代码是否能跑：

```powershell
python train.py --model baseline --epochs 1 --width 32 --smoke --workers 0 --data data --out results_smoke
```

`results_smoke` 只用于确认数据、模型、CUDA 和日志链路正常，不作为实验结果。

## 输出

每个模型训练结束后生成：

```text
results/
├── baseline.json
├── depthwise.json
├── se.json
├── depthwise_se.json
├── gelu.json
├── checkpoints/
│   └── <model>.pt
└── logs/
    └── <model>.jsonl
```

单个 JSON 包含：

- `parameters`：可训练参数量
- `macs` / `gflops`：单张 32×32 输入的乘加次数（只统卷积和全连接，不含 BN、激活、池化）
- `best_val_accuracy` / `best_epoch`：验证集最佳结果
- `test.accuracy` / `test.correct` / `test.total`：测试集结果
- `test.predictions` / `test.labels`：逐样本预测，便于画混淆矩阵
- `history`：每轮训练/验证损失与准确率
- `cpu_latency_ms` / `gpu_latency_ms`：单张推理延迟（机器相关）
- `train_seconds`：训练耗时

汇总五组结果：

```powershell
python analyze.py --results results --figures figures
```

会生成：

- `results/summary.csv`（中文表头）
- `results/summary.json`
- `figures/accuracy_curves.png`（验证准确率曲线）
- `figures/model_comparison.png`（准确率与参数量对比）
- `figures/accuracy_vs_efficiency.png`（准确率 vs 参数量 / 计算量）

## 单张图片预测

```powershell
python predict.py path\to\image.jpg --model baseline --checkpoint results/checkpoints/baseline.pt
```

输出中文类别和概率，例如：

```text
1. 猫（cat）：0.9321
2. 狗（dog）：0.0412
3. 鹿（deer）：0.0087
```

## 目录结构

```text
.
├── train.py               # 训练、验证、测试入口
├── analyze.py             # 汇总结果并画中文图表
├── predict.py             # 单张图片分类
├── preview_data.py        # 生成 32 张图片预览
├── export_images.py       # 把 batch 解码成 jpg 照片
├── download_data.py       # CIFAR-10 镜像下载 + MD5 校验
├── src/
│   ├── models.py          # ResNet-18 及四种结构变体
│   ├── data.py            # 数据划分、DataLoader、中英文类别名
│   ├── engine.py          # 评估、延迟、日志工具
│   └── plotting.py        # 中文字体设置
├── scripts/
│   ├── run_all.ps1        # Windows 一次性跑完五组
│   └── run_all.sh         # Linux/macOS 版本
├── tests/
│   └── test_models.py     # 前向 shape / 参数量 / MACs 检查
├── requirements.txt
└── README.md
```

## 代码检查

```powershell
python -m py_compile train.py analyze.py predict.py preview_data.py export_images.py download_data.py src/models.py src/data.py src/engine.py src/plotting.py tests/test_models.py
python tests/test_models.py
```

## 说明

- 这是 **图像分类**，不是图像增强、去雾、超分或生成；不会修改原图。
- 测试集不参与调参；最佳模型由验证集准确率选择，最后只评估一次测试集。
- 为公平比较，所有变体使用同一组 `seed`、`epochs`、`batch`、学习率和数据增强。
- `--width 32` 是轻量版，适合快速跑通；正式对比建议用默认 `--width 64`。
- 代码不包含预训练权重或现成实验数字，所有结果都需要本地运行得到。
