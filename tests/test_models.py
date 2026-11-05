"""模型检查：前向输出、参数量、MACs；不训练。"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import VARIANTS, count_macs, count_parameters, make_model


def main() -> None:
    for name in VARIANTS:
        model = make_model(name, width=32)
        model.eval()
        with torch.inference_mode():
            logits = model(torch.randn(2, 3, 32, 32))
        assert logits.shape == (2, 10), (name, logits.shape)
        params = count_parameters(model)
        macs = count_macs(model)
        assert params > 0 and macs > 0, (name, params, macs)
        print(f"{name:14s} 参数量={params:>9,d} MACs={macs:>12,d}")
    print("全部模型检查通过")


if __name__ == "__main__":
    main()
