#!/usr/bin/env bash
set -euo pipefail

for model in baseline depthwise se depthwise_se gelu; do
  echo "=== 正在训练：${model} ==="
  python train.py --model "${model}" --epochs 30 --width 32 --batch 128 --workers 0 --data data --out results
done

python analyze.py --results results --figures figures
python robustness.py --results results --data data --figures figures
python error_analysis.py --results results --figures figures --model gelu --compare baseline
