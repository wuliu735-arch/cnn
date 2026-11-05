$ErrorActionPreference = "Stop"
$models = @("baseline", "depthwise", "se", "depthwise_se", "gelu")
foreach ($model in $models) {
    Write-Host "=== 正在训练：$model ==="
    python train.py --model $model --epochs 30 --width 32 --batch 128 --workers 0 --data data --out results
    if ($LASTEXITCODE -ne 0) { throw "训练失败：$model" }
}
python analyze.py --results results --figures figures
python robustness.py --results results --data data --figures figures
python error_analysis.py --results results --figures figures --model gelu --compare baseline
