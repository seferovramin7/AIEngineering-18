# run_three_depths.ps1
$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot
mkdir "$ROOT\runs" -Force | Out-Null

$depths = @(10,20,40)
foreach ($d in $depths) {
  $log  = Join-Path $ROOT ("runs\train_last{0}.log" -f $d)
  $json = Join-Path $ROOT ("runs\final_metrics_last{0}.json" -f $d)
  Remove-Item $log,$json -ErrorAction SilentlyContinue

  Write-Host "=== Training with train-last-n=$d ==="
  python "$PSScriptRoot\..\..\..\app\main.py" demo-train --epochs 2 --image-size 160 --train-last-n $d `
    | Tee-Object -FilePath $log

  # Tunggu JSON muncul (maks 2 menit)
  for ($i=0; $i -lt 120 -and -not (Test-Path $json); $i++) { Start-Sleep -Seconds 1 }

  if (Test-Path $json) {
    Write-Host ">>> Metrics ($json):"
    Get-Content $json | ConvertFrom-Json | Select train_accuracy, val_accuracy, train_last_n
  } else {
    Write-Warning "JSON metrics belum ada untuk train-last-n=$d. Cek $log"
  }
}
