# ==============================================================================
# Script: run_all_pssa_experiments.ps1
# Purpose: Native PowerShell execution for SNN-DT-H Ablations E0-E5
# ==============================================================================

param (
    [string]$Device = "cpu",
    [int]$Updates = 25000,
    [string]$Dataset = "hopper-medium-v2"
)

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Starting SNN-DT-H Ablation Suite: Beating Decision SpikeFormer PSSA" -ForegroundColor Cyan
Write-Host "Device: $Device | Updates: $Updates | Dataset: $Dataset" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

$ablations = @("e0", "e1", "e2", "e3", "e4", "e5")

foreach ($ab in $ablations) {
    Write-Host "`n>>> Running Ablation $ab ($Updates updates)..." -ForegroundColor Yellow
    python offline_rl\experiments\run_pssa_ablations.py --ablation $ab --updates $Updates --device $Device --d4rl-name $Dataset
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Ablation $ab failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "`n==================================================================" -ForegroundColor Green
Write-Host "All Ablation Experiments Completed Successfully!" -ForegroundColor Green
Write-Host "Inspect reports in results/pssa_ablations/" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
