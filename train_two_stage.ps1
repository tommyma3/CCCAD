# Two-Stage Training Script for CompressedAD
# Stage 1: Pre-train encoder with reconstruction
# Stage 2: Train full model with pretrained encoder

param(
    [switch]$PretrainOnly,
    [switch]$TrainOnly,
    [string]$PretrainedPath = "",
    [int]$PretrainSteps = 50000,
    [switch]$FreezeEncoder = $false
)

$ErrorActionPreference = "Stop"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "CompressedAD Two-Stage Training" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check if accelerate config exists
if (-not (Test-Path "accelerate_config.yaml")) {
    Write-Host "ERROR: accelerate_config.yaml not found!" -ForegroundColor Red
    Write-Host "Please create it first using: accelerate config" -ForegroundColor Yellow
    exit 1
}

# Stage 1: Pre-train Encoder
if (-not $TrainOnly) {
    Write-Host "[Stage 1] Pre-training Encoder" -ForegroundColor Green
    Write-Host "Duration: $PretrainSteps steps (~30 min on 1 GPU)" -ForegroundColor Gray
    Write-Host ""
    
    # Run pre-training
    Write-Host "Running: accelerate launch --config_file accelerate_config.yaml pretrain_encoder.py" -ForegroundColor Gray
    accelerate launch --config_file accelerate_config.yaml pretrain_encoder.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Pre-training failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "Pre-training complete!" -ForegroundColor Green
    Write-Host ""
    
    # Find the pretrained checkpoint
    $pretrainDir = Get-ChildItem -Path "./runs" -Filter "pretrain-encoder-*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if ($pretrainDir) {
        $checkpointPath = Join-Path $pretrainDir.FullName "encoder-pretrained-final.pt"
        
        if (Test-Path $checkpointPath) {
            Write-Host "Pretrained encoder saved to:" -ForegroundColor Cyan
            Write-Host "  $checkpointPath" -ForegroundColor White
            Write-Host ""
            
            # Set the path for stage 2
            $PretrainedPath = $checkpointPath
        }
    }
    
    if ($PretrainOnly) {
        Write-Host "Pre-training complete. Use this checkpoint path for main training:" -ForegroundColor Yellow
        Write-Host "  $PretrainedPath" -ForegroundColor White
        exit 0
    }
    
    Write-Host "Waiting 5 seconds before starting Stage 2..." -ForegroundColor Gray
    Start-Sleep -Seconds 5
    Write-Host ""
}

# Stage 2: Train Full Model
Write-Host "[Stage 2] Training Full Model" -ForegroundColor Green

if ($PretrainedPath -eq "") {
    Write-Host "WARNING: No pretrained encoder path specified!" -ForegroundColor Yellow
    Write-Host "Training will start from random initialization." -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne "y") {
        exit 0
    }
} else {
    if (-not (Test-Path $PretrainedPath)) {
        Write-Host "ERROR: Pretrained checkpoint not found: $PretrainedPath" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Using pretrained encoder from:" -ForegroundColor Cyan
    Write-Host "  $PretrainedPath" -ForegroundColor White
    Write-Host ""
    
    # Update config file temporarily
    $configPath = "config/model/ad_compressed_dr.yaml"
    $configContent = Get-Content $configPath -Raw
    
    # Check if pretrained_encoder_path already exists
    if ($configContent -match "pretrained_encoder_path:") {
        # Replace existing path
        $configContent = $configContent -replace "pretrained_encoder_path:.*", "pretrained_encoder_path: '$($PretrainedPath -replace '\\', '/')'"
    } else {
        # Add new path at the end
        $configContent += "`npretrained_encoder_path: '$($PretrainedPath -replace '\\', '/')'"
    }
    
    # Handle freeze_encoder setting
    if ($FreezeEncoder) {
        if ($configContent -match "freeze_encoder:") {
            $configContent = $configContent -replace "freeze_encoder:.*", "freeze_encoder: true"
        } else {
            $configContent += "`nfreeze_encoder: true"
        }
        Write-Host "Encoder will be FROZEN during training" -ForegroundColor Yellow
    } else {
        if ($configContent -match "freeze_encoder:") {
            $configContent = $configContent -replace "freeze_encoder:.*", "freeze_encoder: false"
        } else {
            $configContent += "`nfreeze_encoder: false"
        }
        Write-Host "Encoder will be FINE-TUNED during training" -ForegroundColor Cyan
    }
    
    # Save updated config
    Set-Content -Path $configPath -Value $configContent -NoNewline
    Write-Host "Updated $configPath with pretrained encoder path" -ForegroundColor Gray
    Write-Host ""
}

# Run main training
Write-Host "Running: accelerate launch --config_file accelerate_config.yaml train.py" -ForegroundColor Gray
accelerate launch --config_file accelerate_config.yaml train.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Main training failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Check training progress: tensorboard --logdir ./runs" -ForegroundColor White
Write-Host "2. Evaluate the model: python evaluate.py" -ForegroundColor White
Write-Host ""
