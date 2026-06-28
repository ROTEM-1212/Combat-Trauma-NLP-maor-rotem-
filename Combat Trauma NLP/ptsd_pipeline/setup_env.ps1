# setup_env.ps1
# ---------------------------------------------------------------------------
# Automated project environment setup for Windows (PowerShell 5.1+).
# Run from the project root:
#
#   .\setup_env.ps1          # auto-detect GPU
#   .\setup_env.ps1 -GPU     # force GPU (CUDA 12.8) install
#   .\setup_env.ps1 -CPU     # force CPU-only install
#   .\setup_env.ps1 -Force   # delete and recreate .venv before setup
# ---------------------------------------------------------------------------

param(
    [switch]$GPU,    # Force GPU (CUDA 12.1) PyTorch install
    [switch]$CPU,    # Force CPU-only PyTorch install
    [switch]$Force   # Recreate .venv even if it already exists
)

$ErrorActionPreference = "Stop"

$Root      = $PSScriptRoot
$VenvPath  = Join-Path $Root ".venv"
$Pip       = Join-Path $VenvPath "Scripts\pip.exe"
$Python    = Join-Path $VenvPath "Scripts\python.exe"

# --- Banner ----------------------------------------------------------------
Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Project Environment Setup" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# --- Sanity checks ---------------------------------------------------------
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python launcher 'py' not found. Install Python 3.10+ from python.org." -ForegroundColor Red
    exit 1
}

# --- 1. Create virtual environment -----------------------------------------
if (Test-Path $VenvPath) {
    if ($Force) {
        Write-Host "[1/5] -Force set - removing existing .venv ..." -ForegroundColor Yellow
        Remove-Item $VenvPath -Recurse -Force
    } else {
        Write-Host "[1/5] .venv already exists. Use -Force to recreate." -ForegroundColor Cyan
    }
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "[1/5] Creating virtual environment at .venv ..." -ForegroundColor Cyan
    py -m venv "$VenvPath"
    Write-Host "      Done." -ForegroundColor Green
} else {
    Write-Host "      Skipped (already present)." -ForegroundColor DarkGray
}

# --- 2. Upgrade pip --------------------------------------------------------
Write-Host "[2/5] Upgrading pip ..." -ForegroundColor Cyan
& $Python -m pip install --upgrade pip --quiet
Write-Host "      Done." -ForegroundColor Green

# --- 3. Detect GPU ---------------------------------------------------------
Write-Host "[3/5] Detecting hardware ..." -ForegroundColor Cyan

$HasGPU = $false

if ($GPU) {
    # User forced GPU mode via flag
    $HasGPU = $true
    Write-Host "      -GPU flag set: forcing CUDA 12.8 install." -ForegroundColor Green
} elseif ($CPU) {
    # User forced CPU mode via flag
    $HasGPU = $false
    Write-Host "      -CPU flag set: forcing CPU-only install." -ForegroundColor Cyan
} else {
    # Auto-detect via nvidia-smi
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $gpuOutput = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $gpuOutput) {
            $gpuName = ($gpuOutput | Select-Object -First 1).Trim()
            if ($gpuName) {
                $HasGPU = $true
                Write-Host "      NVIDIA GPU detected: $gpuName" -ForegroundColor Green
            }
        }
    }

    if (-not $HasGPU) {
        Write-Host "      No NVIDIA GPU detected via nvidia-smi." -ForegroundColor Yellow
        $answer = Read-Host "      Install CUDA 12.1 PyTorch anyway? (y/N)"
        if ($answer -match "^[Yy]") {
            $HasGPU = $true
            Write-Host "      Proceeding with CUDA 12.1 install." -ForegroundColor Green
        } else {
            Write-Host "      Proceeding with CPU-only install." -ForegroundColor Cyan
        }
    }
}

# --- 4. Install base requirements ------------------------------------------
Write-Host "[4/5] Installing base requirements (numpy, scikit-learn, transformers) ..." -ForegroundColor Cyan
& $Pip install -r (Join-Path $Root "requirements-base.txt")
Write-Host "      Done." -ForegroundColor Green

# --- 5. Install PyTorch ----------------------------------------------------
if ($HasGPU) {
    Write-Host "[5/5] Installing PyTorch with CUDA 12.8 (Blackwell/Ada/Ampere) ..." -ForegroundColor Green
    & $Pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "[5/5] Installing PyTorch (CPU-only) ..." -ForegroundColor Cyan
    & $Pip install torch torchvision torchaudio
}
Write-Host "      Done." -ForegroundColor Green

# --- Verification ----------------------------------------------------------
Write-Host ""
Write-Host "------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  Verifying installation" -ForegroundColor Cyan
Write-Host "------------------------------------------------------" -ForegroundColor DarkGray

& $Python -c "import torch, sklearn, numpy, transformers; print('  numpy        ', numpy.__version__); print('  scikit-learn ', sklearn.__version__); print('  transformers ', transformers.__version__); print('  torch        ', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"

# --- Done ------------------------------------------------------------------
Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Activate your environment:" -ForegroundColor White
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Then run the pipeline:" -ForegroundColor White
Write-Host "    python run_pipeline.py" -ForegroundColor Yellow
Write-Host ""
