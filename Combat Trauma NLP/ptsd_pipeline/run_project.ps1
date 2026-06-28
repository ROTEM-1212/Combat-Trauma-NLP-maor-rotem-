# run_project.ps1
# ---------------------------------------------------------------------------
# Seamless project runner for Windows (PowerShell 5.1+).
# Activates .venv, runs the pipeline, then deactivates automatically.
#
# Usage (from the project root):
#   .\run_project.ps1
#
# No manual venv activation needed - this script handles everything.
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

$Root     = $PSScriptRoot
$VenvPath = Join-Path $Root ".venv"
$Activate = Join-Path $VenvPath "Scripts\Activate.ps1"
$Python   = Join-Path $VenvPath "Scripts\python.exe"

# --- Banner ----------------------------------------------------------------
Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Project Runner" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# --- 1. Guard: .venv must exist --------------------------------------------
if (-not (Test-Path $VenvPath)) {
    Write-Host "[ERROR] No .venv found. Run setup first:" -ForegroundColor Red
    Write-Host "        .\setup_env.ps1" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

if (-not (Test-Path $Python)) {
    Write-Host "[ERROR] .venv exists but Python executable is missing." -ForegroundColor Red
    Write-Host "        Try recreating it: .\setup_env.ps1 -Force" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# --- 2. Detect main entry point --------------------------------------------
# Priority list - first match wins
$EntryPoint = $null
foreach ($candidate in @("run_pipeline.py", "main.py", "app.py")) {
    $path = Join-Path $Root $candidate
    if (Test-Path $path) {
        $EntryPoint = $path
        break
    }
}

# Fallback: first .py file found in the project root
if (-not $EntryPoint) {
    $EntryPoint = Get-ChildItem -Path $Root -Filter "*.py" -File |
                  Select-Object -First 1 -ExpandProperty FullName
}

if (-not $EntryPoint) {
    Write-Host "[ERROR] No Python entry point found in the project root." -ForegroundColor Red
    exit 1
}

$EntryName = Split-Path $EntryPoint -Leaf

Write-Host "  Entry point  : $EntryName" -ForegroundColor White
Write-Host "  Environment  : .venv" -ForegroundColor White
Write-Host ""

# --- 3. Activate virtual environment ---------------------------------------
Write-Host "[1/3] Activating virtual environment ..." -ForegroundColor Cyan
& $Activate
Write-Host "      Done." -ForegroundColor Green
Write-Host ""

# --- 4. Run the pipeline (try/finally ensures deactivate always runs) ------
$ExitCode = 0

Write-Host "[2/3] Running $EntryName ..." -ForegroundColor Green
Write-Host "------------------------------------------------------" -ForegroundColor DarkGray
Write-Host ""

try {
    & $Python $EntryPoint
    $ExitCode = $LASTEXITCODE
}
catch {
    Write-Host ""
    Write-Host "[ERROR] Unhandled exception while running the pipeline:" -ForegroundColor Red
    Write-Host "        $($_.Exception.Message)" -ForegroundColor Red
    $ExitCode = 1
}
finally {
    # --- 5. Deactivate virtual environment - always runs -------------------
    Write-Host ""
    Write-Host "------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host "[3/3] Deactivating virtual environment ..." -ForegroundColor Cyan
    if (Get-Command deactivate -ErrorAction SilentlyContinue) {
        deactivate
    }
    Write-Host "      Done." -ForegroundColor Green
}

# ─── 6. Final status ───────────────────────────────────────────────────────
Write-Host ""
if ($ExitCode -eq 0) {
    Write-Host "======================================================" -ForegroundColor Green
    Write-Host "  Pipeline finished successfully." -ForegroundColor Green
    Write-Host "======================================================" -ForegroundColor Green
} else {
    Write-Host "======================================================" -ForegroundColor Red
    Write-Host "  Pipeline exited with code $ExitCode." -ForegroundColor Red
    Write-Host "======================================================" -ForegroundColor Red
}
Write-Host ""

exit $ExitCode
