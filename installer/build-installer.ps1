<#
.SYNOPSIS
    Builds the Meeting Translator MSI installer.

.DESCRIPTION
    This script:
    1. Installs cargo-wix (if not present)
    2. Builds the Rust project in release mode
    3. Creates the MSI installer using WiX Toolset

.PARAMETER SkipBuild
    Skip the Rust compilation step (use existing binary in target/release/).

.PARAMETER CudaEnabled
    Build with CUDA support for GPU-accelerated speech recognition.
    Defaults to auto-detect (enabled if nvcc is found).

.NOTES
    Prerequisites:
    - Rust toolchain (rustup + cargo)
    - Visual Studio Build Tools (MSVC)
    - .NET Framework 3.5+ (for WiX Toolset, auto-installed by cargo-wix)

    Usage:
        .\installer\build-installer.ps1
        .\installer\build-installer.ps1 -SkipBuild
#>

param(
    [switch]$SkipBuild,
    [Nullable[bool]]$CudaEnabled = $null
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Meeting Translator - MSI Builder" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Project: $ProjectRoot" -ForegroundColor Gray
Write-Host ""

# --- Step 1: Check prerequisites ---

Write-Host "=== Checking Prerequisites ===" -ForegroundColor Cyan

# Rust
if (-not (Get-Command "cargo" -ErrorAction SilentlyContinue)) {
    Write-Host "  [FAIL] Rust/Cargo not found. Install from https://rustup.rs" -ForegroundColor Red
    exit 1
}
$rustVersion = (rustc --version) -replace "rustc ", ""
Write-Host "  [OK] Rust: $rustVersion" -ForegroundColor Green

# cargo-wix
$hasCargoWix = cargo wix --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing cargo-wix..." -ForegroundColor Gray
    cargo install cargo-wix
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [FAIL] Failed to install cargo-wix" -ForegroundColor Red
        exit 1
    }
}
Write-Host "  [OK] cargo-wix installed" -ForegroundColor Green

# CUDA detection
if ($null -eq $CudaEnabled) {
    $CudaEnabled = (Get-Command "nvcc" -ErrorAction SilentlyContinue) -or
        (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
}
if ($CudaEnabled) {
    Write-Host "  [OK] CUDA detected - building with GPU support" -ForegroundColor Green
} else {
    Write-Host "  [--] No CUDA - building CPU-only" -ForegroundColor Yellow
}

# --- Step 2: Build Rust project ---

if (-not $SkipBuild) {
    Write-Host ""
    Write-Host "=== Building Release Binary ===" -ForegroundColor Cyan

    Push-Location $ProjectRoot
    try {
        if ($CudaEnabled) {
            Write-Host "  cargo build --release (with CUDA)..." -ForegroundColor Gray
            cargo build --release 2>&1 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        } else {
            Write-Host "  cargo build --release --no-default-features..." -ForegroundColor Gray
            cargo build --release --no-default-features 2>&1 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        }

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [FAIL] Build failed" -ForegroundColor Red
            exit 1
        }

        $binaryPath = Join-Path $ProjectRoot "target\release\meeting-translator.exe"
        if (Test-Path $binaryPath) {
            $size = [math]::Round((Get-Item $binaryPath).Length / 1MB, 1)
            Write-Host "  [OK] Built: meeting-translator.exe (${size}MB)" -ForegroundColor Green
        }
    } finally {
        Pop-Location
    }
} else {
    $binaryPath = Join-Path $ProjectRoot "target\release\meeting-translator.exe"
    if (-not (Test-Path $binaryPath)) {
        Write-Host "  [FAIL] Binary not found at $binaryPath. Run without -SkipBuild." -ForegroundColor Red
        exit 1
    }
    Write-Host "  [SKIP] Using existing binary" -ForegroundColor Yellow
}

# --- Step 3: Create MSI ---

Write-Host ""
Write-Host "=== Creating MSI Installer ===" -ForegroundColor Cyan

Push-Location $ProjectRoot
try {
    Write-Host "  Running cargo wix..." -ForegroundColor Gray

    # cargo-wix expects wix/main.wxs by default
    # WixUtilExtension is auto-loaded by cargo-wix (provides util:PermissionEx, WixShellExec)
    $wixBinPath = Join-Path $ProjectRoot "tools\wix\bin"
    $wixBinArg = @()
    if (Test-Path $wixBinPath) {
        $wixBinArg = @("-b", $wixBinPath)
        Write-Host "  Using local WiX binaries: $wixBinPath" -ForegroundColor Gray
    }

    $wixArgs = @("wix", "--no-build", "--nocapture", "-p", "meeting-translator") + $wixBinArg
    & cargo @wixArgs 2>&1 | ForEach-Object {
        Write-Host "  $_" -ForegroundColor Gray
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "  [FAIL] MSI creation failed." -ForegroundColor Red
        Write-Host ""
        Write-Host "  If WiX Toolset is not found, download the binaries:" -ForegroundColor Yellow
        Write-Host "    1. Download wix314-binaries.zip from https://github.com/wixtoolset/wix3/releases" -ForegroundColor Yellow
        Write-Host "    2. Extract to tools\wix\bin\ in the project root" -ForegroundColor Yellow
        Write-Host "    3. Re-run this script" -ForegroundColor Yellow
        exit 1
    }

    # Find the generated MSI
    $msiFile = Get-ChildItem "$ProjectRoot\target\wix\*.msi" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($msiFile) {
        $msiSize = [math]::Round($msiFile.Length / 1MB, 1)
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "  MSI Created Successfully!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "  File: $($msiFile.FullName)" -ForegroundColor White
        Write-Host "  Size: ${msiSize}MB" -ForegroundColor Gray
        Write-Host ""
        Write-Host "  To install:" -ForegroundColor White
        Write-Host "    Double-click the MSI file, or run:" -ForegroundColor Gray
        Write-Host "    msiexec /i `"$($msiFile.FullName)`"" -ForegroundColor Gray
        Write-Host ""
        Write-Host "  The installer will:" -ForegroundColor White
        Write-Host "    1. Install app files to C:\Program Files\Meeting Translator\" -ForegroundColor Gray
        Write-Host "    2. Create Start Menu and Desktop shortcuts" -ForegroundColor Gray
        Write-Host "    3. Offer to run dependency setup (Python, audio drivers, ML models)" -ForegroundColor Gray
        Write-Host ""
    } else {
        Write-Host "  [FAIL] MSI file not found in target\wix\" -ForegroundColor Red
    }
} finally {
    Pop-Location
}
