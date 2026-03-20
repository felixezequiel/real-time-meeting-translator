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

# CUDA: disabled by default for portable MSI distribution.
# CUDA DLLs (cublas64_12.dll, cudart64_12.dll) would be required at runtime,
# making the installer fail on machines without CUDA Toolkit.
# Pass -CudaEnabled $true to build with CUDA if targeting machines that have it.
if ($null -eq $CudaEnabled) {
    $CudaEnabled = $false
}
if ($CudaEnabled) {
    Write-Host "  [!!] Building WITH CUDA - exe will require CUDA Toolkit on target machine" -ForegroundColor Yellow
} else {
    Write-Host "  [OK] Building without CUDA - portable (works on any Windows 11)" -ForegroundColor Green
}

# --- Step 2: Build Rust project ---
# Builds BOTH GPU (CUDA) and CPU binaries to target/dist/.
# Uses a separate target directory to avoid overwriting the dev build.

$DistTargetDir = Join-Path $ProjectRoot "target\dist"
$DistReleaseDir = Join-Path $DistTargetDir "release"
$GpuBinary = Join-Path $DistReleaseDir "meeting-translator-gpu.exe"
$CpuBinary = Join-Path $DistReleaseDir "meeting-translator-cpu.exe"

if (-not $SkipBuild) {
    Write-Host ""
    Write-Host "=== Building Release Binaries (for distribution) ===" -ForegroundColor Cyan
    Write-Host "  Output: target\dist\release\ (does not affect your dev build)" -ForegroundColor Gray

    Push-Location $ProjectRoot
    try {
        $env:CARGO_TARGET_DIR = $DistTargetDir

        # Build GPU binary (with CUDA)
        Write-Host ""
        Write-Host "  [1/2] Building GPU binary (CUDA)..." -ForegroundColor Gray
        cargo build --release 2>&1 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        if ($LASTEXITCODE -ne 0) {
            Remove-Item Env:\CARGO_TARGET_DIR
            Write-Host "  [FAIL] GPU build failed" -ForegroundColor Red
            exit 1
        }
        $builtExe = Join-Path $DistReleaseDir "meeting-translator.exe"
        Copy-Item $builtExe $GpuBinary -Force
        $gpuSize = [math]::Round((Get-Item $GpuBinary).Length / 1MB, 1)
        Write-Host "  [OK] GPU binary: meeting-translator-gpu.exe (${gpuSize}MB)" -ForegroundColor Green

        # Build CPU binary (no CUDA)
        Write-Host ""
        Write-Host "  [2/2] Building CPU binary (portable)..." -ForegroundColor Gray
        cargo build --release --no-default-features 2>&1 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        if ($LASTEXITCODE -ne 0) {
            Remove-Item Env:\CARGO_TARGET_DIR
            Write-Host "  [FAIL] CPU build failed" -ForegroundColor Red
            exit 1
        }
        Copy-Item $builtExe $CpuBinary -Force
        $cpuSize = [math]::Round((Get-Item $CpuBinary).Length / 1MB, 1)
        Write-Host "  [OK] CPU binary: meeting-translator-cpu.exe (${cpuSize}MB)" -ForegroundColor Green

        Remove-Item Env:\CARGO_TARGET_DIR
    } finally {
        Remove-Item Env:\CARGO_TARGET_DIR -ErrorAction SilentlyContinue
        Pop-Location
    }
} else {
    if (-not (Test-Path $GpuBinary) -or -not (Test-Path $CpuBinary)) {
        Write-Host "  [FAIL] Binaries not found in $DistReleaseDir. Run without -SkipBuild." -ForegroundColor Red
        exit 1
    }
    Write-Host "  [SKIP] Using existing binaries" -ForegroundColor Yellow
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

    $wixArgs = @("wix", "--no-build", "--nocapture", "-p", "meeting-translator", "--target-bin-dir", "$DistTargetDir\release") + $wixBinArg
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
