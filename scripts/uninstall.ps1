#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Uninstalls dependencies installed by the Meeting Translator installer.

.DESCRIPTION
    This script removes:
    - Python packages (transformers, torch, speechbrain, ctranslate2, etc.)
    - Downloaded ML models (models/ directory)
    - Whisper GGML models
    - Rust build artifacts (target/ directory)
    - Generated .cargo/config.toml
    - Optionally: VB-Cable, Hi-Fi Cable, CUDA, LLVM, CMake, Rust toolchain

.NOTES
    Run as Administrator for full cleanup.
    Usage: .\scripts\uninstall.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ScriptsDir = Join-Path $ProjectRoot "scripts"
$ModelsDir = Join-Path $ProjectRoot "models"
$TargetDir = Join-Path $ProjectRoot "target"
$CargoDir = Join-Path $ProjectRoot ".cargo"

# --- Helpers ---

function Write-Step {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "  [OK] $Message" -ForegroundColor Green
}

function Write-Skip {
    param([string]$Message)
    Write-Host "  [SKIP] $Message" -ForegroundColor Yellow
}

function Get-PythonExe {
    $candidates = @("python", "python3", "py")
    foreach ($cmd in $candidates) {
        $null = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($?) { return $cmd }
    }
    return $null
}

function Confirm-Action {
    param([string]$Message)
    $choice = Read-Host "$Message (y/N)"
    return ($choice -eq "y" -or $choice -eq "Y")
}

# --- Uninstall Steps ---

function Remove-PythonPackages {
    Write-Step "Python Packages"

    $pythonExe = Get-PythonExe
    if (-not $pythonExe) {
        Write-Skip "Python not found, nothing to uninstall"
        return
    }

    $packages = @(
        "faster-whisper",
        "transformers",
        "sentencepiece",
        "torch",
        "torchaudio",
        "torchvision",
        "numpy",
        "huggingface-hub",
        "tokenizers",
        "safetensors",
        "ctranslate2",
        "onnxruntime",
        # New stack (CosyVoice 2 + SpeechBrain)
        "speechbrain",
        "hyperpyyaml",
        "omegaconf",
        "modelscope",
        "conformer",
        "diffusers",
        "lightning",
        "einops",
        "inflect",
        "WeTextProcessing"
    )

    Write-Host "  Removing packages..." -ForegroundColor Gray
    foreach ($pkg in $packages) {
        $check = & $pythonExe -m pip show $pkg 2>&1
        if ($LASTEXITCODE -eq 0) {
            & $pythonExe -m pip uninstall $pkg -y 2>&1 | Out-Host
            Write-Ok "Removed $pkg"
        }
    }
}

function Remove-MLModels {
    Write-Step "Downloaded ML Models"

    if (Test-Path $ModelsDir) {
        $size = [math]::Round((Get-ChildItem $ModelsDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
        Write-Host "  Found models directory: ${size}MB" -ForegroundColor Gray
        Remove-Item $ModelsDir -Recurse -Force
        Write-Ok "Removed models/ directory (${size}MB freed)"
    } else {
        Write-Skip "No models/ directory found"
    }

    # Also clean HuggingFace cache for our models
    $hfCache = Join-Path $env:USERPROFILE ".cache\huggingface"
    if (Test-Path $hfCache) {
        $cacheSize = [math]::Round((Get-ChildItem $hfCache -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
        if (Confirm-Action "  Remove HuggingFace model cache (~${cacheSize}MB at $hfCache)?") {
            Remove-Item $hfCache -Recurse -Force
            Write-Ok "Removed HuggingFace cache (${cacheSize}MB freed)"
        } else {
            Write-Skip "HuggingFace cache kept"
        }
    }

    # SpeechBrain caches ECAPA-TDNN under a separate path that survives
    # HuggingFace cache cleanup.
    $sbCache = Join-Path $env:USERPROFILE ".cache\speechbrain"
    if (Test-Path $sbCache) {
        $sbSize = [math]::Round((Get-ChildItem $sbCache -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
        Remove-Item $sbCache -Recurse -Force
        Write-Ok "Removed SpeechBrain cache (${sbSize}MB freed)"
    }

    # CosyVoice repo lives under third_party/ — added by Install-CosyVoice.
    $thirdParty = Join-Path $ProjectRoot "third_party"
    if (Test-Path $thirdParty) {
        $tpSize = [math]::Round((Get-ChildItem $thirdParty -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
        Remove-Item $thirdParty -Recurse -Force
        Write-Ok "Removed third_party/ (${tpSize}MB freed)"
    }
}

function Remove-BuildArtifacts {
    Write-Step "Rust Build Artifacts"

    if (Test-Path $TargetDir) {
        $size = [math]::Round((Get-ChildItem $TargetDir -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
        Write-Host "  Found target directory: ${size}MB" -ForegroundColor Gray
        Remove-Item $TargetDir -Recurse -Force
        Write-Ok "Removed target/ directory (${size}MB freed)"
    } else {
        Write-Skip "No target/ directory found"
    }
}

function Remove-CargoConfig {
    Write-Step "Cargo Build Configuration"

    $configPath = Join-Path $CargoDir "config.toml"
    if (Test-Path $configPath) {
        Remove-Item $configPath -Force
        Write-Ok "Removed .cargo/config.toml"

        # Remove .cargo dir if empty
        $remaining = Get-ChildItem $CargoDir -ErrorAction SilentlyContinue
        if (-not $remaining) {
            Remove-Item $CargoDir -Force
            Write-Ok "Removed empty .cargo/ directory"
        }
    } else {
        Write-Skip "No .cargo/config.toml found"
    }
}

function Remove-VBCable {
    Write-Step "VB-Cable Virtual Audio Driver"

    $vbCableDevice = Get-PnpDevice -FriendlyName "*VB-Audio*" -ErrorAction SilentlyContinue
    if (-not $vbCableDevice) {
        Write-Skip "VB-Cable not installed"
        return
    }

    if (Confirm-Action "  Remove VB-Cable virtual audio driver?") {
        Write-Host "  VB-Cable must be removed via Device Manager or Control Panel." -ForegroundColor Yellow
        Write-Host "  Opening Device Manager..." -ForegroundColor Gray
        Start-Process "devmgmt.msc"
        Write-Host "  Look for 'VB-Audio Virtual Cable' under Sound devices and uninstall it." -ForegroundColor Yellow
    } else {
        Write-Skip "VB-Cable kept"
    }
}

function Remove-CUDA {
    Write-Step "CUDA Toolkit"

    $cudaPaths = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue
    if (-not $cudaPaths) {
        Write-Skip "CUDA not installed"
        return
    }

    $latest = $cudaPaths | Sort-Object Name -Descending | Select-Object -First 1
    if (Confirm-Action "  Remove CUDA Toolkit ($($latest.Name))? (This affects ALL CUDA apps)") {
        try {
            winget uninstall --id Nvidia.CUDA 2>&1 | Out-Host
            Write-Ok "CUDA uninstall initiated"
        } catch {
            Write-Host "  Use Windows Settings > Apps to uninstall CUDA manually" -ForegroundColor Yellow
        }
    } else {
        Write-Skip "CUDA kept"
    }
}

function Remove-LLVM {
    Write-Step "LLVM/Clang"

    if (-not (Test-Path "C:\Program Files\LLVM")) {
        Write-Skip "LLVM not installed"
        return
    }

    if (Confirm-Action "  Remove LLVM/Clang? (This affects ALL projects using bindgen/clang)") {
        try {
            winget uninstall --id LLVM.LLVM 2>&1 | Out-Host
            Write-Ok "LLVM uninstall initiated"
        } catch {
            Write-Host "  Use Windows Settings > Apps to uninstall LLVM manually" -ForegroundColor Yellow
        }
    } else {
        Write-Skip "LLVM kept"
    }
}

function Remove-CMake {
    Write-Step "CMake"

    $hasCMake = Get-Command "cmake" -ErrorAction SilentlyContinue
    if (-not $hasCMake) {
        Write-Skip "CMake not installed"
        return
    }

    if (Confirm-Action "  Remove CMake? (This affects ALL projects using CMake)") {
        try {
            winget uninstall --id Kitware.CMake 2>&1 | Out-Host
            Write-Ok "CMake uninstall initiated"
        } catch {
            Write-Host "  Use Windows Settings > Apps to uninstall CMake manually" -ForegroundColor Yellow
        }
    } else {
        Write-Skip "CMake kept"
    }
}

function Remove-RustToolchain {
    Write-Step "Rust Toolchain"

    if (-not (Get-Command "rustup" -ErrorAction SilentlyContinue)) {
        Write-Skip "Rust not installed"
        return
    }

    if (Confirm-Action "  Remove Rust toolchain? (This affects ALL Rust projects on this machine)") {
        & rustup self uninstall -y 2>&1 | Out-Host
        Write-Ok "Rust toolchain removed"
    } else {
        Write-Skip "Rust toolchain kept"
    }
}

# --- Main ---

Write-Host ""
Write-Host "==============================================" -ForegroundColor Red
Write-Host "  Real-Time Meeting Translator - Uninstaller" -ForegroundColor Red
Write-Host "==============================================" -ForegroundColor Red
Write-Host ""
Write-Host "  Project: $ProjectRoot" -ForegroundColor Gray
Write-Host ""
Write-Host "  This will remove dependencies installed by install.ps1." -ForegroundColor Yellow
Write-Host ""

if (-not (Confirm-Action "Continue with uninstall?")) {
    Write-Host "`n  Cancelled." -ForegroundColor Gray
    exit 0
}

# Always safe to remove (project-local)
Remove-PythonPackages
Remove-MLModels
Remove-BuildArtifacts
Remove-CargoConfig

# Ask before removing system-level components
Write-Host ""
if (Confirm-Action "Also remove system-level dependencies (VB-Cable, CUDA, LLVM, CMake, Rust)?") {
    Remove-VBCable
    Remove-CUDA
    Remove-LLVM
    Remove-CMake
    Remove-RustToolchain
} else {
    Write-Skip "System-level dependencies kept"
}

Write-Host "`n" -NoNewline
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Uninstall Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  To reinstall: powershell -ExecutionPolicy Bypass -File scripts\install.ps1" -ForegroundColor Gray
Write-Host ""
