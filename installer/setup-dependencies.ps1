#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Post-install dependency setup for Meeting Translator.
    Runs after MSI installation to configure the runtime environment.

.DESCRIPTION
    This script installs and configures:
    - Python 3.12+ (via winget)
    - VB-Cable virtual audio driver (meeting audio loopback)
    - Hi-Fi Cable virtual audio driver (mic TTS output)
    - Python packages (faster-whisper, transformers, piper-tts, torch, etc.)
    - Whisper GGML model download (~466MB)

.PARAMETER InstallDir
    The directory where Meeting Translator is installed.
    Defaults to the script's own directory.

.NOTES
    This script is designed to run from the Meeting Translator install directory.
    It can be re-run safely to update or repair dependencies.
#>

param(
    [string]$InstallDir = (Split-Path -Parent $MyInvocation.MyCommand.Path)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptsDir = Join-Path $InstallDir "scripts"
$ModelsDir = Join-Path $InstallDir "models"

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

function Write-Fail {
    param([string]$Message)
    Write-Host "  [FAIL] $Message" -ForegroundColor Red
}

function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

function Get-PythonExe {
    $candidates = @("python", "python3", "py")
    foreach ($cmd in $candidates) {
        if (Test-Command $cmd) {
            $version = & $cmd --version 2>&1
            if ($version -match "Python 3\.(\d+)") {
                $minor = [int]$Matches[1]
                if ($minor -ge 10) {
                    return $cmd
                }
            }
        }
    }
    return $null
}

function Refresh-Path {
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"

    $extraPaths = @(
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links",
        "$env:LOCALAPPDATA\Programs\Python\Python312",
        "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts",
        "$env:LOCALAPPDATA\Programs\Python\Python311",
        "$env:LOCALAPPDATA\Programs\Python\Python311\Scripts",
        "C:\Python312",
        "C:\Python311"
    )
    foreach ($p in $extraPaths) {
        if ((Test-Path $p) -and ($env:Path -notlike "*$p*")) {
            $env:Path = "$p;$env:Path"
        }
    }
}

# --- Installation Steps ---

function Install-VCRedist {
    Write-Step "Visual C++ Redistributable (required runtime)"

    # Check if VC++ Redist is already installed
    $vcInstalled = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue
    if ($vcInstalled) {
        Write-Ok "Visual C++ Redistributable already installed (v$($vcInstalled.Version))"
        return
    }

    Write-Host "  Installing Visual C++ Redistributable via winget..." -ForegroundColor Gray
    try {
        winget install --id Microsoft.VCRedist.2015+.x64 --accept-package-agreements --accept-source-agreements --silent 2>&1 | Out-Host
    } catch {}

    Refresh-Path

    $vcInstalled = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue
    if ($vcInstalled) {
        Write-Ok "Visual C++ Redistributable installed"
    } else {
        Write-Fail "VC++ Redistributable installation could not be verified"
        Write-Host "  Download manually from: https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Yellow
    }
}

function Install-Python {
    Write-Step "Python 3.10+ (required for ML models)"

    Refresh-Path

    $pythonExe = Get-PythonExe
    if ($pythonExe) {
        $version = & $pythonExe --version 2>&1
        Write-Ok "Python already installed: $version"
        return $pythonExe
    }

    Write-Host "  Installing Python 3.12 via winget..." -ForegroundColor Gray
    Write-Host "  (This may take a few minutes)" -ForegroundColor Gray
    try {
        winget install --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements --silent 2>&1 | Out-Host
    } catch {
        Write-Host "  winget reported an error (this is often non-fatal)" -ForegroundColor Yellow
    }

    Refresh-Path

    $pythonExe = Get-PythonExe
    if ($pythonExe) {
        $version = & $pythonExe --version 2>&1
        Write-Ok "Python installed: $version"
        return $pythonExe
    } else {
        Write-Fail "Python installation failed."
        Write-Host "  Please install Python 3.10+ manually from https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "  Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
        return $null
    }
}

function Install-VBCable {
    Write-Step "VB-Cable Virtual Audio Driver (meeting audio loopback)"

    $vbCableDevice = Get-PnpDevice -FriendlyName "*VB-Audio*" -ErrorAction SilentlyContinue
    if ($vbCableDevice) {
        Write-Ok "VB-Cable already installed"
        return
    }

    $vbCableZip = Join-Path $env:TEMP "VBCable.zip"
    $vbCableDir = Join-Path $env:TEMP "VBCable"

    Write-Host "  Downloading VB-Cable..." -ForegroundColor Gray
    $downloadUrl = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip"

    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $vbCableZip -UseBasicParsing

        if (Test-Path $vbCableDir) { Remove-Item $vbCableDir -Recurse -Force }
        Expand-Archive -Path $vbCableZip -DestinationPath $vbCableDir -Force

        $setupExe = Get-ChildItem -Path $vbCableDir -Filter "VBCABLE_Setup_x64.exe" -Recurse | Select-Object -First 1

        if ($setupExe) {
            Write-Host "  Installing VB-Cable driver (a dialog may appear)..." -ForegroundColor Gray
            Start-Process -FilePath $setupExe.FullName -ArgumentList "/i" -Wait -Verb RunAs 2>$null

            Start-Sleep -Seconds 3
            $vbCableDevice = Get-PnpDevice -FriendlyName "*VB-Audio*" -ErrorAction SilentlyContinue
            if ($vbCableDevice) {
                Write-Ok "VB-Cable installed successfully"
            } else {
                Write-Skip "VB-Cable installer ran but device not detected. You may need to restart."
            }
        } else {
            Write-Fail "VB-Cable installer not found in downloaded archive"
            Write-Host "  Please install manually from: https://vb-audio.com/Cable/" -ForegroundColor Yellow
        }
    } catch {
        Write-Fail "Failed to download VB-Cable: $_"
        Write-Host "  Please install manually from: https://vb-audio.com/Cable/" -ForegroundColor Yellow
    } finally {
        Remove-Item $vbCableZip -Force -ErrorAction SilentlyContinue
        Remove-Item $vbCableDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Install-HiFiCable {
    Write-Step "Hi-Fi Cable Virtual Audio Driver (translated mic output)"

    $hifiDevice = Get-PnpDevice -FriendlyName "*Hi-Fi*" -ErrorAction SilentlyContinue
    if ($hifiDevice) {
        Write-Ok "Hi-Fi Cable already installed"
        return
    }

    $hifiZip = Join-Path $env:TEMP "HiFiCable.zip"
    $hifiDir = Join-Path $env:TEMP "HiFiCable"

    Write-Host "  Downloading Hi-Fi Cable..." -ForegroundColor Gray
    $downloadUrl = "https://download.vb-audio.com/Download_CABLE/HiFiCableAsioBridgeSetup_v1007.zip"

    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $hifiZip -UseBasicParsing

        if (Test-Path $hifiDir) { Remove-Item $hifiDir -Recurse -Force }
        Expand-Archive -Path $hifiZip -DestinationPath $hifiDir -Force

        $setupExe = Get-ChildItem -Path $hifiDir -Filter "*.exe" -Recurse |
            Where-Object { $_.Name -like "*Setup*" -or $_.Name -like "*HIFI*" } |
            Select-Object -First 1

        if ($setupExe) {
            Write-Host "  Installing Hi-Fi Cable driver (a dialog may appear)..." -ForegroundColor Gray
            Start-Process -FilePath $setupExe.FullName -ArgumentList "/i" -Wait -Verb RunAs 2>$null

            Start-Sleep -Seconds 3
            $hifiDevice = Get-PnpDevice -FriendlyName "*Hi-Fi*" -ErrorAction SilentlyContinue
            if ($hifiDevice) {
                Write-Ok "Hi-Fi Cable installed successfully"
            } else {
                Write-Skip "Hi-Fi Cable installer ran but device not detected. You may need to restart."
            }
        } else {
            Write-Fail "Hi-Fi Cable installer not found in downloaded archive"
            Write-Host "  Please install manually from: https://vb-audio.com/Cable/" -ForegroundColor Yellow
        }
    } catch {
        Write-Fail "Failed to download Hi-Fi Cable: $_"
        Write-Host "  Please install manually from: https://vb-audio.com/Cable/" -ForegroundColor Yellow
    } finally {
        Remove-Item $hifiZip -Force -ErrorAction SilentlyContinue
        Remove-Item $hifiDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Install-CUDA {
    Write-Step "CUDA Toolkit (GPU Acceleration)"

    if (Test-Command "nvcc") {
        $version = (nvcc --version | Select-String "release") -replace ".*release ", "" -replace ",.*", ""
        Write-Ok "CUDA already installed: $version"
        return
    }

    $cudaPaths = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue
    if ($cudaPaths) {
        $latest = $cudaPaths | Sort-Object Name -Descending | Select-Object -First 1
        Write-Ok "CUDA found at: $($latest.FullName)"
        return
    }

    $nvidiaGpu = Get-CimInstance -ClassName Win32_VideoController -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*NVIDIA*" }

    if (-not $nvidiaGpu) {
        Write-Skip "No NVIDIA GPU detected -- CUDA not needed (will use CPU for STT)"
        return
    }

    Write-Host "  NVIDIA GPU detected: $($nvidiaGpu.Name)" -ForegroundColor Gray
    Write-Host "  Installing CUDA Toolkit via winget..." -ForegroundColor Gray
    Write-Host "  (This may take 10-15 minutes)" -ForegroundColor Gray

    try {
        winget install --id Nvidia.CUDA --accept-package-agreements --accept-source-agreements --silent 2>&1 | Out-Host
    } catch {}

    Refresh-Path

    if (Test-Command "nvcc") {
        $version = (nvcc --version | Select-String "release") -replace ".*release ", "" -replace ",.*", ""
        Write-Ok "CUDA installed: $version"
    } else {
        Write-Skip "CUDA installation could not be verified. Install manually from https://developer.nvidia.com/cuda-downloads"
        Write-Host "  The app will still work using CPU (slower STT)." -ForegroundColor Yellow
    }
}

function Install-PythonDeps {
    param([string]$PythonExe)

    Write-Step "Python ML Packages (STT, Translation, TTS)"

    $requirementsFile = Join-Path $ScriptsDir "requirements.txt"
    if (-not (Test-Path $requirementsFile)) {
        Write-Fail "requirements.txt not found at $requirementsFile"
        return
    }

    Write-Host "  Upgrading pip..." -ForegroundColor Gray
    try { & $PythonExe -m pip install --upgrade pip 2>&1 | Out-Null } catch {}

    Write-Host "  Installing packages (this may take 5-10 minutes on first run)..." -ForegroundColor Gray
    Write-Host "  Packages: faster-whisper, transformers, piper-tts, torch, etc." -ForegroundColor Gray

    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonExe -m pip install -r $requirementsFile 2>&1 | ForEach-Object {
        if ($_ -match "^(Collecting|Downloading|Installing|Successfully)") {
            Write-Host "  $_" -ForegroundColor Gray
        }
    }
    $pipExitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevPref

    if ($pipExitCode -eq 0) {
        $packages = @("faster_whisper", "transformers", "torch")
        foreach ($pkg in $packages) {
            $check = & $PythonExe -c "import $pkg; print($pkg.__version__)" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "$pkg $check"
            } else {
                Write-Fail "$pkg not importable"
            }
        }
    } else {
        Write-Fail "pip install failed (exit code: $pipExitCode)"
        Write-Host "  Try running manually: $PythonExe -m pip install -r `"$requirementsFile`"" -ForegroundColor Yellow
    }
}

function Install-WhisperModel {
    Write-Step "Whisper Speech Recognition Model"

    if (-not (Test-Path $ModelsDir)) {
        New-Item -Path $ModelsDir -ItemType Directory -Force | Out-Null
    }

    $modelName = "small"
    $modelFile = Join-Path $ModelsDir "ggml-$modelName.bin"

    if (Test-Path $modelFile) {
        $size = [math]::Round((Get-Item $modelFile).Length / 1MB, 1)
        Write-Ok "Model already downloaded: ggml-$modelName.bin (${size}MB)"
        return
    }

    $modelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$modelName.bin"

    Write-Host "  Downloading ggml-$modelName.bin (~466MB)..." -ForegroundColor Gray
    Write-Host "  This is a one-time download." -ForegroundColor Gray

    try {
        # Use BITS for better download reliability and progress
        $bitsSupported = $true
        try {
            Import-Module BitsTransfer -ErrorAction Stop
        } catch {
            $bitsSupported = $false
        }

        if ($bitsSupported) {
            Start-BitsTransfer -Source $modelUrl -Destination $modelFile -Description "Downloading Whisper model"
        } else {
            Invoke-WebRequest -Uri $modelUrl -OutFile $modelFile -UseBasicParsing
        }

        if (Test-Path $modelFile) {
            $size = [math]::Round((Get-Item $modelFile).Length / 1MB, 1)
            Write-Ok "Model downloaded: ggml-$modelName.bin (${size}MB)"
        } else {
            Write-Fail "Download completed but file not found"
        }
    } catch {
        Write-Fail "Failed to download model: $_"
        Write-Host "  Download manually from: $modelUrl" -ForegroundColor Yellow
        Write-Host "  Place the file at: $modelFile" -ForegroundColor Yellow
    }
}

# --- Main ---

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Meeting Translator - Dependency Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Install directory: $InstallDir" -ForegroundColor Gray
Write-Host ""
Write-Host "  This will install:" -ForegroundColor White
Write-Host "    - Visual C++ Redistributable (runtime)" -ForegroundColor Gray
Write-Host "    - Python 3.12 (if not present)" -ForegroundColor Gray
Write-Host "    - CUDA Toolkit (if NVIDIA GPU detected)" -ForegroundColor Gray
Write-Host "    - VB-Cable virtual audio driver" -ForegroundColor Gray
Write-Host "    - Hi-Fi Cable virtual audio driver" -ForegroundColor Gray
Write-Host "    - Python ML packages (STT, Translation, TTS)" -ForegroundColor Gray
Write-Host "    - Whisper speech recognition model (~466MB)" -ForegroundColor Gray
Write-Host ""

# 1. VC++ Redistributable (required for the exe)
Install-VCRedist

# 2. Python runtime
$pythonExe = Install-Python

# 3. GPU acceleration
Install-CUDA

# 4. Virtual audio drivers
Install-VBCable
Install-HiFiCable

# 5. Python ML packages (only if Python was installed)
if ($pythonExe) {
    Install-PythonDeps -PythonExe $pythonExe
} else {
    Write-Step "Python Packages"
    Write-Skip "Skipped (Python not available). Install Python and re-run this setup."
}

# 6. Whisper model
Install-WhisperModel

# 7. Select GPU or CPU binary based on CUDA availability
Write-Step "Selecting optimal binary"

$gpuExe = Join-Path $InstallDir "meeting-translator-gpu.exe"
$cpuExe = Join-Path $InstallDir "meeting-translator-cpu.exe"
$mainExe = Join-Path $InstallDir "meeting-translator.exe"

Refresh-Path
$hasCudaToolkit = (Test-Command "nvcc") -or
    (Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue)

if ($hasCudaToolkit -and (Test-Path $gpuExe)) {
    Copy-Item $gpuExe $mainExe -Force
    Write-Ok "GPU binary selected (CUDA acceleration enabled)"
} elseif (Test-Path $cpuExe) {
    Copy-Item $cpuExe $mainExe -Force
    Write-Ok "CPU binary selected"
} else {
    Write-Skip "Binary selection skipped (files not found)"
}

# --- Summary ---

Write-Host "`n" -NoNewline
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Dependency Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Verify core dependencies
$allOk = $true
$vcCheck = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue
$pythonCheck = Get-PythonExe
$cudaCheck = (Test-Command "nvcc") -or
    (Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue)
$nvidiaGpu = Get-CimInstance -ClassName Win32_VideoController -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "*NVIDIA*" }
$vbCableCheck = Get-PnpDevice -FriendlyName "*VB-Audio*" -ErrorAction SilentlyContinue
$hifiCheck = Get-PnpDevice -FriendlyName "*Hi-Fi*" -ErrorAction SilentlyContinue
$modelCheck = Test-Path (Join-Path $ModelsDir "ggml-small.bin")

Write-Host "  Dependency Status:" -ForegroundColor White
if ($vcCheck) { Write-Ok "Visual C++ Redistributable" } else { Write-Fail "VC++ Redistributable - NOT INSTALLED"; $allOk = $false }
if ($pythonCheck) { Write-Ok "Python" } else { Write-Fail "Python - NOT INSTALLED"; $allOk = $false }
if ($cudaCheck) { Write-Ok "CUDA" } elseif ($nvidiaGpu) { Write-Fail "CUDA - NOT INSTALLED (NVIDIA GPU detected)"; $allOk = $false } else { Write-Skip "CUDA - not needed (no NVIDIA GPU)" }
if ($vbCableCheck) { Write-Ok "VB-Cable" } else { Write-Fail "VB-Cable - NOT INSTALLED"; $allOk = $false }
if ($hifiCheck) { Write-Ok "Hi-Fi Cable" } else { Write-Fail "Hi-Fi Cable - NOT INSTALLED"; $allOk = $false }
if ($modelCheck) { Write-Ok "Whisper Model" } else { Write-Fail "Whisper Model - NOT DOWNLOADED"; $allOk = $false }

Write-Host ""
if ($allOk) {
    Write-Host "  All dependencies are ready!" -ForegroundColor Green
    Write-Host "  Launch 'Meeting Translator' from the Start Menu or Desktop." -ForegroundColor White
} else {
    Write-Host "  Some dependencies are missing. The app may not work correctly." -ForegroundColor Yellow
    Write-Host "  You can re-run this setup from: Start Menu > Meeting Translator > Setup Dependencies" -ForegroundColor Yellow
}

if (-not $vbCableCheck -or -not $hifiCheck) {
    Write-Host ""
    Write-Host "  NOTE: If audio drivers were just installed, a restart may be required." -ForegroundColor Yellow
}
Write-Host ""
