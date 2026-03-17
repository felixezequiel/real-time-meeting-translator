#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Installs all dependencies for the Real-Time Meeting Translator.

.DESCRIPTION
    This script checks and installs:
    - Rust toolchain (via rustup)
    - Python 3.12+ (via winget)
    - VB-Cable virtual audio driver
    - Python packages (faster-whisper, transformers, piper-tts, etc.)
    - Builds the Rust project
    - Optionally launches the application

.NOTES
    Run as Administrator: Right-click PowerShell -> Run as Administrator
    Usage: .\scripts\install.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ScriptsDir = Join-Path $ProjectRoot "scripts"
$ModelsDir = Join-Path $ProjectRoot "models"

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

    # Also add common install locations
    $extraPaths = @(
        "$env:USERPROFILE\.cargo\bin",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links",
        "C:\Program Files\CMake\bin",
        "C:\Program Files\LLVM\bin"
    )
    foreach ($p in $extraPaths) {
        if ((Test-Path $p) -and ($env:Path -notlike "*$p*")) {
            $env:Path = "$p;$env:Path"
        }
    }
}

# --- Installation Steps ---

function Install-Rust {
    Write-Step "Rust Toolchain"

    Refresh-Path

    if (Test-Command "rustc") {
        $currentVersion = (rustc --version) -replace "rustc ", ""
        Write-Ok "Rust already installed: $currentVersion"

        # Check for updates
        Write-Host "  Checking for updates..." -ForegroundColor Gray
        try {
            $updateOutput = & rustup update stable 2>&1
        } catch {
            # rustup writes info to stderr, ignore errors
        }
        $updatedVersion = (rustc --version) -replace "rustc ", ""
        if ($updatedVersion -ne $currentVersion) {
            Write-Ok "Updated to $updatedVersion"
        }
        return
    }

    Write-Host "  Installing Rust via rustup..." -ForegroundColor Gray
    $rustupUrl = "https://win.rustup.rs/x86_64"
    $rustupExe = Join-Path $env:TEMP "rustup-init.exe"

    Invoke-WebRequest -Uri $rustupUrl -OutFile $rustupExe -UseBasicParsing
    try { & $rustupExe -y --default-toolchain stable 2>&1 | Out-Host } catch {}

    Refresh-Path

    if (Test-Command "rustc") {
        $version = rustc --version
        Write-Ok "Rust installed: $version"
    } else {
        Write-Fail "Rust installation failed. Please install manually from https://rustup.rs"
        exit 1
    }
}

function Install-Python {
    Write-Step "Python 3.10+"

    Refresh-Path

    $pythonExe = Get-PythonExe
    if ($pythonExe) {
        $version = & $pythonExe --version 2>&1
        Write-Ok "Python already installed: $version"
        return $pythonExe
    }

    Write-Host "  Installing Python via winget..." -ForegroundColor Gray
    try { winget install --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements --silent 2>&1 | Out-Host } catch {}

    Refresh-Path

    $pythonExe = Get-PythonExe
    if ($pythonExe) {
        $version = & $pythonExe --version 2>&1
        Write-Ok "Python installed: $version"
        return $pythonExe
    } else {
        Write-Fail "Python installation failed. Please install Python 3.10+ manually."
        exit 1
    }
}

function Install-VBCable {
    Write-Step "VB-Cable Virtual Audio Driver"

    # Check if VB-Cable is already installed by looking for the device
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
            Write-Host "  Installing VB-Cable driver (requires admin)..." -ForegroundColor Gray
            Start-Process -FilePath $setupExe.FullName -ArgumentList "/i" -Wait -Verb RunAs 2>$null

            # Verify installation
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

function Install-PythonDeps {
    param([string]$PythonExe)

    Write-Step "Python Packages"

    $requirementsFile = Join-Path $ScriptsDir "requirements.txt"
    if (-not (Test-Path $requirementsFile)) {
        Write-Fail "requirements.txt not found at $requirementsFile"
        exit 1
    }

    Write-Host "  Upgrading pip..." -ForegroundColor Gray
    try { & $PythonExe -m pip install --upgrade pip 2>&1 | Out-Host } catch {}

    Write-Host "  Installing packages (this may take several minutes on first run)..." -ForegroundColor Gray
    $pipOutput = & $PythonExe -m pip install -r $requirementsFile 2>&1
    $pipExitCode = $LASTEXITCODE

    if ($pipExitCode -eq 0) {
        # Verify key packages
        $packages = @("faster_whisper", "transformers", "torch")
        $allOk = $true
        foreach ($pkg in $packages) {
            $check = & $PythonExe -c "import $pkg; print($pkg.__version__)" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "$pkg $check"
            } else {
                Write-Fail "$pkg not importable"
                $allOk = $false
            }
        }

        if (-not $allOk) {
            Write-Host "  Some packages failed to install. Check the output above." -ForegroundColor Yellow
        }
    } else {
        Write-Fail "pip install failed (exit code: $pipExitCode)"
        Write-Host "  Output: $pipOutput" -ForegroundColor Yellow
        Write-Host "  Try running manually: $PythonExe -m pip install -r $requirementsFile" -ForegroundColor Yellow
    }
}

function Build-RustProject {
    Write-Step "Building Rust Project"

    Refresh-Path

    Push-Location $ProjectRoot
    try {
        Write-Host "  Compiling (release mode)..." -ForegroundColor Gray
        $buildOutput = & cargo build --release 2>&1
        $buildExitCode = $LASTEXITCODE

        if ($buildExitCode -eq 0) {
            $binaryPath = Join-Path $ProjectRoot "target\release\meeting-translator.exe"
            if (Test-Path $binaryPath) {
                $size = [math]::Round((Get-Item $binaryPath).Length / 1MB, 1)
                Write-Ok "Built successfully: meeting-translator.exe (${size}MB)"
            } else {
                Write-Ok "Build completed"
            }
        } else {
            Write-Fail "Build failed"
            Write-Host $buildOutput -ForegroundColor Yellow
            exit 1
        }
    } finally {
        Pop-Location
    }
}

function Test-RustProject {
    Write-Step "Running Tests"

    Refresh-Path

    Push-Location $ProjectRoot
    try {
        $testOutput = & cargo test --workspace 2>&1
        $testExitCode = $LASTEXITCODE

        if ($testExitCode -eq 0) {
            $passed = ($testOutput | Select-String "test result: ok" | Measure-Object).Count
            Write-Ok "All test suites passed ($passed suites)"
        } else {
            Write-Fail "Some tests failed"
            Write-Host $testOutput -ForegroundColor Yellow
        }
    } finally {
        Pop-Location
    }
}

function Show-Summary {
    param([bool]$LaunchApp)

    Write-Host "`n" -NoNewline
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Installation Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  To run manually:" -ForegroundColor White
    Write-Host "    cd $ProjectRoot" -ForegroundColor Gray
    Write-Host "    cargo run --release" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  First run will download ML models (~1GB)." -ForegroundColor Yellow
    Write-Host "  Subsequent runs will be much faster." -ForegroundColor Yellow
    Write-Host ""

    if ($LaunchApp) {
        $binaryPath = Join-Path $ProjectRoot "target\release\meeting-translator.exe"
        if (Test-Path $binaryPath) {
            Write-Host "  Launching Meeting Translator..." -ForegroundColor Cyan
            Start-Process -FilePath $binaryPath -WorkingDirectory $ProjectRoot
        } else {
            Write-Host "  Binary not found. Run: cargo run --release" -ForegroundColor Yellow
        }
    }
}

# --- Main ---

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Real-Time Meeting Translator - Installer" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Project: $ProjectRoot" -ForegroundColor Gray
Write-Host ""

# Check admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "  WARNING: Not running as Administrator." -ForegroundColor Yellow
    Write-Host "  VB-Cable installation requires admin privileges." -ForegroundColor Yellow
    Write-Host "  Other dependencies will still be installed." -ForegroundColor Yellow
    Write-Host ""
}

Install-Rust
$pythonExe = Install-Python

if ($isAdmin) {
    Install-VBCable
} else {
    Write-Step "VB-Cable Virtual Audio Driver"
    Write-Skip "Skipped (requires Administrator). Run again as admin or install from https://vb-audio.com/Cable/"
}

Install-PythonDeps -PythonExe $pythonExe
Build-RustProject
Test-RustProject

$launchChoice = Read-Host "`nLaunch Meeting Translator now? (y/N)"
$shouldLaunch = $launchChoice -eq "y" -or $launchChoice -eq "Y"

Show-Summary -LaunchApp $shouldLaunch
