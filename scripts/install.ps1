#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Installs all dependencies for the Real-Time Meeting Translator.

.DESCRIPTION
    This script checks and installs:
    - Visual Studio Build Tools (MSVC C++ compiler)
    - CMake (for building native dependencies)
    - LLVM/Clang (for bindgen FFI generation)
    - CUDA Toolkit (for GPU-accelerated Whisper inference)
    - Rust toolchain (via rustup)
    - Python 3.12+ (via winget)
    - VB-Cable virtual audio driver (meeting audio loopback)
    - Hi-Fi Cable virtual audio driver (mic TTS output)
    - Python packages (faster-whisper, transformers, piper-tts, etc.)
    - Whisper GGML model download
    - Auto-generates .cargo/config.toml with correct MSVC paths
    - Creates default config.toml if not present
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
        "C:\Program Files\LLVM\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
    )
    foreach ($p in $extraPaths) {
        if ((Test-Path $p) -and ($env:Path -notlike "*$p*")) {
            $env:Path = "$p;$env:Path"
        }
    }
}

# --- Find MSVC paths dynamically ---

function Find-MsvcIncludePath {
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vsWhere)) {
        # Fallback: search common locations
        $vsWhere = "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
    }

    $vsPath = $null
    if (Test-Path $vsWhere) {
        $vsPath = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    }

    if (-not $vsPath) {
        # Search BuildTools locations
        $buildToolsPaths = @(
            "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools",
            "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community",
            "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional",
            "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise"
        )
        foreach ($p in $buildToolsPaths) {
            if (Test-Path "$p\VC\Tools\MSVC") {
                $vsPath = $p
                break
            }
        }
    }

    if (-not $vsPath) { return $null }

    # Find latest MSVC version
    $msvcBase = Join-Path $vsPath "VC\Tools\MSVC"
    if (-not (Test-Path $msvcBase)) { return $null }
    $latestMsvc = Get-ChildItem $msvcBase -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if (-not $latestMsvc) { return $null }

    $msvcInclude = Join-Path $latestMsvc.FullName "include"
    if (-not (Test-Path $msvcInclude)) { return $null }

    return $msvcInclude
}

function Find-UcrtIncludePath {
    $kitsBase = "${env:ProgramFiles(x86)}\Windows Kits\10\Include"
    if (-not (Test-Path $kitsBase)) { return $null }

    $latestKit = Get-ChildItem $kitsBase -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if (-not $latestKit) { return $null }

    $ucrtInclude = Join-Path $latestKit.FullName "ucrt"
    if (-not (Test-Path $ucrtInclude)) { return $null }

    return $ucrtInclude
}

# --- Installation Steps ---

function Install-VSBuildTools {
    Write-Step "Visual Studio Build Tools (MSVC C++ Compiler)"

    $msvcInclude = Find-MsvcIncludePath
    if ($msvcInclude) {
        Write-Ok "MSVC already installed: $msvcInclude"
        return
    }

    Write-Host "  Installing Visual Studio Build Tools via winget..." -ForegroundColor Gray
    Write-Host "  (This may take 5-10 minutes)" -ForegroundColor Gray

    try {
        winget install --id Microsoft.VisualStudio.2022.BuildTools `
            --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" `
            --accept-package-agreements --accept-source-agreements 2>&1 | Out-Host
    } catch {
        Write-Host "  winget reported an error (this is often non-fatal)" -ForegroundColor Yellow
    }

    Refresh-Path

    $msvcInclude = Find-MsvcIncludePath
    if ($msvcInclude) {
        Write-Ok "MSVC Build Tools installed: $msvcInclude"
    } else {
        Write-Fail "MSVC Build Tools installation could not be verified"
        Write-Host "  Install manually: https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Yellow
        Write-Host "  Select 'Desktop development with C++' workload" -ForegroundColor Yellow
    }
}

function Install-CMake {
    Write-Step "CMake"

    Refresh-Path

    if (Test-Command "cmake") {
        $version = (cmake --version | Select-Object -First 1) -replace "cmake version ", ""
        Write-Ok "CMake already installed: $version"
        return
    }

    Write-Host "  Installing CMake via winget..." -ForegroundColor Gray
    try {
        winget install --id Kitware.CMake --accept-package-agreements --accept-source-agreements --silent 2>&1 | Out-Host
    } catch {}

    Refresh-Path

    if (Test-Command "cmake") {
        $version = (cmake --version | Select-Object -First 1) -replace "cmake version ", ""
        Write-Ok "CMake installed: $version"
    } else {
        Write-Fail "CMake installation failed. Install manually from https://cmake.org/download/"
    }
}

function Install-LLVM {
    Write-Step "LLVM/Clang (for bindgen)"

    Refresh-Path

    if (Test-Command "clang") {
        $version = (clang --version | Select-Object -First 1)
        Write-Ok "LLVM already installed: $version"
        return
    }

    # Check if LLVM exists but not in PATH
    if (Test-Path "C:\Program Files\LLVM\bin\clang.exe") {
        Write-Ok "LLVM found at C:\Program Files\LLVM\bin"
        return
    }

    Write-Host "  Installing LLVM via winget..." -ForegroundColor Gray
    try {
        winget install --id LLVM.LLVM --accept-package-agreements --accept-source-agreements --silent 2>&1 | Out-Host
    } catch {}

    Refresh-Path

    if ((Test-Command "clang") -or (Test-Path "C:\Program Files\LLVM\bin\clang.exe")) {
        Write-Ok "LLVM installed"
    } else {
        Write-Fail "LLVM installation failed. Install manually from https://releases.llvm.org/"
    }
}

function Install-CUDA {
    Write-Step "CUDA Toolkit (GPU Acceleration)"

    # Check if CUDA is already available
    if (Test-Command "nvcc") {
        $version = (nvcc --version | Select-String "release") -replace ".*release ", "" -replace ",.*", ""
        Write-Ok "CUDA already installed: $version"
        return
    }

    # Check common install paths
    $cudaPaths = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue
    if ($cudaPaths) {
        $latest = $cudaPaths | Sort-Object Name -Descending | Select-Object -First 1
        Write-Ok "CUDA found at: $($latest.FullName)"
        $env:Path = "$($latest.FullName)\bin;$env:Path"
        return
    }

    # Check if NVIDIA GPU is present
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

function Install-Rust {
    Write-Step "Rust Toolchain (MSVC)"

    Refresh-Path

    if (Test-Command "rustc") {
        $currentVersion = (rustc --version) -replace "rustc ", ""
        Write-Ok "Rust already installed: $currentVersion"

        # Ensure MSVC target is the default (not GNU -- GNU causes dlltool.exe errors)
        $defaultHost = & rustup show | Select-String "Default host"
        if ($defaultHost -and $defaultHost -notlike "*msvc*") {
            Write-Host "  Switching default toolchain to MSVC..." -ForegroundColor Gray
            & rustup default stable-x86_64-pc-windows-msvc 2>&1 | Out-Host
            Write-Ok "Switched to MSVC toolchain"
        } else {
            Write-Ok "MSVC toolchain is default"
        }

        Write-Host "  Checking for updates..." -ForegroundColor Gray
        try {
            $updateOutput = & rustup update stable 2>&1
        } catch {}
        $updatedVersion = (rustc --version) -replace "rustc ", ""
        if ($updatedVersion -ne $currentVersion) {
            Write-Ok "Updated to $updatedVersion"
        }
        return
    }

    Write-Host "  Installing Rust via rustup (MSVC toolchain)..." -ForegroundColor Gray
    $rustupUrl = "https://win.rustup.rs/x86_64"
    $rustupExe = Join-Path $env:TEMP "rustup-init.exe"

    Invoke-WebRequest -Uri $rustupUrl -OutFile $rustupExe -UseBasicParsing
    # Force MSVC host to avoid dlltool.exe dependency from GNU toolchain
    try { & $rustupExe -y --default-toolchain stable --default-host x86_64-pc-windows-msvc 2>&1 | Out-Host } catch {}

    Refresh-Path

    if (Test-Command "rustc") {
        $version = rustc --version
        Write-Ok "Rust installed: $version (MSVC)"
    } else {
        Write-Fail "Rust installation failed. Please install manually from https://rustup.rs"
        Write-Host "  IMPORTANT: Select the MSVC toolchain during installation, NOT GNU" -ForegroundColor Yellow
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
            Write-Host "  Installing VB-Cable driver..." -ForegroundColor Gray
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
    Write-Step "Hi-Fi Cable Virtual Audio Driver (second virtual cable)"

    # Check if Hi-Fi Cable is already installed
    $hifiDevice = Get-PnpDevice -FriendlyName "*Hi-Fi*" -ErrorAction SilentlyContinue
    if ($hifiDevice) {
        Write-Ok "Hi-Fi Cable already installed"
        return
    }

    $hifiZip = Join-Path $env:TEMP "HiFiCable.zip"
    $hifiDir = Join-Path $env:TEMP "HiFiCable"

    Write-Host "  Downloading Hi-Fi Cable & ASIO Bridge..." -ForegroundColor Gray
    $downloadUrl = "https://download.vb-audio.com/Download_CABLE/HiFiCableAsioBridgeSetup_v1007.zip"

    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $hifiZip -UseBasicParsing

        if (Test-Path $hifiDir) { Remove-Item $hifiDir -Recurse -Force }
        Expand-Archive -Path $hifiZip -DestinationPath $hifiDir -Force

        $setupExe = Get-ChildItem -Path $hifiDir -Filter "*.exe" -Recurse |
            Where-Object { $_.Name -like "*Setup*" -or $_.Name -like "*HIFI*" } |
            Select-Object -First 1

        if ($setupExe) {
            Write-Host "  Installing Hi-Fi Cable driver..." -ForegroundColor Gray
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
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $pipOutput = & $PythonExe -m pip install -r $requirementsFile 2>&1
    $pipExitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevPref

    if ($pipExitCode -eq 0) {
        $packages = @("faster_whisper", "transformers", "torch", "ctranslate2")
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

function Install-WhisperModel {
    Write-Step "Whisper GGML Model"

    if (-not (Test-Path $ModelsDir)) {
        New-Item -Path $ModelsDir -ItemType Directory -Force | Out-Null
    }

    # q5_1 quantization: ~181 MB vs 466 MB fp16, ~2x faster inference.
    # Must match DEFAULT_WHISPER_MODEL in crates/shared/src/config.rs.
    $modelName = "small-q5_1"
    $modelFile = Join-Path $ModelsDir "ggml-$modelName.bin"

    if (Test-Path $modelFile) {
        $size = [math]::Round((Get-Item $modelFile).Length / 1MB, 1)
        Write-Ok "Model already downloaded: ggml-$modelName.bin (${size}MB)"
        return
    }

    $modelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$modelName.bin"

    Write-Host "  Downloading ggml-$modelName.bin (~181MB, quantized)..." -ForegroundColor Gray
    Write-Host "  From: $modelUrl" -ForegroundColor Gray

    try {
        Invoke-WebRequest -Uri $modelUrl -OutFile $modelFile -UseBasicParsing

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

function Install-TranslationModels {
    param([string]$PythonExe)

    # Converts Opus-MT models to CTranslate2 int8 format for ~3-5x faster
    # inference. See docs/adr/0001-ctranslate2-translation.md.
    Write-Step "Opus-MT Translation Models (CTranslate2 int8)"

    if (-not (Test-Path $ModelsDir)) {
        New-Item -Path $ModelsDir -ItemType Directory -Force | Out-Null
    }

    $pairs = @(
        @{ Hf = "Helsinki-NLP/opus-mt-en-ROMANCE"; Dir = "opus-mt-en-ROMANCE-ct2" },
        @{ Hf = "Helsinki-NLP/opus-mt-ROMANCE-en"; Dir = "opus-mt-ROMANCE-en-ct2" }
    )

    foreach ($p in $pairs) {
        $targetDir = Join-Path $ModelsDir $p.Dir
        $modelBin = Join-Path $targetDir "model.bin"

        if (Test-Path $modelBin) {
            $size = [math]::Round((Get-Item $modelBin).Length / 1MB, 1)
            Write-Ok "Already converted: $($p.Dir) (${size}MB)"
            continue
        }

        Write-Host "  Converting $($p.Hf) -> $($p.Dir) (int8)..." -ForegroundColor Gray
        $prevPref = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $PythonExe -m ctranslate2.converters.transformers `
            --model $p.Hf `
            --output_dir $targetDir `
            --quantization int8 `
            --force 2>&1 | Out-Host
        $exit = $LASTEXITCODE
        $ErrorActionPreference = $prevPref

        if ($exit -eq 0 -and (Test-Path $modelBin)) {
            $size = [math]::Round((Get-Item $modelBin).Length / 1MB, 1)
            Write-Ok "Converted: $($p.Dir) (${size}MB)"
        } else {
            Write-Fail "Conversion failed for $($p.Hf) (exit $exit)"
            Write-Host "  Run manually: $PythonExe -m ctranslate2.converters.transformers --model $($p.Hf) --output_dir $targetDir --quantization int8" -ForegroundColor Yellow
        }
    }
}

function Configure-CargoConfig {
    Write-Step "Cargo Build Configuration (.cargo/config.toml)"

    $configPath = Join-Path $CargoDir "config.toml"

    # If config already exists, validate it before overwriting
    if (Test-Path $configPath) {
        Write-Host "  Existing .cargo/config.toml found, validating..." -ForegroundColor Gray
        Push-Location $ProjectRoot
        $prevPref = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $checkOutput = & cargo check --message-format short 2>&1 | Select-Object -First 5
        $checkExit = $LASTEXITCODE
        $ErrorActionPreference = $prevPref
        Pop-Location

        # If cargo can at least parse the config (even if build fails for other reasons),
        # the config is fine -- don't overwrite it
        $configBroken = $checkOutput | Select-String "could not parse TOML configuration"
        if (-not $configBroken) {
            Write-Ok ".cargo/config.toml is valid -- keeping existing"
            return
        }
        Write-Host "  Existing config is broken, regenerating..." -ForegroundColor Yellow
    }

    $msvcInclude = Find-MsvcIncludePath
    $ucrtInclude = Find-UcrtIncludePath

    if (-not $msvcInclude) {
        Write-Skip "MSVC include path not found -- skipping .cargo/config.toml generation"
        return
    }

    if (-not $ucrtInclude) {
        Write-Skip "UCRT include path not found -- skipping .cargo/config.toml generation"
        return
    }

    # Normalize paths for TOML (forward slashes)
    $msvcPath = $msvcInclude -replace '\\', '/'
    $ucrtPath = $ucrtInclude -replace '\\', '/'

    # Detect CMake generator
    $cmakeGenerator = "Visual Studio 17 2022"

    if (-not (Test-Path $CargoDir)) {
        New-Item -Path $CargoDir -ItemType Directory -Force | Out-Null
    }

    # Build TOML content line by line to ensure correct escaping.
    # TOML requires \" for literal quotes inside double-quoted strings.
    $bindgenArgs = "-I\""$msvcPath\"" -I\""$ucrtPath\"""
    $lines = @(
        "[env]",
        "BINDGEN_EXTRA_CLANG_ARGS = ""$bindgenArgs""",
        "CMAKE_GENERATOR = ""$cmakeGenerator"""
    )

    Set-Content -Path $configPath -Value ($lines -join "`n") -Encoding UTF8 -NoNewline

    # Verify the generated config is valid TOML
    Push-Location $ProjectRoot
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $verifyOutput = & cargo check --message-format short 2>&1 | Select-Object -First 5
    $ErrorActionPreference = $prevPref
    Pop-Location

    $stillBroken = $verifyOutput | Select-String "could not parse TOML configuration"
    if ($stillBroken) {
        Write-Fail "Generated .cargo/config.toml is invalid"
        Write-Host "  Content:" -ForegroundColor Yellow
        Get-Content $configPath | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
        Write-Host "  You may need to edit it manually." -ForegroundColor Yellow
    } else {
        Write-Ok "Generated .cargo/config.toml"
        Write-Host "  MSVC: $msvcInclude" -ForegroundColor Gray
        Write-Host "  UCRT: $ucrtInclude" -ForegroundColor Gray
    }
}

function Configure-DefaultConfig {
    Write-Step "Application Configuration (config.toml)"

    $configPath = Join-Path $ProjectRoot "config.toml"
    if (Test-Path $configPath) {
        Write-Ok "config.toml already exists"
        return
    }

    $defaultConfig = @"
speaker_source_language = "English"
speaker_target_language = "Portuguese"
mic_source_language = "Portuguese"
mic_target_language = "English"
chunk_duration_ms = 500
whisper_model = "small-q5_1"
tts_speed = 1.1
"@

    Set-Content -Path $configPath -Value $defaultConfig -Encoding UTF8
    Write-Ok "Created default config.toml"
}

function Build-RustProject {
    Write-Step "Building Rust Project"

    Refresh-Path

    # Detect if CUDA is available to decide build features
    $hasCuda = (Test-Command "nvcc") -or
        (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if ($hasCuda) {
        Write-Host "  CUDA detected -- building with GPU acceleration" -ForegroundColor Green
        $cargoArgs = @("build", "--release")
    } else {
        Write-Host "  No CUDA -- building CPU-only (STT will be slower)" -ForegroundColor Yellow
        $cargoArgs = @("build", "--release", "--no-default-features")
    }

    Push-Location $ProjectRoot
    try {
        Write-Host "  Compiling (release mode)..." -ForegroundColor Gray
        Write-Host "  (First build may take 5-10 minutes)" -ForegroundColor Gray
        $prevPref = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & cargo @cargoArgs 2>&1 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        $buildExitCode = $LASTEXITCODE
        $ErrorActionPreference = $prevPref

        if ($buildExitCode -eq 0) {
            $binaryPath = Join-Path $ProjectRoot "target\release\meeting-translator.exe"
            if (Test-Path $binaryPath) {
                $size = [math]::Round((Get-Item $binaryPath).Length / 1MB, 1)
                Write-Ok "Built successfully: meeting-translator.exe (${size}MB)"
            } else {
                Write-Ok "Build completed"
            }
        } else {
            Write-Fail "Build failed (exit code: $buildExitCode)"
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
        $prevPref = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $testOutput = & cargo test --workspace 2>&1
        $testExitCode = $LASTEXITCODE
        $ErrorActionPreference = $prevPref

        if ($testExitCode -eq 0) {
            $passed = ($testOutput | Select-String "test result: ok" | Measure-Object).Count
            Write-Ok "All test suites passed ($passed suites)"
        } else {
            Write-Fail "Some tests failed"
            $testOutput | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
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
    Write-Host "  The app runs in the system tray." -ForegroundColor Yellow
    Write-Host "  Right-click the tray icon to Start/Stop." -ForegroundColor Yellow
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

# 1. Build toolchain (MSVC, CMake, LLVM)
Install-VSBuildTools
Install-CMake
Install-LLVM

# 2. GPU support
Install-CUDA

# 3. Language runtimes
Install-Rust
$pythonExe = Install-Python

# 4. Audio drivers (two virtual cables for pipeline isolation)
Install-VBCable
Install-HiFiCable

# 5. Dependencies
Install-PythonDeps -PythonExe $pythonExe
Install-WhisperModel
Install-TranslationModels -PythonExe $pythonExe

# 6. Configuration
Configure-CargoConfig
Configure-DefaultConfig

# 7. Build and test
Build-RustProject
Test-RustProject

$launchChoice = Read-Host "`nLaunch Meeting Translator now? (y/N)"
$shouldLaunch = $launchChoice -eq "y" -or $launchChoice -eq "Y"

Show-Summary -LaunchApp $shouldLaunch
