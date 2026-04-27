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
    - Python packages (kokoro-onnx, llama-cpp-python, speechbrain, transformers, etc.)
    - Whisper GGML model download
    - CosyVoice 2-0.5B clone + weights download (TTS + zero-shot voice cloning)
    - SpeechBrain ECAPA-TDNN warm-up (online diarisation)
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

    Write-Host "  Upgrading pip / wheel and pinning setuptools..." -ForegroundColor Gray
    # setuptools <75 still ships `pkg_resources.declare_namespace`, which
    # `lightning` (the version cosyvoice pulls) uses inside its package
    # __init__. setuptools 75 removed declare_namespace, so without this
    # pin the bridge crashes with `ModuleNotFoundError: pkg_resources`
    # the moment lightning is imported.
    # Newer pip + wheel are still safe to upgrade unconditionally.
    try { & $PythonExe -m pip install --upgrade pip wheel 2>&1 | Out-Host } catch {}
    try { & $PythonExe -m pip install "setuptools<75" 2>&1 | Out-Host } catch {}

    # Install line-by-line so a build failure on one package (typically
    # pyworld / WeTextProcessing — the C-extension Chinese text deps that
    # don't have Windows wheels) doesn't make pip's resolver atomic-fail
    # the whole list. The cosyvoice runtime tolerates several of these
    # being absent (text normalisers fall back to a default path).
    Write-Host "  Installing packages (line-by-line; failures are isolated)..." -ForegroundColor Gray

    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    $failedPackages = @()
    foreach ($rawLine in (Get-Content $requirementsFile)) {
        $line = $rawLine.Trim()
        # Skip blanks and comments.
        if (-not $line -or $line.StartsWith("#")) { continue }

        # Drop inline comments (`pkg>=1.0  # note`).
        if ($line.Contains("#")) {
            $line = ($line -split "#", 2)[0].Trim()
        }
        if (-not $line) { continue }

        Write-Host "    -> $line" -ForegroundColor DarkGray
        & $PythonExe -m pip install --prefer-binary $line 2>&1 |
            ForEach-Object { Write-Host "       $_" -ForegroundColor DarkGray }
        if ($LASTEXITCODE -ne 0) {
            $failedPackages += $line
            Write-Host "       (failed; will try --no-deps fallback)" -ForegroundColor Yellow
            & $PythonExe -m pip install --prefer-binary --no-deps $line 2>&1 |
                ForEach-Object { Write-Host "       $_" -ForegroundColor DarkGray }
        }
    }

    $ErrorActionPreference = $prevPref

    # Verify the imports the bridges will actually perform — if any of
    # these fail the bridges crash on startup with ModuleNotFoundError.
    # Stack: Qwen via llama-cpp-python (translation), kokoro-onnx (TTS),
    # pyworld (pitch shift), speechbrain (diarisation + separation).
    $criticalImports = @(
        "transformers", "torch", "torchaudio",
        "llama_cpp", "kokoro_onnx", "pyworld",
        "speechbrain", "numpy"
    )
    $missing = @()
    foreach ($pkg in $criticalImports) {
        & $PythonExe -c "import $pkg" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "$pkg importable"
        } else {
            Write-Fail "$pkg NOT importable"
            $missing += $pkg
        }
    }

    if ($missing.Count -gt 0) {
        Write-Host "  Some critical packages are missing: $($missing -join ', ')" -ForegroundColor Yellow
        Write-Host "  Try installing them manually with --no-deps:" -ForegroundColor Yellow
        Write-Host "    $PythonExe -m pip install --no-deps $($missing -join ' ')" -ForegroundColor Yellow
    }

    if ($failedPackages.Count -gt 0) {
        Write-Host "  Packages that needed --no-deps fallback: $($failedPackages -join ', ')" -ForegroundColor Yellow
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

    # NLLB-200 distilled (600M params) converted to CTranslate2 int8. One
    # model covers every supported direction (en<->pt), produces more natural
    # and context-aware translations than Opus-MT. Latency on GPU int8_float16:
    # ~200ms/sentence (vs ~100ms for Opus-MT — still inside the 2-5s budget).
    Write-Step "NLLB-200 Translation Model (CTranslate2 int8)"

    if (-not (Test-Path $ModelsDir)) {
        New-Item -Path $ModelsDir -ItemType Directory -Force | Out-Null
    }

    $hfName = "facebook/nllb-200-distilled-600M"
    $dirName = "nllb-200-distilled-600M-ct2"
    $targetDir = Join-Path $ModelsDir $dirName
    $modelBin = Join-Path $targetDir "model.bin"

    if (Test-Path $modelBin) {
        $size = [math]::Round((Get-Item $modelBin).Length / 1MB, 1)
        Write-Ok "Already converted: $dirName (${size}MB)"
        return
    }

    Write-Host "  Converting $hfName -> $dirName (int8)..." -ForegroundColor Gray
    Write-Host "  Downloads ~2.5GB from HuggingFace; final int8 model ~600MB on disk." -ForegroundColor Gray
    Write-Host "  This may take several minutes on the first run." -ForegroundColor Gray
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonExe -m ctranslate2.converters.transformers `
        --model $hfName `
        --output_dir $targetDir `
        --quantization int8 `
        --force 2>&1 | Out-Host
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $prevPref

    if ($exit -eq 0 -and (Test-Path $modelBin)) {
        $size = [math]::Round((Get-Item $modelBin).Length / 1MB, 1)
        Write-Ok "Converted: $dirName (${size}MB)"
    } else {
        Write-Fail "Conversion failed for $hfName (exit $exit)"
        Write-Host "  Run manually: $PythonExe -m ctranslate2.converters.transformers --model $hfName --output_dir $targetDir --quantization int8" -ForegroundColor Yellow
    }
}

function Install-KokoroModel {
    # Kokoro v1.0 ONNX TTS — replaced Piper. Two files: the ~330 MB
    # ONNX graph and the ~25 MB voice bank. Both live under
    # `models/kokoro/` so the bridge can find them with a single search.
    Write-Step "Kokoro v1.0 TTS Model"

    $kokoroDir = Join-Path $ModelsDir "kokoro"
    if (-not (Test-Path $kokoroDir)) {
        New-Item -Path $kokoroDir -ItemType Directory -Force | Out-Null
    }

    $files = @(
        @{
            name = "kokoro-v1.0.onnx"
            url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
        },
        @{
            name = "voices-v1.0.bin"
            url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
        }
    )

    foreach ($file in $files) {
        $localPath = Join-Path $kokoroDir $file.name
        if (Test-Path $localPath) {
            $size = [math]::Round((Get-Item $localPath).Length / 1MB, 1)
            Write-Ok "$($file.name) already downloaded (${size} MB)"
            continue
        }
        Write-Host "  Downloading $($file.name)..." -ForegroundColor Gray
        try {
            Invoke-WebRequest -Uri $file.url -OutFile $localPath -UseBasicParsing
            $size = [math]::Round((Get-Item $localPath).Length / 1MB, 1)
            Write-Ok "$($file.name) (${size} MB)"
        } catch {
            Write-Fail "Could not download $($file.name) : $_"
        }
    }
}

function Install-LlmModel {
    # Qwen 2.5 1.5B Instruct, Q4_K_M GGUF — local LLM for streaming
    # translation. ~1 GB on disk, ~1 GB VRAM. Replaces NLLB CT2.
    Write-Step "Qwen 2.5 1.5B Instruct (GGUF, Q4_K_M)"

    if (-not (Test-Path $ModelsDir)) {
        New-Item -Path $ModelsDir -ItemType Directory -Force | Out-Null
    }

    $modelName = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    $modelFile = Join-Path $ModelsDir $modelName
    if (Test-Path $modelFile) {
        $size = [math]::Round((Get-Item $modelFile).Length / 1MB, 1)
        Write-Ok "$modelName already downloaded (${size} MB)"
        return
    }

    # Hosted by Qwen team's official GGUF repo on HuggingFace.
    $modelUrl = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    Write-Host "  Downloading $modelName (~1 GB)..." -ForegroundColor Gray
    try {
        Invoke-WebRequest -Uri $modelUrl -OutFile $modelFile -UseBasicParsing
        $size = [math]::Round((Get-Item $modelFile).Length / 1MB, 1)
        Write-Ok "$modelName (${size} MB)"
    } catch {
        Write-Fail "Could not download $modelName : $_"
        Write-Host "  Manual: Invoke-WebRequest -Uri '$modelUrl' -OutFile '$modelFile' -UseBasicParsing" -ForegroundColor Yellow
    }
}

function Install-PiperVoices {
    # Pre-downloads the Piper ONNX voices (Faber pt-BR, Ryan en-US) into
    # %TEMP%\piper_voices\. The bridge auto-downloads on first start
    # too, but doing it here means cold-start of the app stays fast.
    Write-Step "Piper TTS Voices (pt-BR Faber + en-US Ryan)"

    $voicesDir = Join-Path $env:TEMP "piper_voices"
    if (-not (Test-Path $voicesDir)) {
        New-Item -Path $voicesDir -ItemType Directory -Force | Out-Null
    }

    $voices = @(
        @{
            name = "en_US-ryan-medium"
            urls = @(
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"
            )
        },
        @{
            name = "en_US-amy-medium"
            urls = @(
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"
            )
        },
        @{
            name = "pt_BR-faber-medium"
            urls = @(
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx",
                "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json"
            )
        }
    )

    foreach ($voice in $voices) {
        foreach ($url in $voice.urls) {
            $filename = [System.IO.Path]::GetFileName($url)
            $localPath = Join-Path $voicesDir $filename
            if (Test-Path $localPath) {
                continue
            }
            Write-Host "  Downloading $filename..." -ForegroundColor Gray
            try {
                Invoke-WebRequest -Uri $url -OutFile $localPath -UseBasicParsing
                $size = [math]::Round((Get-Item $localPath).Length / 1MB, 1)
                Write-Ok "$filename (${size} MB)"
            } catch {
                Write-Fail "Could not download $filename : $_"
            }
        }
    }
}

# (Install-TorchCuda was used during the CosyVoice attempt; CPU torch is
# enough for Piper + pyworld + speechbrain on the current pipeline. Kept
# here as a no-op stub so the main flow at the bottom of the script
# stays minimal — call site removed in this revision.)
function Install-TorchCuda {
    param([string]$PythonExe)
    return
    Write-Step "PyTorch with CUDA support"

    Refresh-Path

    # Quick check: does the currently installed torch see CUDA?
    $cudaState = (& $PythonExe -c "import torch, sys; sys.stdout.write('1' if torch.cuda.is_available() else '0')" 2>$null)
    if ($cudaState -eq "1") {
        $version = & $PythonExe -c "import torch; print(torch.__version__, '-', torch.version.cuda)" 2>&1
        Write-Ok "PyTorch CUDA already working: $version"
        return
    }

    # Pick a CUDA wheel variant that matches the installed CUDA toolkit.
    # The PyTorch wheels are forward-compatible within a major version,
    # so cu124 wheels work on top of CUDA 12.x runtimes (the user's setup
    # has the 12.4 toolkit + a 13.0 driver).
    $cuVariant = "cu124"
    $nvccVersion = ""
    if (Test-Command "nvcc") {
        $nvccVersion = (& nvcc --version 2>&1 | Select-String "release") -replace ".*release ", "" -replace ",.*", ""
        if ($nvccVersion -match "^12\.1") { $cuVariant = "cu121" }
        elseif ($nvccVersion -match "^12\.6") { $cuVariant = "cu126" }
        elseif ($nvccVersion -match "^12\.8") { $cuVariant = "cu126" }  # cu128 not always available
        # Anything else 12.x → cu124 (default)
    }

    $indexUrl = "https://download.pytorch.org/whl/$cuVariant"
    Write-Host "  Installing PyTorch from $indexUrl (CUDA toolkit: $nvccVersion)..." -ForegroundColor Gray

    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonExe -m pip uninstall -y torch torchaudio torchvision 2>&1 | Out-Host
    & $PythonExe -m pip install --upgrade --index-url $indexUrl torch torchaudio 2>&1 | Out-Host
    $ErrorActionPreference = $prevPref

    $cudaState = (& $PythonExe -c "import torch, sys; sys.stdout.write('1' if torch.cuda.is_available() else '0')" 2>$null)
    if ($cudaState -eq "1") {
        $version = & $PythonExe -c "import torch; print(torch.__version__, '-', torch.version.cuda)" 2>&1
        Write-Ok "PyTorch CUDA installed: $version"
    } else {
        Write-Fail "PyTorch still reports no CUDA. Install manually:"
        Write-Host "    $PythonExe -m pip install --upgrade --index-url $indexUrl torch torchaudio" -ForegroundColor Yellow
    }
}

function Install-CosyVoice {
    param([string]$PythonExe)

    # CosyVoice 2-0.5B is the TTS + zero-shot voice cloner that replaced
    # Piper + KNN-VC. The PyPI package is unstable, so we clone the
    # upstream repo into `third_party/CosyVoice/` and let `tts_bridge.py`
    # add it to sys.path. The 0.5B weights (~600 MB) are downloaded from
    # HuggingFace into `models/CosyVoice2-0.5B/`.
    Write-Step "CosyVoice 2-0.5B (TTS + voice cloning)"

    $thirdPartyDir = Join-Path $ProjectRoot "third_party"
    $repoDir = Join-Path $thirdPartyDir "CosyVoice"
    $matchaDir = Join-Path $repoDir "third_party\Matcha-TTS"

    if (-not (Test-Path $thirdPartyDir)) {
        New-Item -Path $thirdPartyDir -ItemType Directory -Force | Out-Null
    }

    if (-not (Test-Path $repoDir)) {
        Write-Host "  Cloning CosyVoice repo (with submodules)..." -ForegroundColor Gray
        try {
            & git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $repoDir 2>&1 | Out-Host
        } catch {
            Write-Fail "git clone failed: $_"
            Write-Host "  Clone manually: git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $repoDir" -ForegroundColor Yellow
            return
        }
    } else {
        Write-Ok "CosyVoice repo already present at $repoDir"
        # Make sure submodules are in place — Matcha-TTS is required for
        # the matcha vocoder used by the cosyvoice package.
        if (-not (Test-Path "$matchaDir\matcha")) {
            Write-Host "  Initialising submodules (Matcha-TTS)..." -ForegroundColor Gray
            Push-Location $repoDir
            try { & git submodule update --init --recursive 2>&1 | Out-Host } catch {}
            Pop-Location
        }
    }

    # NOTE: We deliberately do NOT run `pip install -r $repoDir/requirements.txt`.
    # CosyVoice's upstream requirements pin grpcio==1.57.0 (no Python 3.12
    # wheel on Windows; builds-from-source crash with `ModuleNotFoundError:
    # pkg_resources`) and pull in heavy unused deps (deepspeed, tensorrt-cu12,
    # gradio, fastapi). The runtime imports the cosyvoice cli actually does
    # are listed in `scripts/requirements.txt` and installed by
    # `Install-PythonDeps`.

    # Install `modelscope` with --no-deps. CosyVoice does
    # `from modelscope import snapshot_download` at module load time, so
    # the package must be importable — but we never call it (weights are
    # downloaded via huggingface_hub below). --no-deps avoids the
    # grpcio<=1.57 transitive pin that breaks the resolver on Python 3.12.
    Write-Host "  Installing modelscope (no-deps, import-only)..." -ForegroundColor Gray
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonExe -m pip install --no-deps "modelscope>=1.20" 2>&1 | Out-Host
    $ErrorActionPreference = $prevPref

    # Best-effort tolerant pass over CosyVoice's upstream requirements.txt.
    # We deliberately don't trust the file (it has Windows-hostile pins —
    # grpcio==1.57.0, deepspeed, tensorrt-cu12, pyworld) but installing
    # line-by-line catches transitive deps we'd otherwise discover one at
    # a time when the bridge crashes (hydra-core, rootutils, ...). Lines
    # that fail are dropped silently; cosyvoice handles most of them with
    # try/except and the bridge tells us if any remaining import fails.
    $upstreamReq = Join-Path $repoDir "requirements.txt"
    if (Test-Path $upstreamReq) {
        Write-Host "  Tolerant pass over CosyVoice's upstream requirements (skipping bad pins)..." -ForegroundColor Gray
        $prevPref = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $skipPatterns = @(
            "grpcio", "deepspeed", "tensorrt", "pyworld",
            "WeTextProcessing", "wetext", "ttsfrd",
            # gradio/uvicorn/fastapi are server-side; we don't run that path.
            "gradio", "uvicorn", "fastapi"
        )
        foreach ($rawLine in (Get-Content $upstreamReq)) {
            $line = $rawLine.Trim()
            if (-not $line -or $line.StartsWith("#") -or $line.StartsWith("-")) { continue }
            if ($line.Contains("#")) { $line = ($line -split "#", 2)[0].Trim() }
            if (-not $line) { continue }
            $skip = $false
            foreach ($pat in $skipPatterns) {
                if ($line -match $pat) { $skip = $true; break }
            }
            if ($skip) { continue }
            & $PythonExe -m pip install --prefer-binary $line 2>&1 | Out-Null
        }
        $ErrorActionPreference = $prevPref
        Write-Ok "Upstream requirements pass complete (failures tolerated)"
    }

    # Download the 0.5B weights via HuggingFace. Using the python `huggingface_hub`
    # snapshot_download keeps us off ModelScope (which has flaky Windows support)
    # and yields a layout that matches what cosyvoice's loader expects.
    $modelDir = Join-Path $ModelsDir "CosyVoice2-0.5B"
    $modelMarker = Join-Path $modelDir "cosyvoice2.yaml"
    if (Test-Path $modelMarker) {
        Write-Ok "CosyVoice2-0.5B weights already downloaded"
        return
    }

    if (-not (Test-Path $ModelsDir)) {
        New-Item -Path $ModelsDir -ItemType Directory -Force | Out-Null
    }

    Write-Host "  Downloading CosyVoice2-0.5B weights from HuggingFace (~600 MB)..." -ForegroundColor Gray
    $downloadScript = @"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='FunAudioLLM/CosyVoice2-0.5B',
    local_dir=r'$modelDir',
)
print('Downloaded')
"@
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonExe -c $downloadScript 2>&1 | Out-Host
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $prevPref

    if ($exit -eq 0 -and (Test-Path $modelMarker)) {
        $size = [math]::Round((Get-ChildItem $modelDir -Recurse | Measure-Object Length -Sum).Sum / 1MB, 1)
        Write-Ok "CosyVoice2-0.5B downloaded (${size}MB)"
    } else {
        Write-Fail "CosyVoice2-0.5B download failed (exit $exit)"
        Write-Host "  Run manually: $PythonExe -c ""from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir=r'$modelDir')""" -ForegroundColor Yellow
    }
}

function Install-OpenVoice {
    param([string]$PythonExe)

    # OpenVoice v2 Tone Color Converter (ADR 0011). Repo cloned into
    # `third_party/OpenVoice/`; the bridge adds it to sys.path at boot.
    # Checkpoint files (~50 MB) live under `models/openvoice/converter/`.
    Write-Step "OpenVoice v2 Tone Color Converter"

    $thirdPartyDir = Join-Path $ProjectRoot "third_party"
    $repoDir = Join-Path $thirdPartyDir "OpenVoice"

    if (-not (Test-Path $thirdPartyDir)) {
        New-Item -Path $thirdPartyDir -ItemType Directory -Force | Out-Null
    }

    if (-not (Test-Path $repoDir)) {
        Write-Host "  Cloning OpenVoice repo..." -ForegroundColor Gray
        try {
            & git clone https://github.com/myshell-ai/OpenVoice.git $repoDir 2>&1 | Out-Host
        } catch {
            Write-Fail "git clone failed: $_"
            Write-Host "  Clone manually: git clone https://github.com/myshell-ai/OpenVoice.git $repoDir" -ForegroundColor Yellow
            return
        }
    } else {
        Write-Ok "OpenVoice repo already present at $repoDir"
    }

    # OpenVoice TCC v2 checkpoint. The myshell-ai/OpenVoiceV2 HF repo
    # ships the converter under `converter/` — we mirror that layout
    # so the bridge can find it under `models/openvoice/converter/`.
    $converterDir = Join-Path $ModelsDir "openvoice\converter"
    $ckptFile = Join-Path $converterDir "checkpoint.pth"
    $cfgFile = Join-Path $converterDir "config.json"

    if ((Test-Path $ckptFile) -and (Test-Path $cfgFile)) {
        $size = [math]::Round((Get-Item $ckptFile).Length / 1MB, 1)
        Write-Ok "OpenVoice checkpoint already present (${size} MB)"
        return
    }

    if (-not (Test-Path $converterDir)) {
        New-Item -Path $converterDir -ItemType Directory -Force | Out-Null
    }

    Write-Host "  Downloading OpenVoice v2 converter checkpoint (~50 MB)..." -ForegroundColor Gray
    $downloadScript = @"
from huggingface_hub import hf_hub_download
import shutil, os
repo = 'myshell-ai/OpenVoiceV2'
target = r'$converterDir'
for fname in ['converter/checkpoint.pth', 'converter/config.json']:
    p = hf_hub_download(repo_id=repo, filename=fname)
    dst = os.path.join(target, os.path.basename(fname))
    shutil.copy(p, dst)
print('Downloaded')
"@
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonExe -c $downloadScript 2>&1 | Out-Host
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $prevPref

    if ($exit -eq 0 -and (Test-Path $ckptFile) -and (Test-Path $cfgFile)) {
        $size = [math]::Round((Get-Item $ckptFile).Length / 1MB, 1)
        Write-Ok "OpenVoice checkpoint downloaded (${size} MB)"
    } else {
        Write-Skip "OpenVoice checkpoint download failed (voice conversion will be disabled at runtime)"
    }
}

function Install-SeparationModel {
    param([string]$PythonExe)

    # Sepformer-libri2mix from SpeechBrain (~120 MB) is the source
    # separator the speaker pipeline uses when `enable_separation = true`
    # in config.toml. Pre-fetched here so flipping the flag at runtime
    # doesn't pay a multi-second model-download stall on the first run.
    Write-Step "Sepformer-libri2mix (source separation, opt-in)"

    $cacheDir = Join-Path $env:USERPROFILE ".cache\speechbrain\sepformer-libri2mix"
    if (Test-Path "$cacheDir\masknet.ckpt") {
        Write-Ok "Sepformer already cached at $cacheDir"
        return
    }

    Write-Host "  Pre-downloading Sepformer (~120 MB)..." -ForegroundColor Gray
    $warmScript = @"
import os
os.environ.setdefault('SEPFORMER_PRETRAINED_DIR', r'$cacheDir')
from speechbrain.inference.separation import SepformerSeparation as Sep
Sep.from_hparams(
    source='speechbrain/sepformer-libri2mix',
    savedir=r'$cacheDir',
    run_opts={'device': 'cpu'},
)
print('Sepformer ready')
"@
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonExe -c $warmScript 2>&1 | Out-Host
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $prevPref

    if ($exit -eq 0) {
        Write-Ok "Sepformer cached at $cacheDir"
    } else {
        Write-Skip "Sepformer warm-up failed (will retry on first run)"
    }
}

function Install-DiarizationModel {
    param([string]$PythonExe)

    # SpeechBrain ECAPA-TDNN is the diarisation embedder that replaced
    # Resemblyzer. Pre-downloading it avoids a stall the first time the
    # diarization bridge starts up — speechbrain otherwise pulls weights
    # on-demand and the Rust pipeline times out waiting for "ready".
    Write-Step "SpeechBrain ECAPA-TDNN (diarisation embeddings)"

    $cacheDir = Join-Path $env:USERPROFILE ".cache\speechbrain\ecapa-tdnn"
    if (Test-Path "$cacheDir\embedding_model.ckpt") {
        Write-Ok "ECAPA-TDNN already cached at $cacheDir"
        return
    }

    Write-Host "  Pre-downloading ECAPA-TDNN (~22 MB)..." -ForegroundColor Gray
    $warmScript = @"
import os
os.environ.setdefault('SPEECHBRAIN_PRETRAINED_DIR', r'$cacheDir')
from speechbrain.inference.speaker import EncoderClassifier
EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir=r'$cacheDir',
    run_opts={'device': 'cpu'},
)
print('ECAPA-TDNN ready')
"@
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonExe -c $warmScript 2>&1 | Out-Host
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $prevPref

    if ($exit -eq 0) {
        Write-Ok "ECAPA-TDNN cached at $cacheDir"
    } else {
        Write-Skip "ECAPA-TDNN warm-up failed (will retry on first run)"
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
chunk_duration_ms = 280
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
Install-LlmModel
Install-KokoroModel
Install-DiarizationModel -PythonExe $pythonExe
Install-SeparationModel -PythonExe $pythonExe
Install-OpenVoice -PythonExe $pythonExe

# 6. Configuration
Configure-CargoConfig
Configure-DefaultConfig

# 7. Build and test
Build-RustProject
Test-RustProject

$launchChoice = Read-Host "`nLaunch Meeting Translator now? (y/N)"
$shouldLaunch = $launchChoice -eq "y" -or $launchChoice -eq "Y"

Show-Summary -LaunchApp $shouldLaunch
