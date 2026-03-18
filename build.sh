#!/bin/bash
# Build script for meeting-translator with whisper-rs CUDA
# Required: MSVC Build Tools 2022, CMake, CUDA Toolkit 12.4
#
# First-time setup (admin PowerShell):
#   Copy-Item "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\extras\visual_studio_integration\MSBuildExtensions\*" "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations\" -Force

MSVC_INC="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/include"
UCRT_INC="C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/ucrt"

export BINDGEN_EXTRA_CLANG_ARGS="-I\"${MSVC_INC}\" -I\"${UCRT_INC}\""
export CMAKE_GENERATOR="Visual Studio 17 2022"

cargo build "$@"
