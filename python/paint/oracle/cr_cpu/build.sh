#!/usr/bin/env bash
# Build the torch-free custom_rasterizer CPU twin as a plain C-ABI shared library (the
# parity oracle), loaded from Python via ctypes. No Python/pybind11 headers — zero ABI coupling.
# Run via: bash oracle/cr_cpu/build.sh
set -euo pipefail
cd "$(dirname "$0")"

OUT="libcr_cpu_oracle.dylib"
echo "Building ${OUT} ..."
# -ffp-contract=off: disable FMA contraction of (a*b - c*d) so the signed-area / barycentric
# math is bit-reproducible across the C++ oracle, numpy, and the future Metal kernel. Without
# it, clang fuses multiply-adds and a few exact-edge pixels flip vs non-FMA implementations.
c++ -O3 -Wall -std=c++14 -ffp-contract=off -dynamiclib -fPIC cr_cpu.cpp -o "${OUT}"
echo "Built $(pwd)/${OUT}"
