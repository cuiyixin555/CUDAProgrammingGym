#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
UTILS_DIR="$REPO_ROOT/utils"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path/to/source.cpp>" >&2
  echo "Example: $0 0_Introduction/asyncAPI.cpp" >&2
  exit 1
fi

SRC="$1"
if [[ ! "$SRC" = /* ]]; then
  SRC="$SCRIPT_DIR/$SRC"
fi

if [[ ! -f "$SRC" ]]; then
  echo "Error: source file not found: $SRC" >&2
  exit 1
fi

NAME="$(basename "$SRC" .cpp)"
CATEGORY="$(basename "$(dirname "$SRC")")"
KERNEL_DIR="$REPO_ROOT/kernel/$CATEGORY"

mkdir -p "$BUILD_DIR"

EXTRA_LIBS=()
EXTRA_FLAGS=()
if grep -q 'nvrtc_helper.h' "$SRC" 2>/dev/null; then
  EXTRA_LIBS+=(-lnvrtc -lcuda)
fi
if grep -q '#include <omp.h>' "$SRC" 2>/dev/null || \
   grep -q '#include <omp.h>' "$KERNEL_DIR"/*.h 2>/dev/null; then
  EXTRA_FLAGS+=(-Xcompiler -fopenmp)
fi

echo "Compiling $SRC -> $BUILD_DIR/$NAME"
nvcc -std=c++17 -x cu -arch=native \
  -I"$UTILS_DIR" \
  -I"$KERNEL_DIR" \
  "${EXTRA_FLAGS[@]}" \
  -o "$BUILD_DIR/$NAME" \
  "$SRC" \
  "${EXTRA_LIBS[@]}"

echo "Done: $BUILD_DIR/$NAME"
