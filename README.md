# CUDAProgrammingGym

A hands-on code repository for CUDA beginners. Samples are organized by topic, with host code and kernels kept separate, and a shared `make.sh` script for building and running.

## Requirements

- An NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (includes `nvcc`)
- Linux or WSL2 (tested on WSL2 with CUDA 13.x)
- OpenMP support in your host compiler (for `cudaOpenMP`; usually included with `g++`)

Verify your setup:

```bash
nvcc --version
nvidia-smi
```

## Project Layout

```
CUDAProgrammingGym/
├── kernel/                  # Kernels (.h for compile-time, .cu for NVRTC/fatbin)
│   └── 0_Introduction/
│       ├── asyncAPIKernel.h
│       ├── clockKernel.h
│       ├── clock_kernel.cu          # NVRTC runtime kernel
│       ├── matrixMulKernel.h
│       ├── matrixMul_kernel.cu      # NVRTC runtime kernel
│       ├── matrixMulDrv_kernel.cu   # Driver API fatbin kernel
│       └── ...
├── tests/                   # Sample host programs
│   ├── make.sh              # Shared build script
│   ├── build/               # Build output (executables, fatbin; auto-generated)
│   └── 0_Introduction/
│       ├── asyncAPI.cpp
│       ├── clock.cpp
│       └── ...
└── utils/                   # CUDA helper headers (from NVIDIA cuda-samples)
    ├── helper_cuda.h
    ├── helper_cuda_drvapi.h
    ├── nvrtc_helper.h
    └── ...
```

Subdirectories under `tests/` and `kernel/` mirror each other. For example, `tests/0_Introduction/asyncAPI.cpp` picks up headers from `kernel/0_Introduction/`.

## Build & Run

All commands below assume you start from the `tests/` directory:

```bash
cd tests
```

General form:

```bash
./make.sh <subdir>/<source>.cpp
./build/<program_name>
```

`make.sh` automatically:

- Compiles `.cpp` files as CUDA source (`-x cu`)
- Selects the GPU architecture for the current machine (`-arch=native`)
- Adds include paths for `utils/` and the matching `kernel/<subdir>/`
- Links `-lnvrtc -lcuda` for NVRTC samples, `-lcuda` for Driver API samples
- Enables OpenMP (`-Xcompiler -fopenmp`) when needed
- Compiles `matrixMulDrv_kernel.cu` to a fatbin when building `matrixMulDrv`
- Writes executables to `tests/build/`

### Runtime API samples

Kernel is included at compile time via a header in `kernel/`.

```bash
# asyncAPI — CUDA events and asynchronous execution
./make.sh 0_Introduction/asyncAPI.cpp
./build/asyncAPI

# clock — per-block cycle measurement with clock()
./make.sh 0_Introduction/clock.cpp
./build/clock

# cudaOpenMP — multi-GPU with OpenMP CPU threads
./make.sh 0_Introduction/cudaOpenMP.cpp
./build/cudaOpenMP

# fp16ScalarProduct — FP16 dot product (requires SM 5.3+)
./make.sh 0_Introduction/fp16ScalarProduct.cpp
./build/fp16ScalarProduct

# matrixMul — tiled matrix multiplication (Runtime API)
./make.sh 0_Introduction/matrixMul.cpp
./build/matrixMul

# Optional matrix dimensions
./build/matrixMul -wA=512 -hA=512 -wB=256 -hB=512
./build/matrixMul -help
```

### NVRTC samples

Kernel source lives in `kernel/0_Introduction/*.cu` and is compiled **at runtime** by NVRTC. Run from `tests/` so the helper can locate the `.cu` file.

```bash
# clock_nvrtc — clock() sample via NVRTC + Driver API
./make.sh 0_Introduction/clock_nvrtc.cpp
./build/clock_nvrtc

# matrixMul_nvrtc — matrix multiplication via NVRTC + Driver API
./make.sh 0_Introduction/matrixMul_nvrtc.cpp
./build/matrixMul_nvrtc

# Optional matrix dimensions
./build/matrixMul_nvrtc -wA=512 -hA=512 -wB=256 -hB=512
```

### Driver API sample (fatbin)

`matrixMulDrv` loads a pre-built **fatbin** (GPU binary bundle). `make.sh` compiles `kernel/0_Introduction/matrixMulDrv_kernel.cu` into `tests/build/matrixMul_kernel64.fatbin` before linking the host program.

```bash
./make.sh 0_Introduction/matrixMulDrv.cpp
./build/matrixMulDrv
```

## Samples

| Program | API | Kernel source | Description |
|---------|-----|---------------|-------------|
| `asyncAPI` | Runtime | `asyncAPIKernel.h` | CUDA events for GPU timing; overlap CPU work with async GPU execution |
| `clock` | Runtime | `clockKernel.h` | Measure per-block execution cycles with `clock()` |
| `clock_nvrtc` | Driver + NVRTC | `clock_kernel.cu` | Same as `clock`, kernel compiled at runtime via NVRTC |
| `cudaOpenMP` | Runtime | `cudaOpenMPKernel.h` | One OpenMP thread per GPU for multi-GPU workloads |
| `fp16ScalarProduct` | Runtime | `fp16ScalarProductKernel.h` | FP16 scalar product (native operators vs intrinsics) |
| `matrixMul` | Runtime | `matrixMulKernel.h` | Tiled matrix multiplication with shared memory |
| `matrixMul_nvrtc` | Driver + NVRTC | `matrixMul_kernel.cu` | Same as `matrixMul`, kernel compiled at runtime via NVRTC |
| `matrixMulDrv` | Driver + fatbin | `matrixMulDrv_kernel.cu` | Matrix multiplication via Driver API and pre-built fatbin |

## Adding a New Sample

1. Add a kernel under `kernel/<topic>/` (`.h` for Runtime API, or `.cu` for NVRTC/fatbin)
2. Add a host program under `tests/<topic>/` (e.g. `mySample.cpp`)
3. Build and run:

```bash
cd tests
./make.sh <topic>/mySample.cpp
./build/mySample
```

## Notes

- Helper headers in `utils/` are adapted from the [NVIDIA cuda-samples](https://github.com/NVIDIA/cuda-samples) Common utilities.
- Some samples are based on official NVIDIA CUDA Samples, with kernels split into separate files for easier reading and practice.
- **NVRTC** (`*_nvrtc`): compiles `.cu` from `kernel/` at runtime; needs `libnvrtc` and `libcuda`.
- **fatbin** (`matrixMulDrv`): compiles `.cu` to a binary bundle at build time; host loads it with `cuModuleLoadData`.
- **fp16ScalarProduct** exits with code 2 if the GPU is below SM 5.3.

## License

MIT License — see [LICENSE](LICENSE).
