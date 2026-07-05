# CUDAProgrammingGym

A hands-on code repository for CUDA beginners. Samples are organized by topic, with host code and kernels kept separate, and a shared `make.sh` script for building and running.

## Requirements

- An NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (includes `nvcc`)
- Linux or WSL2 (tested on WSL2 with CUDA 13.x)

Verify your setup:

```bash
nvcc --version
nvidia-smi
```

## Project Layout

```
CUDAProgrammingGym/
├── kernel/                  # CUDA kernel headers
│   └── 0_Introduction/
│       └── asyncAPIKernel.h
├── tests/                   # Sample host programs
│   ├── make.sh              # Shared build script
│   ├── build/               # Build output (auto-generated)
│   └── 0_Introduction/
│       ├── asyncAPI.cpp
│       └── clock.cpp
└── utils/                   # CUDA helper headers (from NVIDIA cuda-samples)
    ├── helper_cuda.h
    ├── helper_functions.h
    ├── helper_string.h
    ├── helper_timer.h
    └── ...
```

Subdirectories under `tests/` and `kernel/` mirror each other. For example, `tests/0_Introduction/asyncAPI.cpp` picks up headers from `kernel/0_Introduction/`.

## Build & Run

From the `tests/` directory:

```bash
cd tests
./make.sh 0_Introduction/asyncAPI.cpp
./build/asyncAPI
```

General form:

```bash
cd tests
./make.sh <subdir>/<source>.cpp
./build/<program_name>
```

`make.sh` automatically:

- Compiles `.cpp` files as CUDA source (`-x cu`)
- Selects the GPU architecture for the current machine (`-arch=native`)
- Adds include paths for `utils/` and the matching `kernel/<subdir>/`
- Writes executables to `tests/build/`

## Samples

| Program | Description |
|---------|-------------|
| `asyncAPI` | CUDA events for GPU timing and overlapping CPU work with asynchronous GPU execution |
| `clock` | Using `clock()` to measure per-block execution cycles inside a kernel |

```bash
# asyncAPI
./make.sh 0_Introduction/asyncAPI.cpp
./build/asyncAPI

# clock
./make.sh 0_Introduction/clock.cpp
./build/clock
```

## Adding a New Sample

1. Add a kernel header under `kernel/<topic>/` (e.g. `mySampleKernel.h`)
2. Add a host program under `tests/<topic>/` (e.g. `mySample.cpp`) and `#include` the kernel header
3. Build and run:

```bash
cd tests
./make.sh <topic>/mySample.cpp
./build/mySample
```

## Notes

- Helper headers in `utils/` are adapted from the [NVIDIA cuda-samples](https://github.com/NVIDIA/cuda-samples) Common utilities (device selection, error checking, timers, etc.).
- Some samples are based on official NVIDIA CUDA Samples, with kernels split into separate headers for easier reading and practice.

## License

MIT License — see [LICENSE](LICENSE).
