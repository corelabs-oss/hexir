# MLP-MLIR

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-3.13.4+-blue.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

MLP-MLIR is a research compiler built on LLVM MLIR for experimenting with neural-network dialects, lowering pipelines, and heterogeneous CPU/GPU partitioning. It demonstrates end-to-end compilation of synthetic MLP (Multi-Layer Perceptron) programs through custom MLIR dialects to executable code.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Build](#build)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Emit Modes](#emit-modes)
- [Heterogeneous Support](#heterogeneous-support)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Custom MLIR Dialect**: Defines `mlp` operations for neural network primitives
- **Progressive Lowering**: Multi-stage compilation pipeline from high-level ops to LLVM IR
- **Heterogeneous Partitioning**: Automatic CPU/CUDA placement for operations
- **JIT Compilation**: Runtime code generation and execution for CPU path
- **GPU Dialect Generation**: CUDA operations lowered to MLIR GPU dialect
- **Extensible Backend System**: Support for CPU, CUDA, Metal, ROCm, and RISC-V targets

## Architecture

The compiler follows a layered architecture:

```
High-Level IR (mlp dialect)
    ↓ Lowering
Linalg/Arith/Tensor Operations
    ↓ Partitioning
CPU/CUDA Annotated Operations
    ↓ Bufferization
Memory Operations + GPU Launch
    ↓ Target Lowering
LLVM IR / GPU Kernels
```

### Key Components

- **Dialect Definition** (`include/Ops.td`): TableGen definitions for MLP operations
- **Builder** (`src/Builder.cpp`): Constructs synthetic neural network programs
- **Passes**: Lowering and optimization passes in `src/`
- **Backends** (`targets/`): Target-specific code generation
- **JIT Runtime** (`src/Jit.cpp`): Execution engine for CPU path

## Requirements

- **C++17** compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake** 3.13.4 or later
- **LLVM/MLIR** development build with the following components:
  - MLIR Core libraries
  - LLVM Core libraries
  - TableGen
  - OrcJIT

### LLVM/MLIR Setup

This project requires a full LLVM/MLIR build. The code has been developed against recent LLVM trunk. If your MLIR installation uses different library names, update the `target_link_libraries` in `CMakeLists.txt`.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd mlp_mlir
   ```

2. **Set up LLVM/MLIR environment:**
   Ensure `LLVM_DIR` and `MLIR_DIR` point to your LLVM build directory in `CMakeLists.txt`:
   ```cmake
   set(LLVM_DIR "/path/to/llvm/build/lib/cmake/llvm")
   set(MLIR_DIR "/path/to/llvm/build/lib/cmake/mlir")
   ```

## Build

### Initial Build
```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Rebuild
```bash
cd build
make -j$(nproc)
```

### Clean Build
```bash
cd build
make clean
make -j$(nproc)
```

## Quick Start

After building, run the CPU JIT to execute a synthetic MLP program:

```bash
cd build
./mlp_mlir -emit=jit
```

**Expected Output:**
```
8.000000 17.000000
12.000000 14.000000
```

This executes a `linear → relu → print` neural network computation.

## Usage

The main executable `mlp_mlir` supports various emit modes for inspecting the compilation pipeline:

### Basic Commands

```bash
# Inspect initial MLIR
./mlp_mlir -emit=mlir

# View linalg lowering
./mlp_mlir -emit=mlir-linalg

# See CPU/CUDA partitioning
./mlp_mlir -emit=mlir-hetero

# Inspect GPU dialect lowering
./mlp_mlir -emit=mlir-gpu

# Generate LLVM IR
./mlp_mlir -emit=llvm

# JIT compile and run
./mlp_mlir -emit=jit
```

### With Optimizations

Enable MLIR optimizations:

```bash
./mlp_mlir -emit=jit -opt
```

## Emit Modes

| Mode | Description |
|------|-------------|
| `mlir` | Initial MLIR module with custom `mlp` operations |
| `mlir-linalg` | Lowered to `linalg`, `arith`, and tensor operations |
| `mlir-hetero` | Operations annotated with CPU/CUDA device placement |
| `mlir-gpu` | CUDA operations lowered to `gpu.launch` kernels |
| `mlir-llvm` | CPU path lowered to LLVM dialect |
| `llvm` | Translated to LLVM IR text format |
| `jit` | JIT-compiled and executed on CPU |

## Heterogeneous Support

MLP-MLIR demonstrates early heterogeneous compilation by partitioning operations across CPU and CUDA devices:

- **CUDA Partition**: `linalg.matmul` operations marked for GPU execution
- **CPU Partition**: Element-wise operations (`relu`) and I/O (`print`) on CPU

### Example Partitioned IR

```mlir
module attributes {mlp.targets = ["cpu", "cuda"]} {
  %0 = linalg.matmul {device = "cuda"} ...  // GPU matrix multiplication
  %1 = linalg.generic {device = "cpu"} ...   // CPU element-wise ReLU
  mlp.print {device = "cpu"} ...             // CPU output
}
```

### GPU Lowering

CUDA-marked operations are lowered to MLIR GPU dialect:

```mlir
gpu.launch ... {
  scf.for ... {
    // Matrix multiplication kernel
    memref.load ...
    arith.mulf ... arith.addf ...
    memref.store ...
  }
  gpu.terminator
} {device = "cuda"}
```

**Note**: The CUDA path currently generates MLIR GPU IR for inspection. Full CUDA runtime integration (kernel launching, memory transfers) is planned for future development.

## Current Pipeline

The synthetic program (`src/Builder.cpp`) creates a simple MLP:

```text
mlp.constant  →  mlp.linear  →  mlp.relu  →  mlp.print
```

**Lowering Flow:**
```
mlp dialect
  ↓ Shape inference, canonicalization
linalg + arith + tensor operations
  ↓ Partitioning pass
CPU/CUDA placement annotations
  ↓ Bufferization
memref operations + gpu.launch
  ↓ Target-specific lowering
LLVM IR (CPU) / GPU kernels (CUDA)
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Commit** changes: `git commit -am 'Add your feature'`
4. **Push** to the branch: `git push origin feature/your-feature`
5. **Submit** a Pull Request

### Development Setup

- Use the provided `build.sh` script for consistent builds
- Run tests with `./mlp_mlir -emit=jit` to verify functionality
- Follow the existing code style and naming conventions

### Areas for Contribution

- Complete CUDA runtime integration
- Add more neural network operations
- Implement additional backends (Vulkan, OpenCL)
- Performance optimizations and benchmarking
- Documentation improvements

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [LLVM MLIR](https://mlir.llvm.org/) infrastructure
- Inspired by research in heterogeneous compilation for machine learning
- Part of ongoing work in compiler design for neural networks

---

**Note**: This is research software under active development. APIs and behavior may change without notice.