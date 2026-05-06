# MLP-MLIR

MLP-MLIR is a small research compiler built on LLVM MLIR for experimenting with
neural-network dialects, lowering pipelines, and early heterogeneous CPU/CUDA
partitioning.

The project currently builds a synthetic MLP-style program in C++, lowers custom
`mlp` ops through MLIR dialects, can JIT-run the CPU path, and can show an
experimental CUDA partition lowered into the MLIR GPU dialect.

## Requirements

- C++17
- CMake
- LLVM/MLIR build with MLIR libraries available

The code has been developed against a recent LLVM/MLIR tree. If your MLIR build
uses different library names, update the `target_link_libraries` section in
`CMakeLists.txt`.

## Build

From the repository root:

```bash
mkdir -p build
cd build
cmake ..
make
```

## Quick Start

Run the CPU JIT:

```bash
cd build
./mlp_mlir -emit=jit
```

Expected output from the current synthetic `linear -> relu -> print` program:

```text
8.000000 17.000000
12.000000 14.000000
```

Inspect the custom MLIR before lowering:

```bash
./mlp_mlir -emit=mlir
```


## Emit Modes

The main driver supports:

| Command | Meaning |
| --- | --- |
| `-emit=mlir` | Dump the initial MLIR module built by the C++ builder |
| `-emit=mlir-linalg` | Lower `mlp` ops to `linalg`, `arith`, tensor/bufferization support ops |
| `-emit=mlir-hetero` | Annotate graph ops with CPU/CUDA placement |
| `-emit=mlir-gpu` | Bufferize and lower CUDA-marked matmul to `gpu.launch` |
| `-emit=mlir-llvm` | Lower the CPU path to the LLVM dialect |
| `-emit=llvm` | Translate the LLVM dialect module to LLVM IR |
| `-emit=jit` | JIT-compile and run the CPU executable path |

Optimization can be enabled with:

```bash
./mlp_mlir -emit=jit -opt
```

## Current Pipeline

The current synthetic program is created in `src/Builder.cpp` and follows this
shape:

```text
mlp.constant
mlp.linear
mlp.relu
mlp.print
```

The lowering flow is:

```text
mlp dialect
  -> linalg / arith / tensor
  -> CPU/CUDA partition annotations
  -> bufferization
  -> optional gpu.launch for CUDA-marked matmul
  -> linalg/scf/cf/llvm for CPU JIT
```

The CUDA path is currently an IR-generation path. It produces MLIR GPU dialect
IR for inspection and future lowering, but it is not yet a complete executable
CUDA runtime path.

## Heterogeneous CPU/CUDA Support

The partitioner currently marks:

- `linalg.matmul` as `device = "cuda"`
- elementwise `linalg.generic` ReLU as `device = "cpu"`
- `mlp.print` as `device = "cpu"`

Example from:

```bash
./mlp_mlir -emit=mlir-hetero
```

```mlir
module attributes {mlp.targets = ["cpu", "cuda"]} {
  %0 = linalg.matmul {device = "cuda"} ...
  %1 = linalg.generic ... attrs =  {device = "cpu"} ...
  mlp.print ... {device = "cpu"}
}
```

After bufferization, the CUDA-marked matmul can be lowered to:

```mlir
gpu.launch ... {
  scf.for ...
    memref.load ...
    arith.mulf ...
    arith.addf ...
    memref.store ...
  gpu.terminator
} {device = "cuda"}
```

This is intentionally simple: it proves the compiler can partition and enter
the GPU dialect. Kernel outlining, `gpu.module`, NVVM lowering, CUDA runtime
launches, and host/device memory copies are future work.

