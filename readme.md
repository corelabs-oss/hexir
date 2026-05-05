# MLP-MLIR

MLP-MLIR is a small research compiler built on LLVM MLIR for experimenting with
neural-network dialects, lowering pipelines, and early heterogeneous CPU/CUDA
partitioning.

The project currently builds a synthetic MLP-style program in C++, lowers custom
`mlp` ops through MLIR dialects, can JIT-run the CPU path, and can show an
experimental CUDA partition lowered into the MLIR GPU dialect.

## Requirements

- C++17 compiler
- CMake
- LLVM/MLIR build with MLIR libraries available
- This checkout currently expects MLIR from the local LLVM/MLIR installation
  configured in `CMakeLists.txt` or the existing `build/CMakeCache.txt`

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

For later rebuilds:

```bash
cd build
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

Inspect the linalg lowering:

```bash
./mlp_mlir -emit=mlir-linalg
```

Inspect CPU/CUDA graph partitioning:

```bash
./mlp_mlir -emit=mlir-hetero
```

Inspect CUDA partitions lowered to the MLIR GPU dialect:

```bash
./mlp_mlir -emit=mlir-gpu
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

## Tensor Print Lowering

`mlp.print` accepts tensors in the frontend-level IR. During linalg lowering,
tensor prints are converted through bufferization:

```mlir
%buffer = bufferization.to_buffer %tensor read_only
mlp.print %buffer : memref<...>
```

The LLVM lowering then expands `mlp.print memref<...>` into loops that call
`printf`.

## Source Layout

| Path | Purpose |
| --- | --- |
| `include/Ops.td` | ODS definitions for the `mlp` dialect ops |
| `src/Dialect.cpp` | Dialect and custom parser/printer implementation |
| `src/Builder.cpp` | Synthetic test program construction |
| `src/LowerToLinalg.cpp` | Lowering from `mlp` ops to linalg/arith/tensor/buffer ops |
| `src/Partition.cpp` | CPU/CUDA graph partition annotations |
| `src/LowerCudaToGpu.cpp` | CUDA-marked matmul lowering to `gpu.launch` |
| `src/LowerToLLVM.cpp` | CPU-side lowering to LLVM dialect and `printf` printing |
| `src/main.cpp` | Compiler driver and pass pipeline selection |
| `include/TargetInfo.h`, `src/TargetInfo.cpp` | Simple target support/preference table |
| `targets/` | Placeholder backend directories for future backend-specific pipelines |

## Current Limitations

- Input parsing is not wired up yet; the driver builds a synthetic module in
  C++.
- CUDA lowering currently stops at MLIR GPU dialect IR.
- There is no CUDA runtime execution path yet.
- Host/device memory transfer planning is not implemented.
- GPU lowering supports only static rank-2 matmul in the current demo shape.
- The target selection policy is rule-based, not a cost model.

## Roadmap

Near-term:

- Load `.mlir` input files instead of only building synthetic modules.
- Add tests for `mlp -> linalg`, partitioning, print lowering, and GPU lowering.
- Outline `gpu.launch` into `gpu.module` / `gpu.func`.
- Lower GPU modules to NVVM.
- Add explicit host/device allocation and copy operations.

Longer-term:

- Add fusion for `linear + bias + activation`.
- Add shape-aware CPU/GPU cost modeling.
- Add more neural-network ops such as softmax, layer norm, and attention.
- Add backend-specific lowering pipelines for CUDA, ROCm, CPU, and RISC-V.
