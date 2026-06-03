# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Hexir is a research compiler built on LLVM MLIR. It compiles a synthetic MLP (Multi-Layer Perceptron) program — built programmatically, not parsed from a file — through a custom `hexir` dialect, with heterogeneous CPU/CUDA partitioning, down to LLVM IR / JIT execution.

## Build

Requires a local LLVM/MLIR build. `LLVM_DIR` and `MLIR_DIR` are **hardcoded** in `CMakeLists.txt` to `/home/hamza/Repos/llvm-project/build/lib/cmake/{llvm,mlir}`.

```bash
# First time / after CMakeLists changes
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Incremental rebuild (build.sh just does `cd build && make`)
cd build && make -j$(nproc)
```

`src/*.cpp` and `src/Dialects/*.cpp` are globbed with `CONFIGURE_DEPENDS`, so new source files are picked up without editing CMakeLists. New passes still need a declaration in `include/Passes.h` and wiring into the pipeline in `src/main.cpp`.

## Testing

Lit/FileCheck tests live in `test/` — one test per pipeline stage, plus an end-to-end JIT test:

```bash
cd build && make check-hexir    # run the full suite
lit -v test                   # same, from the repo root
lit -v test --filter=jit      # run a single test
```

`test/lit.cfg.py` is standalone (no CMake-configured site config): it finds the binary via `--param build_dir=`, `$HEXIR_BUILD_DIR`, or `<repo>/build`, and FileCheck from PATH or `/usr/lib/llvm-*/bin`. The binary ignores input files (the program is synthetic, from `Builder.cpp`), so tests are RUN-line-only `.mlir` files. **All `-emit=mlir*` and `-emit=llvm` output goes to stderr** (`module->dump()`), hence `2>&1` in every RUN line; only JIT execution prints to stdout. If you change `Builder.cpp` or any lowering pass, expect to update CHECK lines.

You can also inspect any stage manually:

```bash
cd build
./hexir -emit=jit          # end-to-end; expected output:
                              # 8.000000 17.000000
                              # 12.000000 14.000000
./hexir -emit=mlir         # initial hexir-dialect IR
./hexir -emit=mlir-linalg  # after lowering to linalg/arith/tensor
./hexir -emit=mlir-hetero  # after CPU/CUDA partitioning + ls_cpu/ls_gpu materialization
./hexir -emit=mlir-gpu     # CUDA partitions lowered to gpu.launch
./hexir -emit=mlir-llvm    # LLVM dialect
./hexir -emit=llvm         # LLVM IR text
./hexir -emit=jit -opt     # with optimizations
./hexir --print-ir-after-all  # debug pass pipeline
```

## Architecture

### Pipeline (assembled in `src/main.cpp::loadAndProcessMLIR`)

The `Action` enum in main.cpp is **ordered**; pipeline stages are gated by comparisons like `emitAction >= Action::DumpMLIRLinalg`, so each emit mode runs every stage up to its level. Some emit modes (`mlir-hetero`, `mlir-gpu`) early-return rather than continuing.

```
Builder.cpp (synthetic IR: hexir.constant → hexir.linear → hexir.relu → hexir.print)
  ↓ canonicalize, ShapeInferencePass, CSE
  ↓ LowerToLinalgPass            (hexir → linalg/arith/tensor)
  ↓ PartitionPass                (annotates ops with {device = "cpu"|"cuda"} attr)
  ↓ [mlir-hetero only] MaterializeLSTargetsPass  (rewrites linalg ops to ls_cpu/ls_gpu dialect ops)
  ↓ One-shot bufferization       (tensor → memref)
  ↓ [mlir-gpu only] CudaGpuLoweringPass  (cuda-annotated linalg → gpu.launch)
  ↓ linalg → loops → SCF → CF
  ↓ LowerToLLVMPass → LLVM IR / JIT (ExecutionEngine, invokes "main")
```

There is no real frontend: `loadMLIR()` ignores the input file and calls `builder::createMLPLinearFunction()` from `src/Builder.cpp`. To change the test program, edit Builder.cpp.

### Partitioning system

- `src/TargetInfo.cpp` — `TargetSupport` singleton: registry mapping op names → supported targets + preferred target. Register new op/target support here.
- `src/Partition.cpp` — `PartitionPass` walks the module, asks `TargetSupport` for each op's preferred device, sets the `device` string attribute (falls back to "cpu" if unsupported), and tags the module with `hexir.targets = ["cpu", "cuda"]`.
- `src/MaterializeLSTargets.cpp` — converts device-annotated linalg ops to ops in the lightweight `ls_cpu`/`ls_gpu` demo dialects.
- `src/LowerCudaToGpu.cpp` — lowers cuda-annotated ops to the MLIR GPU dialect. GPU IR is for inspection only; only the CPU path actually executes (JIT).

### TableGen

Dialect definitions live in `include/Dialects/` and are processed via `add_subdirectory(include)`:
- `Ops.td` — the `hexir` dialect ops
- `LSDialects.td` — `ls_cpu`/`ls_gpu` dialects
- `ShapeInferenceInterface.td` — op interface used by `ShapeInferencePass`
- `src/HexirCombine.td` — DRR rewrite patterns (generates `HexirCombine.inc`)

Generated `.inc` files land in `build/include/...`; after editing a `.td` file, just rebuild.

### Not compiled

`targets/` (cpu/gpu/metal/riscv/rocm backends) exists but is excluded from the build (commented out in CMakeLists.txt). Don't expect changes there to take effect.
