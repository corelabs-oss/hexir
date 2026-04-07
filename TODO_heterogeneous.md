# Heterogeneous Compilation Implementation TODO

## Steps:

- [x] 1. Update `include/Passes.h`: Add `createHeterogeneousLoweringPass()` declaration
- [x] 2. Create `src/HeterogeneousLowering.cpp`: Implement pass with MatmulToGpuPattern, ReluToCpuPattern
- [x] 3. Extend `include/TargetInfo.h` and `src/TargetInfo.cpp`: Add linalg.matmul/relu target support (GPU/CPU)
- [x] 4. Update `src/Partition.cpp`: Extend PartitionPass to annotate linalg ops with device attr
- [x] 5. Update `src/main.cpp`: Insert PartitionPass → HeterogeneousLoweringPass → GPU/CPU lowers after LowerToLinalg; add GPU/NVVM dialects
- [x] 6. Update `CMakeLists.txt`: Add MLIRGPUDialect/MLIRNVVMDialect/MLIRGPUTransforms, include new src/*.cpp (GLOB handles)
- [x] 7. Update `src/Builder.cpp`: Ensure synthetic module has linear → relu chain for testing
- [ ] 8. Build/test: `bash build.sh`, `./mlp_mlir -mlir-linalg`, verify matmul(gpu.launch) + relu(cpu), gpu.module present

Progress: All implementation steps complete. Building and testing verifies gpu.launch_func for matmul, cpu relu post-lowering.

## Testing Commands:
```bash
# View pipeline
./mlp_mlir --print-ir-after-all

# Emit heterogeneous MLIR
./mlp_mlir -emit=mlir-hetero
```

