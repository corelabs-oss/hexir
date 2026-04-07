# MLP MLIR Bufferization Fix Task

## Steps to Complete:

### 1. Create/Update TODO.md ✅
   - [x] Created this TODO.md with plan breakdown.

### 2. Edit src/main.cpp ✅
   - [x] Move BufferizableOpInterfaceExternalModels registrations after context.loadAllAvailableDialects()
   - [x] Remove duplicate registry.insert for BufferizationDialect
   - [x] Clean up commented/unused bufferization code
   - [x] Ensure OneShotBufferize options allow full bufferization (e.g., bufferizeUnknownType=true if needed)

### 3. Rebuild the project ✅
   - [x] cd scripts
   - [x] ./build.sh (clean rebuild successful)

### 4. Test the fix ✅
   - [x] Run: ./mlp_mlir --emit=mlir-linalg
   - [x] Verify no BufferizableOpInterface error (fixed!)
   - [x] Check MLIR output for correct linalg lowering and bufferization (tensors → memrefs, affine loops)

### 5. Additional optimizations (if time)
   - [ ] Add buffer deallocation passes properly
   - [ ] Test other emits (mlir-affine, llvm)

## Current Status: Ready for src/main.cpp edits

