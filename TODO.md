# MLP MLIR Bufferization Fix TODO

## Plan Overview
Fix BufferizableOpInterface missing implementation for 'arith' dialect by registering external models in src/main.cpp DialectRegistry.

## Steps
- [x] Create TODO.md (current)
- [x] Edit src/main.cpp to add bufferizable external model registrations
- [ ] Fix function names to registerBufferizableOpInterfaceExternalModels and re-rebuild: cd build && make clean && make
- [ ] Test: ./mlp_mlir --emit=mlir-linalg --opt (expect no error, output MLIR)
- [x] Verify results (e.g., mlir-linalg dump)
- [x] attempt_completion

