// LowerToLLVMPass: everything (CPU path) is lowered to the LLVM dialect.
// RUN: %hexir -emit=mlir-llvm 2>&1 | FileCheck %s

// CHECK:       llvm.func @printf(!llvm.ptr, ...)
// CHECK:       llvm.func @main()
// CHECK-NOT:   linalg.
// CHECK-NOT:   scf.
// CHECK-NOT:   hexir.
