// Translation to LLVM IR proper.
// RUN: %hexir -emit=llvm 2>&1 | FileCheck %s

// CHECK: target triple
// CHECK: declare {{.*}}i32 @printf(ptr, ...)
// CHECK: define {{.*}}void @main()
// CHECK: call i32 (ptr, ...) @printf
