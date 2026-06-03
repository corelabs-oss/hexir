// End-to-end: JIT-compile and execute. Checks the numeric result of
// linear(matmul) -> relu -> print, with and without optimizations.
// RUN: %hexir -emit=jit 2>&1 | FileCheck %s
// RUN: %hexir -emit=jit -opt 2>&1 | FileCheck %s

// CHECK:      8.000000 17.000000
// CHECK-NEXT: 12.000000 14.000000
