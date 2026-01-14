#ifndef BUILDER_H
#define BUILDER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

// namespace mlir {
// class MLIRContext;
// class ModuleOp;
// } // namespace mlir

using namespace mlir;

namespace builder1 {

// Basic examples
// mlir::func::FuncOp createMainFunction(mlir::MLIRContext &ctx,
//                                       mlir::ModuleOp module);

func::FuncOp createMainFunction(MLIRContext &ctx, ModuleOp module);

// mlir::func::FuncOp createAddFunction(mlir::MLIRContext &ctx,
//                                      mlir::ModuleOp module);

// mlir::func::FuncOp createMulFunction(mlir::MLIRContext &ctx,
//                                      mlir::ModuleOp module);

// // Toy / MLP examples
// mlir::func::FuncOp createMLPAddFunction(mlir::MLIRContext &ctx,
//                                         mlir::ModuleOp module);

// mlir::func::FuncOp createMLPAddTOSAFunction(mlir::MLIRContext &ctx,
//                                             mlir::ModuleOp module);

// mlir::func::FuncOp createMLPReluFunction(mlir::MLIRContext &ctx,
//                                          mlir::ModuleOp module);

// (Optional / commented-out in cpp)
// mlir::func::FuncOp createMLPTESTFunction(mlir::MLIRContext &ctx,
//                                          mlir::ModuleOp module);

} // namespace builder

#endif // TOY_BUILDER_H
