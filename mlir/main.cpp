//===- main.cpp - The Mlp Compiler ----------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Mlp compiler.
//
//===----------------------------------------------------------------------===//

#include "Builder.h"
#include "Dialect.h"
#include "Jit.h"
#include "Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

// using namespace mlp;
// namespace cl = llvm::cl;

#ifdef PRINT
#define PRINT
#endif
// #define PRINT

using namespace mlir;
using namespace builder;
using namespace jit;

void registerLLVMTranslations(mlir::MLIRContext &context) {
  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);
}

int main() {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  mlir::DialectRegistry registry;

  registry
      .insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
              mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
              mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
              mlir::affine::AffineDialect /*, mlir::math::MathDialect*/,
              mlir::LLVM::LLVMDialect /*, mlir::cf::ControlFlowDialect*/,
              mlir::tosa::TosaDialect, bufferization::BufferizationDialect>();

  mlir::func::registerAllExtensions(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  // mlir::registerAllDialects(registry);

  mlir::MLIRContext ctx(registry);
  ctx.getOrLoadDialect<mlir::mlp::MLPDialect>();

  ctx.loadAllAvailableDialects();

  registry.insert<mlir::mlp::MLPDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();

  registerLLVMTranslations(ctx);
  // Correct way to create a module:
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  // createMainFunction(ctx, *module);
  //  createAddFunction(ctx, *module);
  // createMulFunction(ctx, *module);
  // createMLPAddFunction(ctx, *module);

  // createMLPAddTOSAFunction(ctx, *module);
  // createMLPReluFunction(ctx, *module);
  createMLPTESTFunction(ctx, *module);

  llvm::outs()
      << "\n===================== mlir dialect ========================\n";

  module->print(llvm::outs());
  llvm::outs() << "\n";


  // run pass pipeline (use ctx variable from your main)
  PassManager pm(&ctx);

  if (1) {
    //  Inline all functions into main and then delete them.
    // pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.

    mlir::OpPassManager &optPM = pm.nest<mlir::mlp::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::mlp::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (0) {
    // Partially lower the mlp dialect.
    pm.addPass(mlir::mlp::createLowerToLinalgPass());

    // mlir::bufferization::OneShotBufferizationOptions options;
    // options.allowReturnAllocsFromLoops = true;
    // pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
    //  //pm.addPass(mlir::bufferization::createBufferDeallocationPass());
    // pm.addPass(mlir::createConvertLinalgToLoopsPass());
    //  pm.addPass(mlir::createConvertSCFToCFPass());
    //   ------------------------------------------------------------
    //   pm.addPass(mlir::createConvertArithToLLVMPass());
    //   pm.addPass(mlir::createConvertMemRefToLLVMPass());
    //   pm.addPass(mlir::createConvertFuncToLLVMPass());
    //   pm.addPass(mlir::tosa::createTosaToArith());

    // // Add a few cleanups post lowering.
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    //    Add optimizations if enabled.
    if (auto opt = true; opt) {
      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }
  }

  if (1) {
    // Partially lower the mlp dialect.
    pm.addPass(mlir::mlp::createLowerToAffinePass());
    // Add a few cleanups post lowering.
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
    // Add optimizations if enabled.
    if (auto opt = true; opt) {
      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }
  }

  if (0) {

    // Finish lowering the mlp IR to the LLVM dialect.
    pm.addPass(mlir::mlp::createLowerToLLVMPass());
    // This is necessary to have line tables emitted and basic
    // debugger working. In the future we will add proper debug information
    // emission directly from our frontend.
    pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }

  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Lowering pipeline failed!\n";
    return 1;
  }
  llvm::outs()
      << "\n===================== lowered dialect =====================\n";
  module->print(llvm::outs());

  // translate and print LLVM IR
  if (0) {

    llvm::LLVMContext llvmCtx;
    std::unique_ptr<llvm::Module> llvmModule =
        mlir::translateModuleToLLVMIR(*module, llvmCtx);

    if (!llvmModule) {
      llvm::errs() << "translateModuleToLLVMIR failed\n";
      return 1;
    }
    llvm::outs() << "\n=== LLVM IR ===\n";
    llvmModule->dump();

    if (llvm::verifyModule(*llvmModule, &llvm::errs())) {
      llvm::errs() << "IR verification failed\n";
      return 0;
    }
  }

  // Otherwise, we must be running the jit.
  //  if (emitAction == Action::RunJIT)
  // runJit(*module);
  std::cout << std::endl;
  // llvm::outs() << "\n=== MLIR TEST dialect ===\n";
  // module->print(llvm::outs());
  return 0;
}