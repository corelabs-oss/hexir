
#include "Jit.h"
#include "Dialect.h"
#include "Passes.h"

#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
/**
 * runJit
 *
 * JIT-compiles and executes the "main" function from the provided MLIR module.
 *
 * Behavior:
 *  - Initializes the LLVM native target and native target assembly printer.
 *  - Registers MLIR-to-LLVM IR translations for both builtin and LLVM dialects
 *    on the module's MLIRContext (must be done prior to JIT compilation).
 *  - Constructs an optimization transformer pipeline via
 * makeOptimizingTransformer. The pipeline's optimization level is controlled by
 * the local `enableOpt` flag (currently disabled by default).
 *  - Creates an mlir::ExecutionEngine using the module and the chosen
 * transformer. The creation is asserted to succeed; if it fails the program
 * will terminate due to the assert.
 *  - Invokes the JIT-compiled function named "main" with no arguments using
 *    invokePacked(). If invocation fails, an error message is written to
 *    llvm::errs() and the function returns -1.
 *
 * Parameters:
 *  - module: mlir::ModuleOp representing the MLIR module to JIT-compile and
 * run. The module's MLIRContext (and any required dialect registrations) must
 * remain valid for the duration of this call.
 *
 * Return value:
 *  - 0 on successful invocation of "main".
 *  - -1 if the JIT invocation fails (errors are emitted to llvm::errs()).
 *
 * Notes:
 *  - The function currently asserts on failure to construct the ExecutionEngine
 *    rather than returning an error code; callers should be aware that a failed
 *    creation will abort the process.
 *  - To enable compilation optimizations, set `enableOpt` to a truthy value.
 */
namespace jit {
int runJit(mlir::ModuleOp module) {

  int enableOpt = false;

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before can
  // JIT-compile.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

} // namespace jit
