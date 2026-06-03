
#include "Builder.h"
#include "Dialect.h"
#include "Jit.h"
#include "LSDialects.h"
#include "Passes.h"

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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

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

using namespace hexir;
using namespace builder;
namespace cl = llvm::cl;

using namespace hexir;
using namespace builder;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input hexir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { HEXIR, MLIR };
} // namespace
static cl::opt<enum InputType>
    inputType("x", cl::init(HEXIR),
              cl::desc("Decided the kind of output desired"),
              cl::values(clEnumValN(HEXIR, "hexir",
                                    "load the input file as a hexir source.")),
              cl::values(clEnumValN(MLIR, "mlir",
                                    "load the input file as an MLIR file")));

namespace {
enum Action {
  None,
  DumpAST,
  DumpMLIR,
  DumpMLIRAffine,
  DumpMLIRLinalg,
  DumpMLIRHetero,
  DumpMLIRGPU,
  DumpMLIRLLVM,
  DumpLLVMIR,
  RunJIT
};
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                          "output the MLIR dump after affine lowering")),
    cl::values(clEnumValN(DumpMLIRLinalg, "mlir-linalg",
                          "output the MLIR dump after linalg lowering")),
    cl::values(clEnumValN(DumpMLIRHetero, "mlir-hetero",
                          "output MLIR after CPU/CUDA partitioning")),
    cl::values(clEnumValN(DumpMLIRGPU, "mlir-gpu",
                          "output MLIR after lowering CUDA partitions to GPU")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(
        clEnumValN(RunJIT, "jit",
                   "JIT the code and run it by invoking the main function")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

static int loadMLIR(mlir::MLIRContext &context,
                    mlir::OwningOpRef<mlir::ModuleOp> &module) {

  // CREATE MODULE FIRST
  module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::hexir::HexirDialect>();

  createMLPLinearFunction(context, *module);
  // createMLPAddFunction(context, *module);
  // createMLPReluFunction(context, *module);
  return 0;
}

static int loadAndProcessMLIR(mlir::MLIRContext &context,
                              mlir::OwningOpRef<mlir::ModuleOp> &module) {
  if (int error = loadMLIR(context, module))
    return error;

  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // Check to see what granularity of MLIR we are compiling to.
  bool isLoweringToLinalg = emitAction >= Action::DumpMLIRLinalg;
  bool isPartitioningForHetero = emitAction >= Action::DumpMLIRHetero;
  bool isLoweringCudaToGpu = emitAction == Action::DumpMLIRGPU;
  bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

  if (enableOpt || isLoweringToLinalg) {
    // Inline all functions into main and then delete them.
    // pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    mlir::OpPassManager &optPM = pm.nest<mlir::hexir::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::hexir::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (isLoweringToLinalg) {
    pm.addPass(mlir::hexir::createLowerToLinalgPass());

    if (emitAction == Action::DumpMLIRLinalg) {
      if (mlir::failed(pm.run(*module)))
        return 4;
      return 0;
    }

    if (isPartitioningForHetero)
      pm.addPass(mlir::hexir::createPartitionPass());

    if (emitAction == Action::DumpMLIRHetero) {
      pm.addPass(mlir::hexir::createMaterializeLSTargetsPass());
      if (mlir::failed(pm.run(*module)))
        return 4;
      return 0;
    }

    // Tensor → MemRef
    pm.addPass(mlir::bufferization::createOneShotBufferizePass());
    pm.addPass(
        mlir::bufferization::createBufferDeallocationSimplificationPass());

    if (isLoweringCudaToGpu)
      pm.addPass(mlir::hexir::createCudaGpuLoweringPass());

    if (emitAction == Action::DumpMLIRGPU) {
      if (mlir::failed(pm.run(*module)))
        return 4;
      return 0;
    }

    // Linalg → loops
    pm.addPass(mlir::createConvertLinalgToLoopsPass());

    // SCF → CFG
    pm.addPass(mlir::createSCFToControlFlowPass());

    // LLVM lowering
    // pm.addPass(mlir::createConvertArithToLLVMPass());
    // pm.addPass(mlir::createConvertMemRefToLLVMPass());
    // pm.addPass(mlir::createConvertFuncToLLVMPass());
  }

  //   if (isLoweringToLinalg)
  //   {
  //     // Partially lower the hexir dialect.
  //     pm.addPass(mlir::hexir::createLowerToLinalgPass());

  //     // Add a few cleanups post lowering.
  //     // mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  //     // optPM.addPass(mlir::createCanonicalizerPass());
  //     // optPM.addPass(mlir::createCSEPass());

  //     // Add optimizations if enabled.
  //     if (enableOpt)
  //     {
  //       // optPM.addPass(mlir::affine::createLoopFusionPass());
  //       // optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
  //     }
  //     // 1. One-Shot Bufferization (Converts Tensor constants to MemRef
  //     globals)

  //     mlir::bufferization::OneShotBufferizationOptions options;
  //     options.allowReturnAllocsFromLoops = true;
  //     //pm.addPass(mlir::bufferization::createOneShotBufferizePass());
  //     //  //pm.addPass(mlir::bufferization::createBufferDeallocationPass());
  //     // pm.addPass(mlir::createConvertLinalgToLoopsPass());
  //     //  pm.addPass(mlir::createConvertSCFToCFPass());
  //     //   ------------------------------------------------------------
  //     //   pm.addPass(mlir::createConvertArithToLLVMPass());
  //     //   pm.addPass(mlir::createConvertMemRefToLLVMPass());
  //     //   pm.addPass(mlir::createConvertFuncToLLVMPass());
  //     //   pm.addPass(mlir::tosa::createTosaToArith());
  //     llvm::errs() << "\n=== PASS PIPELINE ===\n";
  // pm.printAsTextualPipeline(llvm::errs());
  // llvm::errs() << "\n====================\n";

  //   }

  if (isLoweringToLLVM) {
    // Finish lowering the hexir IR to the LLVM dialect.
    pm.addPass(mlir::hexir::createLowerToLLVMPass());
    // This is necessary to have line tables emitted and basic
    // debugger working. In the future we will add proper debug information
    // emission directly from our frontend.
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

static int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a hexir AST when the input is MLIR\n";
    return 5;
  }

  // auto moduleAST = parseInputFile(inputFilename);
  //  if (!moduleAST)
  //    return 1;

  // dump(*moduleAST);
  return 0;
}

static int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerGPUDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerNVVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create target machine and configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

static int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerGPUDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerNVVMDialectTranslation(*module->getContext());

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

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "hexir compiler\n");

  if (emitAction == Action::DumpAST)
    return dumpAST();

  // If we aren't dumping the AST, then we are compiling with/to MLIR.
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
                  mlir::gpu::GPUDialect, mlir::NVVM::NVVMDialect,
                  mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                  mlir::affine::AffineDialect, mlir::math::MathDialect,
                  mlir::LLVM::LLVMDialect, mlir::cf::ControlFlowDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::ls_cpu::LSCPUDialect, mlir::ls_gpu::LSGPUDialect>();

  MLIRContext context(registry);

  mlir::func::registerAllExtensions(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  // Register bufferizable op interface external models AFTER dialects loaded
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      const_cast<mlir::DialectRegistry &>(context.getDialectRegistry()));
  mlir::arith::registerBufferizableOpInterfaceExternalModels(
      const_cast<mlir::DialectRegistry &>(context.getDialectRegistry()));
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(
      const_cast<mlir::DialectRegistry &>(context.getDialectRegistry()));
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(
      const_cast<mlir::DialectRegistry &>(context.getDialectRegistry()));

  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::hexir::HexirDialect>();
  context.getOrLoadDialect<mlir::ls_cpu::LSCPUDialect>();
  context.getOrLoadDialect<mlir::ls_gpu::LSGPUDialect>();
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module))
    return error;

  // If we aren't exporting to non-mlir, then we are done.
  bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
  if (isOutputingMLIR) {
    module->dump();
    return 0;
  }

  // Check to see if we are compiling to LLVM IR.
  if (emitAction == Action::DumpLLVMIR)
    return dumpLLVMIR(*module);

  // Otherwise, we must be running the jit.
  if (emitAction == Action::RunJIT)
    return runJit(*module);

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return -1;
}
