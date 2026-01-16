// //===- mlpc.cpp - The Mlp Compiler ----------------------------------------===//
// //
// // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// // See https://llvm.org/LICENSE.txt for license information.
// // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// //
// //===----------------------------------------------------------------------===//
// //
// // This file implements the entry point for the Mlp compiler.
// //
// //===----------------------------------------------------------------------===//

// #include "Builder.h"
// #include "Dialect.h"
// #include "Jit.h"
// #include "Passes.h"
// #include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
// #include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
// #include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
// #include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
// #include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
// #include "mlir/Conversion/LLVMCommon/TypeConverter.h"
// #include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
// #include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
// #include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
// #include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
// #include "mlir/Conversion/TosaToArith/TosaToArith.h"
// #include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
// #include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
// #include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Func/Extensions/AllExtensions.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/IR/Types.h"
// #include "mlir/InitAllDialects.h"
// // #include "mlir/Dialect/Affine/Passes.h"
// #include "mlir/Dialect/Affine/Transforms/Passes.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Arith/Transforms/Passes.h"
// #include "mlir/Dialect/Bufferization/Transforms/Passes.h"
// #include "mlir/Dialect/Func/Extensions/AllExtensions.h"
// #include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/Dialect/LLVMIR/LLVMTypes.h"
// #include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
// #include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
// #include "mlir/Dialect/Linalg/Passes.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/MemRef/Transforms/Passes.h"
// #include "mlir/Dialect/SCF/Transforms/Passes.h"
// #include "mlir/Dialect/Tosa/Transforms/Passes.h"
// #include "mlir/ExecutionEngine/ExecutionEngine.h"
// #include "mlir/ExecutionEngine/OptUtils.h"
// #include "mlir/IR/AsmState.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/IR/DialectRegistry.h"
// #include "mlir/IR/MLIRContext.h"
// #include "mlir/IR/Verifier.h"
// #include "mlir/InitAllExtensions.h"
// #include "mlir/Parser/Parser.h"
// #include "mlir/Pass/PassManager.h"
// #include "mlir/Support/LLVM.h"
// #include "mlir/Support/LogicalResult.h"
// #include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
// #include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
// #include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
// #include "mlir/Target/LLVMIR/Export.h"
// #include "mlir/Target/LLVMIR/ModuleTranslation.h"
// #include "mlir/Transforms/DialectConversion.h"
// #include "mlir/Transforms/Passes.h"
// #include "llvm/ADT/StringRef.h"
// #include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
// #include "llvm/IR/LLVMContext.h"
// #include "llvm/IR/Module.h"
// #include "llvm/IR/Verifier.h"
// #include "llvm/Support/CommandLine.h"
// #include "llvm/Support/ErrorOr.h"
// #include "llvm/Support/MemoryBuffer.h"
// #include "llvm/Support/SourceMgr.h"
// #include "llvm/Support/TargetSelect.h"
// #include "llvm/Support/raw_ostream.h"
// #include <cassert>
// #include <iostream>
// #include <memory>
// #include <string>
// #include <system_error>
// #include <utility>

// // using namespace mlp;
// namespace cl = llvm::cl;

// // static cl::opt<std::string> inputFilename(cl::Positional,
// //                                           cl::desc("<input mlp file>"),
// //                                           cl::init("-"),
// //                                           cl::value_desc("filename"));

// // namespace {
// // enum InputType { Mlp, MLIR };
// // } // namespace

// // static cl::opt<enum InputType> inputType(
// //     "x", cl::init(Mlp), cl::desc("Decided the kind of output desired"),
// //     cl::values(clEnumValN(Mlp, "mlp", "load the input file as a Mlp
// //     source.")), cl::values(clEnumValN(MLIR, "mlir",
// //                           "load the input file as an MLIR file")));

// // namespace {
// // enum Action {
// //   None,
// //   DumpAST,
// //   DumpMLIR,
// //   DumpMLIRAffine,
// //   DumpMLIRLLVM,
// //   DumpLLVMIR,
// //   RunJIT
// // };
// // } // namespace
// // static cl::opt<enum Action> emitAction(
// //     "emit", cl::desc("Select the kind of output desired"),
// //     cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
// //     cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
// //     cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
// //                           "output the MLIR dump after affine lowering")),
// //     cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
// //                           "output the MLIR dump after llvm lowering")),
// //     cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
// //     cl::values(
// //         clEnumValN(RunJIT, "jit",
// //                    "JIT the code and run it by invoking the main
// //                    function")));

// // static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

// // /// Returns a Mlp AST resulting from parsing the file or a nullptr on error.
// // static std::unique_ptr<mlp::ModuleAST>
// // parseInputFile(llvm::StringRef filename) {
// //   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
// //       llvm::MemoryBuffer::getFileOrSTDIN(filename);
// //   if (std::error_code ec = fileOrErr.getError()) {
// //     llvm::errs() << "Could not open input file: " << ec.message() << "\n";
// //     return nullptr;
// //   }
// //   auto buffer = fileOrErr.get()->getBuffer();
// //   LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
// //   Parser parser(lexer);
// //   return parser.parseModule();
// // }

// // static int loadMLIR(mlir::MLIRContext &context,
// //                     mlir::OwningOpRef<mlir::ModuleOp> &module) {
// //   // Handle '.mlp' input to the compiler.
// //   if (inputType != InputType::MLIR &&
// //       !llvm::StringRef(inputFilename).ends_with(".mlir")) {
// //     auto moduleAST = parseInputFile(inputFilename);
// //     if (!moduleAST)
// //       return 6;
// //     module = mlirGen(context, *moduleAST);
// //     return !module ? 1 : 0;
// //   }

// //   // Otherwise, the input is '.mlir'.
// //   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
// //       llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
// //   if (std::error_code ec = fileOrErr.getError()) {
// //     llvm::errs() << "Could not open input file: " << ec.message() << "\n";
// //     return -1;
// //   }

// //   // Parse the input mlir.
// //   llvm::SourceMgr sourceMgr;
// //   sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
// //   module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
// //   if (!module) {
// //     llvm::errs() << "Error can't load file " << inputFilename << "\n";
// //     return 3;
// //   }
// //   return 0;
// // }

// // static int loadAndProcessMLIR(mlir::MLIRContext &context,
// //                               mlir::OwningOpRef<mlir::ModuleOp> &module) {
// //   if (int error = loadMLIR(context, module))
// //     return error;

// //   mlir::PassManager pm(module.get()->getName());
// //   // Apply any generic pass manager command line options and run the
// //   pipeline. if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
// //     return 4;

// //   // Check to see what granularity of MLIR we are compiling to.
// //   bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
// //   bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

// //   if (enableOpt || isLoweringToAffine) {
// //     // Inline all functions into main and then delete them.
// //     pm.addPass(mlir::createInlinerPass());

// //     // Now that there is only one function, we can infer the shapes of each
// //     of
// //     // the operations.
// //     mlir::OpPassManager &optPM = pm.nest<mlir::mlp::FuncOp>();
// //     optPM.addPass(mlir::createCanonicalizerPass());
// //     optPM.addPass(mlir::mlp::createShapeInferencePass());
// //     optPM.addPass(mlir::createCanonicalizerPass());
// //     optPM.addPass(mlir::createCSEPass());
// //   }

// //   if (isLoweringToAffine) {
// //     // Partially lower the mlp dialect.
// //     pm.addPass(mlir::mlp::createLowerToAffinePass());

// //     // Add a few cleanups post lowering.
// //     mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
// //     optPM.addPass(mlir::createCanonicalizerPass());
// //     optPM.addPass(mlir::createCSEPass());

// //     // Add optimizations if enabled.
// //     if (enableOpt) {
// //       optPM.addPass(mlir::affine::createLoopFusionPass());
// //       optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
// //     }
// //   }

// //   if (isLoweringToLLVM) {
// //     // Finish lowering the mlp IR to the LLVM dialect.
// //     pm.addPass(mlir::mlp::createLowerToLLVMPass());
// //     // This is necessary to have line tables emitted and basic
// //     // debugger working. In the future we will add proper debug information
// //     // emission directly from our frontend.
// //     pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
// //   }

// //   if (mlir::failed(pm.run(*module)))
// //     return 4;
// //   return 0;
// // }

// // static int dumpAST() {
// //   if (inputType == InputType::MLIR) {
// //     llvm::errs() << "Can't dump a Mlp AST when the input is MLIR\n";
// //     return 5;
// //   }

// //   auto moduleAST = parseInputFile(inputFilename);
// //   if (!moduleAST)
// //     return 1;

// //   dump(*moduleAST);
// //   return 0;
// // }

// // static int dumpLLVMIR(mlir::ModuleOp module) {
// //   // Register the translation to LLVM IR with the MLIR context.
// //   mlir::registerBuiltinDialectTranslation(*module->getContext());
// //   mlir::registerLLVMDialectTranslation(*module->getContext());

// //   // Convert the module to LLVM IR in a new LLVM IR context.
// //   llvm::LLVMContext llvmContext;
// //   auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
// //   if (!llvmModule) {
// //     llvm::errs() << "Failed to emit LLVM IR\n";
// //     return -1;
// //   }

// //   // Initialize LLVM targets.
// //   llvm::InitializeNativeTarget();
// //   llvm::InitializeNativeTargetAsmPrinter();

// //   // Create target machine and configure the LLVM Module
// //   auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
// //   if (!tmBuilderOrError) {
// //     llvm::errs() << "Could not create JITTargetMachineBuilder\n";
// //     return -1;
// //   }

// //   auto tmOrError = tmBuilderOrError->createTargetMachine();
// //   if (!tmOrError) {
// //     llvm::errs() << "Could not create TargetMachine\n";
// //     return -1;
// //   }
// //   mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
// //                                                         tmOrError.get().get());

// //   /// Optionally run an optimization pipeline over the llvm module.
// //   auto optPipeline = mlir::makeOptimizingTransformer(
// //       /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
// //       /*targetMachine=*/nullptr);
// //   if (auto err = optPipeline(llvmModule.get())) {
// //     llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
// //     return -1;
// //   }
// //   llvm::errs() << *llvmModule << "\n";
// //   return 0;
// // }

// // static int runJit(mlir::ModuleOp module) {
// //   // Initialize LLVM targets.
// //   llvm::InitializeNativeTarget();
// //   llvm::InitializeNativeTargetAsmPrinter();

// //   // Register the translation from MLIR to LLVM IR, which must happen before
// //   we
// //   // can JIT-compile.
// //   mlir::registerBuiltinDialectTranslation(*module->getContext());
// //   mlir::registerLLVMDialectTranslation(*module->getContext());

// //   // An optimization pipeline to use within the execution engine.
// //   auto optPipeline = mlir::makeOptimizingTransformer(
// //       /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
// //       /*targetMachine=*/nullptr);

// //   // Create an MLIR execution engine. The execution engine eagerly
// //   JIT-compiles
// //   // the module.
// //   mlir::ExecutionEngineOptions engineOptions;
// //   engineOptions.transformer = optPipeline;
// //   auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
// //   assert(maybeEngine && "failed to construct an execution engine");
// //   auto &engine = maybeEngine.get();

// //   // Invoke the JIT-compiled function.
// //   auto invocationResult = engine->invokePacked("main");
// //   if (invocationResult) {
// //     llvm::errs() << "JIT invocation failed\n";
// //     return -1;
// //   }

// //   return 0;
// // }

// // int main(int argc, char **argv) {
// //  Register any command line options.
// //  mlir::registerAsmPrinterCLOptions();
// //  mlir::registerMLIRContextCLOptions();
// //  mlir::registerPassManagerCLOptions();

// // cl::ParseCommandLineOptions(argc, argv, "mlp compiler\n");

// // if (emitAction == Action::DumpAST)
// //   return dumpAST();

// // // If we aren't dumping the AST, then we are compiling with/to MLIR.
// // mlir::DialectRegistry registry;
// // mlir::func::registerAllExtensions(registry);
// // mlir::LLVM::registerInlinerInterface(registry);

// // mlir::MLIRContext context(registry);
// // // Load our Dialect in this MLIR Context.
// // context.getOrLoadDialect<mlir::mlp::MlpDialect>();

// // mlir::OwningOpRef<mlir::ModuleOp> module;
// // if (int error = loadAndProcessMLIR(context, module))
// //   return error;

// // // If we aren't exporting to non-mlir, then we are done.
// // bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
// // if (isOutputingMLIR) {
// //   module->dump();
// //   return 0;
// // }

// // // Check to see if we are compiling to LLVM IR.
// // if (emitAction == Action::DumpLLVMIR)
// //   return dumpLLVMIR(*module);

// // // Otherwise, we must be running the jit.
// // if (emitAction == Action::RunJIT)
// //   return runJit(*module);

// // llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
// //  return -1;
// //}

// #ifdef PRINT
// #define PRINT
// #endif
// // #define PRINT

// using namespace mlir;
// using namespace builder1;
// using namespace jit;
// // using namespace dbs;
// // using namespace mlp;

// int main() {
//   // Register any command line options.
//   mlir::registerAsmPrinterCLOptions();
//   mlir::registerMLIRContextCLOptions();
//   mlir::registerPassManagerCLOptions();

//   mlir::DialectRegistry registry;

//   registry
//       .insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
//               mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
//               mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
//               mlir::affine::AffineDialect,
//               // mlir::math::MathDialect,
//               mlir::LLVM::LLVMDialect,
//               // mlir::cf::ControlFlowDialect,
//               mlir::tosa::TosaDialect, bufferization::BufferizationDialect>();

//   mlir::func::registerAllExtensions(registry);
//   mlir::registerBuiltinDialectTranslation(registry);
//   mlir::registerLLVMDialectTranslation(registry);
//   // mlir::registerAllDialects(registry);

//   mlir::MLIRContext ctx(registry);
//   // ctx.appendDialectRegistry(registry);
//   ctx.getOrLoadDialect<mlir::mlp::MLPDialect>();

//   ctx.loadAllAvailableDialects();

//   // registry.insert<mlir::mlp::MLPDialect>();
//   // registry.insert<mlir::bufferization::BufferizationDialect>();

//   // Correct way to create a module:
//   mlir::OwningOpRef<mlir::ModuleOp> module =
//       mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

//   // createMainFunction(ctx, *module);
//   //  createAddFunction(ctx, *module);
//   // createMulFunction(ctx, *module);
//   createMLPAddFunction(ctx, *module);

//   // createMLPAddTOSAFunction(ctx, *module);
//   // createMLPReluFunction(ctx, *module);
//   // createMLPTESTFunction(ctx, *module);

//   llvm::outs()
//       << "\n===================== mlir dialect ========================\n";

//   module->print(llvm::outs());
//   llvm::outs() << "\n";

//   // run pass pipeline (use ctx variable from your main)
//   PassManager pm(&ctx);

//   // if (1) {
//   //   //  Inline all functions into main and then delete them.
//   //   // pm.addPass(mlir::createInlinerPass());

//   //   // Now that there is only one function, we can infer the shapes of each
//   //   of
//   //   // the operations.

//   //   mlir::OpPassManager &optPM = pm.nest<mlir::mlp::FuncOp>();
//   //   optPM.addPass(mlir::createCanonicalizerPass());
//   //   optPM.addPass(mlir::mlp::createShapeInferencePass());
//   //   optPM.addPass(mlir::createCanonicalizerPass());
//   //   optPM.addPass(mlir::createCSEPass());
//   // }

//   // if (1) {
//   //   // Partially lower the mlp dialect.
//   //   pm.addPass(mlir::dbs::createLowerToLinalgPass());

//   //   mlir::bufferization::OneShotBufferizationOptions options;
//   //   options.allowReturnAllocsFromLoops = true;
//   //   // pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
//   //   //  //pm.addPass(mlir::bufferization::createBufferDeallocationPass());
//   //   // pm.addPass(mlir::createConvertLinalgToLoopsPass());
//   //   //  pm.addPass(mlir::createConvertSCFToCFPass());
//   //   //   ------------------------------------------------------------
//   //   //   pm.addPass(mlir::createConvertArithToLLVMPass());
//   //   //   pm.addPass(mlir::createConvertMemRefToLLVMPass());
//   //   //   pm.addPass(mlir::createConvertFuncToLLVMPass());
//   //   //   pm.addPass(mlir::tosa::createTosaToArith());

//   //   // // Add a few cleanups post lowering.
//   //   // mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
//   //   // optPM.addPass(mlir::createCanonicalizerPass());
//   //   // optPM.addPass(mlir::createCSEPass());

//   //   // //    Add optimizations if enabled.
//   //   // if (auto opt = true; opt) {
//   //   //   optPM.addPass(mlir::affine::createLoopFusionPass());
//   //   //   optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
//   //   // }
//   // }

//   // if (0) {
//   //   // Partially lower the mlp dialect.
//   //   pm.addPass(mlir::dbs::createLowerToAffinePass());

//   //   // Add a few cleanups post lowering.
//   //   mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
//   //   optPM.addPass(mlir::createCanonicalizerPass());
//   //   optPM.addPass(mlir::createCSEPass());

//   //   // Add optimizations if enabled.
//   //   if (auto opt = true; opt) {
//   //     optPM.addPass(mlir::affine::createLoopFusionPass());
//   //     optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
//   //   }
//   // }

//   // if (0) {
//   //   // Finish lowering the mlp IR to the LLVM dialect.
//   //   pm.addPass(mlir::mlp::createLowerToLLVMPass());
//   //   // This is necessary to have line tables emitted and basic
//   //   // debugger working. In the future we will add proper debug information
//   //   // emission directly from our frontend.
//   //   pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(
//   //       mlir::LLVM::createDIScopeForLLVMFuncOpPass());
//   // }

//   if (mlir::failed(pm.run(*module))) {
//     llvm::errs() << "Lowering pipeline failed!\n";
//     return 1;
//   }
//   llvm::outs()
//       << "\n===================== lowered dialect =====================\n";
//   module->print(llvm::outs());

//   // translate and print LLVM IR
//   if (0) {

//     llvm::LLVMContext llvmCtx;
//     std::unique_ptr<llvm::Module> llvmModule =
//         mlir::translateModuleToLLVMIR(*module, llvmCtx);

//     if (!llvmModule) {
//       llvm::errs() << "translateModuleToLLVMIR failed\n";
//       return 1;
//     }
//     llvm::outs() << "\n=== LLVM IR ===\n";
//     llvmModule->dump();

//     if (llvm::verifyModule(*llvmModule, &llvm::errs())) {
//       llvm::errs() << "IR verification failed\n";
//       return 0;
//     }
//   }

//   // Otherwise, we must be running the jit.
//   //  if (emitAction == Action::RunJIT)
//   // runJit(*module);
//   std::cout << std::endl;
//   // llvm::outs() << "\n=== MLIR TEST dialect ===\n";
//   // module->print(llvm::outs());
//   return 0;
// }