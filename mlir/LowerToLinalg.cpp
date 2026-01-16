// #include "Dialect.h"
// #include "Passes.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Bufferization/Transforms/Passes.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/Func/Transforms/FuncConversions.h"
// #include "mlir/Dialect/Linalg/IR/Linalg.h"
// #include "mlir/Dialect/Math/IR/Math.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/Tensor/IR/Tensor.h"
// #include "mlir/Dialect/Utils/StructuredOpsUtils.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/IR/BuiltinDialect.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/IR/BuiltinTypes.h"
// #include "mlir/IR/Diagnostics.h"
// #include "mlir/IR/DialectRegistry.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/IR/ValueRange.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Support/LLVM.h"
// #include "mlir/Support/LogicalResult.h"
// #include "mlir/Support/TypeID.h"
// #include "mlir/Transforms/DialectConversion.h"
// #include "llvm/ADT/STLExtras.h"
// #include "llvm/Support/Casting.h"
// #include <memory>
// #include <utility>

// using namespace mlir;

// namespace {

// //===----------------------------------------------------------------------===//
// // MLPTolinalg RewritePatterns: Constant operations
// //===----------------------------------------------------------------------===//

// struct ConstantOpToArith : public OpConversionPattern<mlp::ConstantOp> {
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(mlp::ConstantOp op, OpAdaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     auto attr = op.getValue().dyn_cast<DenseElementsAttr>();
//     if (!attr)
//       return failure();
//     rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getType(), attr);
//     return success();
//   }
// };

// //===----------------------------------------------------------------------===//
// // MLPTolinalg RewritePatterns: Print operations
// //===----------------------------------------------------------------------===//

// struct PrintOpLowering : public OpConversionPattern<mlp::PrintOp> {
//   using OpConversionPattern<mlp::PrintOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(mlp::PrintOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const final {
//     // We don't lower "MLP.print" in this pass, but we need to update its
//     // operands.
//     rewriter.updateRootInPlace(op,
//                                [&] { op->setOperands(adaptor.getOperands()); });
//     return success();
//   }
// };

// struct LinearOpToLinalg : public OpConversionPattern<mlp::LinearOp> {
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(mlp::LinearOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();

//     auto resultTy = op.getType().dyn_cast<RankedTensorType>();
//     if (!resultTy)
//       return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

//     Value lhs = adaptor.getInput();  // %0
//     Value rhs = adaptor.getWeight(); // %1

//     // Create zero-init tensor for matmul output
//     auto zeroAttr = rewriter.getZeroAttr(resultTy);
//     Value init = rewriter.create<arith::ConstantOp>(loc, resultTy, zeroAttr);

//     // Create linalg.matmul
//     auto matmul = rewriter.create<linalg::MatmulOp>(
//         loc,
//         /*resultTensorTypes=*/TypeRange{resultTy},
//         /*inputs=*/ValueRange{lhs, rhs},
//         /*outputs=*/ValueRange{init});

//     rewriter.replaceOp(op, matmul.getResult(0));
//     return success();
//   }
// };

// struct ReluOpToLinalg : public mlir::OpConversionPattern<mlp::ReluOp> {
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(mlp::ReluOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();

//     auto resultTy = op.getType().dyn_cast<RankedTensorType>();
//     if (!resultTy)
//       return rewriter.notifyMatchFailure(op, "expected ranked tensor");

//     Value input = adaptor.getInput();

//     // ------------------------------------------------------------
//     // Create init tensor (zero-filled)
//     // ------------------------------------------------------------
//     auto zeroAttr = rewriter.getZeroAttr(resultTy);
//     Value init = rewriter.create<arith::ConstantOp>(loc, resultTy, zeroAttr);

//     // ------------------------------------------------------------
//     // Build indexing maps
//     // ------------------------------------------------------------
//     auto identity = rewriter.getMultiDimIdentityMap(resultTy.getRank());
//     // DBS_PRINT(identity);

//     SmallVector<AffineMap, 2> indexingMaps = {identity, identity};
//     // ------------------------------------------------------------
//     // Iterator types (all parallel for ReLU)
//     // ------------------------------------------------------------
//     SmallVector<utils::IteratorType> iteratorTypes(
//         resultTy.getRank(), utils::IteratorType::parallel);

//     // ------------------------------------------------------------
//     // Create linalg.generic
//     // ------------------------------------------------------------
//     auto genericOp = rewriter.create<linalg::GenericOp>(
//         loc,
//         /*resultTensorTypes=*/TypeRange{resultTy},
//         /*inputs=*/ValueRange{input},
//         /*outputs=*/ValueRange{init}, indexingMaps, iteratorTypes,
//         [&](OpBuilder &builder, Location loc, ValueRange args) {
//           Value x = args[0];

//           Value zero = builder.create<arith::ConstantOp>(
//               loc, builder.getFloatAttr(x.getType(), 0.0));

//           Value relu = builder.create<arith::MaximumFOp>(loc, x, zero);

//           builder.create<linalg::YieldOp>(loc, relu);
//         });

//     rewriter.replaceOp(op, genericOp.getResult(0));
//     return success();
//   }
// };

// struct SoftmaxToLinalg : public mlir::OpConversionPattern<mlp::SoftmaxOp> {
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(mlp::SoftmaxOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     auto resultTy = op.getType().dyn_cast<RankedTensorType>();
//     if (!resultTy || resultTy.getRank() != 2)
//       return rewriter.notifyMatchFailure(op,
//                                          "Softmax expects a 1D ranked tensor");

//     Value input = adaptor.getInput();
//     Type elemTy = resultTy.getElementType();
//     MLIRContext *ctx = rewriter.getContext();

//     // 1. Create initialization tensors
//     // ------------------------------------------------------------
//     auto zeroAttr = rewriter.getZeroAttr(resultTy);
//     Value initVec = rewriter.create<arith::ConstantOp>(loc, resultTy, zeroAttr);

//     RankedTensorType scalarTy = RankedTensorType::get({}, elemTy);
//     auto zeroScalarAttr =
//         DenseElementsAttr::get(scalarTy, rewriter.getFloatAttr(elemTy, 0.0));
//     Value initScalar =
//         rewriter.create<arith::ConstantOp>(loc, scalarTy, zeroScalarAttr);

//     // 2. Step 1: Compute exp(x) elementwise
//     // ------------------------------------------------------------
//     auto map1D = rewriter.getMultiDimIdentityMap(resultTy.getRank());
//     SmallVector<AffineMap, 2> expMaps = {map1D, map1D};
//     SmallVector<utils::IteratorType, 2> parallelIter = {
//         utils::IteratorType::parallel, utils::IteratorType::parallel};
//     // DBS_PRINT(expMaps[0] << expMaps[1] );
//     auto expOp = rewriter.create<linalg::GenericOp>(
//         loc, resultTy, ValueRange{input}, ValueRange{initVec}, expMaps,
//         parallelIter, [&](OpBuilder &b, Location loc, ValueRange args) {
//           Value x = args[0];
//           Value ex = b.create<math::ExpOp>(loc, x); // Compute e^x
//           b.create<linalg::YieldOp>(loc, ex);
//         });

//     // 3. Step 2: Compute Sum Reduction of exp(x)
//     // ------------------------------------------------------------
//     SmallVector<int64_t, 1> reductionDims = {0};
//     auto sumOp = rewriter.create<linalg::ReduceOp>(
//         loc, expOp.getResult(0), initScalar, reductionDims,
//         [&](OpBuilder &b, Location loc, ValueRange args) {
//           Value val = args[0];
//           Value acc = args[1];
//           Value sum = b.create<arith::AddFOp>(loc, acc, val);
//           b.create<linalg::YieldOp>(loc, sum);
//         });

//     // 4. Step 3: Divide exp(x) by the sum (Normalization)
//     // ------------------------------------------------------------
//     // The "Magic": Map scalar sum (0D) to the vector (1D) using (d0) -> ()
//     auto mapScalar = AffineMap::get(1, 0, ctx); // This creates (d0) -> ()
//     SmallVector<AffineMap, 3> divMaps = {
//         map1D,     // expOp result (1D)
//         mapScalar, // sumOp result (0D broadcasted to 1D)
//         map1D      // Output (1D)
//     };

//     auto divOp = rewriter.create<linalg::GenericOp>(
//         loc, resultTy, ValueRange{expOp.getResult(0), sumOp.getResult(0)},
//         ValueRange{initVec}, divMaps, parallelIter,
//         [&](OpBuilder &b, Location loc, ValueRange args) {
//           Value ex = args[0];
//           Value totalSum = args[1];
//           Value result = b.create<arith::DivFOp>(loc, ex, totalSum);
//           b.create<linalg::YieldOp>(loc, result);
//         });

//     // Replace the original op with the final divided result
//     rewriter.replaceOp(op, expOp.getResult(0));

//     return success();
//   }
// };
// } // namespace

// //===----------------------------------------------------------------------===//
// // MlpTolinalgLoweringPass
// //===----------------------------------------------------------------------===//

// // / This is a partial lowering to linalg loops of the MLP operations that are
// // / computationally intensive (like matmul for example...) while keeping the
// // / rest of the code in the MLP dialect.

// //===----------------------------------------------------------------------===//
// // // MLPToLinalgLoweringPass
// //
// //===----------------------------------------------------------------------===//

// namespace {

// struct MLPToLinalgLoweringPass
//     : public PassWrapper<MLPToLinalgLoweringPass, OperationPass<ModuleOp>> {

//   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MLPToLinalgLoweringPass)

//   StringRef getArgument() const override { return "mlp-lower-to-linalg"; }
//   StringRef getDescription() const override {
//     return "Lower MLP dialect matmul operations to Linalg dialect";
//   }

//   void getDependentDialects(DialectRegistry &registry) const override {
//     registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
//   }

//   void runOnOperation() override {
//     ConversionTarget target(getContext());
//     target.addLegalDialect<linalg::LinalgDialect, BuiltinDialect,
//                            arith::ArithDialect, func::FuncDialect,
//                            memref::MemRefDialect, mlir::math::MathDialect>();

//     ModuleOp module = getOperation();
//     MLIRContext *ctx = module.getContext();

//     RewritePatternSet patterns(ctx);

//     target.addIllegalDialect<mlir::mlp::MLPDialect>();

//     // target.addDynamicallyLegalOp<mlp::PrintOp>([](mlp::PrintOp op) {
//     //   return llvm::none_of(op->getOperandTypes(), [](Type type) {
//     //     return llvm::isa<TensorType>(type);
//     //   });
//     // });

//     target.addDynamicallyLegalOp<mlp::PrintOp>([](mlp::PrintOp op) {
//       return llvm::none_of(op->getOperandTypes(), [](Type type) {
//         return llvm::isa<MemRefType>(type);
//       });
//     });

//     patterns.add<LinearOpToLinalg, ReluOpToLinalg, SoftmaxToLinalg,
//                  ConstantOpToArith>(ctx);

//     if (failed(applyPartialConversion(getOperation(), target,
//                                       std::move(patterns))))
//       signalPassFailure();
//   }
// };

// } // namespace

// //===----------------------------------------------------------------------===//
// // // Pass Registration
// //
// //===----------------------------------------------------------------------===//

// std::unique_ptr<Pass> mlir::mlp::createLowerToLinalgPass() {
//   return std::make_unique<MLPToLinalgLoweringPass>();
// }