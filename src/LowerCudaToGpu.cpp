#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct CudaGpuLoweringPass
    : public PassWrapper<CudaGpuLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CudaGpuLoweringPass)

  StringRef getArgument() const final { return "mlp-lower-cuda-to-gpu"; }

  StringRef getDescription() const final {
    return "Lower CUDA-partitioned linalg ops to the MLIR GPU dialect.";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<linalg::MatmulOp> cudaMatmuls;

    module.walk([&](linalg::MatmulOp op) {
      auto device = op->getAttrOfType<StringAttr>("device");
      if (device && device.getValue() == "cuda")
        cudaMatmuls.push_back(op);
    });

    for (linalg::MatmulOp matmul : cudaMatmuls) {
      if (failed(lowerMatmul(matmul))) {
        signalPassFailure();
        return;
      }
    }
  }

  LogicalResult lowerMatmul(linalg::MatmulOp matmul) {
    if (matmul->getNumResults() != 0)
      return matmul.emitOpError()
             << "expected bufferized memref form before GPU lowering";

    Value lhs = matmul.getInputs()[0];
    Value rhs = matmul.getInputs()[1];
    Value out = matmul.getOutputs()[0];

    auto lhsType = dyn_cast<MemRefType>(lhs.getType());
    auto rhsType = dyn_cast<MemRefType>(rhs.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!lhsType || !rhsType || !outType)
      return matmul.emitOpError()
             << "expected memref operands before GPU lowering";

    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
        !outType.hasStaticShape())
      return matmul.emitOpError()
             << "only static-shape matmul is supported for now";

    if (outType.getRank() != 2 || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2)
      return matmul.emitOpError() << "expected rank-2 matmul operands";

    int64_t m = outType.getDimSize(0);
    int64_t n = outType.getDimSize(1);
    int64_t k = lhsType.getDimSize(1);
    if (rhsType.getDimSize(0) != k)
      return matmul.emitOpError() << "matmul K dimension mismatch";

    OpBuilder builder(matmul);
    Location loc = matmul.getLoc();

    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
    Value cM = arith::ConstantIndexOp::create(builder, loc, m);
    Value cN = arith::ConstantIndexOp::create(builder, loc, n);
    Value cK = arith::ConstantIndexOp::create(builder, loc, k);

    gpu::LaunchOp launch =
        gpu::LaunchOp::create(builder, loc, c1, c1, c1, c1, c1, c1);
    launch->setAttr("device", builder.getStringAttr("cuda"));

    Block &body = launch.getBody().front();
    builder.setInsertionPointToEnd(&body);
    gpu::TerminatorOp::create(builder, loc);
    builder.setInsertionPointToStart(&body);

    scf::ForOp iLoop = scf::ForOp::create(builder, loc, c0, cM, c1);
    builder.setInsertionPointToStart(iLoop.getBody());
    Value i = iLoop.getInductionVar();

    scf::ForOp jLoop = scf::ForOp::create(builder, loc, c0, cN, c1);
    builder.setInsertionPointToStart(jLoop.getBody());
    Value j = jLoop.getInductionVar();

    scf::ForOp kLoop = scf::ForOp::create(builder, loc, c0, cK, c1);
    builder.setInsertionPointToStart(kLoop.getBody());
    Value kk = kLoop.getInductionVar();

    Value a = memref::LoadOp::create(builder, loc, lhs, ValueRange{i, kk});
    Value b = memref::LoadOp::create(builder, loc, rhs, ValueRange{kk, j});
    Value current = memref::LoadOp::create(builder, loc, out, ValueRange{i, j});
    Value prod = arith::MulFOp::create(builder, loc, a, b);
    Value sum = arith::AddFOp::create(builder, loc, current, prod);
    memref::StoreOp::create(builder, loc, sum, out, ValueRange{i, j});

    matmul.erase();
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::mlp::createCudaGpuLoweringPass() {
  return std::make_unique<CudaGpuLoweringPass>();
}
