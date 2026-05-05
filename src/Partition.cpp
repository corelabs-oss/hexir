#include "Dialect.h"
#include "TargetInfo.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace mlir {
namespace mlp {

struct PartitionPass
    : public PassWrapper<PartitionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionPass)

  StringRef getArgument() const final { return "mlp-partition"; }
  StringRef getDescription() const final {
    return "Partition MLP and Linalg ops to CPU/GPU based on target support.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    TargetSupport &targets = TargetSupport::getInstance();

    auto assignDevice = [&targets](Operation *op) -> StringRef {
      StringRef preferred = targets.getPreferredTarget(op);
      if (targets.isSupported(op, preferred))
        return preferred;
      return "cpu";
    };

    module->setAttr("mlp.targets",
                    ArrayAttr::get(ctx, {StringAttr::get(ctx, "cpu"),
                                         StringAttr::get(ctx, "cuda")}));

    module->walk([ctx, &assignDevice](Operation *op) {
      // Handle MLP dialect ops (keep existing)
      if (isa<MLPDialect>(op->getDialect())) {
        StringRef device = assignDevice(op);
        if (!device.empty()) {
          op->setAttr("device", StringAttr::get(ctx, device));
        }
        return;
      }

      // Handle linalg ops - explicit matching
      if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
        op->setAttr("device", StringAttr::get(ctx, "cuda"));
//        op->setAttr("device", StringAttr::get(ctx, "cpu"));
        return;
      }

      if (isa<linalg::AddOp, linalg::GenericOp>(op)) {
        op->setAttr("device", StringAttr::get(ctx, assignDevice(op)));
        return;
      }
    });
  }
};

std::unique_ptr<Pass> createPartitionPass() {
  return std::make_unique<PartitionPass>();
}

} // namespace mlp
} // namespace mlir
