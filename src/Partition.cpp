#include "Dialect.h"
#include "TargetInfo.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

    auto assignDevice = [](Operation *op) -> std::string {
      // Always assign CPU for now; future: GPU for matmul
      return "cpu";
    };

    module->walk([ctx, &assignDevice](Operation *op) {
      // Handle MLP dialect ops (keep existing)
      if (isa<MLPDialect>(op->getDialect())) {
        std::string device = assignDevice(op);
        if (!device.empty()) {
          op->setAttr("device", StringAttr::get(ctx, device));
        }
        return;
      }

      // Handle linalg ops - explicit matching
      if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
        op->setAttr("device", StringAttr::get(ctx, assignDevice(op)));
        return;
      }

      // if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      // ReLU: 1 input, body: yield maxf(0.0f, arg0)
      // Add: 2 inputs, body: yield addf(arg0, arg1)
      // Generic: tag all for infra

      // if (generic.getNumInputs() == 1 || generic.getNumInputs() == 2) {
      //   op->setAttr("device", StringAttr::get(ctx, assignDevice(op)));
      // }
      // return;
      // }
    });
  }
};

std::unique_ptr<Pass> createPartitionPass() {
  return std::make_unique<PartitionPass>();
}

} // namespace mlp
} // namespace mlir
