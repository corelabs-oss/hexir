#include "TargetInfo.h"
#include "Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace mlp {

// TargetSupport &amp;TargetSupport::getInstance() {
//   static TargetSupport instance;
//   return instance;
// }

// TargetSupport::TargetSupport() {
//   static MLIRContext staticCtx;
//   MLIRContext *ctx = &amp;staticCtx;

//   // MLP ops support (CPU only)
//   registerSupport("mlp.linear", {"cpu"});
//   registerSupport("mlp.add", {"cpu"});
//   registerSupport("mlp.relu", {"cpu"});

//   // Linalg ops support (CPU only)
//   registerSupport("linalg.matmul", {"cpu"});
//   registerSupport("linalg.add", {"cpu"});
//   registerSupport("linalg.generic", {"cpu"}); // ReLU

//   // Preferred: CPU for all
//   opPreferred_[StringAttr::get(ctx, "mlp.linear")] = StringAttr::get(ctx,
//   "cpu"); opPreferred_[StringAttr::get(ctx, "linalg.matmul")] =
//   StringAttr::get(ctx, "cpu"); opPreferred_[StringAttr::get(ctx,
//   "linalg.add")] = StringAttr::get(ctx, "cpu");
//   opPreferred_[StringAttr::get(ctx, "linalg.generic")] = StringAttr::get(ctx,
//   "cpu");
// }

// bool TargetSupport::isSupported(Operation *op, StringRef target) const {
//   MLIRContext *ctx = op->getContext();
//   auto key = StringAttr::get(ctx, op->getName().getStringRef());
//   auto it = opSupports_.find(key);
//   if (it == opSupports_.end()) return target == "cpu"; // CPU fallback
//   return it->second.count(target);
// }

// StringRef TargetSupport::getPreferredTarget(Operation *op) const {
//   MLIRContext *ctx = op->getContext();
//   auto key = StringAttr::get(ctx, op->getName().getStringRef());
//   auto it = opPreferred_.find(key);
//   return it != opPreferred_.end() ? it->second.getValue() : "cpu";
// }

// void TargetSupport::registerSupport(StringRef opName,
// llvm::ArrayRef<StringRef> targets) {
//   static MLIRContext staticCtx;
//   MLIRContext *ctx = &amp;staticCtx;
//   auto key = StringAttr::get(ctx, opName);
//   llvm::StringSet<> &amp;opSet = opSupports_[key];
//   for (auto tgt : targets)
//     opSet.insert(tgt);
// }

} // namespace mlp
} // namespace mlir
