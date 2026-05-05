#include "TargetInfo.h"
#include "Dialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace mlp {

TargetSupport &TargetSupport::getInstance() {
  static TargetSupport instance;
  return instance;
}

TargetSupport::TargetSupport() {
  registerSupport("mlp.linear", {"cpu", "cuda"});
  registerSupport("mlp.add", {"cpu", "cuda"});
  registerSupport("mlp.relu", {"cpu", "cuda"});

  registerSupport("linalg.matmul", {"cpu", "cuda"});
  registerSupport("linalg.add", {"cpu", "cuda"});
  registerSupport("linalg.generic", {"cpu", "cuda"});

  opPreferred_["mlp.linear"] = "cpu";
  opPreferred_["linalg.matmul"] = "cpu";
  opPreferred_["mlp.add"] = "cpu";
  opPreferred_["mlp.relu"] = "cpu";
  opPreferred_["linalg.add"] = "cpu";
  opPreferred_["linalg.generic"] = "cpu";
}

bool TargetSupport::isSupported(Operation *op, StringRef target) const {
  auto it = opSupports_.find(op->getName().getStringRef());
  if (it == opSupports_.end())
    return target == "cpu";
  return it->second.contains(target);
}

StringRef TargetSupport::getPreferredTarget(Operation *op) const {
  auto it = opPreferred_.find(op->getName().getStringRef());
  if (it == opPreferred_.end())
    return "cpu";
  return it->second;
}

void TargetSupport::registerSupport(StringRef opName,
                                    llvm::ArrayRef<StringRef> targets) {
  llvm::StringSet<> &opSet = opSupports_[opName];
  for (StringRef target : targets)
    opSet.insert(target);
}

} // namespace mlp
} // namespace mlir
