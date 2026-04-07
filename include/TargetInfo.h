#pragma once

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
namespace mlp {

/// Target support info for operators.
class TargetSupport {
public:
  static TargetSupport &getInstance();

  /// Check if op is supported on target (e.g., "cpu", "gpu").
  bool isSupported(Operation *op, StringRef target) const;

  /// Get preferred target for op (empty if none).
  StringRef getPreferredTarget(Operation *op) const;

  /// Register support: opName -> targets.
  void registerSupport(StringRef opName, llvm::ArrayRef<StringRef> targets);

private:
  TargetSupport();

  llvm::DenseMap<StringAttr, llvm::StringSet<>> opSupports_;
  llvm::DenseMap<StringAttr, StringAttr> opPreferred_;
};

} // namespace mlp
} // namespace mlir
