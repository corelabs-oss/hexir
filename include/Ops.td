//===- Ops.td - MLP dialect operation definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the MLP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLP_OPS
#define MLP_OPS

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ShapeInferenceInterface.td"

// Provide a definition of the 'mlp' dialect in the ODS framework so that we
// can define our operations.
def MLP_Dialect : Dialect {
  let name = "mlp";
  let cppNamespace = "::mlir::mlp";

  // We set this bit to generate a declaration of the `materializeConstant`
  // method so that we can materialize constants for our mlp operations.
  //let hasConstantMaterializer = 1; Original is 1
  let hasConstantMaterializer = 0;

  // We set this bit to generate the declarations for the dialect's type parsing
  // and printing hooks.
  let useDefaultTypePrinterParser = 1;

}

// Base class for mlp dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class MLP_OP<string mnemonic, list<Trait> traits = []> :
    Op<MLP_Dialect, mnemonic, traits>;

// Provide a definition for the MLP StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef. We use `DialectType`
// to demarcate the StructType as belonging to the MLP dialect.
def MLP_StructType :
    DialectType<MLP_Dialect, CPred<"::llvm::isa<StructType>($_self)">,
                "MLP struct type">;

// Provide a definition of the types that are used within the MLP dialect.
def MLP_Type : AnyTypeOf<[F64Tensor, MLP_StructType]>;

//===----------------------------------------------------------------------===//
// MLP Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// We define a MLP operation by inheriting from our base 'MLP_OP' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'Pure' as it is a pure operation
// and may be removed if dead.
def ConstantOp : MLP_OP<"constant",
    [Pure]> {//      ConstantLike,Pure 
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = MLP.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `ConstantOp::create(builder, ...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "double":$value)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;

  // Set the folder bit so that we can implement constant folders.
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : MLP_OP<"add",
    [Pure]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

def ReluOp : MLP_OP<"relu",
    [Pure]> {

  let summary = "element-wise ReLU operation";
  let description = [{
    The `relu` operation applies the Rectified Linear Unit function
    element-wise to the input tensor:

      relu(x) = max(0, x)

    The input and output tensor shapes are identical.
  }];

  let arguments = (ins F64Tensor:$input);
  let results   = (outs F64Tensor:$result);

  // Custom printer/parser (optional, but consistent with AddOp)
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

// def SoftmaxOp : MLP_OP<"softmax",
//     [Pure]> {

//   let summary = "softmax normalization operation";
//   let description = [{
//     The `softmax` operation applies the softmax function to the input tensor,
//     normalizing values along the last dimension so that they sum to 1.

//     This operation is typically used in classification layers.
//   }];

//   let arguments = (ins F64Tensor:$input);
//   let results   = (outs F64Tensor:$result);

//   let hasCustomAssemblyFormat = 1;

//   let builders = [
//     OpBuilder<(ins "Value":$input)>
//   ];
// }

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//

// def LinearOp : MLP_OP<"linear",
//     [Pure]> {
//   let summary = "linear (fully-connected) layer operation";
//   let description = [{
//     The "linear" operation represents a fully-connected layer.
//     It computes the matrix multiplication of input and weight tensors,
//     optionally adding a bias tensor. Shapes must be compatible for matmul:
//     [batch, in_features] x [in_features, out_features] -> [batch, out_features].
//   }];

//   // Inputs: lhs=input tensor, rhs=weight tensor, optional bias
//   let arguments = (ins F64Tensor:$input, F64Tensor:$weight);
//   let results = (outs F64Tensor);

//   // Indicate that the operation has a custom parser and printer method.
//   let hasCustomAssemblyFormat = 1;

//   // Builder for LinearOp
//   let builders = [
//     OpBuilder<(ins "Value":$input, "Value":$weight)>
//   ];
// }


def LinearOp : MLP_OP<"linear",
    [Pure]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$input, F64Tensor:$weight);
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$weight)>
  ];
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

// def CastOp : MLP_OP<"cast", [
//      DeclareOpInterfaceMethods<CastOpInterface>,
//      DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
//      Pure,
//      SameOperandsAndResultShape
//   ]> {
//   let summary = "shape cast operation";
//   let description = [{
//     The "cast" operation converts a tensor from one type to an equivalent type
//     without changing any data elements. The source and destination types must
//     both be tensor types with the same element type. If both are ranked, then
//     shape is required to match. The operation is invalid if converting to a
//     mismatching constant dimension.
//   }];

//   let arguments = (ins F64Tensor:$input);
//   let results = (outs F64Tensor:$output);

//   let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
// }

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : MLP_OP<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "mlp.func" operation represents a user defined function. These are
    callable SSA-region operations that contain mlp computations.

    Example:

    ```mlir
    mlp.func @main() {
      %0 = mlp.constant dense<5.500000e+00> : tensor<f64>
      %1 = mlp.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
      mlp.print %1 : tensor<2x2xf64>
      mlp.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

// def GenericCallOp : MLP_OP<"generic_call",
//     [DeclareOpInterfaceMethods<CallOpInterface>]> {
//   let summary = "generic call operation";
//   let description = [{
//     Generic calls represent calls to a user defined function that needs to
//     be specialized for the shape of its arguments. The callee name is attached
//     as a symbol reference via an attribute. The arguments list must match the
//     arguments expected by the callee. For example:

//     ```mlir
//      %4 = mlp.generic_call @my_func(%1, %3)
//            : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
//     ```

//     This is only valid if a function named "my_func" exists and takes two
//     arguments.
//   }];

//   // The generic call operation takes a symbol reference attribute as the
//   // callee, and inputs for the call.
//   let arguments = (ins
//     FlatSymbolRefAttr:$callee,
//     Variadic<MLP_Type>:$inputs,
//     OptionalAttr<DictArrayAttr>:$arg_attrs,
//     OptionalAttr<DictArrayAttr>:$res_attrs
//   );

//   // The generic call operation returns a single value of TensorType or
//   // StructType.
//   let results = (outs MLP_Type);

//   // Specialize assembly printing and parsing using a declarative format.
//   let assemblyFormat = [{
//     $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
//   }];

//   // Add custom build methods for the generic call operation.
//   let builders = [
//     OpBuilder<(ins "Type":$result_type, "StringRef":$callee,
//               "ArrayRef<Value>":$arguments)>
//   ];
// }

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

// def MulOp : MLP_OP<"mul",
//     [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
//   let summary = "element-wise multiplication operation";
//   let description = [{
//     The "mul" operation performs element-wise multiplication between two
//     tensors. The shapes of the tensor operands are expected to match.
//   }];

//   let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
//   let results = (outs F64Tensor);

//   // Indicate that the operation has a custom parser and printer method.
//   let hasCustomAssemblyFormat = 1;

//   // Allow building a MulOp with from the two input operands.
//   let builders = [
//     OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
//   ];
// }

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : MLP_OP<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// def ReshapeOp : MLP_OP<"reshape", [Pure]> {
//   let summary = "tensor reshape operation";
//   let description = [{
//     Reshape operation is transforming its input tensor into a new tensor with
//     the same number of elements but different shapes. For example:

//     ```mlir
//        %0 = mlp.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
//     ```
//   }];

//   let arguments = (ins F64Tensor:$input);

//   let assemblyFormat = [{
//     `(` $input `:` type($input) `)` attr-dict `to` type(results)
//   }];

//   // Enable registering canonicalization patterns with this operation.
//   let hasCanonicalizer = 1;

//   // We expect that the reshape operation returns a statically shaped tensor.
//   let results = (outs StaticShapeTensorOf<[F64]>);
// }

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

// def ReturnOp : MLP_OP<"return", [Pure, HasParent<"FuncOp">,
//                                  Terminator]> {
//   let summary = "return operation";
//   let description = [{
//     The "return" operation represents a return operation within a function.
//     The operation takes an optional operand and produces no results.
//     The operand type must match the signature of the function that contains
//     the operation. For example:

//     ```mlir
//       mlp.func @foo() -> tensor<2xf64> {
//         ...
//         mlp.return %0 : tensor<2xf64>
//       }
//     ```
//   }];

//   // The return operation takes an optional input operand to return. This
//   // value must match the return type of the enclosing function.
//   let arguments = (ins Variadic<MLP_Type>:$input);

//   // The return operation only emits the input in the format if it is present.
//   let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

//   // Allow building a ReturnOp with no return operand.
//   let builders = [
//     OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
//   ];

//   // Provide extra utility definitions on the c++ operation class definition.
//   let extraClassDeclaration = [{
//     bool hasOperand() { return getNumOperands() != 0; }
//   }];

//   // Indicate that additional verification for this operation is necessary.
//   let hasVerifier = 1;
// }

//===----------------------------------------------------------------------===//
// StructAccessOp
//===----------------------------------------------------------------------===//

// def StructAccessOp : MLP_OP<"struct_access", [Pure]> {
//   let summary = "struct access";
//   let description = [{
//     Access the Nth element of a value returning a struct type.
//   }];

//   let arguments = (ins MLP_StructType:$input, I64Attr:$index);
//   let results = (outs MLP_Type:$output);

//   let assemblyFormat = [{
//     $input `[` $index `]` attr-dict `:` type($input) `->` type($output)
//   }];

//   // Allow building a StructAccessOp with just a struct value and an index.
//   let builders = [
//     OpBuilder<(ins "Value":$input, "size_t":$index)>
//   ];

//   // Indicate that additional verification for this operation is necessary.
//   let hasVerifier = 1;

//   // Set the folder bit so that we can fold constant accesses.
//   let hasFolder = 1;
// }

//===----------------------------------------------------------------------===//
// StructConstantOp
//===----------------------------------------------------------------------===//

// def StructConstantOp : MLP_OP<"struct_constant", [ConstantLike, Pure]> {
//   let summary = "struct constant";
//   let description = [{
//     Constant operation turns a literal struct value into an SSA value. The data
//     is attached to the operation as an attribute. The struct constant is encoded
//     as an array of other constant values. For example:

//     ```mlir
//       %0 = mlp.struct_constant [
//         dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
//       ] : !mlp.struct<tensor<*xf64>>
//     ```
//   }];

//   let arguments = (ins ArrayAttr:$value);
//   let results = (outs MLP_StructType:$output);

//   let assemblyFormat = "$value attr-dict `:` type($output)";

//   // Indicate that additional verification for this operation is necessary.
//   let hasVerifier = 1;
//   let hasFolder = 1;
// }

// //===----------------------------------------------------------------------===//
// // TransposeOp
// //===----------------------------------------------------------------------===//

// def TransposeOp : MLP_OP<"transpose",
//     [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
//   let summary = "transpose operation";

//   let arguments = (ins F64Tensor:$input);
//   let results = (outs F64Tensor);

//   let assemblyFormat = [{
//     `(` $input `:` type($input) `)` attr-dict `to` type(results)
//   }];

//   // Enable registering canonicalization patterns with this operation.
//   let hasCanonicalizer = 1;

//   // Allow building a TransposeOp with from the input operand.
//   let builders = [
//     OpBuilder<(ins "Value":$input)>
//   ];

//   // Indicate that additional verification for this operation is necessary.
//   let hasVerifier = 1;
// }

#endif // MLP_OPS
