cmake_minimum_required(VERSION 3.13.4)
project(mlp_mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(LLVM_DIR "/home/hamza/Repos/llvm-project/build/lib/cmake/llvm")
set(MLIR_DIR "/home/hamza/Repos/llvm-project/build/lib/cmake/mlir")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)
include(TableGen)
include(HandleLLVMOptions)

set(LLVM_LINK_COMPONENTS Core Support nativecodegen OrcJIT)

include_directories(include)

include_directories(
  ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include # For TableGen-generated .inc files
  ${CMAKE_CURRENT_BINARY_DIR})

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)



set(LLVM_TARGET_DEFINITIONS src/MlpCombine.td)
mlir_tablegen(MlpCombine.inc -gen-rewriters)
add_public_tablegen_target(CombineIncGen)


file(
  GLOB CCMLIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  GLOB CCMLIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/targets/cpu/*.cpp"
  GLOB CCMLIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/targets/gpu/*.cpp"
  GLOB CCMLIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/targets/metal/*.cpp"
  GLOB CCMLIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/targets/cpu/*.cpp"
  GLOB CCMLIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/targets/rocm/*.cpp"
)
add_executable(mlp_mlir ${CCMLIR_SRC})


# add_dependencies(mlp_mlir Ops ShapeInfer CombineIncGen)
add_dependencies(mlp_mlir OpsIncGen ShapeInferenceInterfaceIncGen CombineIncGen)

llvm_update_compile_flags(mlp_mlir)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(
  mlp_mlir
  PRIVATE ${dialect_libs}
          ${conversion_libs}
          ${extension_libs}
          LLVMSupport
          MLIRAnalysis
          MLIRBuiltinToLLVMIRTranslation
          MLIRCallInterfaces
          MLIRCastInterfaces
          MLIRExecutionEngine
          MLIRFunctionInterfaces
          MLIRIR
          MLIRLLVMCommonConversion
          MLIRLLVMDialect
          MLIRLLVMToLLVMIRTranslation
          MLIRMemRefDialect
          MLIRParser
          MLIRPass
          MLIRSideEffectInterfaces
          MLIRSupport
          MLIRTargetLLVMIRExport
          MLIRTransforms
          MLIRLinalgDialect
          MLIRTensorDialect
          MLIRArithDialect
          MLIRIR
          MLIRSupport
          MLIRParser
          MLIRAnalysis
          MLIRPass
          MLIRTransforms
          MLIRFuncDialect
          MLIRArithDialect
          MLIRMemRefDialect
          MLIRLinalgDialect
          MLIRAffineDialect
          MLIRSCFDialect
          MLIRLLVMDialect
          MLIRTensorDialect
          MLIRFuncTransforms
          MLIRLinalgTransforms
          MLIRAffineToStandard
          MLIRSCFToControlFlow
          MLIRArithToLLVM
          MLIRMemRefToLLVM
          MLIRFuncToLLVM
          MLIRLLVMCommonConversion
          MLIRLLVMToLLVMIRTranslation
          MLIRTargetLLVMIRExport
          MLIRBuiltinToLLVMIRTranslation
          MLIRTranslateLib
          MLIRExecutionEngine
          MLIRCallInterfaces
          MLIRCastInterfaces
          MLIRFunctionInterfaces
          MLIRSideEffectInterfaces
          LLVMSupport)

install(TARGETS mlp_mlir RUNTIME DESTINATION bin)