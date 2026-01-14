# # cmake_minimum_required(VERSION 3.16)
# # project(ellvm CXX)

# # set(CMAKE_CXX_STANDARD 17)
# # set(CMAKE_CXX_STANDARD_REQUIRED ON)
# # #set(LLVM_DIR "/home/user/miniconda3/envs/tvm5/lib/cmake/llvm")
# # set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# # find_package(LLVM REQUIRED CONFIG)

# # link_directories(${LLVM_LIBRARY_DIRS})

# # message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
# # message(STATUS "LLVM DIR: ${LLVM_DIR}")
# # message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")


# # include_directories(
# #   ${LLVM_INCLUDE_DIRS}
# #   ${LLVM_BUILD_INCLUDE_DIR}
# #   ${CMAKE_SOURCE_DIR}/include
# # )


# # add_definitions(${LLVM_DEFINITIONS})

# # file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
# # add_executable(ellvm ${SOURCES})



# # target_link_libraries(ellvm
# #   PRIVATE
# #     LLVMCore
# #     LLVMSupport
# #     LLVMIRReader
# #     LLVMAsmPrinter
# #     LLVMAsmParser
# #     LLVMBitWriter
# #     LLVMExecutionEngine
# #     LLVMOrcJIT
# #     LLVMTarget
# #     LLVMX86CodeGen
# #     LLVMX86AsmParser
# #     LLVMX86Desc
# #     LLVMX86Info
# # )


# # target_compile_options(ellvm PRIVATE 
# #     -Wall                  # Enable all common warnings
# #     -Wextra                # Extra warnings
# #     -Wno-unused-parameter  # Suppress unused parameter warnings
# #     -Wunused-variable      # Warn about unused variables
# #     -Wreturn-type          # Warn if a function might not return a value
# #     -Wno-redundant-move    # Suppress LLVM redundant move warning
# #     -g                     # Include debug information
# # )


# cmake_minimum_required(VERSION 3.13.4)
# project(standalone_mlir LANGUAGES CXX C)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED YES)

# set(LLVM_DIR "/home/hamza/Repos/llvm-project/build/lib/cmake/llvm")
# set(MLIR_DIR "/home/hamza/Repos/llvm-project/build/lib/cmake/mlir")


# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find_package(LLVM REQUIRED CONFIG)
# find_package(MLIR REQUIRED CONFIG)

# list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# include(AddLLVM)
# include(AddMLIR)
# include(TableGen)
# include(HandleLLVMOptions)

# set(LLVM_LINK_COMPONENTS Core Support nativecodegen OrcJIT)

# include_directories(include)

# include_directories(
#   ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include
#   ${CMAKE_CURRENT_BINARY_DIR}/include # For TableGen-generated .inc files
#   ${CMAKE_CURRENT_BINARY_DIR})

# add_definitions(${LLVM_DEFINITIONS})

# add_subdirectory(include)

# set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
# mlir_tablegen(ToyCombine.inc -gen-rewriters)
# add_public_tablegen_target(CombineIncGen)


# # file(
# #   GLOB CCMLIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
# # )
# add_executable(standalone_mlir
#   main.cpp
#   parser/*.cpp
#   mlir/*.cpp
#   )

# #add_executable(standalone_mlir ${CCMLIR_SRC})

# # add_dependencies(standalone_mlir Ops ShapeInfer CombineIncGen)
# add_dependencies(standalone_mlir ToyCh7OpsIncGen ToyCh7ShapeInferenceInterfaceIncGen CombineIncGen)

# llvm_update_compile_flags(standalone_mlir)

# get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
# get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
# get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# target_link_libraries(
#   standalone_mlir
#   PRIVATE ${dialect_libs}
#           ${conversion_libs}
#           ${extension_libs}
#           LLVMSupport
#           MLIRAnalysis
#           MLIRBuiltinToLLVMIRTranslation
#           MLIRCallInterfaces
#           MLIRCastInterfaces
#           MLIRExecutionEngine
#           MLIRFunctionInterfaces
#           MLIRIR
#           MLIRLLVMCommonConversion
#           MLIRLLVMDialect
#           MLIRLLVMToLLVMIRTranslation
#           MLIRMemRefDialect
#           MLIRParser
#           MLIRPass
#           MLIRSideEffectInterfaces
#           MLIRSupport
#           MLIRTargetLLVMIRExport
#           MLIRTransforms
#           MLIRLinalgDialect
#           MLIRTensorDialect
#           MLIRArithDialect
#           MLIRIR
#           MLIRSupport
#           MLIRParser
#           MLIRAnalysis
#           MLIRPass
#           MLIRTransforms
#           MLIRFuncDialect
#           MLIRArithDialect
#           MLIRMemRefDialect
#           MLIRLinalgDialect
#           MLIRAffineDialect
#           MLIRSCFDialect
#           MLIRLLVMDialect
#           MLIRTensorDialect
#           MLIRFuncTransforms
#           MLIRLinalgTransforms
#           MLIRAffineToStandard
#           MLIRSCFToControlFlow
#           MLIRArithToLLVM
#           MLIRMemRefToLLVM
#           MLIRFuncToLLVM
#           MLIRLLVMCommonConversion
#           MLIRLLVMToLLVMIRTranslation
#           MLIRTargetLLVMIRExport
#           MLIRBuiltinToLLVMIRTranslation
#           MLIRTranslateLib
#           MLIRExecutionEngine
#           MLIRCallInterfaces
#           MLIRCastInterfaces
#           MLIRFunctionInterfaces
#           MLIRSideEffectInterfaces
#           LLVMSupport)

# install(TARGETS standalone_mlir RUNTIME DESTINATION bin)


cmake_minimum_required(VERSION 3.13.4)
project(standalone_mlir LANGUAGES CXX C)

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

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(CombineIncGen)


# file(
#   GLOB CCMLIR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
# )
add_executable(standalone_mlir
  main.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/LowerToLLVM.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp
  )



#add_executable(standalone_mlir ${CCMLIR_SRC})

# add_dependencies(standalone_mlir Ops ShapeInfer CombineIncGen)
add_dependencies(standalone_mlir ToyCh7OpsIncGen ToyCh7ShapeInferenceInterfaceIncGen CombineIncGen)

llvm_update_compile_flags(standalone_mlir)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(
  standalone_mlir
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

install(TARGETS standalone_mlir RUNTIME DESTINATION bin)