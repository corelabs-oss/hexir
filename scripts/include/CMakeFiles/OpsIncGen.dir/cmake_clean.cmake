file(REMOVE_RECURSE
  "CMakeFiles/OpsIncGen"
  "Dialect.cpp.inc"
  "Dialect.h.inc"
  "Ops.cpp.inc"
  "Ops.h.inc"
  "ShapeInferenceOpInterfaces.cpp.inc"
  "ShapeInferenceOpInterfaces.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/OpsIncGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
