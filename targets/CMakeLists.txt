
# Automatically detect all backend directories under targets/
file(GLOB BACKEND_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(BACKEND ${BACKEND_DIRS})
    set(BACKEND_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${BACKEND})
    if(IS_DIRECTORY ${BACKEND_PATH})
        # Expect each backend to have a single backend source file named <Backend>Backend.cpp
        set(BACKEND_SRC ${BACKEND_PATH}/${BACKEND}Backend.cpp)

        if(EXISTS ${BACKEND_SRC})
            message(STATUS "Adding backend: ${BACKEND}")
            add_library(ppytorch_${BACKEND} ${BACKEND_SRC})
            target_include_directories(ppytorch_${BACKEND} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
            target_link_libraries(ppytorch_${BACKEND} PUBLIC ppytorch_core MLIRIR MLIRLLVM)
        else()
            message(WARNING "No source file found for backend ${BACKEND}, skipping...")
        endif()
    endif()
endforeach()
