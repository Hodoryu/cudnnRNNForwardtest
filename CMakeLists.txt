cmake_minimum_required(VERSION 3.18)
project(cudnn_rnn_performance_test)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
# Use user-set CUDA_HOME environment variable if available
if(DEFINED ENV{CUDA_HOME})
    set(CUDA_HOME $ENV{CUDA_HOME})
    message(STATUS "Using CUDA_HOME from environment: ${CUDA_HOME}")
else()
    set(CUDA_HOME "/usr/local/cuda")
    message(STATUS "Using default CUDA_HOME: ${CUDA_HOME}")
endif()

# Set CUDA toolkit root directory for find_package
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_HOME})

# Find Threads first (required by CUDA)
find_package(Threads REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)

# Find ONNX Runtime
message(STATUS "ONNX Runtime ========>${ONNXRUNTIME_HOME}.")
set(ONNXRUNTIME_HOME "/usr/local/onnxruntime-gpu" CACHE PATH "ONNX Runtime installation directory")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_HOME}/include")
set(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_HOME}/lib")

# Find ONNX Protobuf headers (usually included with ONNX Runtime)
set(ONNX_PROTOBUF_INCLUDE_DIR "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime")

# Check if ONNX Runtime was found
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime headers not found. Please set ONNXRUNTIME_HOME to the ONNX Runtime installation directory.")
endif()

if(NOT EXISTS "${ONNXRUNTIME_LIBRARY_DIR}/libonnxruntime.so")
    message(FATAL_ERROR "ONNX Runtime library not found. Please set ONNXRUNTIME_HOME to the ONNX Runtime installation directory.")
endif()

# Find cuDNN
set(CUDNN_HOME "/usr/local/cudnn" CACHE PATH "cuDNN installation directory")
find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    PATHS ${CUDNN_HOME}/include
          /usr/include
          /usr/local/include
          ${CUDA_INCLUDE_DIRS}
    DOC "cuDNN include directory"
)

find_library(CUDNN_LIBRARY
    NAMES cudnn
    PATHS ${CUDNN_HOME}/lib64
          ${CUDNN_HOME}/lib
          /usr/lib64
          /usr/local/lib64
          ${CUDA_LIBRARY_DIRS}
    DOC "cuDNN library"
)

# Check if cuDNN was found
if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN not found. Please set CUDNN_HOME to the cuDNN installation directory.")
endif()

# Include directories
include_directories(${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR} ${ONNXRUNTIME_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Add CUDA executable
cuda_add_executable(rnn_performance_test rnn_performance_test.cpp)

# Link libraries
target_link_libraries(rnn_performance_test 
    ${CUDNN_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_CUBLAS_LIBRARY}
    "${ONNXRUNTIME_LIBRARY_DIR}/libonnxruntime.so"
    cuda
    Threads::Threads
)

# Set compile options
target_compile_options(rnn_performance_test PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g -DDEBUG>
    -Wall -Wextra -m64
)

# Installation
install(TARGETS rnn_performance_test DESTINATION bin)

# Testing
enable_testing()
add_test(NAME rnn_performance_test COMMAND rnn_performance_test)

# Print configuration
message(STATUS "CUDA_HOME: ${CUDA_HOME}")
message(STATUS "CUDNN_HOME: ${CUDNN_HOME}")
message(STATUS "CUDNN_INCLUDE_DIR: ${CUDNN_INCLUDE_DIR}")
message(STATUS "CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
message(STATUS "ONNXRUNTIME_HOME: ${ONNXRUNTIME_HOME}")
message(STATUS "ONNXRUNTIME_INCLUDE_DIR: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNXRUNTIME_LIBRARY: ${ONNXRUNTIME_LIBRARY_DIR}/libonnxruntime.so")

# Build targets
add_custom_target(check 
    COMMAND ${CMAKE_COMMAND} -E echo "Checking dependencies..."
    COMMAND ${CMAKE_COMMAND} -E echo "CUDA found at: ${CUDA_HOME}"
    COMMAND ${CMAKE_COMMAND} -E echo "cuDNN found at: ${CUDNN_HOME}"
    COMMAND ${CMAKE_COMMAND} -E echo "ONNX Runtime found at: ${ONNXRUNTIME_HOME}"
    COMMAND ${CMAKE_COMMAND} -E echo "All dependencies found!"
    VERBATIM
)

# Debug configuration
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release, Debug, RelWithDebInfo, MinSizeRel)")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Debug RelWithDebInfo MinSizeRel)

# Profile build
if(CMAKE_BUILD_TYPE STREQUAL "Profile")
    target_compile_options(rnn_performance_test PRIVATE -pg)
    target_link_libraries(rnn_performance_test PRIVATE -pg)
endif()
