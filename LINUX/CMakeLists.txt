cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(NukeOptiXDenoiser LANGUAGES CXX)

# Set the C++ Standard for the project (C++17 required for OptiX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# RPATH Configuration - Make plugin portable
# ============================================================================
# Don't set RPATH - rely on Nuke's LD_LIBRARY_PATH environment
# This makes the plugin work regardless of where Nuke is installed
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)

# Allow global toolchain to be set using GLOBAL_TOOLCHAIN env
if(DEFINED ENV{GLOBAL_TOOLCHAIN})
    message("Set toolchain $ENV{GLOBAL_TOOLCHAIN}")
    set(CMAKE_TOOLCHAIN_FILE $ENV{GLOBAL_TOOLCHAIN})
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Build configuration - defaults for production release
option(ENABLE_OPTIX_DEBUG "Enable extensive debug logging" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)

# Release mode with optimizations
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ============================================================================
# CUDA Configuration (Static Runtime)
# ============================================================================
set(CUDA_AVAILABLE FALSE)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA found - Version: ${CUDAToolkit_VERSION}")
else()
    # Manual CUDA detection
    find_program(NVCC_EXECUTABLE nvcc 
        PATHS 
        /usr/local/cuda-12.8/bin
        /usr/local/cuda-12.1/bin
        /usr/local/cuda/bin
        ENV PATH
    )
    
    if(NVCC_EXECUTABLE)
        get_filename_component(CUDA_BIN_DIR ${NVCC_EXECUTABLE} DIRECTORY)
        get_filename_component(CUDA_ROOT_DIR ${CUDA_BIN_DIR} DIRECTORY)
        set(CUDA_INCLUDE_DIR "${CUDA_ROOT_DIR}/include")
        set(CUDA_LIBRARY_DIR "${CUDA_ROOT_DIR}/lib64")
        set(CUDA_CUDART_STATIC "${CUDA_LIBRARY_DIR}/libcudart_static.a")
        
        if(EXISTS "${CUDA_CUDART_STATIC}")
            set(CUDA_AVAILABLE TRUE)
            set(CMAKE_CUDA_COMPILER ${NVCC_EXECUTABLE})
            enable_language(CUDA)
            set(CMAKE_CUDA_STANDARD 17)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            message(STATUS "CUDA found manually at: ${CUDA_ROOT_DIR}")
        endif()
    endif()
    
    if(NOT CUDA_AVAILABLE)
        message(FATAL_ERROR "CUDA not found!")
    endif()
endif()

# ============================================================================
# OptiX Configuration (Header-only)
# ============================================================================
set(OPTIX_ROOT_DIR "/opt/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64")
set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT_DIR}/include")
set(OPTIX_SDK_DIR "${OPTIX_ROOT_DIR}/SDK")

if(NOT EXISTS ${OPTIX_INCLUDE_DIR}/optix.h)
    message(FATAL_ERROR "OptiX not found at ${OPTIX_INCLUDE_DIR}")
endif()
message(STATUS "OptiX found: ${OPTIX_INCLUDE_DIR}")

# ============================================================================
# Nuke Configuration
# ============================================================================
if(NOT DEFINED NUKE_VERSION)
    set(NUKE_VERSION "16.0v6")
endif()
set(NUKE_INSTALL_DIR "/opt/Nuke${NUKE_VERSION}")
set(NUKE_INCLUDE_DIR "${NUKE_INSTALL_DIR}/include")

if(NOT EXISTS ${NUKE_INCLUDE_DIR})
    message(FATAL_ERROR "Nuke include directory not found: ${NUKE_INCLUDE_DIR}")
endif()
message(STATUS "Nuke ${NUKE_VERSION}: ${NUKE_INSTALL_DIR}")

# ABI compatibility with Nuke
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

# Find Nuke library
find_library(NUKE_LIBRARIES DDImage PATHS "${NUKE_INSTALL_DIR}" NO_DEFAULT_PATH)
if(NOT NUKE_LIBRARIES)
    message(FATAL_ERROR "Could not find libDDImage.so")
endif()

# ============================================================================
# Build Plugin
# ============================================================================
set(OPTIX_DENOISER_SOURCES
    "${CMAKE_SOURCE_DIR}/src/OptiXDenoiser.cpp"
    "${CMAKE_SOURCE_DIR}/src/OptiXDenoiser.h"
)

foreach(SRC_FILE ${OPTIX_DENOISER_SOURCES})
    if(NOT EXISTS ${SRC_FILE})
        message(FATAL_ERROR "Source file not found: ${SRC_FILE}")
    endif()
endforeach()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(OptiXDenoiser SHARED ${OPTIX_DENOISER_SOURCES})

# Remove "lib" prefix (Nuke convention)
set_target_properties(OptiXDenoiser PROPERTIES PREFIX "")

# Include directories
target_include_directories(OptiXDenoiser PRIVATE
    ${NUKE_INCLUDE_DIR}
    ${OPTIX_INCLUDE_DIR}
    ${OPTIX_SDK_DIR}
)

if(CUDAToolkit_FOUND)
    target_include_directories(OptiXDenoiser PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
else()
    target_include_directories(OptiXDenoiser PRIVATE ${CUDA_INCLUDE_DIR})
endif()

# ============================================================================
# Linking - Static CUDA Runtime
# ============================================================================
# Link Nuke (will show "not found" in ldd - that's correct for portability)
target_link_libraries(OptiXDenoiser PRIVATE ${NUKE_LIBRARIES})

# Link CUDA statically
if(CUDAToolkit_FOUND)
    target_link_libraries(OptiXDenoiser PRIVATE 
        CUDA::cudart_static    # Static CUDA Runtime
        CUDA::cuda_driver      # CUDA Driver API (always dynamic - that's fine)
        ${CMAKE_DL_LIBS}       # libdl (for dlopen)
        pthread                # pthreads
        rt                     # librt
    )
else()
    # Manual static linking
    target_link_libraries(OptiXDenoiser PRIVATE 
        ${CUDA_CUDART_STATIC}
        ${CMAKE_DL_LIBS}
        pthread
        rt
    )
    
    # Find CUDA driver library
    find_library(CUDA_DRIVER_LIBRARY cuda
        PATHS 
        ${CUDA_LIBRARY_DIR}
        ${CUDA_LIBRARY_DIR}/stubs
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
    )
    if(CUDA_DRIVER_LIBRARY)
        target_link_libraries(OptiXDenoiser PRIVATE ${CUDA_DRIVER_LIBRARY})
    else()
        target_link_libraries(OptiXDenoiser PRIVATE cuda)
    endif()
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Build Configuration:")
message(STATUS "  Plugin: OptiXDenoiser.so")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Runtime: STATIC")
message(STATUS "  RPATH: DISABLED (portable)")
message(STATUS "  Output: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "========================================")
message(STATUS "")
message(STATUS "Expected ldd output:")
message(STATUS "  libDDImage.so => not found")
message(STATUS "  libcuda.so.1 => /lib64/libcuda.so.1")
message(STATUS "  (no libcudart.so - it's static)")
message(STATUS "")

# Installation
install(TARGETS OptiXDenoiser DESTINATION plugins)
