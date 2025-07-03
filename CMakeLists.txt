cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# First try to find CUDA, then enable CUDA language if found
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    project(NukeOptiXDenoiser LANGUAGES CXX CUDA)
    set(CUDA_AVAILABLE TRUE)
else()
    project(NukeOptiXDenoiser LANGUAGES CXX)
    set(CUDA_AVAILABLE FALSE)
    message(WARNING "CUDA not found via CUDAToolkit, trying manual detection...")
endif()

# Set the C++ Standard for the project (C++17 required for OptiX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Allow global toolchain to be set using GLOBAL_TOOLCHAIN env
if(DEFINED ENV{GLOBAL_TOOLCHAIN})
    message("Set toolchain $ENV{GLOBAL_TOOLCHAIN}")
    set(CMAKE_TOOLCHAIN_FILE $ENV{GLOBAL_TOOLCHAIN})
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Find CUDA package (preferred method)
if(NOT CUDA_AVAILABLE)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        set(CUDA_AVAILABLE TRUE)
        message(STATUS "CUDA found via legacy find_package")
    endif()
endif()

# Manual CUDA detection if packages fail
if(NOT CUDA_AVAILABLE)
    # Try to find CUDA installation manually
    find_program(NVCC_EXECUTABLE nvcc 
        PATHS 
        /usr/local/cuda-12.9/bin
        /usr/local/cuda/bin
        /opt/cuda/bin
        /usr/bin
        ENV PATH
    )
    
    if(NVCC_EXECUTABLE)
        get_filename_component(CUDA_BIN_DIR ${NVCC_EXECUTABLE} DIRECTORY)
        get_filename_component(CUDA_ROOT_DIR ${CUDA_BIN_DIR} DIRECTORY)
        set(CUDA_INCLUDE_DIR "${CUDA_ROOT_DIR}/include")
        set(CUDA_LIBRARY_DIR "${CUDA_ROOT_DIR}/lib64")
        
        if(EXISTS "${CUDA_LIBRARY_DIR}/libcudart.so")
            set(CUDA_CUDART_LIBRARY "${CUDA_LIBRARY_DIR}/libcudart.so")
            set(CUDA_AVAILABLE TRUE)
            message(STATUS "CUDA found manually at: ${CUDA_ROOT_DIR}")
            
            # Set CMAKE variables for CUDA
            set(CMAKE_CUDA_COMPILER ${NVCC_EXECUTABLE})
            # Try to enable CUDA language now that we found it
            enable_language(CUDA)
        endif()
    endif()
endif()

# If still no CUDA, use hardcoded fallback
if(NOT CUDA_AVAILABLE)
    message(WARNING "CUDA not found automatically, using hardcoded paths...")
    set(CUDA_ROOT_DIR "/usr/local/cuda-12.9")
    set(CUDA_INCLUDE_DIR "${CUDA_ROOT_DIR}/include")
    set(CUDA_LIBRARY_DIR "${CUDA_ROOT_DIR}/lib64")
    set(CUDA_CUDART_LIBRARY "${CUDA_LIBRARY_DIR}/libcudart.so")
    set(NVCC_EXECUTABLE "${CUDA_ROOT_DIR}/bin/nvcc")
    
    # Check if hardcoded paths exist
    if(EXISTS ${CUDA_INCLUDE_DIR} AND EXISTS ${CUDA_CUDART_LIBRARY} AND EXISTS ${NVCC_EXECUTABLE})
        set(CUDA_AVAILABLE TRUE)
        set(CMAKE_CUDA_COMPILER ${NVCC_EXECUTABLE})
        message(STATUS "Using hardcoded CUDA paths")
        # Try to enable CUDA language
        enable_language(CUDA)
    else()
        message(FATAL_ERROR "CUDA not found! Please install CUDA or adjust paths in CMakeLists.txt")
    endif()
endif()

# Hardcoded paths for OptiX (header-only SDK)
set(OPTIX_ROOT_DIR "/opt/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64")
set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT_DIR}/include")
set(OPTIX_SDK_DIR "${OPTIX_ROOT_DIR}/SDK")

# Debug output to verify Nuke include paths
set(NUKE_INCLUDE_DIR "/opt/Nuke16.0v4/include")
if(EXISTS ${NUKE_INCLUDE_DIR})
    include_directories(${NUKE_INCLUDE_DIR})
    message(STATUS "Include directory found: ${NUKE_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Include directory not found: ${NUKE_INCLUDE_DIR}")
endif()

# Find the Nuke core library dynamically
find_library(NUKE_LIBRARIES DDImage PATHS "/opt/Nuke16.0v4/" NO_DEFAULT_PATH)
if(NOT NUKE_LIBRARIES)
    message(FATAL_ERROR "Could not find the Nuke library (libDDImage.so).")
else()
    message(STATUS "Nuke Libraries found: ${NUKE_LIBRARIES}")
endif()

# Compatibility flags for Nuke 15+ (uncomment if needed)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

# Verify CUDA paths
if(CUDA_AVAILABLE)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found via CUDAToolkit")
        message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
    elseif(CUDA_FOUND)
        message(STATUS "CUDA found via find_package")
        message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
        message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
    else()
        message(STATUS "Using manual/hardcoded CUDA paths")
        if(EXISTS ${CUDA_INCLUDE_DIR}/cuda.h)
            message(STATUS "CUDA include directory found: ${CUDA_INCLUDE_DIR}")
        else()
            message(FATAL_ERROR "CUDA include directory not found at ${CUDA_INCLUDE_DIR}")
        endif()
        
        if(EXISTS ${CUDA_CUDART_LIBRARY})
            message(STATUS "CUDA runtime library found: ${CUDA_CUDART_LIBRARY}")
        else()
            message(FATAL_ERROR "CUDA runtime library not found at ${CUDA_CUDART_LIBRARY}")
        endif()
    endif()
else()
    message(FATAL_ERROR "CUDA is required but not found!")
endif()

# Verify OptiX paths (header-only, no library needed)
if(EXISTS ${OPTIX_INCLUDE_DIR}/optix.h)
    message(STATUS "OptiX include directory found: ${OPTIX_INCLUDE_DIR}")
else()
    message(WARNING "OptiX include directory not found at ${OPTIX_INCLUDE_DIR}")
    message(WARNING "Please adjust OPTIX_INCLUDE_DIR to the correct path")
endif()

if(EXISTS ${OPTIX_SDK_DIR})
    message(STATUS "OptiX SDK directory found: ${OPTIX_SDK_DIR}")
else()
    message(WARNING "OptiX SDK directory not found at ${OPTIX_SDK_DIR}")
endif()

# Set the path to the OptiXDenoiser source files
set(OPTIX_DENOISER_SOURCES
    "${CMAKE_SOURCE_DIR}/src/OptiXDenoiser.cpp"
    "${CMAKE_SOURCE_DIR}/src/OptiXDenoiser.h"
)

# Verify source files exist
foreach(SRC_FILE ${OPTIX_DENOISER_SOURCES})
    if(EXISTS ${SRC_FILE})
        message(STATUS "Found source file: ${SRC_FILE}")
    else()
        message(FATAL_ERROR "Source file not found: ${SRC_FILE}")
    endif()
endforeach()

# Set the library output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Create the OptiXDenoiser plugin
add_library(OptiXDenoiser SHARED ${OPTIX_DENOISER_SOURCES})

# Include directories
target_include_directories(OptiXDenoiser PRIVATE
    ${NUKE_INCLUDE_DIR}
    ${OPTIX_INCLUDE_DIR}
    ${OPTIX_SDK_DIR}
)

# Add CUDA include directories
if(CUDAToolkit_FOUND)
    target_include_directories(OptiXDenoiser PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
elseif(CUDA_FOUND)
    target_include_directories(OptiXDenoiser PRIVATE ${CUDA_INCLUDE_DIRS})
else()
    target_include_directories(OptiXDenoiser PRIVATE ${CUDA_INCLUDE_DIR})
endif()

# Link libraries - Only Nuke and CUDA runtime needed
target_link_libraries(OptiXDenoiser PRIVATE
    ${NUKE_LIBRARIES}
)

# Link CUDA runtime
if(CUDAToolkit_FOUND)
    target_link_libraries(OptiXDenoiser PRIVATE CUDA::cudart)
elseif(CUDA_FOUND)
    target_link_libraries(OptiXDenoiser PRIVATE ${CUDA_LIBRARIES})
else()
    target_link_libraries(OptiXDenoiser PRIVATE ${CUDA_CUDART_LIBRARY})
endif()

# Set target properties
set_target_properties(OptiXDenoiser PROPERTIES
    PREFIX ""  # Remove 'lib' prefix
    INSTALL_RPATH "$ORIGIN/../lib"
)

# Only set CUDA properties if CUDA language is available
if(CUDA_AVAILABLE AND CMAKE_CUDA_COMPILER)
    set_target_properties(OptiXDenoiser PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    # CUDA-specific flags
    set_property(TARGET OptiXDenoiser PROPERTY CUDA_ARCHITECTURES "50;60;70;75;80;86")
endif()

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(OptiXDenoiser PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wweak-vtables -fPIC>
    )
elseif(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(OptiXDenoiser PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -fPIC>
    )
endif()

# CUDA-specific flags
# (moved to target properties section above)

# Debug information
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CUDA Available: ${CUDA_AVAILABLE}")
if(CUDAToolkit_FOUND)
    message(STATUS "Using CUDAToolkit")
elseif(CUDA_FOUND)
    message(STATUS "Using legacy CUDA package")
else()
    message(STATUS "CUDA Include Dir: ${CUDA_INCLUDE_DIR}")
    message(STATUS "CUDA Library Dir: ${CUDA_LIBRARY_DIR}")
endif()
message(STATUS "OptiX Include Dir: ${OPTIX_INCLUDE_DIR}")
message(STATUS "OptiX SDK Dir: ${OPTIX_SDK_DIR}")
message(STATUS "OptiX is header-only - no library linking required")

# Installation
install(
    TARGETS OptiXDenoiser
    DESTINATION plugins
    COMPONENT Nuke
)
