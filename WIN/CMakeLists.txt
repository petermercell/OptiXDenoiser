cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# ============================================================================
# CRITICAL: Set MSVC runtime library policy BEFORE project()
# This enables modern runtime library control for static linking
# ============================================================================
cmake_policy(SET CMP0091 NEW)

# First try to find CUDA, then enable CUDA language if found
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    project(NukeOptiXDenoiser LANGUAGES CXX CUDA)
    set(CUDA_AVAILABLE TRUE)
else()
    project(NukeOptiXDenoiser LANGUAGES CXX)
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA not found via CUDAToolkit, trying manual detection...")
endif()

# ============================================================================
# Windows-specific Configuration
# ============================================================================
if(NOT WIN32)
    message(FATAL_ERROR "This CMakeLists.txt is for Windows only")
endif()

# Compiler requirements - Visual Studio 2022
if(MSVC)
    if(MSVC_VERSION LESS 1930)
        message(FATAL_ERROR "Visual Studio 2022 or later required (MSVC 19.30+)")
    else()
        message(STATUS "Using MSVC version ${MSVC_VERSION}")
    endif()
else()
    message(FATAL_ERROR "This build requires MSVC (Visual Studio)")
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================================
# CRITICAL: Force static runtime library globally
# This eliminates MSVCP140.dll, VCRUNTIME140.dll, api-ms-win-crt dependencies
# ============================================================================
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Set the C++ Standard (C++17 required for OptiX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# ============================================================================
# Windows Compiler Flags and Definitions  
# ============================================================================
add_definitions(
    -DWIN32
    -D_WINDOWS
    -DNDEBUG
    -DNOMINMAX                    # Prevent min/max macro conflicts
    -D_CRT_SECURE_NO_WARNINGS     # Disable CRT security warnings
    -D_SCL_SECURE_NO_WARNINGS     # Disable SCL security warnings
    -D_USE_MATH_DEFINES           # Enable M_PI and other math constants (required by Nuke)
    -DWIN32_LEAN_AND_MEAN         # Exclude rarely-used stuff from Windows headers
    -DCM_GETIDLIST_FILTER_CLASS=0x00000200
    -DCM_GETIDLIST_FILTER_PRESENT=0x00000100
)

if(MSVC)
    add_compile_options(
        /W3          # Warning level 3
        /O2          # Maximize speed
        /Ob2         # Inline expansion
        /EHsc        # Exception handling
        /FS          # Force synchronous PDB writes
        /nologo      # Suppress startup banner
        /MP          # Multi-processor compilation
    )
    
    # Suppress noisy warnings
    add_compile_options(
        /wd4996      # Deprecated functions
        /wd4244      # Conversion, possible loss of data
        /wd4267      # size_t to smaller type
        /wd4305      # Truncation from double to float
        /wd4251      # DLL interface warnings
        /wd4018      # Signed/unsigned mismatch
        /wd4514      # Unreferenced inline function
        /wd4710      # Function not inlined
        /wd4711      # Function selected for inline
        /wd4668      # Undefined preprocessor macro
    )
endif()

# ============================================================================
# CUDA Configuration (Static Runtime)
# ============================================================================
if(NOT CUDA_AVAILABLE)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        set(CUDA_AVAILABLE TRUE)
        message(STATUS "CUDA found via legacy find_package")
    endif()
endif()

# Manual CUDA detection if packages fail
if(NOT CUDA_AVAILABLE)
    # Windows CUDA paths - check multiple versions
    find_program(NVCC_EXECUTABLE nvcc 
        PATHS 
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin"
        ENV PATH
    )
    
    if(NVCC_EXECUTABLE)
        get_filename_component(CUDA_BIN_DIR ${NVCC_EXECUTABLE} DIRECTORY)
        get_filename_component(CUDA_ROOT_DIR ${CUDA_BIN_DIR} DIRECTORY)
        set(CUDA_INCLUDE_DIR "${CUDA_ROOT_DIR}/include")
        set(CUDA_LIBRARY_DIR "${CUDA_ROOT_DIR}/lib/x64")
        
        # Look for static CUDA runtime first
        if(EXISTS "${CUDA_LIBRARY_DIR}/cudart_static.lib")
            set(CUDA_CUDART_LIBRARY "${CUDA_LIBRARY_DIR}/cudart_static.lib")
            set(CUDA_RUNTIME_STATIC TRUE)
            set(CUDA_AVAILABLE TRUE)
            message(STATUS "CUDA found with static runtime at: ${CUDA_ROOT_DIR}")
        elseif(EXISTS "${CUDA_LIBRARY_DIR}/cudart.lib")
            set(CUDA_CUDART_LIBRARY "${CUDA_LIBRARY_DIR}/cudart.lib")
            set(CUDA_RUNTIME_STATIC FALSE)
            set(CUDA_AVAILABLE TRUE)
            message(STATUS "CUDA found (dynamic runtime) at: ${CUDA_ROOT_DIR}")
        endif()
        
        if(CUDA_AVAILABLE)
            set(CMAKE_CUDA_COMPILER ${NVCC_EXECUTABLE})
            enable_language(CUDA)
        endif()
    endif()
endif()

if(NOT CUDA_AVAILABLE)
    message(FATAL_ERROR "CUDA not found! Please install CUDA Toolkit or adjust paths.")
endif()

# ============================================================================
# OptiX Configuration (Header-only)
# ============================================================================
set(OPTIX_ROOT_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0")
set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT_DIR}/include")
set(OPTIX_SDK_DIR "${OPTIX_ROOT_DIR}/SDK")

if(NOT EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
    message(FATAL_ERROR "OptiX not found at ${OPTIX_INCLUDE_DIR}")
endif()
message(STATUS "OptiX found: ${OPTIX_INCLUDE_DIR}")

# ============================================================================
# Nuke Configuration
# ============================================================================
if(NOT DEFINED NUKE_VERSION)
    set(NUKE_VERSION "16.0v6")
endif()

if(NOT DEFINED NDKDIR)
    set(NDKDIR "C:/Program Files/Nuke${NUKE_VERSION}")
endif()

set(NUKE_INCLUDE_DIR "${NDKDIR}/include")
set(NUKE_LIBRARY_DIR "${NDKDIR}")

if(NOT EXISTS "${NUKE_INCLUDE_DIR}")
    message(FATAL_ERROR "Nuke include directory not found: ${NUKE_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${NDKDIR}/DDImage.lib")
    message(FATAL_ERROR "DDImage.lib not found in: ${NDKDIR}")
endif()

message(STATUS "Nuke ${NUKE_VERSION}: ${NDKDIR}")

# ============================================================================
# Source Files
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

# ============================================================================
# Build Plugin
# ============================================================================
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(OptiXDenoiser SHARED ${OPTIX_DENOISER_SOURCES})

# CRITICAL: Set static runtime for this target
set_property(TARGET OptiXDenoiser PROPERTY 
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

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

# ============================================================================
# Linking - Static where possible
# ============================================================================
# Link Nuke
target_link_libraries(OptiXDenoiser PRIVATE "${NDKDIR}/DDImage.lib")

# Link CUDA - prefer static runtime
if(CUDAToolkit_FOUND)
    # Use static CUDA runtime if available
    if(TARGET CUDA::cudart_static)
        target_link_libraries(OptiXDenoiser PRIVATE CUDA::cudart_static)
        message(STATUS "Using static CUDA runtime (CUDAToolkit)")
    else()
        target_link_libraries(OptiXDenoiser PRIVATE CUDA::cudart)
        message(STATUS "Using dynamic CUDA runtime (CUDAToolkit)")
    endif()
    target_link_libraries(OptiXDenoiser PRIVATE CUDA::cuda_driver)
elseif(CUDA_FOUND)
    target_link_libraries(OptiXDenoiser PRIVATE ${CUDA_LIBRARIES})
else()
    target_link_libraries(OptiXDenoiser PRIVATE ${CUDA_CUDART_LIBRARY})
    # Link CUDA driver library
    if(EXISTS "${CUDA_LIBRARY_DIR}/cuda.lib")
        target_link_libraries(OptiXDenoiser PRIVATE "${CUDA_LIBRARY_DIR}/cuda.lib")
    endif()
endif()

# ============================================================================
# Target Properties
# ============================================================================
set_target_properties(OptiXDenoiser PROPERTIES
    OUTPUT_NAME "OptiXDenoiser"
    SUFFIX ".dll"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib
)

# CUDA properties if available
if(CUDA_AVAILABLE AND CMAKE_CUDA_COMPILER)
    set_target_properties(OptiXDenoiser PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    # Modern GPU architectures
    set_property(TARGET OptiXDenoiser PROPERTY CUDA_ARCHITECTURES "50;60;70;75;80;86;89;90")
endif()

# ============================================================================
# Installation
# ============================================================================
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "$ENV{USERPROFILE}/.nuke" 
        CACHE PATH "Install path" FORCE)
endif()

install(TARGETS OptiXDenoiser
    RUNTIME DESTINATION plugins
    LIBRARY DESTINATION plugins
)

# ============================================================================
# Build Summary
# ============================================================================
message(STATUS "")
message(STATUS "===========================================================================")
message(STATUS "OptiXDenoiser Windows Portable Build Configuration:")
message(STATUS "  Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler:          MSVC ${MSVC_VERSION}")
message(STATUS "  C++ Standard:      C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Runtime Linking:   STATIC (/MT) - No VC++ Redistributable required!")
message(STATUS "")
message(STATUS "CUDA Configuration:")
if(CUDAToolkit_FOUND)
    message(STATUS "  CUDA Version:      ${CUDAToolkit_VERSION}")
    message(STATUS "  Detection:         CUDAToolkit")
elseif(CUDA_FOUND)
    message(STATUS "  Detection:         Legacy find_package")
else()
    message(STATUS "  CUDA Root:         ${CUDA_ROOT_DIR}")
    message(STATUS "  Detection:         Manual")
endif()
if(CUDA_RUNTIME_STATIC)
    message(STATUS "  CUDA Runtime:      STATIC (cudart_static.lib)")
else()
    message(STATUS "  CUDA Runtime:      Dynamic (cudart.lib)")
endif()
message(STATUS "")
message(STATUS "OptiX Configuration:")
message(STATUS "  OptiX SDK:         ${OPTIX_ROOT_DIR}")
message(STATUS "  OptiX:             Header-only (no library)")
message(STATUS "")
message(STATUS "Nuke Configuration:")
message(STATUS "  Nuke Version:      ${NUKE_VERSION}")
message(STATUS "  Nuke Directory:    ${NDKDIR}")
message(STATUS "")
message(STATUS "Output:")
message(STATUS "  Plugin:            OptiXDenoiser.dll")
message(STATUS "  Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Portability:")
message(STATUS "  Required DLLs:     DDImage.dll (from Nuke)")
message(STATUS "                     nvcuda.dll (NVIDIA driver)")
message(STATUS "  NOT Required:      Visual C++ Redistributable")
message(STATUS "                     cudart64_*.dll (if static)")
message(STATUS "")
message(STATUS "Build Commands:")
message(STATUS "  cmake -B build -G \"Visual Studio 17 2022\" -A x64 -DNUKE_VERSION=16.0v4")
message(STATUS "  cmake --build build --config Release")
message(STATUS "  cmake --install build --config Release")
message(STATUS "===========================================================================")
message(STATUS "")