cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Set the project with both languages from the start
project(NukeOptiXDenoiser LANGUAGES CXX)

# Set the C++ Standard for the project (C++17 required for OptiX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow global toolchain to be set using GLOBAL_TOOLCHAIN env
if(DEFINED ENV{GLOBAL_TOOLCHAIN})
    message("Set toolchain $ENV{GLOBAL_TOOLCHAIN}")
    set(CMAKE_TOOLCHAIN_FILE $ENV{GLOBAL_TOOLCHAIN})
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Debug configuration for OptiX Denoiser
option(ENABLE_OPTIX_DEBUG "Enable extensive debug logging for OptiX Denoiser" ON)
option(ENABLE_CUDA_MEMCHECK "Enable CUDA memory checking support" ON)
option(ENABLE_ASAN "Enable AddressSanitizer (memory debugging)" ON)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer (thread safety debugging)" OFF)

# Set build type based on debug option
if(ENABLE_OPTIX_DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    message(STATUS "Building in Debug mode with extensive logging")
else()
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Building in Release mode")
endif()

# Initialize CUDA availability flag
set(CUDA_AVAILABLE FALSE)

# First try to find CUDA via CUDAToolkit (recommended modern approach)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA found via CUDAToolkit - Version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit root: ${CUDAToolkit_BIN_DIR}/..")
else()
    message(STATUS "CUDA not found via CUDAToolkit, trying legacy method...")
    
    # Try legacy CUDA package
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        set(CUDA_AVAILABLE TRUE)
        message(STATUS "CUDA found via legacy find_package - Version: ${CUDA_VERSION}")
    else()
        message(STATUS "CUDA not found via legacy method, trying manual detection...")
        
        # Manual CUDA detection
        find_program(NVCC_EXECUTABLE nvcc 
            PATHS 
            /usr/local/cuda-12.1/bin
            /usr/local/cuda/bin
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
                # Enable CUDA language now that we found it
                enable_language(CUDA)
                set(CMAKE_CUDA_STANDARD 17)
                set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            endif()
        else()
            # If still no CUDA, use hardcoded fallback
            message(STATUS "CUDA not found automatically, trying hardcoded paths...")
            set(CUDA_ROOT_DIR "/usr/local/cuda-12.1")
            set(CUDA_INCLUDE_DIR "${CUDA_ROOT_DIR}/include")
            set(CUDA_LIBRARY_DIR "${CUDA_ROOT_DIR}/lib64")
            set(CUDA_CUDART_LIBRARY "${CUDA_LIBRARY_DIR}/libcudart.so")
            set(NVCC_EXECUTABLE "${CUDA_ROOT_DIR}/bin/nvcc")
            
            # Check if hardcoded paths exist
            if(EXISTS ${CUDA_INCLUDE_DIR} AND EXISTS ${CUDA_CUDART_LIBRARY} AND EXISTS ${NVCC_EXECUTABLE})
                set(CUDA_AVAILABLE TRUE)
                set(CMAKE_CUDA_COMPILER ${NVCC_EXECUTABLE})
                message(STATUS "Using hardcoded CUDA paths")
                # Enable CUDA language
                enable_language(CUDA)
                set(CMAKE_CUDA_STANDARD 17)
                set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            else()
                message(FATAL_ERROR "CUDA not found! Please install CUDA or adjust paths in CMakeLists.txt")
            endif()
        endif()
    endif()
endif()

# Hardcoded paths for OptiX (header-only SDK)
set(OPTIX_ROOT_DIR "/opt/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64")
set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT_DIR}/include")
set(OPTIX_SDK_DIR "${OPTIX_ROOT_DIR}/SDK")

# Debug output to verify Nuke include paths
set(NUKE_INCLUDE_DIR "/opt/Nuke14.1v2/include")
if(EXISTS ${NUKE_INCLUDE_DIR})
    include_directories(${NUKE_INCLUDE_DIR})
    message(STATUS "Include directory found: ${NUKE_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Include directory not found: ${NUKE_INCLUDE_DIR}")
endif()

# Find the Nuke core library dynamically
find_library(NUKE_LIBRARIES DDImage PATHS "/opt/Nuke14.1v2" NO_DEFAULT_PATH)
if(NOT NUKE_LIBRARIES)
    message(FATAL_ERROR "Could not find the Nuke library (libDDImage.so).")
else()
    message(STATUS "Nuke Libraries found: ${NUKE_LIBRARIES}")
endif()

# Compatibility flags for Nuke 15+ (uncomment if needed)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

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

# Remove the "lib" prefix from the output filename
set_target_properties(OptiXDenoiser PROPERTIES PREFIX "")

# =====================================================
# SANITIZER CONFIGURATION - FIXED VERSION
# =====================================================

# Function to check if sanitizer is supported - IMPROVED VERSION
function(check_sanitizer_support SANITIZER_NAME SANITIZER_FLAG RESULT_VAR)
    # Create a simple test program
    set(TEST_CODE "
#include <iostream>
int main() {
    int* ptr = new int[10];
    delete[] ptr;
    return 0;
}
")
    
    # Write test file
    file(WRITE "${CMAKE_BINARY_DIR}/test_sanitizer.cpp" "${TEST_CODE}")
    
    # Try to compile with sanitizer
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} ${SANITIZER_FLAG} -o "${CMAKE_BINARY_DIR}/test_sanitizer" "${CMAKE_BINARY_DIR}/test_sanitizer.cpp"
        RESULT_VARIABLE COMPILE_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    
    # Clean up
    file(REMOVE "${CMAKE_BINARY_DIR}/test_sanitizer.cpp")
    if(EXISTS "${CMAKE_BINARY_DIR}/test_sanitizer")
        file(REMOVE "${CMAKE_BINARY_DIR}/test_sanitizer")
    endif()
    
    if(COMPILE_RESULT EQUAL 0)
        set(${RESULT_VAR} TRUE PARENT_SCOPE)
        message(STATUS "${SANITIZER_NAME} is supported")
    else()
        set(${RESULT_VAR} FALSE PARENT_SCOPE)
        message(STATUS "${SANITIZER_NAME} is NOT supported by this compiler")
    endif()
endfunction()

# Sanitizer flags and configuration
set(SANITIZER_FLAGS "")
set(SANITIZER_LINK_FLAGS "")

# AddressSanitizer (Memory debugging - detects buffer overruns, use-after-free, etc.)
if(ENABLE_ASAN)
    check_sanitizer_support("AddressSanitizer" "-fsanitize=address" ASAN_SUPPORTED)
    if(ASAN_SUPPORTED)
        set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fsanitize=address")
        set(SANITIZER_LINK_FLAGS "${SANITIZER_LINK_FLAGS} -fsanitize=address")
        
        # Additional ASan options for better debugging
        set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fno-omit-frame-pointer")
        set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fno-optimize-sibling-calls")
        
        # Define ASan options at compile time
        target_compile_definitions(OptiXDenoiser PRIVATE ASAN_ENABLED=1)
        
        message(STATUS "AddressSanitizer enabled")
        message(STATUS "This will help catch buffer overruns and memory corruption")
    else()
        message(STATUS "AddressSanitizer not supported, trying fallback manual detection...")
        
        # Manual fallback - force enable if GCC version is recent enough
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "4.8")
                message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION} should support AddressSanitizer, forcing enable")
                set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fsanitize=address")
                set(SANITIZER_LINK_FLAGS "${SANITIZER_LINK_FLAGS} -fsanitize=address")
                set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fno-omit-frame-pointer")
                target_compile_definitions(OptiXDenoiser PRIVATE ASAN_ENABLED=1)
                set(ENABLE_ASAN ON)
                message(STATUS "AddressSanitizer force-enabled for GCC")
            else()
                message(WARNING "GCC version too old for AddressSanitizer")
                set(ENABLE_ASAN OFF)
            endif()
        else()
            message(WARNING "AddressSanitizer requested but not supported")
            set(ENABLE_ASAN OFF)
        endif()
    endif()
endif()

# UndefinedBehaviorSanitizer (Catches undefined behavior)
if(ENABLE_UBSAN)
    check_sanitizer_support("UndefinedBehaviorSanitizer" "-fsanitize=undefined" UBSAN_SUPPORTED)
    if(UBSAN_SUPPORTED)
        set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fsanitize=undefined")
        set(SANITIZER_LINK_FLAGS "${SANITIZER_LINK_FLAGS} -fsanitize=undefined")
        
        # Additional UBSan options
        set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fno-sanitize-recover=undefined")
        
        target_compile_definitions(OptiXDenoiser PRIVATE UBSAN_ENABLED=1)
        message(STATUS "UndefinedBehaviorSanitizer enabled")
    else()
        message(WARNING "UndefinedBehaviorSanitizer requested but not supported")
        set(ENABLE_UBSAN OFF)
    endif()
endif()

# ThreadSanitizer (Thread safety debugging - detects race conditions)
if(ENABLE_TSAN)
    check_sanitizer_support("ThreadSanitizer" "-fsanitize=thread" TSAN_SUPPORTED)
    if(TSAN_SUPPORTED)
        # Note: ThreadSanitizer is incompatible with AddressSanitizer
        if(ENABLE_ASAN)
            message(WARNING "ThreadSanitizer is incompatible with AddressSanitizer")
            message(WARNING "Disabling ThreadSanitizer in favor of AddressSanitizer")
            set(ENABLE_TSAN OFF)
        else()
            set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fsanitize=thread")
            set(SANITIZER_LINK_FLAGS "${SANITIZER_LINK_FLAGS} -fsanitize=thread")
            
            target_compile_definitions(OptiXDenoiser PRIVATE TSAN_ENABLED=1)
            message(STATUS "ThreadSanitizer enabled")
        endif()
    else()
        message(WARNING "ThreadSanitizer requested but not supported")
        set(ENABLE_TSAN OFF)
    endif()
endif()

# If no sanitizers are working, enable manual memory debugging
if(NOT ENABLE_ASAN AND NOT ENABLE_UBSAN AND NOT ENABLE_TSAN)
    message(STATUS "No sanitizers available, enabling manual memory debugging")
    target_compile_definitions(OptiXDenoiser PRIVATE MANUAL_MEMORY_DEBUG=1)
endif()

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

# Link libraries - Nuke, CUDA runtime, and CUDA driver API
target_link_libraries(OptiXDenoiser PRIVATE
    ${NUKE_LIBRARIES}
)

# Link CUDA libraries (both runtime and driver API) - CRITICAL FIX FOR cuCtxDestroy_v2
if(CUDAToolkit_FOUND)
    target_link_libraries(OptiXDenoiser PRIVATE 
        CUDA::cudart      # CUDA Runtime API
        CUDA::cuda_driver # CUDA Driver API (fixes cuCtxDestroy_v2 undefined symbol)
    )
    message(STATUS "Linking against CUDA::cudart and CUDA::cuda_driver")
elseif(CUDA_FOUND)
    target_link_libraries(OptiXDenoiser PRIVATE 
        ${CUDA_LIBRARIES}
        cuda              # Add CUDA driver library
    )
    message(STATUS "Linking against legacy CUDA libraries and cuda driver")
else()
    # Manual CUDA library linking
    target_link_libraries(OptiXDenoiser PRIVATE 
        ${CUDA_CUDART_LIBRARY}
    )
    
    # Find and link CUDA driver library manually
    find_library(CUDA_DRIVER_LIBRARY cuda
        PATHS 
        ${CUDA_LIBRARY_DIR}
        ${CUDA_LIBRARY_DIR}/stubs
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/local/cuda/lib64
        /usr/local/cuda/lib64/stubs
        NO_DEFAULT_PATH
    )
    
    if(CUDA_DRIVER_LIBRARY)
        target_link_libraries(OptiXDenoiser PRIVATE ${CUDA_DRIVER_LIBRARY})
        message(STATUS "Found CUDA driver library: ${CUDA_DRIVER_LIBRARY}")
    else()
        # Fallback to system search
        find_library(CUDA_DRIVER_LIBRARY cuda)
        if(CUDA_DRIVER_LIBRARY)
            target_link_libraries(OptiXDenoiser PRIVATE ${CUDA_DRIVER_LIBRARY})
            message(STATUS "Found CUDA driver library (system): ${CUDA_DRIVER_LIBRARY}")
        else()
            # Last resort: try linking directly with -lcuda
            target_link_libraries(OptiXDenoiser PRIVATE cuda)
            message(WARNING "CUDA driver library not found, trying direct -lcuda link")
        endif()
    endif()
endif()

# Additional verification after linking setup
message(STATUS "=== LINKING VERIFICATION ===")
get_target_property(LINKED_LIBS OptiXDenoiser LINK_LIBRARIES)
message(STATUS "Libraries linked to OptiXDenoiser: ${LINKED_LIBS}")

# Debug information summary
message(STATUS "=== BUILD CONFIGURATION SUMMARY ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Debug Mode: ${ENABLE_OPTIX_DEBUG}")
message(STATUS "CUDA Memory Check: ${ENABLE_CUDA_MEMCHECK}")
message(STATUS "AddressSanitizer: ${ENABLE_ASAN}")
message(STATUS "UndefinedBehaviorSanitizer: ${ENABLE_UBSAN}")
message(STATUS "ThreadSanitizer: ${ENABLE_TSAN}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CUDA Available: ${CUDA_AVAILABLE}")
if(CUDAToolkit_FOUND)
    message(STATUS "Using CUDAToolkit")
    message(STATUS "CUDA Runtime Library: Available via CUDA::cudart")
    message(STATUS "CUDA Driver Library: Available via CUDA::cuda_driver")
elseif(CUDA_FOUND)
    message(STATUS "Using legacy CUDA package")
    message(STATUS "CUDA Runtime Library: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA Driver Library: Added via 'cuda' target")
else()
    message(STATUS "CUDA Include Dir: ${CUDA_INCLUDE_DIR}")
    message(STATUS "CUDA Library Dir: ${CUDA_LIBRARY_DIR}")
    message(STATUS "CUDA Runtime Library: ${CUDA_CUDART_LIBRARY}")
    if(CUDA_DRIVER_LIBRARY)
        message(STATUS "CUDA Driver Library: ${CUDA_DRIVER_LIBRARY}")
    else()
        message(STATUS "CUDA Driver Library: Using direct -lcuda link")
    endif()
endif()
message(STATUS "OptiX Include Dir: ${OPTIX_INCLUDE_DIR}")
message(STATUS "OptiX SDK Dir: ${OPTIX_SDK_DIR}")
message(STATUS "OptiX is header-only - no library linking required")

# Sanitizer-specific instructions
if(ENABLE_ASAN OR ENABLE_UBSAN OR ENABLE_TSAN)
    message(STATUS "=== SANITIZER BUILD INSTRUCTIONS ===")
    message(STATUS "1. Build: make -j$(nproc)")
    message(STATUS "2. Test linking: ldd lib/OptiXDenoiser.so | grep -E '(asan|ubsan|tsan)'")
    
    if(ENABLE_ASAN)
        message(STATUS "3. AddressSanitizer environment variables:")
        message(STATUS "   export ASAN_OPTIONS=abort_on_error=1:halt_on_error=1:detect_leaks=1")
        message(STATUS "   export ASAN_SYMBOLIZER_PATH=$(which llvm-symbolizer)")
        message(STATUS "   This will catch buffer overruns and memory corruption immediately")
    endif()
    
    if(ENABLE_UBSAN)
        message(STATUS "4. UndefinedBehaviorSanitizer environment variables:")
        message(STATUS "   export UBSAN_OPTIONS=abort_on_error=1:print_stacktrace=1")
    endif()
    
    if(ENABLE_TSAN)
        message(STATUS "5. ThreadSanitizer environment variables:")
        message(STATUS "   export TSAN_OPTIONS=abort_on_error=1:halt_on_error=1")
    endif()
    
    message(STATUS "6. Run Nuke with sanitizers:")
    message(STATUS "   /usr/local/Nuke14.1v2/Nuke14.1 your_script.nk")
    message(STATUS "   The plugin will abort immediately on memory errors")
endif()

if(ENABLE_OPTIX_DEBUG)
    message(STATUS "=== DEBUG BUILD INSTRUCTIONS ===")
    message(STATUS "1. Build: make -j$(nproc)")
    message(STATUS "2. Test linking: ldd lib/OptiXDenoiser.so | grep cuda")
    message(STATUS "   You should see both libcudart.so and libcuda.so")
    message(STATUS "3. Set environment variables:")
    message(STATUS "   export CUDA_LAUNCH_BLOCKING=1")
    message(STATUS "   export CUDA_MEMCHECK=1")
    message(STATUS "4. Run with memory checker:")
    message(STATUS "   cuda-memcheck --tool memcheck nuke your_script.nk")
    message(STATUS "5. Or use compute-sanitizer (CUDA 11.8+):")
    message(STATUS "   compute-sanitizer --tool memcheck nuke your_script.nk")
endif()

message(STATUS "=== MEMORY DEBUGGING PRIORITY ===")
message(STATUS "For finding the buffer overrun causing the crash:")
message(STATUS "1. HIGHEST PRIORITY: Use AddressSanitizer (ENABLE_ASAN=ON)")
message(STATUS "2. MEDIUM PRIORITY: Use CUDA memory checker")
message(STATUS "3. LOWEST PRIORITY: Use UndefinedBehaviorSanitizer")
message(STATUS "")
message(STATUS "AddressSanitizer will catch the exact location of buffer overrun")
message(STATUS "and provide a detailed stack trace showing where memory corruption occurs.")

message(STATUS "=== POST-BUILD VERIFICATION ===")
message(STATUS "After building, verify CUDA Driver API linking with:")
message(STATUS "  ldd lib/OptiXDenoiser.so | grep -E '(cuda|CUDA)'")
message(STATUS "Expected output should include:")
message(STATUS "  libcudart.so.XX => ...")
message(STATUS "  libcuda.so.1 => ...")
message(STATUS "If libcuda.so.1 is missing, the cuCtxDestroy_v2 error will occur.")

# Installation
install(
    TARGETS OptiXDenoiser
    DESTINATION plugins
    COMPONENT Nuke
)
