// Copyright (c) 2025 - OptiX Denoiser for Nuke
// Based on NVIDIA OptiX SDK examples
// DEBUG VERSION with extensive memory checking and logging

#include "OptiXDenoiser.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <chrono>
#include <mutex>

// Ensure consistent macro definitions between header and source
#define ENABLE_CUDA_DEBUG 1
#define ENABLE_VERBOSE_LOGGING 1

#if ENABLE_CUDA_DEBUG
    #define CUDA_CHECK_DEBUG(call) \
        do { \
            cudaError_t error = call; \
            if (error != cudaSuccess) { \
                std::cout << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
                throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
            } \
            cudaDeviceSynchronize(); \
            error = cudaGetLastError(); \
            if (error != cudaSuccess) { \
                std::cout << "CUDA error after sync at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
                throw std::runtime_error(std::string("CUDA sync error: ") + cudaGetErrorString(error)); \
            } \
        } while(0)
#else
    #define CUDA_CHECK_DEBUG(call) CUDA_CHECK(call)
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

#define OPTIX_CHECK(call) \
    do { \
        OptixResult result = call; \
        if (result != OPTIX_SUCCESS) { \
            throw std::runtime_error(std::string("OptiX error: ") + std::to_string(result)); \
        } \
    } while(0)

#if ENABLE_VERBOSE_LOGGING
    #define DEBUG_LOG(msg) std::cout << "[DEBUG] " << msg << std::endl
    #define DEBUG_LOG_VALUE(name, value) std::cout << "[DEBUG] " << name << ": " << (value) << std::endl
    #define DEBUG_LOG_PTR(name, ptr) std::cout << "[DEBUG] " << name << ": " << static_cast<const void*>(ptr) << std::endl
#else
    #define DEBUG_LOG(msg)
    #define DEBUG_LOG_VALUE(name, value)
    #define DEBUG_LOG_PTR(name, ptr)
#endif

// Memory guard functions
#if ENABLE_MEMORY_GUARDS
void checkCudaMemory(const char* location) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "[MEMORY_GUARD] CUDA error at " << location << ": " << cudaGetErrorString(error) << std::endl;
    }
}

void validatePointer(const void* ptr, size_t /*size*/, const char* name) {
    if (!ptr) {
        throw std::runtime_error(std::string("Null pointer: ") + name);
    }
    
    // Try to verify CUDA pointer attributes if it's a device pointer
    cudaPointerAttributes attributes;
    cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);
    if (error == cudaSuccess) {
        std::string memType = (attributes.type == cudaMemoryTypeDevice) ? "Device" : 
                             (attributes.type == cudaMemoryTypeHost) ? "Host" : "Other";
        DEBUG_LOG_VALUE(std::string(name) + " memory type", memType);
    }
}

#else
    #define checkCudaMemory(location)
    #define validatePointer(ptr, size, name)
#endif

// OptiX context callback
void context_log_cb(uint32_t level, const char* tag, const char* message, void* /*cbdata*/)
{
    if (level < 4) {
        std::cerr << "[OptiX][" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
    }
}

// OptiXDenoiserCore Implementation
void* OptiXDenoiserCore::allocateAlignedCudaMemory(size_t size, size_t& actualSize, const std::string& description)
{
    DEBUG_LOG("=== allocateAlignedCudaMemory START ===");
    DEBUG_LOG_VALUE("Requested size", size);
    DEBUG_LOG_VALUE("Description", description);
    
    // Ensure larger alignment as required by OptiX
    const size_t alignment = 128;
    actualSize = ((size + alignment - 1) / alignment) * alignment;
    
    DEBUG_LOG_VALUE("Aligned size", actualSize);
    
    void* ptr = nullptr;
    
    try {
        CUDA_CHECK_DEBUG(cudaMalloc(&ptr, actualSize));
        DEBUG_LOG_PTR("Allocated pointer", ptr);
        
        // Zero-initialize the memory
        CUDA_CHECK_DEBUG(cudaMemset(ptr, 0, actualSize));
        
        // Track allocated pointers for cleanup
        {
            std::lock_guard<std::mutex> lock(m_cleanupMutex);
            m_allocatedPointers.emplace_back(ptr, actualSize, description);
            m_totalMemoryUsed += actualSize;
        }
        
        // Verify allocation
        cudaPointerAttributes attributes;
        CUDA_CHECK_DEBUG(cudaPointerGetAttributes(&attributes, ptr));
        DEBUG_LOG_VALUE("Memory type", 
            (attributes.type == cudaMemoryTypeDevice) ? "Device" : 
            (attributes.type == cudaMemoryTypeHost) ? "Host" : "Other");
        
        DEBUG_LOG_VALUE("Total memory used", m_totalMemoryUsed);
        DEBUG_LOG_VALUE("Total allocations", m_allocatedPointers.size());
        
    } catch (...) {
        // Clean up on allocation failure
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
        DEBUG_LOG("Allocation failed, cleaned up pointer");
        throw;
    }
    
    checkCudaMemory("allocateAlignedCudaMemory");
    DEBUG_LOG("=== allocateAlignedCudaMemory END ===");
    
    return ptr;
}

void OptiXDenoiserCore::dumpMemoryUsage() const
{
    DEBUG_LOG("=== MEMORY USAGE DUMP ===");
    DEBUG_LOG_VALUE("Total allocations", m_allocatedPointers.size());
    DEBUG_LOG_VALUE("Total memory used", m_totalMemoryUsed);
    
    for (size_t i = 0; i < m_allocatedPointers.size(); ++i) {
        const auto& alloc = m_allocatedPointers[i];
        DEBUG_LOG_VALUE("Allocation " + std::to_string(i), 
                       alloc.description + " - " + std::to_string(alloc.size) + " bytes");
    }
}

bool OptiXDenoiserCore::validateAllAllocations() const
{
    DEBUG_LOG("=== validateAllAllocations START ===");
    
    for (size_t i = 0; i < m_allocatedPointers.size(); ++i) {
        const auto& alloc = m_allocatedPointers[i];
        if (!alloc.ptr) {
            DEBUG_LOG_VALUE("Invalid allocation found at index", i);
            return false;
        }
        
        cudaPointerAttributes attributes;
        cudaError_t error = cudaPointerGetAttributes(&attributes, alloc.ptr);
        if (error != cudaSuccess) {
            DEBUG_LOG_VALUE("Allocation validation failed at index", i);
            DEBUG_LOG_VALUE("CUDA error", cudaGetErrorString(error));
            return false;
        }
    }
    
    DEBUG_LOG("All allocations validated successfully");
    return true;
}

void OptiXDenoiserCore::clampNormalBuffer(float* normalData, size_t pixelCount)
{
    DEBUG_LOG("=== clampNormalBuffer START ===");
    DEBUG_LOG_VALUE("Pixel count", pixelCount);
    
    if (!normalData) {
        DEBUG_LOG("Normal data is null, skipping");
        return;
    }
    
    validatePointer(normalData, pixelCount * 4 * sizeof(float), "normalData");
    
    size_t clamped_count = 0;
    // Clamp normal values to [0.0, 1.0] range as required by OptiX
    for (size_t i = 0; i < pixelCount; ++i) {
        // Process RGB channels (3 floats per pixel for normals)
        for (int c = 0; c < 3; ++c) {
            size_t index = i * 4 + c; // Assuming RGBA format
            float original = normalData[index];
            float clamped = std::max(0.0f, std::min(1.0f, original));
            if (original != clamped) {
                clamped_count++;
            }
            normalData[index] = clamped;
        }
    }
    
    DEBUG_LOG_VALUE("Clamped pixels", clamped_count);
    DEBUG_LOG("=== clampNormalBuffer END ===");
}

void OptiXDenoiserCore::validateHDRRange(const float* imageData, size_t pixelCount, bool& outOfRange)
{
    DEBUG_LOG("=== validateHDRRange START ===");
    DEBUG_LOG_VALUE("Pixel count", pixelCount);
    
    outOfRange = false;
    if (!imageData) {
        DEBUG_LOG("Image data is null, skipping");
        return;
    }
    
    validatePointer(imageData, pixelCount * 4 * sizeof(float), "imageData");
    
    const float maxRecommendedValue = 10000.0f;
    size_t outOfRangeCount = 0;
    float maxValue = 0.0f;
    
    for (size_t i = 0; i < pixelCount; ++i) {
        float r = imageData[i * 4 + 0];
        float g = imageData[i * 4 + 1];
        float b = imageData[i * 4 + 2];
        
        float pixelMax = std::max({r, g, b});
        maxValue = std::max(maxValue, pixelMax);
        
        if (r > maxRecommendedValue || g > maxRecommendedValue || b > maxRecommendedValue) {
            outOfRangeCount++;
        }
    }
    
    DEBUG_LOG_VALUE("Max pixel value found", maxValue);
    DEBUG_LOG_VALUE("Out of range pixels", outOfRangeCount);
    
    if (outOfRangeCount > 0) {
        outOfRange = true;
        std::cerr << "Warning: " << outOfRangeCount << " pixels exceed recommended HDR range [0-10000]. "
                  << "Max value found: " << maxValue << ". This may affect denoising quality." << std::endl;
    }
    
    DEBUG_LOG("=== validateHDRRange END ===");
}

OptixImage2D OptiXDenoiserCore::createOptixImage2D(unsigned int width, unsigned int height, 
                                                   const float* hmem, OptixPixelFormat format)
{
    DEBUG_LOG("=== createOptixImage2D START ===");
    DEBUG_LOG_VALUE("Width", width);
    DEBUG_LOG_VALUE("Height", height);
    DEBUG_LOG_VALUE("Format", static_cast<int>(format));
    DEBUG_LOG_VALUE("Has host memory", (hmem != nullptr));
    
    OptixImage2D oi = {};
    
    // Calculate stride based on format
    size_t pixelSize = 0;
    std::string formatName;
    switch (format) {
        case OPTIX_PIXEL_FORMAT_FLOAT4:
            pixelSize = 4 * sizeof(float);
            formatName = "FLOAT4";
            break;
        case OPTIX_PIXEL_FORMAT_FLOAT3:
            pixelSize = 3 * sizeof(float);
            formatName = "FLOAT3";
            break;
        case OPTIX_PIXEL_FORMAT_FLOAT2:
            pixelSize = 2 * sizeof(float);
            formatName = "FLOAT2";
            break;
        case OPTIX_PIXEL_FORMAT_HALF2:
            pixelSize = 2 * sizeof(uint16_t);
            formatName = "HALF2";
            break;
        default:
            throw std::runtime_error("Unsupported pixel format: " + std::to_string(static_cast<int>(format)));
    }
    
    DEBUG_LOG_VALUE("Using format", formatName);
    DEBUG_LOG_VALUE("Pixel size", pixelSize);
    
    const size_t frame_byte_size = width * height * pixelSize;
    DEBUG_LOG_VALUE("Frame byte size", frame_byte_size);
    
    // Add extra alignment for OptiX requirements
    const size_t alignment = 128;
    const size_t aligned_frame_size = ((frame_byte_size + alignment - 1) / alignment) * alignment;
    DEBUG_LOG_VALUE("Aligned frame size", aligned_frame_size);
    
    size_t actualSize = 0;
    
    // Use aligned allocation with description
    std::string description = "OptixImage2D_" + formatName + "_" + std::to_string(width) + "x" + std::to_string(height);
    oi.data = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(aligned_frame_size, actualSize, description));
    
    if (hmem) {
        DEBUG_LOG("Copying host memory to device");
        validatePointer(hmem, frame_byte_size, "hmem");
        CUDA_CHECK_DEBUG(cudaMemcpy(reinterpret_cast<void*>(oi.data), hmem, frame_byte_size, cudaMemcpyHostToDevice));
        checkCudaMemory("createOptixImage2D - after memcpy");
    }
    
    oi.width = width;
    oi.height = height;
    oi.rowStrideInBytes = width * pixelSize;
    oi.pixelStrideInBytes = pixelSize;
    oi.format = format;
    
    DEBUG_LOG_VALUE("Row stride", oi.rowStrideInBytes);
    DEBUG_LOG_VALUE("Pixel stride", oi.pixelStrideInBytes);
    DEBUG_LOG_PTR("Device pointer", reinterpret_cast<void*>(oi.data));
    
    DEBUG_LOG("=== createOptixImage2D END ===");
    
    return oi;
}

void OptiXDenoiserCore::copyOptixImage2D(OptixImage2D& dest, const OptixImage2D& src)
{
    DEBUG_LOG("=== copyOptixImage2D START ===");
    
    if (dest.format != src.format) {
        throw std::runtime_error("Cannot copy images with different formats");
    }
    
    DEBUG_LOG_VALUE("Source dimensions", std::to_string(src.width) + "x" + std::to_string(src.height));
    DEBUG_LOG_VALUE("Dest dimensions", std::to_string(dest.width) + "x" + std::to_string(dest.height));
    
    const size_t pixelSize = dest.pixelStrideInBytes;
    const size_t copySize = src.width * src.height * pixelSize;
    
    DEBUG_LOG_VALUE("Copy size", copySize);
    DEBUG_LOG_VALUE("Pixel size", pixelSize);
    
    validatePointer(reinterpret_cast<void*>(src.data), copySize, "src.data");
    validatePointer(reinterpret_cast<void*>(dest.data), copySize, "dest.data");
    
    CUDA_CHECK_DEBUG(cudaMemcpy(reinterpret_cast<void*>(dest.data), reinterpret_cast<void*>(src.data), 
                         copySize, cudaMemcpyDeviceToDevice));
    
    checkCudaMemory("copyOptixImage2D");
    DEBUG_LOG("=== copyOptixImage2D END ===");
}

void OptiXDenoiserCore::cleanupCudaResources()
{
    DEBUG_LOG("=== cleanupCudaResources START ===");
    
    std::lock_guard<std::mutex> lock(m_cleanupMutex);
    
    if (m_cleanupInProgress) {
        DEBUG_LOG("Cleanup already in progress, returning");
        return;
    }
    m_cleanupInProgress = true;
    
    try {
        // Wait for any pending CUDA operations
        if (m_stream) {
            DEBUG_LOG("Synchronizing CUDA stream before cleanup");
            CUDA_CHECK(cudaStreamSynchronize(m_stream));
        }
        
        // Free tracked allocations in reverse order
        DEBUG_LOG_VALUE("Number of tracked allocations", m_allocatedPointers.size());
        for (auto it = m_allocatedPointers.rbegin(); it != m_allocatedPointers.rend(); ++it) {
            if (it->ptr) {
                DEBUG_LOG_PTR("Freeing tracked allocation", it->ptr);
                DEBUG_LOG_VALUE("Allocation description", it->description);
                DEBUG_LOG_VALUE("Allocation size", it->size);
                CUDA_CHECK(cudaFree(it->ptr));
                it->ptr = nullptr;
            }
        }
        m_allocatedPointers.clear();
        
        // Clean up stream
        if (m_stream) {
            DEBUG_LOG("Destroying CUDA stream");
            CUDA_CHECK(cudaStreamDestroy(m_stream));
            m_stream = nullptr;
        }
        
        // Clean up CUDA context if we own it
        if (m_cuContext && m_ownsCudaContext) {
            DEBUG_LOG("Destroying CUDA context");
            cuCtxDestroy(m_cuContext);
            m_cuContext = nullptr;
            m_ownsCudaContext = false;
        }
        
        // Reset pointers to zero (they're already freed through tracked allocations)
        m_intensity = 0;
        m_avgColor = 0;
        m_scratch = 0;
        m_state = 0;
        
        // Reset guide layer pointers (they're already freed)
        m_guideLayer.albedo.data = 0;
        m_guideLayer.normal.data = 0;
        m_guideLayer.flow.data = 0;
        
        // Reset layer pointers (they're already freed)
        for (auto& layer : m_layers) {
            layer.input.data = 0;
            layer.output.data = 0;
            layer.previousOutput.data = 0;
        }
        
        DEBUG_LOG_VALUE("Total memory that was used", m_totalMemoryUsed);
        m_totalMemoryUsed = 0;
        
    } catch (const std::exception& e) {
        DEBUG_LOG_VALUE("Exception during CUDA cleanup", e.what());
        // Continue cleanup even if some operations fail
    }
    
    m_cleanupInProgress = false;
    DEBUG_LOG("=== cleanupCudaResources END ===");
}

void OptiXDenoiserCore::cleanupOptixResources()
{
    DEBUG_LOG("=== cleanupOptixResources START ===");
    
    if (m_denoiser) {
        DEBUG_LOG("Destroying OptiX denoiser");
        optixDenoiserDestroy(m_denoiser);
        m_denoiser = nullptr;
    }
    if (m_context) {
        DEBUG_LOG("Destroying OptiX context");
        optixDeviceContextDestroy(m_context);
        m_context = nullptr;
    }
    
    DEBUG_LOG("=== cleanupOptixResources END ===");
}

void OptiXDenoiserCore::init(const Data& data, unsigned int tileWidth, unsigned int tileHeight, bool temporalMode)
{
    DEBUG_LOG("=== OptiXDenoiserCore::init START ===");
    DEBUG_LOG_VALUE("Width", data.width);
    DEBUG_LOG_VALUE("Height", data.height);
    DEBUG_LOG_VALUE("Tile width", tileWidth);
    DEBUG_LOG_VALUE("Tile height", tileHeight);
    DEBUG_LOG_VALUE("Temporal mode", temporalMode);
    DEBUG_LOG_VALUE("Has color", (data.color != nullptr));
    DEBUG_LOG_VALUE("Has albedo", (data.albedo != nullptr));
    DEBUG_LOG_VALUE("Has normal", (data.normal != nullptr));
    DEBUG_LOG_VALUE("Has flow", (data.flow != nullptr));
    DEBUG_LOG_VALUE("Output count", data.outputs.size());

    // Validate input data more thoroughly
    if (!data.color) {
        throw std::runtime_error("Color data is null");
    }
    
    if (data.outputs.empty()) {
        throw std::runtime_error("No output buffers provided");
    }
    
    if (data.width == 0 || data.height == 0) {
        throw std::runtime_error("Invalid dimensions: " + std::to_string(data.width) + "x" + std::to_string(data.height));
    }
    
    if (data.width > 16384 || data.height > 16384) {
        throw std::runtime_error("Dimensions too large for OptiX: " + std::to_string(data.width) + "x" + std::to_string(data.height));
    }

    if (data.normal && !data.albedo) {
        throw std::runtime_error("Albedo is required when normal is provided");
    }

    // Validate buffer sizes
    const size_t expectedPixelCount = static_cast<size_t>(data.width) * data.height;
    DEBUG_LOG_VALUE("Expected pixel count", expectedPixelCount);

    // Clean up any existing resources
    if (m_initialized) {
        DEBUG_LOG("Already initialized, cleaning up first");
        finish();
    }

    m_host_outputs = data.outputs;
    m_temporalMode = temporalMode;
    m_tileWidth = (tileWidth > 0) ? tileWidth : data.width;
    m_tileHeight = (tileHeight > 0) ? tileHeight : data.height;
    m_frameIndex = 0;

    DEBUG_LOG_VALUE("Final tile width", m_tileWidth);
    DEBUG_LOG_VALUE("Final tile height", m_tileHeight);

    // Initialize CUDA with proper context creation
    DEBUG_LOG("Initializing CUDA");
    int deviceCount;
    CUDA_CHECK_DEBUG(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    // Initialize CUDA driver first
    CUresult cuResult = cuInit(0);
    if (cuResult != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to initialize CUDA driver");
    }

    // GET CURRENT CUDA CONTEXT
    CUcontext cuContext = nullptr;
    cuResult = cuCtxGetCurrent(&cuContext);

    // We need a valid context - create one if none exists or if getting current failed
    if (cuResult != CUDA_SUCCESS || cuContext == nullptr) {
        // Create a new CUDA context
        int device;
        CUDA_CHECK_DEBUG(cudaGetDevice(&device));
        
        CUdevice cuDevice;
        cuResult = cuDeviceGet(&cuDevice, device);
        if (cuResult != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to get CUDA device: " + std::to_string(cuResult));
        }
        
        cuResult = cuCtxCreate(&cuContext, 0, cuDevice);
        if (cuResult != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to create CUDA context: " + std::to_string(cuResult));
        }
        
        // Make it current
        cuResult = cuCtxSetCurrent(cuContext);
        if (cuResult != CUDA_SUCCESS) {
            cuCtxDestroy(cuContext);  // Clean up on failure
            throw std::runtime_error("Failed to set CUDA context current: " + std::to_string(cuResult));
        }
        
        m_ownsCudaContext = true;
        DEBUG_LOG("Created new CUDA context");
    } else {
        // We have a valid existing context
        m_ownsCudaContext = false;
        DEBUG_LOG("Using existing CUDA context");
    }

    // Store the context
    m_cuContext = cuContext;

    CUDA_CHECK_DEBUG(cudaStreamCreate(&m_stream));

    // Check CUDA device properties
    int device;
    CUDA_CHECK_DEBUG(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK_DEBUG(cudaGetDeviceProperties(&prop, device));
    DEBUG_LOG_VALUE("CUDA device", std::string(prop.name));
    DEBUG_LOG_VALUE("Compute capability", std::to_string(prop.major) + "." + std::to_string(prop.minor));
    DEBUG_LOG_VALUE("Total global memory", prop.totalGlobalMem);

    checkCudaMemory("After CUDA initialization");

    // Initialize OptiX with the REAL CUDA context
    DEBUG_LOG("Initializing OptiX");
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &m_context));

    // Create denoiser - Always use HDR mode
    DEBUG_LOG("Creating OptiX denoiser");
    OptixDenoiserOptions denoiser_options = {};
    denoiser_options.guideAlbedo = data.albedo ? 1 : 0;
    denoiser_options.guideNormal = data.normal ? 1 : 0;

    DEBUG_LOG_VALUE("Guide albedo enabled", denoiser_options.guideAlbedo);
    DEBUG_LOG_VALUE("Guide normal enabled", denoiser_options.guideNormal);

    OptixDenoiserModelKind modelKind = temporalMode ? 
        OPTIX_DENOISER_MODEL_KIND_TEMPORAL : OPTIX_DENOISER_MODEL_KIND_HDR;

    DEBUG_LOG_VALUE("Model kind", (modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL) ? "Temporal" : "HDR");

    OPTIX_CHECK(optixDenoiserCreate(m_context, modelKind, &denoiser_options, &m_denoiser));

    // Allocate device memory
    DEBUG_LOG("Computing memory requirements");
    OptixDenoiserSizes denoiser_sizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, m_tileWidth, m_tileHeight, &denoiser_sizes));

    DEBUG_LOG_VALUE("State size", denoiser_sizes.stateSizeInBytes);
    DEBUG_LOG_VALUE("Scratch size (with overlap)", denoiser_sizes.withOverlapScratchSizeInBytes);
    DEBUG_LOG_VALUE("Scratch size (without overlap)", denoiser_sizes.withoutOverlapScratchSizeInBytes);
    DEBUG_LOG_VALUE("Overlap window size", denoiser_sizes.overlapWindowSizeInPixels);

    if (tileWidth == 0) {
        m_scratch_size = static_cast<uint32_t>(denoiser_sizes.withoutOverlapScratchSizeInBytes);
        m_overlap = 0;
        DEBUG_LOG("Using non-tiled mode");
    } else {
        m_scratch_size = static_cast<uint32_t>(denoiser_sizes.withOverlapScratchSizeInBytes);
        m_overlap = denoiser_sizes.overlapWindowSizeInPixels;
        DEBUG_LOG("Using tiled mode");
    }

    DEBUG_LOG_VALUE("Final scratch size", m_scratch_size);
    DEBUG_LOG_VALUE("Final overlap", m_overlap);

    // Allocate buffers with proper alignment and descriptions
    DEBUG_LOG("Allocating OptiX buffers");
    size_t actualSize;
    m_intensity = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(sizeof(float), actualSize, "intensity"));
    m_avgColor = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(3 * sizeof(float), actualSize, "avgColor"));
    m_scratch = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(m_scratch_size, actualSize, "scratch"));
    m_state = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(denoiser_sizes.stateSizeInBytes, actualSize, "state"));

    m_state_size = static_cast<uint32_t>(denoiser_sizes.stateSizeInBytes);

    checkCudaMemory("After buffer allocation");

    // Create image layers
    DEBUG_LOG("Creating image layers");
    OptixDenoiserLayer layer = {};
    layer.input = createOptixImage2D(data.width, data.height, data.color);
    layer.output = createOptixImage2D(data.width, data.height);

    if (m_temporalMode) {
        DEBUG_LOG("Creating temporal layer");
        layer.previousOutput = createOptixImage2D(data.width, data.height);
        // Initialize previous output with current input for first frame
        copyOptixImage2D(layer.previousOutput, layer.input);
    }

    m_layers.push_back(layer);

    // Setup guide layers
    DEBUG_LOG("Setting up guide layers");
    if (data.albedo) {
        DEBUG_LOG("Creating albedo guide layer");
        m_guideLayer.albedo = createOptixImage2D(data.width, data.height, data.albedo);
    }
    if (data.normal) {
        DEBUG_LOG("Creating normal guide layer");
        m_guideLayer.normal = createOptixImage2D(data.width, data.height, data.normal);
    }
    if (data.flow && m_temporalMode) {
        DEBUG_LOG("Creating flow guide layer");
        m_guideLayer.flow = createOptixImage2D(data.width, data.height, data.flow, OPTIX_PIXEL_FORMAT_FLOAT2);
    }

    checkCudaMemory("After guide layer creation");

    // Setup denoiser
    DEBUG_LOG("Setting up OptiX denoiser");
    OPTIX_CHECK(optixDenoiserSetup(m_denoiser, m_stream,
                                  m_tileWidth + 2 * m_overlap,
                                  m_tileHeight + 2 * m_overlap,
                                  m_state, m_state_size,
                                  m_scratch, m_scratch_size));

    checkCudaMemory("After denoiser setup");

    // Initialize denoiser parameters for OptiX 7.6
    DEBUG_LOG("Initializing denoiser parameters");
    m_params = {};  // Zero initialize the structure
    m_params.hdrIntensity = m_intensity;
    m_params.hdrAverageColor = m_avgColor;
    m_params.blendFactor = 0.0f;
    m_params.temporalModeUsePreviousLayers = 0;

    m_initialized = true;
    
    // Dump memory usage for debugging
    dumpMemoryUsage();
    
    DEBUG_LOG("=== OptiXDenoiserCore::init END ===");
}

void OptiXDenoiserCore::update(const Data& data)
{
    DEBUG_LOG("=== OptiXDenoiserCore::update START ===");
    
    if (!m_initialized) {
        DEBUG_LOG("Not initialized, skipping update");
        return;
    }

    DEBUG_LOG_VALUE("Frame index", m_frameIndex);

    m_host_outputs = data.outputs;

    // Update input image
    const size_t colorBufferSize = data.width * data.height * sizeof(float4);
    DEBUG_LOG_VALUE("Color buffer size", colorBufferSize);
    
    validatePointer(data.color, colorBufferSize, "data.color");
    validatePointer(reinterpret_cast<void*>(m_layers[0].input.data), colorBufferSize, "input layer data");
    
    CUDA_CHECK_DEBUG(cudaMemcpy(reinterpret_cast<void*>(m_layers[0].input.data), data.color,
                         colorBufferSize, cudaMemcpyHostToDevice));

    checkCudaMemory("After input image update");

    // Update guide layers
    if (data.albedo && m_guideLayer.albedo.data) {
        DEBUG_LOG("Updating albedo guide layer");
        validatePointer(data.albedo, colorBufferSize, "data.albedo");
        CUDA_CHECK_DEBUG(cudaMemcpy(reinterpret_cast<void*>(m_guideLayer.albedo.data), data.albedo,
                             colorBufferSize, cudaMemcpyHostToDevice));
        checkCudaMemory("After albedo update");
    }

    if (data.normal && m_guideLayer.normal.data) {
        DEBUG_LOG("Updating normal guide layer");
        validatePointer(data.normal, colorBufferSize, "data.normal");
        CUDA_CHECK_DEBUG(cudaMemcpy(reinterpret_cast<void*>(m_guideLayer.normal.data), data.normal,
                             colorBufferSize, cudaMemcpyHostToDevice));
        checkCudaMemory("After normal update");
    }

    if (data.flow && m_guideLayer.flow.data && m_temporalMode) {
        DEBUG_LOG("Updating flow guide layer");
        const size_t flowBufferSize = data.width * data.height * 2 * sizeof(float);
        DEBUG_LOG_VALUE("Flow buffer size", flowBufferSize);
        validatePointer(data.flow, flowBufferSize, "data.flow");
        CUDA_CHECK_DEBUG(cudaMemcpy(reinterpret_cast<void*>(m_guideLayer.flow.data), data.flow,
                             flowBufferSize, cudaMemcpyHostToDevice));
        checkCudaMemory("After flow update");
    }

    // Handle temporal mode - swap buffers for next frame
    if (m_temporalMode && m_frameIndex > 0) {
        DEBUG_LOG("Copying output to previous output for temporal mode");
        copyOptixImage2D(m_layers[0].previousOutput, m_layers[0].output);
        m_params.temporalModeUsePreviousLayers = 1;
    }

    m_frameIndex++;
    DEBUG_LOG_VALUE("New frame index", m_frameIndex);
    
    DEBUG_LOG("=== OptiXDenoiserCore::update END ===");
}

void OptiXDenoiserCore::exec()
{
    DEBUG_LOG("=== OptiXDenoiserCore::exec START ===");
    
    if (!m_initialized) {
        DEBUG_LOG("Not initialized, skipping execution");
        return;
    }

    checkCudaMemory("Before OptiX operations");

    // Compute intensity
    DEBUG_LOG("Computing intensity");
    OPTIX_CHECK(optixDenoiserComputeIntensity(m_denoiser, m_stream, &m_layers[0].input,
                                             m_intensity, m_scratch, m_scratch_size));
    
    CUDA_CHECK_DEBUG(cudaStreamSynchronize(m_stream));
    checkCudaMemory("After intensity computation");

    // Compute average color
    DEBUG_LOG("Computing average color");
    OPTIX_CHECK(optixDenoiserComputeAverageColor(m_denoiser, m_stream, &m_layers[0].input,
                                                m_avgColor, m_scratch, m_scratch_size));
    
    CUDA_CHECK_DEBUG(cudaStreamSynchronize(m_stream));
    checkCudaMemory("After average color computation");

    // Denoise - Use tiled denoising for OptiX 7.6 compatibility
    DEBUG_LOG("Starting denoising");
    if (m_tileWidth == m_layers[0].input.width && m_tileHeight == m_layers[0].input.height) {
        DEBUG_LOG("Using non-tiled denoising");
        OPTIX_CHECK(optixDenoiserInvoke(m_denoiser, m_stream, &m_params,
                                       m_state, m_state_size, &m_guideLayer,
                                       m_layers.data(), static_cast<unsigned int>(m_layers.size()),
                                       0, 0, m_scratch, m_scratch_size));
    } else {
        DEBUG_LOG("Using tiled denoising");
        DEBUG_LOG_VALUE("Tile size", std::to_string(m_tileWidth) + "x" + std::to_string(m_tileHeight));
        DEBUG_LOG_VALUE("Overlap", m_overlap);
        OPTIX_CHECK(optixUtilDenoiserInvokeTiled(m_denoiser, m_stream, &m_params,
                                                m_state, m_state_size, &m_guideLayer,
                                                m_layers.data(), static_cast<unsigned int>(m_layers.size()),
                                                m_scratch, m_scratch_size, m_overlap,
                                                m_tileWidth, m_tileHeight));
    }

    CUDA_CHECK_DEBUG(cudaStreamSynchronize(m_stream));
    checkCudaMemory("After denoising");
    
    DEBUG_LOG("=== OptiXDenoiserCore::exec END ===");
}

void OptiXDenoiserCore::getResults()
{
    DEBUG_LOG("=== OptiXDenoiserCore::getResults START ===");
    
    if (!m_initialized || m_host_outputs.empty()) {
        DEBUG_LOG("Not initialized or no output buffers");
        return;
    }

    const uint64_t frame_byte_size = m_layers[0].output.width * m_layers[0].output.height * sizeof(float4);
    DEBUG_LOG_VALUE("Frame byte size", frame_byte_size);
    DEBUG_LOG_VALUE("Output buffer count", m_host_outputs.size());
    
    validatePointer(reinterpret_cast<void*>(m_layers[0].output.data), frame_byte_size, "output layer data");
    validatePointer(m_host_outputs[0], frame_byte_size, "host output buffer");
    
    CUDA_CHECK_DEBUG(cudaMemcpy(m_host_outputs[0], reinterpret_cast<void*>(m_layers[0].output.data),
                         frame_byte_size, cudaMemcpyDeviceToHost));
    
    checkCudaMemory("After result copy");
    DEBUG_LOG("=== OptiXDenoiserCore::getResults END ===");
}

void OptiXDenoiserCore::finish()
{
    DEBUG_LOG("=== OptiXDenoiserCore::finish START ===");
    
    if (!m_initialized) {
        DEBUG_LOG("Not initialized, nothing to finish");
        return;
    }

    try {
        // Dump final memory usage before cleanup
        dumpMemoryUsage();
        
        cleanupCudaResources();
        cleanupOptixResources();
    } catch (const std::exception& e) {
        std::cerr << "Error during cleanup: " << e.what() << std::endl;
    }

    m_layers.clear();
    m_host_outputs.clear();
    m_guideLayer = {};
    m_params = {};
    m_frameIndex = 0;
    
    m_initialized = false;
    
    DEBUG_LOG("=== OptiXDenoiserCore::finish END ===");
}

void OptiXDenoiserCore::convertMotionVectors(const float* motionRGBA, float* motionXY, 
                                            unsigned int width, unsigned int height)
{
    DEBUG_LOG("=== convertMotionVectors START ===");
    DEBUG_LOG_VALUE("Width", width);
    DEBUG_LOG_VALUE("Height", height);
    
    if (!motionRGBA || !motionXY) {
        throw std::runtime_error("Null motion vector buffers");
    }
    
    const size_t pixelCount = static_cast<size_t>(width) * height;
    DEBUG_LOG_VALUE("Pixel count", pixelCount);
    
    validatePointer(motionRGBA, pixelCount * 4 * sizeof(float), "motionRGBA");
    validatePointer(motionXY, pixelCount * 2 * sizeof(float), "motionXY");
    
    for (unsigned int i = 0; i < pixelCount; ++i) {
        motionXY[i * 2 + 0] = motionRGBA[i * 4 + 0]; // X component
        motionXY[i * 2 + 1] = motionRGBA[i * 4 + 1]; // Y component
    }
    
    DEBUG_LOG("=== convertMotionVectors END ===");
}

#ifdef ENABLE_MEMORY_GUARDS
void OptiXDenoiserIop::validateBufferIntegrity(const std::vector<float>& buffer, const std::string& name) {
    if (buffer.empty()) {
        DEBUG_LOG_VALUE("Buffer " + name + " is empty", "WARNING");
        return;
    }
    
    // Check for NaN or infinite values
    size_t invalidCount = 0;
    for (size_t i = 0; i < buffer.size(); ++i) {
        if (!std::isfinite(buffer[i])) {
            invalidCount++;
        }
    }
    
    if (invalidCount > 0) {
        DEBUG_LOG_VALUE("Buffer " + name + " has invalid values", invalidCount);
    }
}
#endif

// Nuke Plugin Implementation
OptiXDenoiserIop::OptiXDenoiserIop(Node *node) : PlanarIop(node)
{
    DEBUG_LOG("=== OptiXDenoiserIop::OptiXDenoiserIop START ===");
    
    // Initialize parameters
    m_bTemporal = false;
    m_bInitialized = false;
    m_tileWidth = 2048;   // DEFAULT: 2048 for better performance
    m_tileHeight = 2048;  // DEFAULT: 2048 for better performance
    m_numRuns = 1;
    m_blendFactor = 0.0f;

    m_width = 0;
    m_height = 0;

    m_defaultChannels.clear();
    m_defaultChannels += Chan_Red;
    m_defaultChannels += Chan_Green;
    m_defaultChannels += Chan_Blue;
    m_defaultChannels += Chan_Alpha;
    
    m_defaultNumberOfChannels = 4;  // Always 4 (RGBA)

    m_denoiser = std::make_unique<OptiXDenoiserCore>();
    m_frameCounter = 0;
    
    // Initialize timing
    m_lastAbortCheck = std::chrono::steady_clock::now();
    
    DEBUG_LOG_VALUE("Default channels size", m_defaultNumberOfChannels);
    DEBUG_LOG_VALUE("Default tile size", std::to_string(m_tileWidth) + "x" + std::to_string(m_tileHeight));
    DEBUG_LOG("=== OptiXDenoiserIop::OptiXDenoiserIop END ===");
}

OptiXDenoiserIop::~OptiXDenoiserIop()
{
    DEBUG_LOG("=== OptiXDenoiserIop::~OptiXDenoiserIop START ===");
    
    try {
        // Ensure denoiser is properly cleaned up
        if (m_denoiser) {
            if (m_bInitialized) {
                DEBUG_LOG("Finishing denoiser in destructor");
                m_denoiser->finish();
                m_bInitialized = false;
            }
            m_denoiser.reset();
        }
        
        // Force cleanup of host buffers with explicit memory release
        safeBufferClear(m_colorBuffer, "color");
        safeBufferClear(m_albedoBuffer, "albedo");
        safeBufferClear(m_normalBuffer, "normal");
        safeBufferClear(m_motionBuffer, "motion");
        safeBufferClear(m_motionXYBuffer, "motionXY");
        safeBufferClear(m_outputBuffer, "output");
        
    } catch (const std::exception& e) {
        DEBUG_LOG_VALUE("Exception in destructor", e.what());
        // Don't throw from destructor
    } catch (...) {
        DEBUG_LOG("Unknown exception in destructor");
        // Don't throw from destructor
    }
    
    DEBUG_LOG("=== OptiXDenoiserIop::~OptiXDenoiserIop END ===");
}

void OptiXDenoiserIop::safeBufferClear(std::vector<float>& buffer, const std::string& name)
{
    if (!buffer.empty()) {
        DEBUG_LOG_VALUE("Clearing buffer", name);
        DEBUG_LOG_VALUE("Buffer size before clear", buffer.size());
        buffer.clear();
        buffer.shrink_to_fit();
        
        // Force memory release by swapping with empty vector
        std::vector<float> empty;
        buffer.swap(empty);
    }
}

bool OptiXDenoiserIop::checkAbortWithTimeout(const std::chrono::steady_clock::time_point& startTime)
{
    auto now = std::chrono::steady_clock::now();
    
    // Check abort status every ABORT_CHECK_INTERVAL
    if (now - m_lastAbortCheck >= ABORT_CHECK_INTERVAL) {
        m_lastAbortCheck = now;
        
        if (aborted() || cancelled()) {
            DEBUG_LOG("Operation aborted by user");
            return true;
        }
        
        // Check for timeout
        if (now - startTime >= OPERATION_TIMEOUT) {
            DEBUG_LOG("Operation timed out");
            return true;
        }
    }
    
    return false;
}

void OptiXDenoiserIop::knobs(Knob_Callback f)
{
    Divider(f, "Performance Settings");
    
    Int_knob(f, &m_tileWidth, "tile_width", "Tile Width");
    Tooltip(f, "Tile width for memory optimization (default: 2048, 0 = no tiling). 2048 is recommended for best performance.");
    SetFlags(f, Knob::STARTLINE);

    Int_knob(f, &m_tileHeight, "tile_height", "Tile Height");
    Tooltip(f, "Tile height for memory optimization (default: 2048, 0 = no tiling). 2048 is recommended for best performance.");

    Divider(f, "Cache Management");
    
    Button(f, "clear_cache", "Clear GPU Cache");
    Tooltip(f, "Manually clear OptiX GPU cache to free memory. Useful for large images or memory issues.");
    SetFlags(f, Knob::STARTLINE);

    Divider(f, "");
    
    Text_knob(f, "OptiXDenoiser by Peter Mercell v1.01 / 2025 - PERFORMANCE BUILD", "");
    SetFlags(f, Knob::STARTLINE);
    
    Text_knob(f, "Default: 2048x2048 tiling for optimal performance", "");
    SetFlags(f, Knob::STARTLINE);
}

int OptiXDenoiserIop::knob_changed(Knob* k)
{
    DEBUG_LOG("=== OptiXDenoiserIop::knob_changed START ===");
    DEBUG_LOG_VALUE("Knob name", k->name().c_str());
    
    if (k->is("inputs")) {
        DEBUG_LOG("Inputs knob changed");
        bool shouldUseTemporal = hasMotionVectorsConnected();
        DEBUG_LOG_VALUE("Should use temporal", shouldUseTemporal);
        DEBUG_LOG_VALUE("Current temporal", m_bTemporal);
        
        if (m_bTemporal != shouldUseTemporal) {
            DEBUG_LOG("Temporal mode changed, forcing re-initialization");
            m_bTemporal = shouldUseTemporal;
            // Force re-initialization of denoiser
            if (m_denoiser && m_bInitialized) {
                DEBUG_LOG("Cleaning up existing denoiser");
                m_denoiser->finish();
                m_bInitialized = false;
                m_frameCounter = 0;  // Reset frame counter
            }
        }
        invalidate();
        DEBUG_LOG("=== OptiXDenoiserIop::knob_changed END ===");
        return 1;
    }
    
    if (k->is("tile_width") || k->is("tile_height")) {
        DEBUG_LOG("Tile size knob changed");
        DEBUG_LOG_VALUE("New tile width", m_tileWidth);
        DEBUG_LOG_VALUE("New tile height", m_tileHeight);
        
        // Force re-initialization if memory settings changed
        if (m_denoiser && m_bInitialized) {
            DEBUG_LOG("Cleaning up existing denoiser due to tile size change");
            m_denoiser->finish();
            m_bInitialized = false;
            m_frameCounter = 0;  // Reset frame counter
        }
        invalidate();
        DEBUG_LOG("=== OptiXDenoiserIop::knob_changed END ===");
        return 1;
    }
    
    // NEW: Manual cache clear button
    if (k->is("clear_cache")) {
        DEBUG_LOG("Manual cache clear requested");
        if (m_denoiser && m_bInitialized) {
            DEBUG_LOG("Clearing OptiX GPU cache manually");
            m_denoiser->finish();
            m_bInitialized = false;
            
            // Also clear host buffers to free system memory
            safeBufferClear(m_colorBuffer, "color");
            safeBufferClear(m_albedoBuffer, "albedo");
            safeBufferClear(m_normalBuffer, "normal");
            safeBufferClear(m_motionBuffer, "motion");
            safeBufferClear(m_motionXYBuffer, "motionXY");
            safeBufferClear(m_outputBuffer, "output");
            
            DEBUG_LOG("GPU and host memory cache cleared");
            
            // Print memory usage info
            if (m_denoiser) {
                DEBUG_LOG_VALUE("GPU allocations before clear", m_denoiser->getAllocationCount());
                DEBUG_LOG_VALUE("GPU memory before clear (bytes)", m_denoiser->getTotalMemoryUsed());
            }
        } else {
            DEBUG_LOG("No cache to clear - denoiser not initialized");
        }
        // Don't invalidate - this is just a cleanup operation
        DEBUG_LOG("=== OptiXDenoiserIop::knob_changed END ===");
        return 1;
    }
    
    DEBUG_LOG("=== OptiXDenoiserIop::knob_changed END ===");
    return 0;
}

bool OptiXDenoiserIop::hasMotionVectorsConnected() const
{
    bool hasMotion = (node_inputs() > 3 && input(3) != nullptr);
    DEBUG_LOG_VALUE("Has motion vectors", hasMotion);
    DEBUG_LOG_VALUE("Node inputs", node_inputs());
    return hasMotion;
}

const char *OptiXDenoiserIop::input_label(int n, char *) const
{
    switch (n) {
    case 0: return "beauty";
    case 1: return "albedo";
    case 2: return "normal";
    case 3: return "motion";
    default: return nullptr;
    }
}

void OptiXDenoiserIop::_validate(bool for_real)
{
    DEBUG_LOG("=== OptiXDenoiserIop::_validate START ===");
    DEBUG_LOG_VALUE("For real", for_real);
    
    copy_info();
    
    ChannelSet newChannels;
    newChannels += Chan_Red;
    newChannels += Chan_Green;
    newChannels += Chan_Blue;
    newChannels += Chan_Alpha;
    info_.channels() = newChannels;
    
    bool shouldUseTemporal = hasMotionVectorsConnected();
    
    if (m_bTemporal != shouldUseTemporal) {
        DEBUG_LOG("Temporal mode mismatch in validate, forcing re-init");
        m_bTemporal = shouldUseTemporal;
        // Force re-initialization
        if (m_denoiser) {
            m_denoiser->finish();
            m_bInitialized = false;
        }
    }
    
    DEBUG_LOG("=== OptiXDenoiserIop::_validate END ===");
}

void OptiXDenoiserIop::getRequests(const Box& box, const ChannelSet& channels, int count, RequestOutput& /*reqData*/) const
{
    // Cache the last request to avoid redundant processing
    static Box lastBox;
    static ChannelSet lastChannels;
    static int lastCount = -1;
    
    // Check if this is the same request as last time
    if (box == lastBox && channels == lastChannels && count == lastCount) {
        // Same request - don't spam the log
        return;
    }
    
    DEBUG_LOG("=== OptiXDenoiserIop::getRequests START ===");
    DEBUG_LOG_VALUE("Requested box", std::string("[") + std::to_string(box.x()) + "," + std::to_string(box.y()) + 
                    " " + std::to_string(box.r()) + "," + std::to_string(box.t()) + "]");
    DEBUG_LOG_VALUE("Output channels requested", channels.size());
    DEBUG_LOG_VALUE("Input count", getInputs().size());
    
    // Update cache
    lastBox = box;
    lastChannels = channels;
    lastCount = count;
    
    // Define RGBA channels we actually need
    ChannelSet rgbaChannels;
    rgbaChannels += Chan_Red;
    rgbaChannels += Chan_Green;
    rgbaChannels += Chan_Blue;
    rgbaChannels += Chan_Alpha;
    
    // ALWAYS REQUEST THE FULL REQUESTED AREA - no size limitations
    const Box requestBox = box;  // Always request exactly what's needed
    
    DEBUG_LOG_VALUE("Requesting from inputs", std::string("[") + std::to_string(requestBox.x()) + "," + 
                    std::to_string(requestBox.y()) + " " + std::to_string(requestBox.r()) + "," + 
                    std::to_string(requestBox.t()) + "]");
    DEBUG_LOG("Requesting full area from all inputs (no size limitations)");
    
    for (int i = 0, endI = getInputs().size(); i < endI; i++) {
        if (input(i)) {
            // Only request RGBA channels from each input
            const ChannelSet availableChannels = input(i)->info().channels();
            
            ChannelSet requestChannels;
            foreach(z, rgbaChannels) {
                if (availableChannels.contains(z)) {
                    requestChannels += z;
                }
            }
            
            if (!requestChannels.empty()) {
                // CRITICAL: Always request the full box area from each input
                input(i)->request(requestBox, requestChannels, count);
                
                DEBUG_LOG_VALUE(std::string("Input ") + std::to_string(i) + " requesting channels", requestChannels.size());
            }
        }
    }
    
    DEBUG_LOG("=== OptiXDenoiserIop::getRequests END ===");
}

bool OptiXDenoiserIop::validateCUDA()
{
    DEBUG_LOG("=== validateCUDA START ===");
    
    try {
        // First check if CUDA runtime is available
        cudaError_t error = cudaGetLastError();  // Clear any previous errors
        
        int deviceCount;
        error = cudaGetDeviceCount(&deviceCount);
        DEBUG_LOG_VALUE("CUDA device count", deviceCount);
        DEBUG_LOG_VALUE("CUDA error", cudaGetErrorString(error));
        
        if (error != cudaSuccess) {
            DEBUG_LOG_VALUE("CUDA runtime error", cudaGetErrorString(error));
            return false;
        }
        
        if (deviceCount == 0) {
            DEBUG_LOG("No CUDA devices available");
            return false;
        }
        
        // Check if we can actually use the device
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, 0);
        if (error != cudaSuccess) {
            DEBUG_LOG_VALUE("Failed to get device properties", cudaGetErrorString(error));
            return false;
        }
        
        DEBUG_LOG_VALUE("Device name", std::string(prop.name));
        DEBUG_LOG_VALUE("Compute capability", std::to_string(prop.major) + "." + std::to_string(prop.minor));
        DEBUG_LOG_VALUE("Total global memory", prop.totalGlobalMem);
        
        if (prop.major < 5) {
            DEBUG_LOG("Compute capability too low (< 5.0)");
            return false;
        }
        
        // Test basic CUDA operation
        void* testPtr;
        error = cudaMalloc(&testPtr, 1024);
        if (error == cudaSuccess) {
            cudaFree(testPtr);
            DEBUG_LOG("Basic CUDA allocation test passed");
        } else {
            DEBUG_LOG_VALUE("Basic CUDA allocation test failed", cudaGetErrorString(error));
            return false;
        }
        
        DEBUG_LOG("CUDA validation successful");
        return true;
    } catch (...) {
        DEBUG_LOG("Exception during CUDA validation");
        return false;
    }
}

void OptiXDenoiserIop::allocateBuffers()
{
    DEBUG_LOG("=== allocateBuffers START ===");
    DEBUG_LOG_VALUE("Image dimensions", std::to_string(m_width) + "x" + std::to_string(m_height));
    
    if (m_width == 0 || m_height == 0) {
        throw std::runtime_error("Invalid dimensions for buffer allocation");
    }
    
    const size_t pixelCount = static_cast<size_t>(m_width) * m_height;
    const size_t bufferSizeRGBA = pixelCount * 4; // 4 floats per pixel
    const size_t bufferSizeXY = pixelCount * 2;   // 2 floats per pixel for motion
    
    DEBUG_LOG_VALUE("Pixel count", pixelCount);
    DEBUG_LOG_VALUE("RGBA buffer size (floats)", bufferSizeRGBA);
    DEBUG_LOG_VALUE("XY buffer size (floats)", bufferSizeXY);
    
    // Clear existing buffers safely
    safeBufferClear(m_colorBuffer, "color");
    safeBufferClear(m_albedoBuffer, "albedo");
    safeBufferClear(m_normalBuffer, "normal");
    safeBufferClear(m_motionBuffer, "motion");
    safeBufferClear(m_outputBuffer, "output");
    safeBufferClear(m_motionXYBuffer, "motionXY");
    
    // Allocate with proper sizes and initialization
    try {
        m_colorBuffer.resize(bufferSizeRGBA, 0.0f);
        m_albedoBuffer.resize(bufferSizeRGBA, 0.0f);
        m_normalBuffer.resize(bufferSizeRGBA, 0.0f);
        m_motionBuffer.resize(bufferSizeRGBA, 0.0f);
        m_outputBuffer.resize(bufferSizeRGBA, 0.0f);
        m_motionXYBuffer.resize(bufferSizeXY, 0.0f);
    } catch (const std::bad_alloc& e) {
        throw std::runtime_error("Failed to allocate host buffers: " + std::string(e.what()));
    }
    
    DEBUG_LOG_VALUE("Color buffer actual size", m_colorBuffer.size());
    DEBUG_LOG_VALUE("Albedo buffer actual size", m_albedoBuffer.size());
    DEBUG_LOG_VALUE("Normal buffer actual size", m_normalBuffer.size());
    DEBUG_LOG_VALUE("Motion buffer actual size", m_motionBuffer.size());
    DEBUG_LOG_VALUE("Output buffer actual size", m_outputBuffer.size());
    DEBUG_LOG_VALUE("Motion XY buffer actual size", m_motionXYBuffer.size());
    
    // Verify buffer allocation
    if (m_colorBuffer.size() != bufferSizeRGBA) {
        throw std::runtime_error("Color buffer allocation failed");
    }
    if (m_outputBuffer.size() != bufferSizeRGBA) {
        throw std::runtime_error("Output buffer allocation failed");
    }
    
    DEBUG_LOG("=== allocateBuffers END ===");
}

void OptiXDenoiserIop::readInputPlanes()
{
    DEBUG_LOG("=== readInputPlanes START ===");
    
    // Use processing bounds instead of format bounds
    const Box processingBounds = m_processingBounds;
    const Box imageFormat = info().format();
    
    DEBUG_LOG_VALUE("Processing bounds", std::string("[") + std::to_string(processingBounds.x()) + "," + 
                    std::to_string(processingBounds.y()) + " " + std::to_string(processingBounds.r()) + "," + 
                    std::to_string(processingBounds.t()) + "]");
    DEBUG_LOG_VALUE("Image format", std::string("[") + std::to_string(imageFormat.x()) + "," + 
                    std::to_string(imageFormat.y()) + " " + std::to_string(imageFormat.r()) + "," + 
                    std::to_string(imageFormat.t()) + "]");
    
    // Define RGBA channels we actually need
    ChannelSet rgbaChannels;
    rgbaChannels += Chan_Red;
    rgbaChannels += Chan_Green;
    rgbaChannels += Chan_Blue;
    rgbaChannels += Chan_Alpha;
    
    // CRITICAL: Our buffers are now sized to processing bounds
    const size_t expectedBufferSize = static_cast<size_t>(m_width) * m_height * 4;
    DEBUG_LOG_VALUE("Expected buffer size (processing-based)", expectedBufferSize);
    DEBUG_LOG_VALUE("Buffer dimensions", std::to_string(m_width) + "×" + std::to_string(m_height));
    
    // CRITICAL: Validate buffer dimensions match expected size
    if (m_colorBuffer.size() != expectedBufferSize) {
        throw std::runtime_error("Color buffer size mismatch: expected " + 
                               std::to_string(expectedBufferSize) + 
                               ", got " + std::to_string(m_colorBuffer.size()));
    }
    
    const auto startTime = std::chrono::steady_clock::now();
    
    for (int i = 0; i < node_inputs(); ++i) {
        if (checkAbortWithTimeout(startTime)) {
            DEBUG_LOG("Operation aborted/cancelled or timed out during input reading");
            throw std::runtime_error("Operation cancelled by user or timed out");
        }

        DEBUG_LOG_VALUE("Processing input", i);

        Iop* inputIop = dynamic_cast<Iop*>(input(i));
        if (!inputIop || !inputIop->tryValidate(true)) {
            DEBUG_LOG_VALUE("Input invalid or unavailable", i);
            continue;
        }

        try {
            // CRITICAL: Request the full processing bounds from input
            Box inputBounds = processingBounds;
            
            DEBUG_LOG_VALUE(std::string("Input ") + std::to_string(i) + " requested bounds", 
                           std::string("[") + std::to_string(inputBounds.x()) + "," + 
                           std::to_string(inputBounds.y()) + " " + std::to_string(inputBounds.r()) + "," + 
                           std::to_string(inputBounds.t()) + "]");

            // Since we're requesting exactly what we need, offsets are 0
            int inputWidth = inputBounds.w();
            int inputHeight = inputBounds.h();
            int offsetX = 0;  // No offset since we match exactly
            int offsetY = 0;
            
            DEBUG_LOG_VALUE(std::string("Input ") + std::to_string(i) + " size", 
                           std::to_string(inputWidth) + "×" + std::to_string(inputHeight));

            // Only request RGBA channels from this input
            const ChannelSet availableChannels = inputIop->info().channels();
            ChannelSet requestChannels;
            foreach(z, rgbaChannels) {
                if (availableChannels.contains(z)) {
                    requestChannels += z;
                }
            }

            DEBUG_LOG_VALUE(std::string("Input ") + std::to_string(i) + " available channels", availableChannels.size());
            DEBUG_LOG_VALUE(std::string("Input ") + std::to_string(i) + " requesting channels", requestChannels.size());
            
            if (requestChannels.empty()) {
                DEBUG_LOG_VALUE("No RGBA channels available in input", i);
                continue;
            }
            
            inputIop->request(inputBounds, requestChannels, 0);
            
            if (checkAbortWithTimeout(startTime)) {
                DEBUG_LOG("Operation aborted before fetchPlane");
                throw std::runtime_error("Operation cancelled by user");
            }
            
            // Create plane with only the channels we requested
            ImagePlane inputPlane(inputBounds, false, requestChannels, requestChannels.size());
            inputIop->fetchPlane(inputPlane);
            
            // Select target buffer
            float* targetBuffer = nullptr;
            size_t targetBufferSize = 0;
            std::string bufferName;
            
            switch (i) {
            case 0: 
                targetBuffer = m_colorBuffer.data(); 
                targetBufferSize = m_colorBuffer.size();
                bufferName = "color";
                break;
            case 1: 
                targetBuffer = m_albedoBuffer.data(); 
                targetBufferSize = m_albedoBuffer.size();
                bufferName = "albedo";
                break;
            case 2: 
                targetBuffer = m_normalBuffer.data(); 
                targetBufferSize = m_normalBuffer.size();
                bufferName = "normal";
                break;
            case 3: 
                targetBuffer = m_motionBuffer.data(); 
                targetBufferSize = m_motionBuffer.size();
                bufferName = "motion";
                break;
            }

            if (targetBuffer) {
                DEBUG_LOG_VALUE(std::string("Target buffer (") + bufferName + ") size", targetBufferSize);
                
                if (targetBufferSize < expectedBufferSize) {
                    throw std::runtime_error("Buffer size mismatch for " + bufferName);
                }
                
                validatePointer(targetBuffer, targetBufferSize * sizeof(float), (bufferName + " buffer").c_str());

                auto chanStride = inputPlane.chanStride();
                DEBUG_LOG_VALUE("Channel stride", chanStride);
                
                // Define RGBA channel mapping
                Channel rgbaChannelOrder[4] = {Chan_Red, Chan_Green, Chan_Blue, Chan_Alpha};
                int channelMap[4] = {-1, -1, -1, -1};
                int planeChannelIndex = 0;
                
                for (Channel ch : requestChannels) {
                    for (int rgbaIndex = 0; rgbaIndex < 4; rgbaIndex++) {
                        if (ch == rgbaChannelOrder[rgbaIndex]) {
                            channelMap[rgbaIndex] = planeChannelIndex;
                            break;
                        }
                    }
                    planeChannelIndex++;
                }
                
                DEBUG_LOG_VALUE("Red channel index", channelMap[0]);
                DEBUG_LOG_VALUE("Green channel index", channelMap[1]);
                DEBUG_LOG_VALUE("Blue channel index", channelMap[2]);
                DEBUG_LOG_VALUE("Alpha channel index", channelMap[3]);
                
                // SIMPLIFIED: Direct 1:1 mapping since bounds match exactly
                size_t pixelsProcessed = 0;
                size_t channelsProcessed = 0;
                
                for (unsigned int y = 0; y < m_height; y++) {
                    if (y % 100 == 0 && checkAbortWithTimeout(startTime)) {
                        DEBUG_LOG("Operation aborted during pixel processing");
                        throw std::runtime_error("Operation cancelled by user");
                    }
                    
                    for (unsigned int x = 0; x < m_width; x++) {
                        for (int c = 0; c < 4; c++) {
                            size_t targetIndex = (static_cast<size_t>(y) * m_width + x) * 4 + c;
                            SAFE_BUFFER_ACCESS(targetBuffer, targetIndex, targetBufferSize, bufferName.c_str());
                            
                            if (channelMap[c] >= 0) {
                                // Direct mapping since bounds match
                                const float* indata = &inputPlane.readable()[chanStride * channelMap[c]];
                                size_t inputIndex = static_cast<size_t>(y) * inputWidth + x;
                                
                                size_t inputPlaneSize = static_cast<size_t>(inputWidth) * inputHeight;
                                if (inputIndex >= inputPlaneSize) {
                                    std::cerr << "FATAL: Input index out of bounds: " << inputIndex 
                                              << " >= " << inputPlaneSize << std::endl;
                                    std::abort();
                                }
                                
                                float value = indata[inputIndex];
                                if (!std::isfinite(value)) {
                                    value = 0.0f;
                                }
                                targetBuffer[targetIndex] = value;
                            } else {
                                // Channel doesn't exist, set default value
                                targetBuffer[targetIndex] = (c == 3) ? 1.0f : 0.0f;
                            }
                            channelsProcessed++;
                        }
                        pixelsProcessed++;
                    }
                }
                
                DEBUG_LOG_VALUE(std::string("Pixels processed for ") + bufferName, pixelsProcessed);
                DEBUG_LOG_VALUE(std::string("Channels processed for ") + bufferName, channelsProcessed);
            }
            
        } catch (const std::exception& e) {
            DEBUG_LOG_VALUE("Exception during input processing", e.what());
            throw;
        }
    }
    
    // Process normal data if available
    if (node_inputs() > 2 && input(2) && !m_normalBuffer.empty()) {
        DEBUG_LOG("Processing normal buffer");
        m_denoiser->clampNormalBuffer(m_normalBuffer.data(), m_width * m_height);
    }
    
    // Validate HDR range
    if (!m_colorBuffer.empty()) {
        DEBUG_LOG("Validating HDR range");
        bool outOfRange = false;
        m_denoiser->validateHDRRange(m_colorBuffer.data(), m_width * m_height, outOfRange);
    }
    
    // Convert motion vectors if needed
    if (node_inputs() > 3 && input(3) && m_bTemporal) {
        DEBUG_LOG("Converting motion vectors to XY format");
        m_denoiser->convertMotionVectors(m_motionBuffer.data(), m_motionXYBuffer.data(), m_width, m_height);
    }
    
    DEBUG_LOG("=== readInputPlanes END ===");
}

void OptiXDenoiserIop::writeOutputPlane(ImagePlane& plane)
{
    DEBUG_LOG("=== writeOutputPlane START ===");
    
    float* outputData = m_outputBuffer.data();
    const int bufferWidth = m_width;
    const int bufferHeight = m_height;
    
    // Get plane info
    const Box planeBounds = plane.bounds();
    const int planeWidth = planeBounds.w();
    const int planeHeight = planeBounds.h();
    
    DEBUG_LOG_VALUE("Buffer dimensions", std::to_string(bufferWidth) + "x" + std::to_string(bufferHeight));
    DEBUG_LOG_VALUE("Plane dimensions", std::to_string(planeWidth) + "x" + std::to_string(planeHeight));
    DEBUG_LOG_VALUE("Processing channels", 4);
    
    if (!outputData) {
        throw std::runtime_error("Output buffer is null");
    }
    
    // SIMPLE: Direct 1:1 mapping since we processed the exact requested area
    if (bufferWidth != planeWidth || bufferHeight != planeHeight) {
        throw std::runtime_error("Buffer and plane dimensions mismatch");
    }
    
    validatePointer(outputData, m_outputBuffer.size() * sizeof(float), "output buffer");

    size_t pixelsWritten = 0;
    size_t channelsWritten = 0;

    // Direct copy - no coordinate mapping needed
    for (int chanNo = 0; chanNo < 4; chanNo++) {
        DEBUG_LOG_VALUE("Processing channel", chanNo);
        
        float* outdata = &plane.writable()[plane.chanStride() * chanNo];
        size_t expectedPlaneSize = static_cast<size_t>(planeWidth) * planeHeight;
        validatePointer(outdata, expectedPlaneSize * sizeof(float), ("plane channel " + std::to_string(chanNo)).c_str());

        for (int y = 0; y < planeHeight; y++) {
            for (int x = 0; x < planeWidth; x++) {
                size_t bufferIndex = (static_cast<size_t>(y) * bufferWidth + x) * 4 + chanNo;
                size_t planeIndex = static_cast<size_t>(y) * planeWidth + x;
                
                if (bufferIndex >= m_outputBuffer.size()) {
                    throw std::runtime_error("Buffer index out of bounds: " + std::to_string(bufferIndex));
                }
                if (planeIndex >= expectedPlaneSize) {
                    throw std::runtime_error("Plane index out of bounds: " + std::to_string(planeIndex));
                }
                
                outdata[planeIndex] = outputData[bufferIndex];
                channelsWritten++;
            }
        }
        pixelsWritten = static_cast<size_t>(planeWidth) * planeHeight;
    }
    
    DEBUG_LOG_VALUE("Pixels written per channel", pixelsWritten);
    DEBUG_LOG_VALUE("Total channels written", channelsWritten);
    
    DEBUG_LOG("=== writeOutputPlane END ===");
}

void OptiXDenoiserIop::handleDenoiserError(const std::exception& e)
{
    DEBUG_LOG("=== handleDenoiserError START ===");
    DEBUG_LOG_VALUE("Error message", e.what());
    
    std::string errorMsg = "OptiX Denoiser error: ";
    errorMsg += e.what();
    error(errorMsg.c_str());
    
    // Reset denoiser state
    if (m_denoiser) {
        DEBUG_LOG("Finishing denoiser due to error");
        try {
            m_denoiser->finish();
        } catch (...) {
            DEBUG_LOG("Exception during error cleanup - continuing");
        }
    }
    m_bInitialized = false;
    
    // Copy input to output as fallback
    if (!m_colorBuffer.empty() && !m_outputBuffer.empty()) {
        DEBUG_LOG("Copying input to output as fallback");
        if (m_colorBuffer.size() == m_outputBuffer.size()) {
            std::copy(m_colorBuffer.begin(), m_colorBuffer.end(), m_outputBuffer.begin());
        } else {
            DEBUG_LOG("Buffer size mismatch, cannot copy");
        }
    }
    
    DEBUG_LOG("=== handleDenoiserError END ===");
}

void OptiXDenoiserIop::renderStripe(ImagePlane& plane)
{
    DEBUG_LOG("=== OptiXDenoiserIop::renderStripe START ===");
    
    // Performance timing
    auto renderStart = std::chrono::steady_clock::now();
    
    // Add early abort check to prevent cascade
    static int consecutiveFailures = 0;
    if (consecutiveFailures > 3) {
        DEBUG_LOG("Too many consecutive failures, skipping");
        error("OptiX denoiser failed multiple times - check CUDA/OptiX installation");
        return;
    }
    
    // RAII cleanup guard for exception safety
    CleanupGuard guard(this);
    
    if (aborted() || cancelled()) {
        DEBUG_LOG("Operation aborted/cancelled");
        return;
    }

    // Validate all required inputs are properly connected
    if (!input(0)) {
        error("Primary input (beauty) is required");
        return;
    }
    
    // Check for temporal mode requirements
    bool shouldUseTemporal = hasMotionVectorsConnected();
    if (m_bTemporal != shouldUseTemporal) {
        DEBUG_LOG("Temporal mode mismatch, forcing re-init");
        m_bTemporal = shouldUseTemporal;
        if (m_denoiser) {
            m_denoiser->finish();
            m_bInitialized = false;
            m_frameCounter = 0;
        }
    }

    // Get processing area information
    const Box requestedBounds = plane.bounds();
    const Box imageFormat = info().format();
    
    unsigned int processWidth = requestedBounds.w();
    unsigned int processHeight = requestedBounds.h();
    Box processBounds = requestedBounds;
    
    DEBUG_LOG_VALUE("Requested bounds", std::string("[") + std::to_string(requestedBounds.x()) + "," + 
                    std::to_string(requestedBounds.y()) + " " + std::to_string(requestedBounds.r()) + "," + 
                    std::to_string(requestedBounds.t()) + "]");
    DEBUG_LOG_VALUE("Format bounds", std::string("[") + std::to_string(imageFormat.x()) + "," + 
                    std::to_string(imageFormat.y()) + " " + std::to_string(imageFormat.r()) + "," + 
                    std::to_string(imageFormat.t()) + "]");
    DEBUG_LOG_VALUE("Processing dimensions", std::to_string(processWidth) + "x" + std::to_string(processHeight));
    
    // Performance analysis
    const size_t totalPixels = static_cast<size_t>(processWidth) * processHeight;
    const float overscanFactor = (static_cast<float>(processWidth) * processHeight) / 
                                (static_cast<float>(imageFormat.w()) * imageFormat.h());
    
    DEBUG_LOG_VALUE("Total pixels", totalPixels);
    DEBUG_LOG_VALUE("Overscan factor", overscanFactor);
    
    // Performance recommendations
    const size_t LARGE_IMAGE_THRESHOLD = 5000000;  // 5M pixels
    const size_t HUGE_IMAGE_THRESHOLD = 15000000;  // 15M pixels
    
    if (totalPixels > HUGE_IMAGE_THRESHOLD) {
        DEBUG_LOG("HUGE IMAGE: Processing >15M pixels - expect longer processing time");
        if (m_tileWidth == 0) {
            DEBUG_LOG("RECOMMENDATION: Enable tiling (2048x2048) for much better performance");
        }
    } else if (totalPixels > LARGE_IMAGE_THRESHOLD) {
        DEBUG_LOG("LARGE IMAGE: Processing >5M pixels");
        if (m_tileWidth == 0) {
            DEBUG_LOG("SUGGESTION: Consider enabling tiling for better performance");
        }
    }

    // Memory estimation
    const size_t estimatedGPUMemoryMB = (totalPixels * 16 * 6) / (1024 * 1024); // ~6 RGBA buffers
    DEBUG_LOG_VALUE("Estimated GPU memory (MB)", estimatedGPUMemoryMB);

    // Check if dimensions changed - cleanup if needed
    if (m_bInitialized && (processWidth != m_width || processHeight != m_height)) {
        DEBUG_LOG("Processing dimensions changed, cleaning up denoiser");
        DEBUG_LOG_VALUE("Old dimensions", std::to_string(m_width) + "x" + std::to_string(m_height));
        DEBUG_LOG_VALUE("New dimensions", std::to_string(processWidth) + "x" + std::to_string(processHeight));
        if (m_denoiser) {
            m_denoiser->finish();
            m_bInitialized = false;
            m_frameCounter = 0;
        }
    }

    // Smart cache management - clear every few frames for large images
    const int CACHE_CLEAR_INTERVAL = (totalPixels > LARGE_IMAGE_THRESHOLD) ? 3 : 10;
    if (m_frameCounter > 0 && (m_frameCounter % CACHE_CLEAR_INTERVAL == 0)) {
        DEBUG_LOG_VALUE("Periodic cache cleanup after frames", m_frameCounter);
        if (m_denoiser && m_bInitialized) {
            DEBUG_LOG("Clearing OptiX cache for performance");
            m_denoiser->finish();
            m_bInitialized = false;
        }
    }

    // Validate CUDA
    if (!validateCUDA()) {
        consecutiveFailures++;
        error("CUDA not available or no CUDA devices found");
        DEBUG_LOG("CUDA validation failed");
        return;
    }
    
    // Update processing state
    m_width = processWidth;
    m_height = processHeight;
    m_processingBounds = processBounds;
    
    DEBUG_LOG_VALUE("Final processing dimensions", std::to_string(m_width) + "x" + std::to_string(m_height));
    DEBUG_LOG_VALUE("Frame counter", m_frameCounter);
    DEBUG_LOG_VALUE("Temporal mode", m_bTemporal);
    DEBUG_LOG_VALUE("Initialized", m_bInitialized);
    DEBUG_LOG_VALUE("Tile size", std::to_string(m_tileWidth) + "x" + std::to_string(m_tileHeight));

    try {
        // Buffer allocation
        DEBUG_LOG("Allocating host buffers");
        auto bufferStart = std::chrono::steady_clock::now();
        
        try {
            allocateBuffers();
        } catch (const std::exception& e) {
            DEBUG_LOG_VALUE("Buffer allocation failed", e.what());
            guard.cleanupNeeded = true;
            throw;
        }
        
        auto bufferEnd = std::chrono::steady_clock::now();
        auto bufferDuration = std::chrono::duration_cast<std::chrono::milliseconds>(bufferEnd - bufferStart);
        DEBUG_LOG_VALUE("Buffer allocation took (ms)", bufferDuration.count());

        // Check for abort after buffer allocation
        if (aborted() || cancelled()) {
            DEBUG_LOG("Operation aborted during buffer allocation");
            guard.cleanupNeeded = true;
            return;
        }

        // Read input data
        DEBUG_LOG("Reading input planes");
        auto readStart = std::chrono::steady_clock::now();
        
        try {
            readInputPlanes();
        } catch (const std::exception& e) {
            DEBUG_LOG_VALUE("Input plane reading failed", e.what());
            guard.cleanupNeeded = true;
            throw;
        }
        
        auto readEnd = std::chrono::steady_clock::now();
        auto readDuration = std::chrono::duration_cast<std::chrono::milliseconds>(readEnd - readStart);
        DEBUG_LOG_VALUE("Input reading took (ms)", readDuration.count());

        // Setup denoiser data structure
        DEBUG_LOG("Setting up denoiser data");
        OptiXDenoiserCore::Data data;
        data.width = m_width;
        data.height = m_height;
        data.color = m_colorBuffer.data();
        data.albedo = (node_inputs() > 1 && input(1)) ? m_albedoBuffer.data() : nullptr;
        data.normal = (node_inputs() > 2 && input(2)) ? m_normalBuffer.data() : nullptr;
        data.flow = (node_inputs() > 3 && input(3) && m_bTemporal) ? m_motionXYBuffer.data() : nullptr;
        data.outputs.push_back(m_outputBuffer.data());

        DEBUG_LOG_VALUE("Data has albedo", (data.albedo != nullptr));
        DEBUG_LOG_VALUE("Data has normal", (data.normal != nullptr));
        DEBUG_LOG_VALUE("Data has flow", (data.flow != nullptr));

        // Initialize or update denoiser
        if (!m_bInitialized) {
            DEBUG_LOG("Initializing OptiX denoiser");
            auto initStart = std::chrono::steady_clock::now();
            
            try {
                // Use configured tile size (defaults to 2048x2048)
                DEBUG_LOG_VALUE("Using tile size", std::to_string(m_tileWidth) + "x" + std::to_string(m_tileHeight));
                
                if (m_tileWidth > 0 && m_tileHeight > 0) {
                    DEBUG_LOG("Tiled denoising enabled for optimal performance");
                    DEBUG_LOG_VALUE("Tiles needed (approx)", 
                                   std::to_string((processWidth + m_tileWidth - 1) / m_tileWidth) + "x" + 
                                   std::to_string((processHeight + m_tileHeight - 1) / m_tileHeight));
                } else {
                    DEBUG_LOG("Non-tiled denoising (may be slower for large images)");
                }
                
                m_denoiser->init(data, m_tileWidth, m_tileHeight, m_bTemporal);
                m_bInitialized = true;
                
                auto initEnd = std::chrono::steady_clock::now();
                auto initDuration = std::chrono::duration_cast<std::chrono::milliseconds>(initEnd - initStart);
                DEBUG_LOG_VALUE("OptiX initialization took (ms)", initDuration.count());
                
            } catch (const std::exception& e) {
                DEBUG_LOG_VALUE("Denoiser initialization failed", e.what());
                guard.cleanupNeeded = true;
                throw;
            }
        } else {
            DEBUG_LOG("Updating denoiser with new frame data");
            auto updateStart = std::chrono::steady_clock::now();
            
            m_denoiser->update(data);
            
            auto updateEnd = std::chrono::steady_clock::now();
            auto updateDuration = std::chrono::duration_cast<std::chrono::milliseconds>(updateEnd - updateStart);
            DEBUG_LOG_VALUE("Data update took (ms)", updateDuration.count());
        }

        // Check for abort before denoising
        if (aborted() || cancelled()) {
            DEBUG_LOG("Operation aborted before denoising");
            guard.cleanupNeeded = true;
            return;
        }
        
        // Execute denoising
        DEBUG_LOG("=== STARTING DENOISING PROCESS ===");
        auto denoiseStart = std::chrono::steady_clock::now();
        
        m_denoiser->exec();
        
        auto denoiseEnd = std::chrono::steady_clock::now();
        auto denoiseDuration = std::chrono::duration_cast<std::chrono::milliseconds>(denoiseEnd - denoiseStart);
        DEBUG_LOG_VALUE("=== DENOISING COMPLETED (ms)", denoiseDuration.count());
        
        // Performance analysis
        const float pixelsPerSecond = (denoiseDuration.count() > 0) ? 
            (static_cast<float>(totalPixels) / denoiseDuration.count() * 1000.0f) : 0.0f;
        DEBUG_LOG_VALUE("Denoising rate (pixels/sec)", static_cast<size_t>(pixelsPerSecond));
        
        // Get results from GPU
        DEBUG_LOG("Retrieving results from GPU");
        auto resultStart = std::chrono::steady_clock::now();
        
        m_denoiser->getResults();
        
        auto resultEnd = std::chrono::steady_clock::now();
        auto resultDuration = std::chrono::duration_cast<std::chrono::milliseconds>(resultEnd - resultStart);
        DEBUG_LOG_VALUE("Result retrieval took (ms)", resultDuration.count());

        // Write output to plane
        DEBUG_LOG("Writing output plane");
        auto writeStart = std::chrono::steady_clock::now();
        
        writeOutputPlane(plane);
        
        auto writeEnd = std::chrono::steady_clock::now();
        auto writeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(writeEnd - writeStart);
        DEBUG_LOG_VALUE("Output writing took (ms)", writeDuration.count());
        
        // Update frame counter
        m_frameCounter++;
        DEBUG_LOG_VALUE("New frame counter", m_frameCounter);

        // Smart cleanup for very large single-frame operations
        if (!m_bTemporal && totalPixels > HUGE_IMAGE_THRESHOLD) {
            DEBUG_LOG("Auto-cleanup after huge single-frame operation");
            if (m_denoiser && m_bInitialized) {
                m_denoiser->finish();
                m_bInitialized = false;
            }
        }

        // Calculate and log total performance
        auto renderEnd = std::chrono::steady_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(renderEnd - renderStart);
        DEBUG_LOG_VALUE("=== TOTAL RENDER TIME (ms)", totalDuration.count());
        DEBUG_LOG_VALUE("=== TOTAL RENDER TIME (sec)", totalDuration.count() / 1000.0f);
        
        // Performance breakdown
        DEBUG_LOG("=== PERFORMANCE BREAKDOWN ===");
        DEBUG_LOG_VALUE("Buffer allocation", std::to_string(bufferDuration.count()) + "ms");
        DEBUG_LOG_VALUE("Input reading", std::to_string(readDuration.count()) + "ms");
        DEBUG_LOG_VALUE("Denoising", std::to_string(denoiseDuration.count()) + "ms");
        DEBUG_LOG_VALUE("Result retrieval", std::to_string(resultDuration.count()) + "ms");
        DEBUG_LOG_VALUE("Output writing", std::to_string(writeDuration.count()) + "ms");
        
        // SUCCESS - Reset failure counter
        consecutiveFailures = 0;
        DEBUG_LOG("=== FRAME PROCESSED SUCCESSFULLY ===");

    } catch (const std::exception& e) {
        consecutiveFailures++;
        DEBUG_LOG_VALUE("Exception in renderStripe (failure #" + std::to_string(consecutiveFailures) + ")", e.what());
        guard.cleanupNeeded = true;
        handleDenoiserError(e);
        
        // Fallback: copy input to output
        if (!m_colorBuffer.empty() && !m_outputBuffer.empty() && 
            m_colorBuffer.size() == m_outputBuffer.size()) {
            DEBUG_LOG("Copying input to output as fallback");
            std::copy(m_colorBuffer.begin(), m_colorBuffer.end(), m_outputBuffer.begin());
            writeOutputPlane(plane);
        } else {
            DEBUG_LOG("Cannot copy input to output - buffer size mismatch or empty buffers");
        }
    } catch (...) {
        consecutiveFailures++;
        DEBUG_LOG("Unknown exception in renderStripe (failure #" + std::to_string(consecutiveFailures) + ")");
        guard.cleanupNeeded = true;
        error("Unknown error occurred during denoising");
        
        // Emergency fallback
        if (!m_colorBuffer.empty() && !m_outputBuffer.empty() && 
            m_colorBuffer.size() == m_outputBuffer.size()) {
            std::copy(m_colorBuffer.begin(), m_colorBuffer.end(), m_outputBuffer.begin());
            writeOutputPlane(plane);
        }
    }
    
    DEBUG_LOG("=== OptiXDenoiserIop::renderStripe END ===");
}

static Iop *build(Node *node) { 
    return new OptiXDenoiserIop(node); 
}

const Iop::Description OptiXDenoiserIop::d("OptiXDenoiser", "Filter/OptiXDenoiser", build);
