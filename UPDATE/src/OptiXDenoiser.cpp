// Copyright (c) 2025 - OptiX Denoiser for Nuke
// Based on NVIDIA OptiX SDK examples

#include "OptiXDenoiser.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <thread>
#include <chrono>

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

// OptiX context callback
static void context_log_cb(uint32_t level, const char* tag, const char* message, void* /*cbdata*/)
{
    if (level < 4) {
        std::cerr << "[OptiX][" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
    }
}

// OptiXDenoiserCore Implementation
void* OptiXDenoiserCore::allocateAlignedCudaMemory(size_t size, size_t& actualSize, const std::string& description)
{
    const size_t alignment = 128;
    actualSize = ((size + alignment - 1) / alignment) * alignment;
    
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, actualSize));
    CUDA_CHECK(cudaMemset(ptr, 0, actualSize));
    
    {
        std::lock_guard<std::mutex> lock(m_cleanupMutex);
        m_allocatedPointers.emplace_back(ptr, actualSize, description);
        m_totalMemoryUsed += actualSize;
    }
    
    return ptr;
}

void OptiXDenoiserCore::clampNormalBuffer(float* normalData, size_t pixelCount)
{
    if (!normalData) return;
    
    for (size_t i = 0; i < pixelCount; ++i) {
        for (int c = 0; c < 3; ++c) {
            size_t index = i * 4 + c;
            normalData[index] = std::max(0.0f, std::min(1.0f, normalData[index]));
        }
    }
}

void OptiXDenoiserCore::validateHDRRange(const float* imageData, size_t pixelCount, bool& outOfRange)
{
    outOfRange = false;
    if (!imageData) return;
    
    const float maxRecommendedValue = 10000.0f;
    size_t outOfRangeCount = 0;
    
    for (size_t i = 0; i < pixelCount; ++i) {
        float r = imageData[i * 4 + 0];
        float g = imageData[i * 4 + 1];
        float b = imageData[i * 4 + 2];
        
        if (r > maxRecommendedValue || g > maxRecommendedValue || b > maxRecommendedValue) {
            outOfRangeCount++;
        }
    }
    
    if (outOfRangeCount > 0) {
        outOfRange = true;
        std::cerr << "Warning: " << outOfRangeCount << " pixels exceed recommended HDR range [0-10000]. "
                  << "This may affect denoising quality." << std::endl;
    }
}

OptixImage2D OptiXDenoiserCore::createOptixImage2D(unsigned int width, unsigned int height, 
                                                   const float* hmem, OptixPixelFormat format)
{
    OptixImage2D oi = {};
    
    size_t pixelSize = 0;
    switch (format) {
        case OPTIX_PIXEL_FORMAT_FLOAT4:
            pixelSize = 4 * sizeof(float);
            break;
        case OPTIX_PIXEL_FORMAT_FLOAT3:
            pixelSize = 3 * sizeof(float);
            break;
        case OPTIX_PIXEL_FORMAT_FLOAT2:
            pixelSize = 2 * sizeof(float);
            break;
        case OPTIX_PIXEL_FORMAT_HALF2:
            pixelSize = 2 * sizeof(uint16_t);
            break;
        default:
            throw std::runtime_error("Unsupported pixel format");
    }
    
    const size_t frame_byte_size = width * height * pixelSize;
    size_t actualSize = 0;
    
    std::string description = "OptixImage2D_" + std::to_string(width) + "x" + std::to_string(height);
    oi.data = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(frame_byte_size, actualSize, description));
    
    if (hmem) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(oi.data), hmem, frame_byte_size, cudaMemcpyHostToDevice));
    }
    
    oi.width = width;
    oi.height = height;
    oi.rowStrideInBytes = width * pixelSize;
    oi.pixelStrideInBytes = pixelSize;
    oi.format = format;
    
    return oi;
}

void OptiXDenoiserCore::copyOptixImage2D(OptixImage2D& dest, const OptixImage2D& src)
{
    if (dest.format != src.format) {
        throw std::runtime_error("Cannot copy images with different formats");
    }
    
    const size_t pixelSize = dest.pixelStrideInBytes;
    const size_t copySize = src.width * src.height * pixelSize;
    
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dest.data), reinterpret_cast<void*>(src.data), 
                         copySize, cudaMemcpyDeviceToDevice));
}

void OptiXDenoiserCore::cleanupCudaResources()
{
    MEMORY_LOG("=== CUDA CLEANUP START ===");
    std::lock_guard<std::mutex> lock(m_cleanupMutex);
    
    // Ensure we're not already cleaning up to prevent recursion
    static bool cleanupInProgress = false;
    if (cleanupInProgress) {
        MEMORY_LOG("Cleanup already in progress - returning");
        return;
    }
    cleanupInProgress = true;
    
    try {
        MEMORY_LOG("Syncing stream if present");
        // Wait for any pending CUDA operations
        if (m_stream) {
            cudaError_t streamResult = cudaStreamSynchronize(m_stream);
            if (streamResult != cudaSuccess) {
                MEMORY_LOG("Stream sync failed: " << cudaGetErrorString(streamResult));
            } else {
                MEMORY_LOG("Stream sync successful");
            }
        }
        
        MEMORY_LOG("Freeing " << m_allocatedPointers.size() << " tracked allocations");
        // Free tracked allocations in reverse order
        for (auto it = m_allocatedPointers.rbegin(); it != m_allocatedPointers.rend(); ++it) {
            if (it->ptr) {
                MEMORY_LOG("Freeing allocation: " << it->description << " ptr=" << it->ptr);
                cudaError_t freeResult = cudaFree(it->ptr);
                if (freeResult != cudaSuccess) {
                    MEMORY_LOG("cudaFree failed for " << it->description << ": " << cudaGetErrorString(freeResult));
                } else {
                    MEMORY_LOG("Successfully freed: " << it->description);
                }
                it->ptr = nullptr;
            }
        }
        m_allocatedPointers.clear();
        MEMORY_LOG("All allocations freed");
        
        // Clean up stream
        if (m_stream) {
            MEMORY_LOG("Destroying CUDA stream");
            cudaError_t destroyResult = cudaStreamDestroy(m_stream);
            if (destroyResult != cudaSuccess) {
                MEMORY_LOG("Stream destroy failed: " << cudaGetErrorString(destroyResult));
            } else {
                MEMORY_LOG("Stream destroyed successfully");
            }
            m_stream = nullptr;
        }
        
        MEMORY_LOG("Resetting pointers");
        // Reset pointers
        m_intensity = 0;
        m_avgColor = 0;
        m_scratch = 0;
        m_state = 0;
        
        m_guideLayer.albedo.data = 0;
        m_guideLayer.normal.data = 0;
        m_guideLayer.flow.data = 0;
        
        for (auto& layer : m_layers) {
            layer.input.data = 0;
            layer.output.data = 0;
            layer.previousOutput.data = 0;
        }
        
        m_totalMemoryUsed = 0;
        MEMORY_LOG("All pointers reset");
        
    } catch (const std::exception& e) {
        MEMORY_LOG("Exception in CUDA cleanup: " << e.what());
        std::cerr << "Exception during CUDA cleanup: " << e.what() << std::endl;
    } catch (...) {
        MEMORY_LOG("Unknown exception in CUDA cleanup");
        std::cerr << "Unknown exception during CUDA cleanup" << std::endl;
    }
    
    cleanupInProgress = false;
    MEMORY_LOG("=== CUDA CLEANUP END ===");
}

void OptiXDenoiserCore::cleanupOptixResources()
{
    try {
        if (m_denoiser) {
            OptixResult result = optixDenoiserDestroy(m_denoiser);
            if (result != OPTIX_SUCCESS) {
                std::cerr << "Warning: OptiX denoiser destroy failed: " << result << std::endl;
            }
            m_denoiser = nullptr;
        }
        if (m_context) {
            OptixResult result = optixDeviceContextDestroy(m_context);
            if (result != OPTIX_SUCCESS) {
                std::cerr << "Warning: OptiX context destroy failed: " << result << std::endl;
            }
            m_context = nullptr;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during OptiX cleanup: " << e.what() << std::endl;
        // Don't rethrow during cleanup
    } catch (...) {
        std::cerr << "Unknown exception during OptiX cleanup" << std::endl;
        // Don't rethrow during cleanup
    }
}

void OptiXDenoiserCore::init(const Data& data, unsigned int tileWidth, unsigned int tileHeight, bool temporalMode)
{
    if (!data.color || data.outputs.empty() || !data.width || !data.height) {
        throw std::runtime_error("Invalid denoiser input data");
    }

    if (data.normal && !data.albedo) {
        throw std::runtime_error("Albedo is required when normal is provided");
    }

    // Clean up any existing resources
    if (m_initialized) {
        finish();
    }

    m_host_outputs = data.outputs;
    m_temporalMode = temporalMode;
    m_frameIndex = 0;

    // Initialize CUDA driver
    CUresult cuResult = cuInit(0);
    if (cuResult != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to initialize CUDA driver: " + std::to_string(cuResult));
    }

    // Get or create CUDA context
    CUcontext cuContext = nullptr;
    cuResult = cuCtxGetCurrent(&cuContext);

    if (cuResult != CUDA_SUCCESS || cuContext == nullptr) {
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        
        CUdevice cuDevice;
        cuResult = cuDeviceGet(&cuDevice, device);
        if (cuResult != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to get CUDA device: " + std::to_string(cuResult));
        }
        
        cuResult = cuCtxCreate(&cuContext, 0, cuDevice);
        if (cuResult != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to create CUDA context: " + std::to_string(cuResult));
        }
        
        cuResult = cuCtxSetCurrent(cuContext);
        if (cuResult != CUDA_SUCCESS) {
            cuCtxDestroy(cuContext);
            throw std::runtime_error("Failed to set CUDA context current: " + std::to_string(cuResult));
        }
        
        m_ownsCudaContext = true;
    } else {
        m_ownsCudaContext = false;
    }

    m_cuContext = cuContext;

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&m_stream));

    // Initialize OptiX
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &m_context));

    // Create denoiser with version compatibility
    OptixDenoiserOptions denoiser_options = {};
    denoiser_options.guideAlbedo = data.albedo ? 1 : 0;
    denoiser_options.guideNormal = data.normal ? 1 : 0;

    // Check OptiX version for alpha denoising support
#if OPTIX_VERSION >= 80000
    // OptiX 8.0+ supports alpha denoising
    denoiser_options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
    std::cout << "OptiX 8.0+ detected - Alpha denoising enabled" << std::endl;
#else
    // OptiX 7.x doesn't have alpha denoising
    std::cout << "OptiX 7.x detected - Alpha denoising not available" << std::endl;
#endif

    OptixDenoiserModelKind modelKind = temporalMode ? 
        OPTIX_DENOISER_MODEL_KIND_TEMPORAL : OPTIX_DENOISER_MODEL_KIND_HDR;
    
    OPTIX_CHECK(optixDenoiserCreate(m_context, modelKind, &denoiser_options, &m_denoiser));

    // Enhanced tiling setup - always use full image size for memory calculation
    OptixDenoiserSizes denoiser_sizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, data.width, data.height, &denoiser_sizes));

    if (tileWidth == 0 || tileHeight == 0) {
        // No tiling mode - process entire image
        m_tileWidth = data.width;
        m_tileHeight = data.height;
        m_scratch_size = static_cast<uint32_t>(denoiser_sizes.withoutOverlapScratchSizeInBytes);
        m_overlap = 0;
    } else {
        // Tiling mode - validate and set tile dimensions
        m_tileWidth = std::min(tileWidth, data.width);
        m_tileHeight = std::min(tileHeight, data.height);
        
        // Ensure minimum tile size for stability
        m_tileWidth = std::max(m_tileWidth, 256u);
        m_tileHeight = std::max(m_tileHeight, 256u);
        
        m_scratch_size = static_cast<uint32_t>(denoiser_sizes.withOverlapScratchSizeInBytes);
        m_overlap = denoiser_sizes.overlapWindowSizeInPixels;
        
        // Safety check for overlap
        if (m_overlap >= m_tileWidth / 2 || m_overlap >= m_tileHeight / 2) {
            // Fallback to no tiling if overlap is too large
            m_tileWidth = data.width;
            m_tileHeight = data.height;
            m_scratch_size = static_cast<uint32_t>(denoiser_sizes.withoutOverlapScratchSizeInBytes);
            m_overlap = 0;
        }
    }

    // Allocate buffers
    size_t actualSize;
    m_intensity = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(sizeof(float), actualSize, "intensity"));
    m_avgColor = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(3 * sizeof(float), actualSize, "avgColor"));
    m_scratch = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(m_scratch_size, actualSize, "scratch"));
    m_state = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(denoiser_sizes.stateSizeInBytes, actualSize, "state"));

    m_state_size = static_cast<uint32_t>(denoiser_sizes.stateSizeInBytes);

    // Create image layers
    OptixDenoiserLayer layer = {};
    layer.input = createOptixImage2D(data.width, data.height, data.color);
    layer.output = createOptixImage2D(data.width, data.height);

    if (m_temporalMode) {
        layer.previousOutput = createOptixImage2D(data.width, data.height);
        copyOptixImage2D(layer.previousOutput, layer.input);
    }

    m_layers.push_back(layer);

    // Setup guide layers
    if (data.albedo) {
        m_guideLayer.albedo = createOptixImage2D(data.width, data.height, data.albedo);
    }
    if (data.normal) {
        m_guideLayer.normal = createOptixImage2D(data.width, data.height, data.normal);
    }
    if (data.flow && m_temporalMode) {
        m_guideLayer.flow = createOptixImage2D(data.width, data.height, data.flow, OPTIX_PIXEL_FORMAT_FLOAT2);
    }

    // Setup denoiser with proper tile dimensions
    OPTIX_CHECK(optixDenoiserSetup(m_denoiser, m_stream,
                                  m_tileWidth + 2 * m_overlap,
                                  m_tileHeight + 2 * m_overlap,
                                  m_state, m_state_size,
                                  m_scratch, m_scratch_size));

    m_params.hdrIntensity = m_intensity;
    m_params.hdrAverageColor = m_avgColor;
    m_params.blendFactor = 0.0f;
    m_params.temporalModeUsePreviousLayers = 0;

    m_initialized = true;
    
    std::cout << "OptiX Denoiser initialized successfully:" << std::endl;
    std::cout << "  Version: " << OPTIX_VERSION << std::endl;
    std::cout << "  Temporal mode: " << (temporalMode ? "enabled" : "disabled") << std::endl;
    std::cout << "  Tile size: " << m_tileWidth << "x" << m_tileHeight << std::endl;
    std::cout << "  Memory usage: " << (m_totalMemoryUsed / 1024 / 1024) << " MB" << std::endl;
}

void OptiXDenoiserCore::update(const Data& data)
{
    if (!m_initialized) return;

    m_host_outputs = data.outputs;

    // Update input image
    const size_t colorBufferSize = data.width * data.height * sizeof(float4);
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_layers[0].input.data), data.color,
                         colorBufferSize, cudaMemcpyHostToDevice));

    // Update guide layers
    if (data.albedo && m_guideLayer.albedo.data) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_guideLayer.albedo.data), data.albedo,
                             colorBufferSize, cudaMemcpyHostToDevice));
    }

    if (data.normal && m_guideLayer.normal.data) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_guideLayer.normal.data), data.normal,
                             colorBufferSize, cudaMemcpyHostToDevice));
    }

    if (data.flow && m_guideLayer.flow.data && m_temporalMode) {
        const size_t flowBufferSize = data.width * data.height * 2 * sizeof(float);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_guideLayer.flow.data), data.flow,
                             flowBufferSize, cudaMemcpyHostToDevice));
    }

    // Handle temporal mode - swap buffers for next frame
    if (m_temporalMode && m_frameIndex > 0) {
        copyOptixImage2D(m_layers[0].previousOutput, m_layers[0].output);
        m_params.temporalModeUsePreviousLayers = 1;
    }

    m_frameIndex++;
}

void OptiXDenoiserCore::exec()
{
    if (!m_initialized) return;

    // Compute intensity
    OPTIX_CHECK(optixDenoiserComputeIntensity(m_denoiser, m_stream, &m_layers[0].input,
                                             m_intensity, m_scratch, m_scratch_size));

    // Compute average color
    OPTIX_CHECK(optixDenoiserComputeAverageColor(m_denoiser, m_stream, &m_layers[0].input,
                                                m_avgColor, m_scratch, m_scratch_size));

    // Enhanced denoising with robust tiling
    bool useTiling = (m_tileWidth < m_layers[0].input.width || m_tileHeight < m_layers[0].input.height);

    if (!useTiling) {
        // No tiling - process entire image
        OPTIX_CHECK(optixDenoiserInvoke(m_denoiser, m_stream, &m_params,
                                       m_state, m_state_size, &m_guideLayer,
                                       m_layers.data(), static_cast<unsigned int>(m_layers.size()),
                                       0, 0, m_scratch, m_scratch_size));
    } else {
        // Tiling mode - use OptiX utility function
        OPTIX_CHECK(optixUtilDenoiserInvokeTiled(m_denoiser, m_stream, &m_params,
                                                m_state, m_state_size, &m_guideLayer,
                                                m_layers.data(), static_cast<unsigned int>(m_layers.size()),
                                                m_scratch, m_scratch_size, m_overlap,
                                                m_tileWidth, m_tileHeight));
    }

    CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

void OptiXDenoiserCore::getResults()
{
    if (!m_initialized || m_host_outputs.empty()) return;

    const uint64_t frame_byte_size = m_layers[0].output.width * m_layers[0].output.height * sizeof(float4);
    
    CUDA_CHECK(cudaMemcpy(m_host_outputs[0], reinterpret_cast<void*>(m_layers[0].output.data),
                         frame_byte_size, cudaMemcpyDeviceToHost));
}

void OptiXDenoiserCore::finish()
{
    if (!m_initialized) return;

    MEMORY_LOG("=== FINISH START ===");
    MEMORY_LOG("Temporal mode: " << m_temporalMode);
    MEMORY_LOG("Total allocations: " << m_allocatedPointers.size());

    try {
        // CRITICAL: Temporal-specific cleanup first
        if (m_temporalMode) {
            MEMORY_LOG("Temporal mode cleanup - syncing streams");
            if (m_stream) {
                MEMORY_LOG("Synchronizing CUDA stream");
                cudaStreamSynchronize(m_stream);
                MEMORY_LOG("Stream sync complete");
            }
            
            // Clear temporal layers explicitly
            MEMORY_LOG("Clearing temporal layers - count: " << m_layers.size());
            for (size_t i = 0; i < m_layers.size(); ++i) {
                MEMORY_LOG("Processing layer " << i);
                auto& layer = m_layers[i];
                if (layer.previousOutput.data != 0) {
                    MEMORY_LOG("Clearing temporal previousOutput buffer for layer " << i);
                    layer.previousOutput.data = 0;
                    MEMORY_LOG("Temporal buffer cleared for layer " << i);
                }
            }
            MEMORY_LOG("All temporal layers processed");
        }
        
        MEMORY_LOG("Starting CUDA cleanup");
        cleanupCudaResources();
        MEMORY_LOG("CUDA cleanup complete");
        
        MEMORY_LOG("Starting OptiX cleanup");
        cleanupOptixResources();
        MEMORY_LOG("OptiX cleanup complete");
        
    } catch (const std::exception& e) {
        MEMORY_LOG("Exception during cleanup: " << e.what());
        std::cerr << "Error during cleanup: " << e.what() << std::endl;
    }

    MEMORY_LOG("Clearing containers");
    m_layers.clear();
    m_host_outputs.clear();
    m_guideLayer = {};
    m_params = {};
    m_frameIndex = 0;
    
    m_initialized = false;
    MEMORY_LOG("=== FINISH END ===");
}

void OptiXDenoiserCore::convertMotionVectors(const float* motionRGBA, float* motionXY, 
                                            unsigned int width, unsigned int height)
{
    for (unsigned int i = 0; i < width * height; ++i) {
        motionXY[i * 2 + 0] = motionRGBA[i * 4 + 0]; // X component
        motionXY[i * 2 + 1] = motionRGBA[i * 4 + 1]; // Y component
    }
}

// Debug helpers for OptiXDenoiserCore
void OptiXDenoiserCore::dumpMemoryState() const
{
    std::lock_guard<std::mutex> lock(m_cleanupMutex);
    MEMORY_LOG("=== MEMORY STATE DUMP ===");
    MEMORY_LOG("Total allocations: " << m_allocatedPointers.size());
    MEMORY_LOG("Total memory used: " << m_totalMemoryUsed << " bytes");
    MEMORY_LOG("Cleanup in progress: " << m_cleanupInProgress);
    MEMORY_LOG("Initialized: " << m_initialized);
    
    for (size_t i = 0; i < m_allocatedPointers.size(); ++i) {
        const auto& alloc = m_allocatedPointers[i];
        MEMORY_LOG("Allocation " << i << ": " << alloc.description << 
                   " - ptr=" << alloc.ptr << 
                   " - size=" << alloc.size << 
                   " - valid=" << alloc.isValid);
    }
    MEMORY_LOG("=== END MEMORY DUMP ===");
}

void OptiXDenoiserCore::validateAllPointers() const
{
    MEMORY_LOG("Validating all pointers...");
    for (size_t i = 0; i < m_allocatedPointers.size(); ++i) {
        const auto& alloc = m_allocatedPointers[i];
        if (!alloc.isValid || !alloc.ptr) {
            MEMORY_LOG("INVALID POINTER FOUND at index " << i << ": " << alloc.description);
        }
    }
    MEMORY_LOG("Pointer validation complete");
}

void OptiXDenoiserCore::logMemoryState(const std::string& context) const
{
    MEMORY_LOG("Memory state at " << context << ":");
    MEMORY_LOG("  Allocations: " << m_allocatedPointers.size());
    MEMORY_LOG("  Total memory: " << m_totalMemoryUsed);
    MEMORY_LOG("  Initialized: " << m_initialized);
    MEMORY_LOG("  Cleanup in progress: " << m_cleanupInProgress);
}

// Debug helpers for OptiXDenoiserIop
void OptiXDenoiserIop::logBufferSizes(const std::string& context) const
{
    MEMORY_LOG("Buffer sizes at " << context << ":");
    MEMORY_LOG("  Color buffer: " << m_colorBuffer.size());
    MEMORY_LOG("  Albedo buffer: " << m_albedoBuffer.size());
    MEMORY_LOG("  Normal buffer: " << m_normalBuffer.size());
    MEMORY_LOG("  Motion buffer: " << m_motionBuffer.size());
    MEMORY_LOG("  Output buffer: " << m_outputBuffer.size());
    MEMORY_LOG("  Motion XY buffer: " << m_motionXYBuffer.size());
}

void OptiXDenoiserIop::logTileChange(int oldPreset, int newPreset) const
{
    TILE_LOG("=== TILE CHANGE DETECTED ===");
    TILE_LOG("Old preset: " << oldPreset << " (" << getTilePresetName(oldPreset) << ")");
    TILE_LOG("New preset: " << newPreset << " (" << getTilePresetName(newPreset) << ")");
    
    unsigned int oldW, oldH, newW, newH;
    
    // Get old dimensions
    int savedPreset = m_tilePreset;
    const_cast<OptiXDenoiserIop*>(this)->m_tilePreset = oldPreset;
    getTileDimensions(oldW, oldH);
    
    // Get new dimensions
    const_cast<OptiXDenoiserIop*>(this)->m_tilePreset = newPreset;
    getTileDimensions(newW, newH);
    
    // Restore preset
    const_cast<OptiXDenoiserIop*>(this)->m_tilePreset = savedPreset;
    
    TILE_LOG("Old tile size: " << oldW << "x" << oldH);
    TILE_LOG("New tile size: " << newW << "x" << newH);
    TILE_LOG("Denoiser initialized: " << m_bInitialized);
    if (m_denoiser) {
        TILE_LOG("GPU allocations: " << m_denoiser->getTotalAllocations());
        TILE_LOG("GPU memory used: " << m_denoiser->getTotalMemoryUsed());
    }
    TILE_LOG("=== END TILE CHANGE ===");
}

void OptiXDenoiserIop::dumpCompleteState(const std::string& context) const
{
    CRASH_LOG("=== COMPLETE STATE DUMP: " << context << " ===");
    CRASH_LOG("Class state:");
    CRASH_LOG("  m_bInitialized: " << m_bInitialized);
    CRASH_LOG("  m_bTemporal: " << m_bTemporal);
    CRASH_LOG("  m_tilePreset: " << m_tilePreset);
    CRASH_LOG("  m_tileWidth: " << m_tileWidth);
    CRASH_LOG("  m_tileHeight: " << m_tileHeight);
    CRASH_LOG("  m_width: " << m_width);
    CRASH_LOG("  m_height: " << m_height);
    CRASH_LOG("  m_frameCounter: " << m_frameCounter);
    
    logBufferSizes(context);
    
    if (m_denoiser) {
        CRASH_LOG("Denoiser state:");
        CRASH_LOG("  Initialized: " << m_denoiser->isInitialized());
        CRASH_LOG("  Allocations: " << m_denoiser->getTotalAllocations());
        CRASH_LOG("  Memory used: " << m_denoiser->getTotalMemoryUsed());
        m_denoiser->dumpMemoryState();
    } else {
        CRASH_LOG("Denoiser is NULL!");
    }
    CRASH_LOG("=== END COMPLETE STATE DUMP ===");
}

// Nuke Plugin Implementation
OptiXDenoiserIop::OptiXDenoiserIop(Node *node) : PlanarIop(node)
{
    CRASH_LOG("=== CONSTRUCTOR START ===");
    
    // Initialize parameters - clean and simple
    m_bTemporal = false;
    m_bInitialized = false;
    m_tileWidth = 2048;
    m_tileHeight = 2048;
    m_numRuns = 1;

    // Frame range control
    m_startFrame = 1;
    m_endFrame = 100;
    
    // Tile preset - default to 2048x2048 for better performance
    m_tilePreset = TILE_2048;
    m_lastTilePreset = m_tilePreset;

    m_width = 0;
    m_height = 0;
    m_lastProcessWidth = 0;
    m_lastProcessHeight = 0;

    m_defaultChannels = Mask_RGB;
    m_defaultNumberOfChannels = m_defaultChannels.size();

    m_denoiser = std::make_unique<OptiXDenoiserCore>();
    m_frameCounter = 0;
    
    CRASH_LOG("Constructor initialized with:");
    CRASH_LOG("  Tile preset: " << m_tilePreset);
    CRASH_LOG("  Default tile size: " << m_tileWidth << "x" << m_tileHeight);
    CRASH_LOG("=== CONSTRUCTOR END ===");
}

OptiXDenoiserIop::~OptiXDenoiserIop()
{
    CRASH_LOG("=== DESTRUCTOR START ===");
    
    try {
        // Ensure denoiser is properly cleaned up FIRST
        if (m_denoiser && m_bInitialized) {
            CRASH_LOG("Cleaning up initialized denoiser");
            m_denoiser->dumpMemoryState();
            m_denoiser->finish();
            m_bInitialized = false;
            CRASH_LOG("Denoiser cleanup successful");
        }
        
        // Reset the unique_ptr 
        if (m_denoiser) {
            CRASH_LOG("Resetting denoiser unique_ptr");
            m_denoiser.reset();
            CRASH_LOG("Denoiser reset complete");
        }
        
        // Clear host buffers AFTER denoiser cleanup
        CRASH_LOG("Clearing host buffers");
        safeBufferClear(m_colorBuffer, "color");
        safeBufferClear(m_albedoBuffer, "albedo");
        safeBufferClear(m_normalBuffer, "normal");
        safeBufferClear(m_motionBuffer, "motion");
        safeBufferClear(m_motionXYBuffer, "motionXY");
        safeBufferClear(m_outputBuffer, "output");
        CRASH_LOG("All host buffers cleared");
        
    } catch (const std::exception& e) {
        CRASH_LOG("Exception in destructor: " << e.what());
        // Don't throw from destructor
    } catch (...) {
        CRASH_LOG("Unknown exception in destructor");
        // Don't throw from destructor
    }
    
    CRASH_LOG("=== DESTRUCTOR END ===");
}

void OptiXDenoiserIop::getTileDimensions(unsigned int& tileWidth, unsigned int& tileHeight) const
{
    switch (m_tilePreset) {
        case TILE_512:
            tileWidth = 512;
            tileHeight = 512;
            break;
        case TILE_1024:
            tileWidth = 1024;
            tileHeight = 1024;
            break;
        case TILE_2048:
            tileWidth = 2048;
            tileHeight = 2048;
            break;
        case TILE_NONE:
        default:
            tileWidth = 0;
            tileHeight = 0;
            break;
    }
}

const char* OptiXDenoiserIop::getTilePresetName(int preset) const
{
    switch (preset) {
        case TILE_NONE: return "No Tiling (Full Image)";
        case TILE_512: return "512x512 Tiles";
        case TILE_1024: return "1024x1024 Tiles";
        case TILE_2048: return "2048x2048 Tiles";
        default: return "Unknown";
    }
}

void OptiXDenoiserIop::knobs(Knob_Callback f)
{
    Divider(f, "Temporal Settings");
    
    Int_knob(f, &m_startFrame, "start_frame", "Start Frame");
    Tooltip(f, "Reference frame - temporal state resets here\n"
             "Set to first frame of your sequence");
    SetFlags(f, Knob::STARTLINE);
    
    Int_knob(f, &m_endFrame, "end_frame", "End Frame");
    Tooltip(f, "Last frame for temporal denoising\n"
             "Frames outside range use non-temporal mode");

    Text_knob(f, "Frame Info");
    SetFlags(f, Knob::STARTLINE);
    
    Divider(f, "Memory Settings");
    
    static const char* tile_preset_names[] = {"No Tiling", "512x512", "1024x1024", "2048x2048", nullptr};
    Enumeration_knob(f, &m_tilePreset, tile_preset_names, "tile_preset", "Tile Size");
    Tooltip(f, "Tile size preset for memory optimization\n"
             "No Tiling: Process entire image (uses more VRAM)\n"
             "512x512: Good for 8GB VRAM\n"
             "1024x1024: Good for 12GB+ VRAM\n"
             "2048x2048: Good for 24GB+ VRAM");
    SetFlags(f, Knob::STARTLINE);

    Divider(f, "");
    
    // Version-aware footer
#if OPTIX_VERSION >= 80000
    Text_knob(f, "OptiXDenoiser by Peter Mercell v1.03 / 2025 (OptiX 8.0+)");
#else
    Text_knob(f, "OptiXDenoiser by Peter Mercell v1.03 / 2025 (OptiX 7.x)");
#endif
    SetFlags(f, Knob::STARTLINE);
}

int OptiXDenoiserIop::knob_changed(Knob* k)
{
    CRASH_LOG("=== KNOB_CHANGED START ===");
    CRASH_LOG("Knob: " << k->name().c_str());
    
    dumpCompleteState("knob_changed start");
    
    if (k->is("inputs")) {
        CRASH_LOG("Processing inputs knob change");
        bool shouldUseTemporal = hasMotionVectorsConnected();
        if (m_bTemporal != shouldUseTemporal) {
            CRASH_LOG("Temporal mode change detected: " << m_bTemporal << " -> " << shouldUseTemporal);
            m_bTemporal = shouldUseTemporal;
            
            if (m_denoiser) {
                CRASH_LOG("Cleaning up denoiser for temporal mode change");
                try {
                    // Extra sync for temporal mode changes
                    CRASH_LOG("Forcing CUDA device synchronization for temporal change");
                    cudaDeviceSynchronize();
                    
                    m_denoiser->dumpMemoryState();
                    m_denoiser->finish();
                    CRASH_LOG("Denoiser finished successfully");
                } catch (const std::exception& e) {
                    CRASH_LOG("Exception during denoiser cleanup: " << e.what());
                } catch (...) {
                    CRASH_LOG("Unknown exception during denoiser cleanup");
                }
                m_bInitialized = false;
                m_frameCounter = 0;
            }
        }
        CRASH_LOG("Inputs knob change complete");
        return 1;
    }
    
    if (k->is("tile_preset") || k->is("start_frame") || k->is("end_frame")) {
        CRASH_LOG("Processing tile/frame knob change");
        
        if (k->is("tile_preset")) {
            logTileChange(m_lastTilePreset, m_tilePreset);
            m_lastTilePreset = m_tilePreset;
        }
        
        if (m_denoiser) {
            CRASH_LOG("Cleaning up denoiser for tile/frame change");
            try {
                // Simple but effective sync for temporal mode
                if (m_bTemporal) {
                    CRASH_LOG("Temporal mode detected - basic sync");
                    cudaDeviceSynchronize();
                    CRASH_LOG("Basic temporal sync complete");
                }
                
                CRASH_LOG("Starting denoiser cleanup");
                m_denoiser->finish();
                CRASH_LOG("Denoiser finished successfully");
            } catch (const std::exception& e) {
                CRASH_LOG("Exception during denoiser cleanup: " << e.what());
            } catch (...) {
                CRASH_LOG("Unknown exception during denoiser cleanup");
            }
            m_bInitialized = false;
            m_frameCounter = 0;
        }
        
        // Clear host buffers
        CRASH_LOG("Clearing host buffers");
        logBufferSizes("before buffer clear");
        
        try {
            safeBufferClear(m_colorBuffer, "color");
            safeBufferClear(m_albedoBuffer, "albedo");
            safeBufferClear(m_normalBuffer, "normal");
            safeBufferClear(m_motionBuffer, "motion");
            safeBufferClear(m_outputBuffer, "output");
            safeBufferClear(m_motionXYBuffer, "motionXY");
            CRASH_LOG("All buffers cleared successfully");
        } catch (const std::exception& e) {
            CRASH_LOG("Exception during buffer clearing: " << e.what());
        } catch (...) {
            CRASH_LOG("Unknown exception during buffer clearing");
        }
        
        logBufferSizes("after buffer clear");
        CRASH_LOG("Tile/frame knob change complete");
        return 1;
    }
    
    dumpCompleteState("knob_changed end");
    CRASH_LOG("=== KNOB_CHANGED END ===");
    return 0;
}

bool OptiXDenoiserIop::hasMotionVectorsConnected() const
{
    return (node_inputs() > 3 && input(3) != nullptr);
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

bool OptiXDenoiserIop::isWithinFrameRange(int currentFrame) const
{
    return (currentFrame >= m_startFrame && currentFrame <= m_endFrame);
}

bool OptiXDenoiserIop::isStartFrame(int currentFrame) const
{
    return (currentFrame == m_startFrame);
}

int OptiXDenoiserIop::getAnimationLength() const
{
    return m_endFrame - m_startFrame + 1;
}

void OptiXDenoiserIop::_validate(bool for_real)
{
    copy_info();
    
    // Use temporal mode when motion vectors are connected
    bool shouldUseTemporal = hasMotionVectorsConnected();
    
    if (m_bTemporal != shouldUseTemporal) {
        m_bTemporal = shouldUseTemporal;
        if (m_denoiser) {
            m_denoiser->finish();
            m_bInitialized = false;
        }
    }
    
    // Set output channels to RGBA only
    ChannelSet newChannels;
    newChannels += Chan_Red;
    newChannels += Chan_Green;
    newChannels += Chan_Blue;
    newChannels += Chan_Alpha;
    info_.channels() = newChannels;
}

void OptiXDenoiserIop::getRequests(const Box& box, const ChannelSet& channels, int count, RequestOutput& reqData) const
{
    // Define RGBA channels we need
    ChannelSet rgbaChannels;
    rgbaChannels += Chan_Red;
    rgbaChannels += Chan_Green;
    rgbaChannels += Chan_Blue;
    rgbaChannels += Chan_Alpha;
    
    // Request from all connected inputs - use the requested box as-is
    for (int i = 0, endI = getInputs().size(); i < endI; i++) {
        if (input(i)) {
            const ChannelSet availableChannels = input(i)->info().channels();
            ChannelSet requestChannels;
            foreach(z, rgbaChannels) {
                if (availableChannels.contains(z)) {
                    requestChannels += z;
                }
            }
            if (!requestChannels.empty()) {
                input(i)->request(box, requestChannels, count);
            }
        }
    }
}

bool OptiXDenoiserIop::validateCUDA()
{
    try {
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            return false;
        }
        
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, 0);
        if (error != cudaSuccess) {
            return false;
        }
        
        if (prop.major < 5) {
            return false;
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

void OptiXDenoiserIop::allocateBuffers()
{
    if (m_width == 0 || m_height == 0) {
        throw std::runtime_error("Invalid dimensions for buffer allocation");
    }
    
    const size_t pixelCount = static_cast<size_t>(m_width) * m_height;
    const size_t bufferSizeRGBA = pixelCount * 4; // 4 floats per pixel
    const size_t bufferSizeXY = pixelCount * 2;   // 2 floats per pixel for motion
    
    // Clear existing buffers
    safeBufferClear(m_colorBuffer, "color");
    safeBufferClear(m_albedoBuffer, "albedo");
    safeBufferClear(m_normalBuffer, "normal");
    safeBufferClear(m_motionBuffer, "motion");
    safeBufferClear(m_outputBuffer, "output");
    safeBufferClear(m_motionXYBuffer, "motionXY");
    
    // Allocate with proper sizes
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
}

void OptiXDenoiserIop::readInputPlanes()
{
    const Box processingBounds = m_processingBounds;
    
    // Define RGBA channels
    ChannelSet rgbaChannels;
    rgbaChannels += Chan_Red;
    rgbaChannels += Chan_Green;
    rgbaChannels += Chan_Blue;
    rgbaChannels += Chan_Alpha;
    
    for (int i = 0; i < node_inputs(); ++i) {
        if (aborted() || cancelled()) return;

        Iop* inputIop = dynamic_cast<Iop*>(input(i));
        if (!inputIop || !inputIop->tryValidate(true)) continue;

        try {
            // Only request RGBA channels from this input
            const ChannelSet availableChannels = inputIop->info().channels();
            ChannelSet requestChannels;
            foreach(z, rgbaChannels) {
                if (availableChannels.contains(z)) {
                    requestChannels += z;
                }
            }
            
            if (requestChannels.empty()) continue;
            
            inputIop->request(processingBounds, requestChannels, 0);
            ImagePlane inputPlane(processingBounds, false, requestChannels, requestChannels.size());
            inputIop->fetchPlane(inputPlane);
            
            // Select target buffer
            float* targetBuffer = nullptr;
            switch (i) {
            case 0: targetBuffer = m_colorBuffer.data(); break;
            case 1: targetBuffer = m_albedoBuffer.data(); break;
            case 2: targetBuffer = m_normalBuffer.data(); break;
            case 3: targetBuffer = m_motionBuffer.data(); break;
            }

            if (targetBuffer) {
                auto chanStride = inputPlane.chanStride();
                
                // Channel mapping
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
                
                // Get plane bounds for coordinate mapping
                const Box planeBounds = inputPlane.bounds();
                const int planeX = planeBounds.x();
                const int planeY = planeBounds.y();
                const int planeWidth = planeBounds.w();
                const int planeHeight = planeBounds.h();
                
                // Buffer bounds (where we're writing to)
                const int bufferX = processingBounds.x();
                const int bufferY = processingBounds.y();
                
                // Copy pixel data with proper coordinate mapping
                for (int py = 0; py < planeHeight; py++) {
                    for (int px = 0; px < planeWidth; px++) {
                        // Map from plane coordinates to buffer coordinates
                        int bufferPosX = (planeX + px) - bufferX;
                        int bufferPosY = (planeY + py) - bufferY;
                        
                        // Check if this pixel is within our buffer bounds
                        if (bufferPosX >= 0 && bufferPosX < static_cast<int>(m_width) &&
                            bufferPosY >= 0 && bufferPosY < static_cast<int>(m_height)) {
                            
                            for (int c = 0; c < 4; c++) {
                                size_t targetIndex = (static_cast<size_t>(bufferPosY) * m_width + bufferPosX) * 4 + c;
                                
                                if (channelMap[c] >= 0) {
                                    const float* indata = &inputPlane.readable()[chanStride * channelMap[c]];
                                    size_t inputIndex = static_cast<size_t>(py) * planeWidth + px;
                                    
                                    float value = indata[inputIndex];
                                    if (!std::isfinite(value)) {
                                        value = 0.0f;
                                    }
                                    targetBuffer[targetIndex] = value;
                                } else {
                                    // Channel doesn't exist, set default value
                                    targetBuffer[targetIndex] = (c == 3) ? 1.0f : 0.0f;
                                }
                            }
                        }
                    }
                }
            }
            
        } catch (const std::exception& e) {
            throw;
        }
    }
    
    // Process normal data if available
    if (node_inputs() > 2 && input(2) && !m_normalBuffer.empty()) {
        m_denoiser->clampNormalBuffer(m_normalBuffer.data(), m_width * m_height);
    }
    
    // Validate HDR range
    if (!m_colorBuffer.empty()) {
        bool outOfRange = false;
        m_denoiser->validateHDRRange(m_colorBuffer.data(), m_width * m_height, outOfRange);
    }
    
    // Convert motion vectors if needed
    if (node_inputs() > 3 && input(3) && m_bTemporal) {
        m_denoiser->convertMotionVectors(m_motionBuffer.data(), m_motionXYBuffer.data(), m_width, m_height);
    }
}

void OptiXDenoiserIop::writeOutputPlane(ImagePlane& plane)
{
    float* outputData = m_outputBuffer.data();
    
    // Get plane info
    const Box planeBounds = plane.bounds();
    const int planeWidth = planeBounds.w();
    const int planeHeight = planeBounds.h();
    
    if (!outputData) {
        throw std::runtime_error("Output buffer is null");
    }
    
    // Since we processed exactly the requested area, this should be a direct 1:1 copy
    if (static_cast<int>(m_width) != planeWidth || static_cast<int>(m_height) != planeHeight) {
        throw std::runtime_error("Buffer and plane dimensions mismatch: buffer " + 
                                std::to_string(m_width) + "x" + std::to_string(m_height) + 
                                " vs plane " + std::to_string(planeWidth) + "x" + std::to_string(planeHeight));
    }
    
    // Direct copy - no coordinate mapping needed
    for (int chanNo = 0; chanNo < 4; chanNo++) {
        float* outdata = &plane.writable()[plane.chanStride() * chanNo];

        for (int y = 0; y < planeHeight; y++) {
            for (int x = 0; x < planeWidth; x++) {
                size_t bufferIndex = (static_cast<size_t>(y) * m_width + x) * 4 + chanNo;
                size_t planeIndex = static_cast<size_t>(y) * planeWidth + x;
                
                if (bufferIndex < m_outputBuffer.size()) {
                    outdata[planeIndex] = outputData[bufferIndex];
                } else {
                    outdata[planeIndex] = 0.0f;
                }
            }
        }
    }
}

void OptiXDenoiserIop::handleDenoiserError(const std::exception& e)
{
    std::string errorMsg = "OptiX Denoiser error: ";
    errorMsg += e.what();
    error(errorMsg.c_str());
    
    // Reset denoiser state
    if (m_denoiser) {
        m_denoiser->finish();
    }
    m_bInitialized = false;
    
    // Copy input to output as fallback
    if (!m_colorBuffer.empty() && !m_outputBuffer.empty()) {
        if (m_colorBuffer.size() == m_outputBuffer.size()) {
            std::copy(m_colorBuffer.begin(), m_colorBuffer.end(), m_outputBuffer.begin());
        }
    }
}

void OptiXDenoiserIop::safeBufferClear(std::vector<float>& buffer, const std::string& name)
{
    MEMORY_LOG("Clearing buffer: " << name << " (size: " << buffer.size() << ")");
    
    if (!buffer.empty()) {
        try {
            buffer.clear();
            buffer.shrink_to_fit();
            
            // Force memory release by swapping with empty vector
            std::vector<float> empty;
            buffer.swap(empty);
            
            MEMORY_LOG("Buffer " << name << " cleared and memory released");
        } catch (const std::exception& e) {
            MEMORY_LOG("Exception clearing buffer " << name << ": " << e.what());
        } catch (...) {
            MEMORY_LOG("Unknown exception clearing buffer " << name);
        }
    } else {
        MEMORY_LOG("Buffer " << name << " was already empty");
    }
}

void OptiXDenoiserIop::renderStripe(ImagePlane& plane)
{
    CRASH_LOG("=== RENDER_STRIPE START ===");
    dumpCompleteState("renderStripe start");

    if (aborted() || cancelled()) {
        CRASH_LOG("Operation aborted/cancelled");
        return;
    }

    if (!validateCUDA()) {
        CRASH_LOG("CUDA validation failed");
        error("CUDA not available or no CUDA devices found");
        return;
    }

    // Safety check for input
    if (!input(0)) {
        CRASH_LOG("No input connected");
        error("No input connected");
        return;
    }

    // Get the requested area from the plane
    const Box requestedBounds = plane.bounds();
    
    unsigned int processWidth = requestedBounds.w();
    unsigned int processHeight = requestedBounds.h();
    
    TILE_LOG("Processing area: " << processWidth << "x" << processHeight);
    TILE_LOG("Last processed: " << m_lastProcessWidth << "x" << m_lastProcessHeight);
    
    // Get tile dimensions from current preset
    unsigned int tileWidth, tileHeight;
    getTileDimensions(tileWidth, tileHeight);
    
    TILE_LOG("Current tile dimensions: " << tileWidth << "x" << tileHeight);
    
    // Check if dimensions OR tile settings changed - cleanup if needed
    bool needsReinit = false;
    if (m_bInitialized) {
        if (processWidth != m_width || processHeight != m_height) {
            TILE_LOG("Dimensions changed: " << m_width << "x" << m_height << " -> " << processWidth << "x" << processHeight);
            needsReinit = true;
        }
        
        // Also check if tile dimensions changed
        unsigned int currentTileW = (m_tileWidth > 0) ? m_tileWidth : 0;
        unsigned int currentTileH = (m_tileHeight > 0) ? m_tileHeight : 0;
        if (currentTileW != tileWidth || currentTileH != tileHeight) {
            TILE_LOG("Tile dimensions changed: " << currentTileW << "x" << currentTileH << " -> " << tileWidth << "x" << tileHeight);
            needsReinit = true;
        }
    }
    
    if (needsReinit && m_denoiser) {
        CRASH_LOG("Reinitialization needed - cleaning up denoiser");
        dumpCompleteState("before reinit cleanup");
        
        try {
            m_denoiser->dumpMemoryState();
            m_denoiser->finish();
            CRASH_LOG("Denoiser finished successfully");
        } catch (const std::exception& e) {
            CRASH_LOG("Exception during reinit cleanup: " << e.what());
        } catch (...) {
            CRASH_LOG("Unknown exception during reinit cleanup");
        }
        m_bInitialized = false;
        m_frameCounter = 0;
        
        // Clear host buffers to prevent size mismatches
        CRASH_LOG("Clearing host buffers for reinit");
        logBufferSizes("before reinit buffer clear");
        
        try {
            safeBufferClear(m_colorBuffer, "color");
            safeBufferClear(m_albedoBuffer, "albedo");
            safeBufferClear(m_normalBuffer, "normal");
            safeBufferClear(m_motionBuffer, "motion");
            safeBufferClear(m_outputBuffer, "output");
            safeBufferClear(m_motionXYBuffer, "motionXY");
            CRASH_LOG("Reinit buffer clear successful");
        } catch (const std::exception& e) {
            CRASH_LOG("Exception during reinit buffer clear: " << e.what());
        } catch (...) {
            CRASH_LOG("Unknown exception during reinit buffer clear");
        }
        
        logBufferSizes("after reinit buffer clear");
    }

    // Enhanced temporal frame management
    bool shouldUseTemporal = hasMotionVectorsConnected();
    int currentFrame = outputContext().frame();

    // Check temporal mode and frame continuity
    // CRITICAL: Handle temporal frame logic - Per-instance tracking
    if (m_bTemporal) {
        bool isStartFrame = (currentFrame == m_startFrame);
        bool needsTemporalReset = false;
        
        // Only reset at start frame or major changes
        if (isStartFrame) {
            CRASH_LOG("At start frame " << currentFrame << " - resetting temporal state");
            needsTemporalReset = true;
        } else if (!isWithinFrameRange(currentFrame)) {
            CRASH_LOG("Frame " << currentFrame << " outside range [" << m_startFrame << "-" << m_endFrame << "] - using non-temporal");
            // Temporarily disable temporal for this frame
            shouldUseTemporal = false;
        }
        
        // Only reset if at start frame OR if denoiser isn't initialized
        if (needsTemporalReset && m_denoiser && m_bInitialized) {
            CRASH_LOG("Resetting temporal denoiser state");
            try {
                m_denoiser->finish();
                m_bInitialized = false;
                m_frameCounter = 0;
                CRASH_LOG("Temporal reset successful");
            } catch (...) {
                CRASH_LOG("Exception during temporal reset");
            }
        }
    }

    // Update processing state - process exactly what was requested
    m_width = processWidth;
    m_height = processHeight;
    m_processingBounds = requestedBounds;
    
    // Store tile dimensions for comparison in next frame
    m_tileWidth = tileWidth;
    m_tileHeight = tileHeight;
    
    // Update tracking variables
    m_lastProcessWidth = processWidth;
    m_lastProcessHeight = processHeight;

    CRASH_LOG("State updated - proceeding with processing");
    dumpCompleteState("before processing");

    try {
        // Allocate buffers for the requested area
        CRASH_LOG("Allocating buffers");
        allocateBuffers();
        logBufferSizes("after allocation");

        // Check for abort after buffer allocation
        if (aborted() || cancelled()) {
            CRASH_LOG("Aborted after buffer allocation");
            return;
        }

        // Read input data from the requested area
        CRASH_LOG("Reading input planes");
        readInputPlanes();

        // Setup denoiser data structure
        CRASH_LOG("Setting up denoiser data");
        OptiXDenoiserCore::Data data;
        data.width = m_width;
        data.height = m_height;
        data.color = m_colorBuffer.data();
        data.albedo = (node_inputs() > 1 && input(1)) ? m_albedoBuffer.data() : nullptr;
        data.normal = (node_inputs() > 2 && input(2)) ? m_normalBuffer.data() : nullptr;
        data.flow = (node_inputs() > 3 && input(3) && m_bTemporal) ? m_motionXYBuffer.data() : nullptr;
        data.outputs.push_back(m_outputBuffer.data());

        // Initialize or update denoiser
        if (!m_bInitialized) {
            CRASH_LOG("Initializing denoiser with tile size: " << tileWidth << "x" << tileHeight);
            try {
                m_denoiser->init(data, tileWidth, tileHeight, m_bTemporal);
                m_bInitialized = true;
                m_frameCounter = 0;
                CRASH_LOG("Denoiser initialized successfully");
                m_denoiser->dumpMemoryState();
            } catch (const std::exception& e) {
                CRASH_LOG("Exception during denoiser init: " << e.what());
                throw;
            }
        } else {
            CRASH_LOG("Updating denoiser");
            try {
                m_denoiser->update(data);
                CRASH_LOG("Denoiser updated successfully");
            } catch (const std::exception& e) {
                CRASH_LOG("Exception during denoiser update: " << e.what());
                throw;
            }
        }

        // Check for abort before denoising
        if (aborted() || cancelled()) {
            CRASH_LOG("Aborted before denoising");
            return;
        }
        
        // Execute denoising
        CRASH_LOG("Executing denoising");
        m_denoiser->exec();
        CRASH_LOG("Denoising completed");
        
        // Get results from GPU
        CRASH_LOG("Getting results from GPU");
        m_denoiser->getResults();
        CRASH_LOG("Results retrieved");

        // Write output to plane - direct 1:1 mapping since we processed exactly what was requested
        CRASH_LOG("Writing output plane");
        writeOutputPlane(plane);
        CRASH_LOG("Output plane written");
        
        // Increment frame counter after successful processing
        m_frameCounter++;
        CRASH_LOG("Frame counter incremented to: " << m_frameCounter);

    } catch (const std::exception& e) {
        CRASH_LOG("Exception in renderStripe: " << e.what());
        dumpCompleteState("exception occurred");
        handleDenoiserError(e);
        
        // Fallback: copy input to output
        if (!m_colorBuffer.empty() && !m_outputBuffer.empty() && 
            m_colorBuffer.size() == m_outputBuffer.size()) {
            CRASH_LOG("Using fallback - copying input to output");
            std::copy(m_colorBuffer.begin(), m_colorBuffer.end(), m_outputBuffer.begin());
            writeOutputPlane(plane);
        }
    } catch (...) {
        CRASH_LOG("Unknown exception in renderStripe");
        dumpCompleteState("unknown exception occurred");
        error("Unknown error occurred during denoising");
        
        // Emergency fallback
        if (!m_colorBuffer.empty() && !m_outputBuffer.empty() && 
            m_colorBuffer.size() == m_outputBuffer.size()) {
            CRASH_LOG("Using emergency fallback");
            std::copy(m_colorBuffer.begin(), m_colorBuffer.end(), m_outputBuffer.begin());
            writeOutputPlane(plane);
        }
    }
    
    dumpCompleteState("renderStripe end");
    CRASH_LOG("=== RENDER_STRIPE END ===");
}

static Iop *build(Node *node) { 
    return new OptiXDenoiserIop(node); 
}

const Iop::Description OptiXDenoiserIop::d("OptiXDenoiser", "Filter/OptiXDenoiser", build);
