// Copyright (c) 2025 - OptiX Denoiser for Nuke
// Based on NVIDIA OptiX SDK examples
// COMPLETE HEADER FILE

#pragma once

#include <DDImage/PlanarIop.h>
#include <DDImage/Interest.h>
#include <DDImage/Row.h>
#include <DDImage/Knobs.h>
#include <DDImage/Knob.h>
#include <DDImage/DDMath.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_denoiser_tiling.h>
#include <cuda_runtime.h>
#include <cuda.h>  // For CUcontext, cuCtxCreate, etc.

#include <vector>
#include <memory>
#include <chrono>  // Add for timeout functionality
#include <mutex>   // Add for thread safety

// Define ENABLE_MEMORY_GUARDS in the header so it's available everywhere
#ifndef ENABLE_MEMORY_GUARDS
#define ENABLE_MEMORY_GUARDS 1
#endif

// Memory debugging macros - GLOBAL SCOPE
#ifdef ENABLE_MEMORY_GUARDS
#define MEMORY_GUARD_PATTERN 0xDEADBEEF
#define MEMORY_GUARD_SIZE 16

#define SAFE_BUFFER_ACCESS(buffer, index, size, name) \
    do { \
        if ((index) >= (size)) { \
            std::cerr << "\n=== BUFFER OVERRUN DETECTED ===" << std::endl; \
            std::cerr << "Buffer: " << (name) << std::endl; \
            std::cerr << "Index: " << (index) << std::endl; \
            std::cerr << "Size: " << (size) << std::endl; \
            std::cerr << "Location: " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "Function: " << __func__ << std::endl; \
            std::cerr << "================================" << std::endl; \
            std::abort(); \
        } \
    } while(0)
#else
    #define SAFE_BUFFER_ACCESS(buffer, index, size, name)
#endif

static const char *const HELP = "AI-based denoiser using NVIDIA OptiX technology";
static const char *const CLASS = "OptiXDenoiser";

using namespace DD::Image;

// Forward declarations
class OptiXDenoiserCore;

// OptiX context callback function declaration
void context_log_cb(uint32_t level, const char* tag, const char* message, void* cbdata);

// OptiX Denoiser wrapper class
class OptiXDenoiserCore
{
public:
    struct Data
    {
        uint32_t  width     = 0;
        uint32_t  height    = 0;
        float*    color     = nullptr;
        float*    albedo    = nullptr;
        float*    normal    = nullptr;
        float*    flow      = nullptr;
        std::vector<float*> outputs;
    };

    // Constructor/Destructor
    OptiXDenoiserCore() = default;
    ~OptiXDenoiserCore() { finish(); }

    // Disable copy constructor and assignment to prevent memory issues
    OptiXDenoiserCore(const OptiXDenoiserCore&) = delete;
    OptiXDenoiserCore& operator=(const OptiXDenoiserCore&) = delete;

    // Main interface methods
    void init(const Data& data, 
              unsigned int tileWidth = 0, 
              unsigned int tileHeight = 0,
              bool temporalMode = false);
    
    void update(const Data& data);
    void exec();
    void getResults();
    void finish();

    // State query
    bool isInitialized() const { return m_initialized; }

    // Memory usage query
    size_t getTotalMemoryUsed() const { return m_totalMemoryUsed; }
    size_t getAllocationCount() const { return m_allocatedPointers.size(); }

    // Static utility methods
    static void convertMotionVectors(const float* motionRGBA, float* motionXY, 
                                   unsigned int width, unsigned int height);
    static void clampNormalBuffer(float* normalData, size_t pixelCount);
    static void validateHDRRange(const float* imageData, size_t pixelCount, bool& outOfRange);

private:
#ifdef ENABLE_MEMORY_GUARDS
    // Memory guard structure
    struct MemoryGuard {
        uint32_t guards[MEMORY_GUARD_SIZE];
        
        MemoryGuard() {
            for (int i = 0; i < MEMORY_GUARD_SIZE; i++) {
                guards[i] = MEMORY_GUARD_PATTERN;
            }
        }
        
        bool isValid() const {
            for (int i = 0; i < MEMORY_GUARD_SIZE; i++) {
                if (guards[i] != MEMORY_GUARD_PATTERN) {
                    return false;
                }
            }
            return true;
        }
    };
#endif

    // Memory allocation tracking
    struct AllocationInfo {
        void* ptr;
        size_t size;
        std::string description;  // Add description for better debugging
        
        AllocationInfo(void* p, size_t s, const std::string& desc = "") 
            : ptr(p), size(s), description(desc) {}
    };
    
    std::vector<AllocationInfo> m_allocatedPointers;
    mutable bool m_cleanupInProgress = false;
    mutable std::mutex m_cleanupMutex;  // Thread safety for cleanup
    
    // OptiX context and denoiser
    OptixDeviceContext    m_context      = nullptr;
    OptixDenoiser         m_denoiser     = nullptr;
    OptixDenoiserParams   m_params       = {};

    // CUDA context management
    CUcontext             m_cuContext    = nullptr;
    bool                  m_ownsCudaContext = false;  // Track if we created it

    // Configuration state
    bool                  m_temporalMode = false;
    bool                  m_initialized  = false;
    uint32_t              m_frameIndex   = 0;

    // CUDA stream for async operations
    cudaStream_t          m_stream       = nullptr;

    // Device memory pointers
    CUdeviceptr           m_intensity    = 0;
    CUdeviceptr           m_avgColor     = 0;
    CUdeviceptr           m_scratch      = 0;
    uint32_t              m_scratch_size = 0;
    CUdeviceptr           m_state        = 0;
    uint32_t              m_state_size   = 0;

    // Tiling configuration
    unsigned int          m_tileWidth    = 0;
    unsigned int          m_tileHeight   = 0;
    unsigned int          m_overlap      = 0;

    // OptiX image layers
    OptixDenoiserGuideLayer           m_guideLayer = {};
    std::vector<OptixDenoiserLayer>   m_layers;
    std::vector<float*>               m_host_outputs;

    // Memory tracking
    mutable size_t m_totalMemoryUsed = 0;

    // Helper functions
    OptixImage2D createOptixImage2D(unsigned int width, unsigned int height, 
                                   const float* hmem = nullptr, 
                                   OptixPixelFormat format = OPTIX_PIXEL_FORMAT_FLOAT4);
    void copyOptixImage2D(OptixImage2D& dest, const OptixImage2D& src);
    void* allocateAlignedCudaMemory(size_t size, size_t& actualSize, 
                                   const std::string& description = "");
    
    // Internal cleanup helpers
    void cleanupCudaResources();
    void cleanupOptixResources();
    
    // Memory debugging helpers
    void dumpMemoryUsage() const;
    bool validateAllAllocations() const;
};

// Nuke Plugin Class
class OptiXDenoiserIop : public PlanarIop
{
public:
    // Constructor/Destructor
    OptiXDenoiserIop(Node *node);
    virtual ~OptiXDenoiserIop();

    // Disable copy constructor and assignment
    OptiXDenoiserIop(const OptiXDenoiserIop&) = delete;
    OptiXDenoiserIop& operator=(const OptiXDenoiserIop&) = delete;

    // Nuke interface requirements
    int minimum_inputs() const override { return 1; }
    int maximum_inputs() const override { return 4; }

    PackedPreference packedPreference() const override { return ePackedPreferenceUnpacked; }

    // Knob interface
    void knobs(Knob_Callback f) override;
    int knob_changed(Knob* k) override;

    // Processing pipeline
    void _validate(bool) override;
    void getRequests(const Box& box, const ChannelSet& channels, int count, RequestOutput& /*reqData*/) const override;
    void renderStripe(ImagePlane& plane) override;

    // Processing preferences
    bool useStripes() const override { return false; }
    bool renderFullPlanes() const override { return true; }

    // Input/output interface
    const char *input_label(int n, char *) const override;
    static const Iop::Description d;

    // Node information
    const char *Class() const override { return d.name; }
    const char *node_help() const override { return HELP; }

private:
    // RAII cleanup guard for exception safety
    struct CleanupGuard {
        OptiXDenoiserIop* iop;
        bool cleanupNeeded = false;
        
        explicit CleanupGuard(OptiXDenoiserIop* i) : iop(i) {}
        
        ~CleanupGuard() {
            if (cleanupNeeded && iop && iop->m_denoiser) {
                try {
                    iop->m_denoiser->finish();
                    iop->m_bInitialized = false;
                } catch (...) {
                    // Don't throw from destructor
                }
            }
        }
    };

    // UI Parameters - Simplified
    int m_tileWidth;      // Tile width for memory optimization
    int m_tileHeight;     // Tile height for memory optimization
    int m_numRuns;        // Number of denoising iterations
    float m_blendFactor;  // Blend factor for temporal denoising

    // Internal state
    bool m_bTemporal;     // Auto-detected temporal mode
    bool m_bInitialized;  // Initialization state

    // Image dimensions
    unsigned int m_width, m_height;

    // Channel configuration
    ChannelSet m_defaultChannels;
    int m_defaultNumberOfChannels;

    // OptiX denoiser core instance
    std::unique_ptr<OptiXDenoiserCore> m_denoiser;
    uint32_t m_frameCounter;

    // Host image buffers (RGBA format)
    std::vector<float> m_colorBuffer;
    std::vector<float> m_albedoBuffer;
    std::vector<float> m_normalBuffer;
    std::vector<float> m_motionBuffer;
    std::vector<float> m_motionXYBuffer;
    std::vector<float> m_outputBuffer;

    
    // ADD IT HERE - right after the buffer declarations:
    Box m_processingBounds;  // Track what area we're processing

    // Timing and abort handling
    std::chrono::steady_clock::time_point m_lastAbortCheck;
    static constexpr std::chrono::milliseconds ABORT_CHECK_INTERVAL{100}; // Check every 100ms
    static constexpr std::chrono::seconds OPERATION_TIMEOUT{300}; // 300 second timeout

    // Helper methods
    bool validateCUDA();
    void allocateBuffers();
    void readInputPlanes();
    void writeOutputPlane(ImagePlane& plane);
    bool hasMotionVectorsConnected() const;
    
    // Error handling
    void handleDenoiserError(const std::exception& e);
    
    // Safety helpers
    bool checkAbortWithTimeout(const std::chrono::steady_clock::time_point& startTime);
    void safeBufferClear(std::vector<float>& buffer, const std::string& name);
    
    // Buffer integrity validation - NOW ALWAYS DECLARED
    void validateBufferIntegrity(const std::vector<float>& buffer, const std::string& name);
};
