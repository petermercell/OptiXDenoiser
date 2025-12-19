// Copyright (c) 2025 - OptiX Denoiser for Nuke
// Based on NVIDIA OptiX SDK examples

#pragma once

#include <DDImage/PlanarIop.h>
#include <DDImage/Interest.h>
#include <DDImage/Row.h>
#include <DDImage/Knobs.h>
#include <DDImage/Knob.h>
#include <DDImage/DDMath.h>
#include <DDImage/OutputContext.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_denoiser_tiling.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <iostream>
#include <fstream>

// Debug flags - all disabled for production
#define ENABLE_MEMORY_DEBUG 0
#define ENABLE_TILE_DEBUG 0
#define ENABLE_CRASH_DEBUG 0

#if ENABLE_MEMORY_DEBUG
    #define MEMORY_LOG(msg) \
        do { \
            std::cout << "[MEMORY] " << __FUNCTION__ << ":" << __LINE__ << " - " << msg << std::endl; \
            std::cout.flush(); \
        } while(0)
    
    #define MEMORY_LOG_PTR(name, ptr) \
        do { \
            std::cout << "[MEMORY] " << __FUNCTION__ << ":" << __LINE__ << " - " << name << ": " << static_cast<const void*>(ptr) << std::endl; \
            std::cout.flush(); \
        } while(0)
#else
    #define MEMORY_LOG(msg)
    #define MEMORY_LOG_PTR(name, ptr)
#endif

#if ENABLE_TILE_DEBUG
    #define TILE_LOG(msg) \
        do { \
            std::cout << "[TILE] " << __FUNCTION__ << ":" << __LINE__ << " - " << msg << std::endl; \
            std::cout.flush(); \
        } while(0)
#else
    #define TILE_LOG(msg)
#endif

#if ENABLE_CRASH_DEBUG
    #define CRASH_LOG(msg) \
        do { \
            std::cout << "[CRASH_DEBUG] " << __FUNCTION__ << ":" << __LINE__ << " - " << msg << std::endl; \
            std::cout.flush(); \
        } while(0)
#else
    #define CRASH_LOG(msg)
#endif

static const char *const HELP = "AI-based denoiser using NVIDIA OptiX technology";
static const char *const CLASS = "OptiXDenoiser";

using namespace DD::Image;

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

    // Disable copy constructor and assignment
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
    uint32_t getFrameIndex() const { return m_frameIndex; }

    // Debug methods
    void dumpMemoryState() const;
    size_t getTotalAllocations() const { return m_allocatedPointers.size(); }
    size_t getTotalMemoryUsed() const { return m_totalMemoryUsed; }

    // Static utility methods
    static void convertMotionVectors(const float* motionRGBA, float* motionXY, 
                                   unsigned int width, unsigned int height);
    static void clampNormalBuffer(float* normalData, size_t pixelCount);
    static void validateHDRRange(const float* imageData, size_t pixelCount, bool& outOfRange);

private:
    // Memory allocation tracking with enhanced debugging
    struct AllocationInfo {
        void* ptr;
        size_t size;
        std::string description;
        bool isValid;
        
        AllocationInfo(void* p, size_t s, const std::string& desc = "") 
            : ptr(p), size(s), description(desc), isValid(true) {}
    };
    
    std::vector<AllocationInfo> m_allocatedPointers;
    mutable std::mutex m_cleanupMutex;
    mutable bool m_cleanupInProgress = false;
    
    // OptiX context and denoiser
    OptixDeviceContext    m_context      = nullptr;
    OptixDenoiser         m_denoiser     = nullptr;
    OptixDenoiserParams   m_params       = {};

    // CUDA context management
    CUcontext             m_cuContext    = nullptr;
    bool                  m_ownsCudaContext = false;

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
    
    // Debug helpers
    void validateAllPointers() const;
    void logMemoryState(const std::string& context) const;
};

// Nuke Plugin Class
class OptiXDenoiserIop : public PlanarIop
{
public:
    // Tile size presets
    enum TilePreset {
        TILE_NONE = 0,      // No tiling (full image)
        TILE_512 = 1,       // 512x512 tiles
        TILE_1024 = 2,      // 1024x1024 tiles  
        TILE_2048 = 3       // 2048x2048 tiles
    };

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
    void getRequests(const Box& box, const ChannelSet& channels, int count, RequestOutput& reqData) const override;
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
    // UI Parameters - Frame Range Control
    int m_startFrame;          // Start frame (reference frame)
    int m_endFrame;            // End frame
    
    // Tile preset selection
    int m_tilePreset;          // Selected tile preset (TilePreset enum)

    // Internal state with debugging
    bool m_bTemporal;
    bool m_bInitialized;
    int m_tileWidth;
    int m_tileHeight;
    int m_numRuns;
    
    // Debug state tracking
    int m_lastTilePreset = -1;
    unsigned int m_lastProcessWidth = 0;
    unsigned int m_lastProcessHeight = 0;

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

    // Processing bounds tracking
    Box m_processingBounds;

    // Helper methods
    bool validateCUDA();
    void allocateBuffers();
    void readInputPlanes();
    void writeOutputPlane(ImagePlane& plane);
    bool hasMotionVectorsConnected() const;
    void handleDenoiserError(const std::exception& e);
    void safeBufferClear(std::vector<float>& buffer, const std::string& name);
    
    // Frame range helpers
    bool isWithinFrameRange(int currentFrame) const;
    bool isStartFrame(int currentFrame) const;
    int getAnimationLength() const;
    
    // Tile preset helpers
    void getTileDimensions(unsigned int& tileWidth, unsigned int& tileHeight) const;
    const char* getTilePresetName(int preset) const;
    
    // Debug helpers
    void logBufferSizes(const std::string& context) const;
    void logTileChange(int oldPreset, int newPreset) const;
    void dumpCompleteState(const std::string& context) const;
};
