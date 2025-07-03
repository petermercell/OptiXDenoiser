// Copyright (c) 2025 - OptiX Denoiser for Nuke
// Based on NVIDIA OptiX SDK examples

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

#include <vector>
#include <memory>

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

    // Static utility methods
    static void convertMotionVectors(const float* motionRGBA, float* motionXY, 
                                   unsigned int width, unsigned int height);
    static void clampNormalBuffer(float* normalData, size_t pixelCount);
    static void validateHDRRange(const float* imageData, size_t pixelCount, bool& outOfRange);

private:
    // OptiX context and denoiser
    OptixDeviceContext    m_context      = nullptr;
    OptixDenoiser         m_denoiser     = nullptr;
    OptixDenoiserParams   m_params       = {};

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

    // Helper functions
    OptixImage2D createOptixImage2D(unsigned int width, unsigned int height, 
                                   const float* hmem = nullptr, 
                                   OptixPixelFormat format = OPTIX_PIXEL_FORMAT_FLOAT4);
    void copyOptixImage2D(OptixImage2D& dest, const OptixImage2D& src);
    void* allocateAlignedCudaMemory(size_t size, size_t& actualSize);
    static void context_log_cb(uint32_t level, const char* tag, const char* message, void* cbdata);
    
    // Internal cleanup helpers
    void cleanupCudaResources();
    void cleanupOptixResources();
    
    // Memory tracking
    mutable size_t m_totalMemoryUsed = 0;
};

// Nuke Plugin Class
class OptiXDenoiserIop : public PlanarIop
{
public:
    // Constructor/Destructor
    OptiXDenoiserIop(Node *node);
    virtual ~OptiXDenoiserIop();

    // Nuke interface requirements
    int minimum_inputs() const override { return 1; }
    int maximum_inputs() const override { return 4; }

    PackedPreference packedPreference() const override { return ePackedPreferenceUnpacked; }

    // Knob interface
    void knobs(Knob_Callback f) override;
    int knob_changed(Knob* k) override;

    // Processing pipeline
    void _validate(bool) override;
    void getRequests(const Box& box, const ChannelSet& channels, int count, RequestOutput &reqData) const override;
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

    // Helper methods
    bool validateCUDA();
    void allocateBuffers();
    void readInputPlanes();
    void writeOutputPlane(ImagePlane& plane);
    bool hasMotionVectorsConnected() const;
    
    // Error handling
    void handleDenoiserError(const std::exception& e);
};
