// Copyright (c) 2025 - OptiX Denoiser for Nuke
// Based on NVIDIA OptiX SDK examples

#include "OptiXDenoiser.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <cstring>

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

// OptiXDenoiserCore Implementation
void OptiXDenoiserCore::context_log_cb(uint32_t level, const char* tag, const char* message, void* /*cbdata*/)
{
    if (level < 4)
        std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

void* OptiXDenoiserCore::allocateAlignedCudaMemory(size_t size, size_t& actualSize)
{
    // Ensure 16-byte alignment as required by OptiX
    const size_t alignment = 16;
    actualSize = ((size + alignment - 1) / alignment) * alignment;
    
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, actualSize));
    
    // Zero-initialize the memory
    CUDA_CHECK(cudaMemset(ptr, 0, actualSize));
    
    m_totalMemoryUsed += actualSize;
    return ptr;
}

void OptiXDenoiserCore::clampNormalBuffer(float* normalData, size_t pixelCount)
{
    if (!normalData) return;
    
    // Clamp normal values to [0.0, 1.0] range as required by OptiX
    for (size_t i = 0; i < pixelCount; ++i) {
        // Process RGB channels (3 floats per pixel for normals)
        for (int c = 0; c < 3; ++c) {
            size_t index = i * 4 + c; // Assuming RGBA format
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
    
    // Calculate stride based on format
    size_t pixelSize = 0;
    switch (format) {
        case OPTIX_PIXEL_FORMAT_FLOAT4:
            pixelSize = sizeof(float4);
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
    
    // Use aligned allocation
    oi.data = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(frame_byte_size, actualSize));
    
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
    if (m_intensity) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_intensity)));
        m_intensity = 0;
    }
    if (m_avgColor) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_avgColor)));
        m_avgColor = 0;
    }
    if (m_scratch) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_scratch)));
        m_scratch = 0;
    }
    if (m_state) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state)));
        m_state = 0;
    }
    
    // Clean up guide layer resources
    if (m_guideLayer.albedo.data) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_guideLayer.albedo.data)));
        m_guideLayer.albedo.data = 0;
    }
    if (m_guideLayer.normal.data) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_guideLayer.normal.data)));
        m_guideLayer.normal.data = 0;
    }
    if (m_guideLayer.flow.data) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_guideLayer.flow.data)));
        m_guideLayer.flow.data = 0;
    }

    // Clean up layer resources
    for (auto& layer : m_layers) {
        if (layer.input.data) {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(layer.input.data)));
            layer.input.data = 0;
        }
        if (layer.output.data) {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(layer.output.data)));
            layer.output.data = 0;
        }
        if (layer.previousOutput.data) {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(layer.previousOutput.data)));
            layer.previousOutput.data = 0;
        }
    }
    
    if (m_stream) {
        CUDA_CHECK(cudaStreamDestroy(m_stream));
        m_stream = nullptr;
    }
    
    m_totalMemoryUsed = 0;
}

void OptiXDenoiserCore::cleanupOptixResources()
{
    if (m_denoiser) {
        optixDenoiserDestroy(m_denoiser);
        m_denoiser = nullptr;
    }
    if (m_context) {
        optixDeviceContextDestroy(m_context);
        m_context = nullptr;
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
    m_tileWidth = (tileWidth > 0) ? tileWidth : data.width;
    m_tileHeight = (tileHeight > 0) ? tileHeight : data.height;
    m_frameIndex = 0;

    // Initialize CUDA
    CUDA_CHECK(cudaFree(nullptr));
    CUDA_CHECK(cudaStreamCreate(&m_stream));

    // Initialize OptiX
    CUcontext cu_ctx = nullptr;
    OPTIX_CHECK(optixInit());
    
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &m_context));

    // Create denoiser - Always use HDR mode
    OptixDenoiserOptions denoiser_options = {};
    denoiser_options.guideAlbedo = data.albedo ? 1 : 0;
    denoiser_options.guideNormal = data.normal ? 1 : 0;
    denoiser_options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY; // Always copy alpha

    OptixDenoiserModelKind modelKind = temporalMode ? 
        OPTIX_DENOISER_MODEL_KIND_TEMPORAL : OPTIX_DENOISER_MODEL_KIND_HDR;
    
    OPTIX_CHECK(optixDenoiserCreate(m_context, modelKind, &denoiser_options, &m_denoiser));

    // Allocate device memory
    OptixDenoiserSizes denoiser_sizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, m_tileWidth, m_tileHeight, &denoiser_sizes));

    if (tileWidth == 0) {
        m_scratch_size = static_cast<uint32_t>(denoiser_sizes.withoutOverlapScratchSizeInBytes);
        m_overlap = 0;
    } else {
        m_scratch_size = static_cast<uint32_t>(denoiser_sizes.withOverlapScratchSizeInBytes);
        m_overlap = denoiser_sizes.overlapWindowSizeInPixels;
    }

    // Allocate buffers with proper alignment
    size_t actualSize;
    m_intensity = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(sizeof(float), actualSize));
    m_avgColor = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(3 * sizeof(float), actualSize));
    m_scratch = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(m_scratch_size, actualSize));
    m_state = reinterpret_cast<CUdeviceptr>(allocateAlignedCudaMemory(denoiser_sizes.stateSizeInBytes, actualSize));

    m_state_size = static_cast<uint32_t>(denoiser_sizes.stateSizeInBytes);

    // Create image layers
    OptixDenoiserLayer layer = {};
    layer.input = createOptixImage2D(data.width, data.height, data.color);
    layer.output = createOptixImage2D(data.width, data.height);

    if (m_temporalMode) {
        layer.previousOutput = createOptixImage2D(data.width, data.height);
        // Initialize previous output with current input for first frame
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

    // Setup denoiser
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

    // Denoise
    if (m_tileWidth == m_layers[0].input.width && m_tileHeight == m_layers[0].input.height) {
        OPTIX_CHECK(optixDenoiserInvoke(m_denoiser, m_stream, &m_params,
                                       m_state, m_state_size, &m_guideLayer,
                                       m_layers.data(), static_cast<unsigned int>(m_layers.size()),
                                       0, 0, m_scratch, m_scratch_size));
    } else {
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

    try {
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
}

void OptiXDenoiserCore::convertMotionVectors(const float* motionRGBA, float* motionXY, 
                                            unsigned int width, unsigned int height)
{
    for (unsigned int i = 0; i < width * height; ++i) {
        motionXY[i * 2 + 0] = motionRGBA[i * 4 + 0]; // X component
        motionXY[i * 2 + 1] = motionRGBA[i * 4 + 1]; // Y component
    }
}

// Nuke Plugin Implementation
OptiXDenoiserIop::OptiXDenoiserIop(Node *node) : PlanarIop(node)
{
    // Initialize parameters
    m_bTemporal = false;
    m_bInitialized = false;
    m_tileWidth = 0;
    m_tileHeight = 0;
    m_numRuns = 1;
    m_blendFactor = 0.0f;

    m_width = 0;
    m_height = 0;

    m_defaultChannels = Mask_RGB;
    m_defaultNumberOfChannels = m_defaultChannels.size();

    m_denoiser = std::make_unique<OptiXDenoiserCore>();
    m_frameCounter = 0;
}

OptiXDenoiserIop::~OptiXDenoiserIop()
{
    if (m_denoiser) {
        m_denoiser->finish();
    }
}

void OptiXDenoiserIop::knobs(Knob_Callback f)
{
    Divider(f, "Memory Settings");
    
    Int_knob(f, &m_tileWidth, "tile_width", "Tile Width");
    Tooltip(f, "Tile width for memory optimization (0 = no tiling)");
    SetFlags(f, Knob::STARTLINE);

    Int_knob(f, &m_tileHeight, "tile_height", "Tile Height");
    Tooltip(f, "Tile height for memory optimization (0 = no tiling)");

    Divider(f, "");
    
    Text_knob(f, "OptiXDenoiser by Peter Mercell v1.00 / 2025", "");
    SetFlags(f, Knob::STARTLINE);
}

// Updated knob_changed function:
int OptiXDenoiserIop::knob_changed(Knob* k)
{
    if (k->is("inputs")) {
        bool shouldUseTemporal = hasMotionVectorsConnected();
        if (m_bTemporal != shouldUseTemporal) {
            m_bTemporal = shouldUseTemporal;
            // Force re-initialization of denoiser
            if (m_denoiser) {
                m_denoiser->finish();
                m_bInitialized = false;
            }
        }
        return 1;
    }
    
    if (k->is("tile_width") || k->is("tile_height")) {
        // Force re-initialization if memory settings changed
        if (m_denoiser) {
            m_denoiser->finish();
            m_bInitialized = false;
        }
        return 1;
    }
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

void OptiXDenoiserIop::_validate(bool for_real)
{
    copy_info();
    
    bool shouldUseTemporal = hasMotionVectorsConnected();
    
    if (m_bTemporal != shouldUseTemporal) {
        m_bTemporal = shouldUseTemporal;
        // Force re-initialization
        if (m_denoiser) {
            m_denoiser->finish();
            m_bInitialized = false;
        }
    }
}

void OptiXDenoiserIop::getRequests(const Box& box, const ChannelSet& channels, int count, RequestOutput& reqData) const
{
    for (int i = 0, endI = getInputs().size(); i < endI; i++) {
        if (input(i)) {
            const ChannelSet readChannels = input(i)->info().channels();
            input(i)->request(readChannels, count);
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
        
        // Check compute capability
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
    size_t bufferSize = m_width * m_height * 4 * sizeof(float); // 4 channels (RGBA)
    m_colorBuffer.resize(bufferSize / sizeof(float), 0.0f);
    m_albedoBuffer.resize(bufferSize / sizeof(float), 0.0f);
    m_normalBuffer.resize(bufferSize / sizeof(float), 0.0f);
    m_motionBuffer.resize(bufferSize / sizeof(float), 0.0f);
    m_outputBuffer.resize(bufferSize / sizeof(float), 0.0f);
    
    // Allocate motion vector buffer in XY format for OptiX
    m_motionXYBuffer.resize(m_width * m_height * 2, 0.0f);
}

void OptiXDenoiserIop::readInputPlanes()
{
    const Box imageFormat = info().format();
    
    for (int i = 0; i < node_inputs(); ++i) {
        if (aborted() || cancelled()) return;

        Iop* inputIop = dynamic_cast<Iop*>(input(i));
        if (!inputIop || !inputIop->tryValidate(true)) continue;

        Box imageBounds = inputIop->info();
        imageBounds.intersect(imageFormat);

        inputIop->request(imageBounds, m_defaultChannels, 0);

        ImagePlane inputPlane(imageBounds, false, m_defaultChannels, m_defaultNumberOfChannels);
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
            for (int y = 0; y < m_height; y++) {
                for (int x = 0; x < m_width; x++) {
                    for (int c = 0; c < std::min(4, m_defaultNumberOfChannels); c++) {
                        const float* indata = &inputPlane.readable()[chanStride * c];
                        size_t index = (y * m_width + x) * 4 + c;
                        targetBuffer[index] = indata[y * m_width + x];
                    }
                    // Set alpha to 1.0 for channels that don't have alpha
                    if (m_defaultNumberOfChannels < 4) {
                        size_t alphaIndex = (y * m_width + x) * 4 + 3;
                        targetBuffer[alphaIndex] = 1.0f;
                    }
                }
            }
        }
    }
    
    // Process normal data if available (clamp to [0,1] range)
    if (node_inputs() > 2 && input(2) && !m_normalBuffer.empty()) {
        m_denoiser->clampNormalBuffer(m_normalBuffer.data(), m_width * m_height);
    }
    
    // Validate HDR range
    if (!m_colorBuffer.empty()) {
        bool outOfRange = false;
        m_denoiser->validateHDRRange(m_colorBuffer.data(), m_width * m_height, outOfRange);
    }
    
    // Convert motion vectors to XY format if we have motion input
    if (node_inputs() > 3 && input(3) && m_bTemporal) {
        m_denoiser->convertMotionVectors(m_motionBuffer.data(), m_motionXYBuffer.data(), m_width, m_height);
    }
}

void OptiXDenoiserIop::writeOutputPlane(ImagePlane& plane)
{
    float* outputData = m_outputBuffer.data();
    const int w = m_width;
    const int h = m_height;

    for (int chanNo = 0; chanNo < m_defaultNumberOfChannels; chanNo++) {
        float* outdata = &plane.writable()[plane.chanStride() * chanNo];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                size_t index = (y * w + x) * 4 + chanNo;
                outdata[y * w + x] = outputData[index];
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
        std::copy(m_colorBuffer.begin(), m_colorBuffer.end(), m_outputBuffer.begin());
    }
}

void OptiXDenoiserIop::renderStripe(ImagePlane& plane)
{
    if (aborted() || cancelled()) return;

    if (!validateCUDA()) {
        error("CUDA not available or no CUDA devices found");
        return;
    }

    const Box imageFormat = info().format();
    m_width = imageFormat.w();
    m_height = imageFormat.h();

    try {
        // Allocate buffers
        allocateBuffers();

        // Read input data
        readInputPlanes();

        // Setup denoiser data
        OptiXDenoiserCore::Data data;
        data.width = m_width;
        data.height = m_height;
        data.color = m_colorBuffer.data();
        data.albedo = (node_inputs() > 1 && input(1)) ? m_albedoBuffer.data() : nullptr;
        data.normal = (node_inputs() > 2 && input(2)) ? m_normalBuffer.data() : nullptr;
        data.flow = (node_inputs() > 3 && input(3) && m_bTemporal) ? m_motionXYBuffer.data() : nullptr;
        data.outputs.push_back(m_outputBuffer.data());

        // Initialize denoiser if needed
        if (!m_bInitialized) {
            m_denoiser->init(data, m_tileWidth, m_tileHeight, m_bTemporal);
            m_bInitialized = true;
        } else {
            // Update with new data
            m_denoiser->update(data);
        }

        // Run denoiser multiple times if requested
        for (int i = 0; i < m_numRuns; ++i) {
            if (aborted() || cancelled()) return;
            
            if (i > 0) {
                // For multiple runs, update input with previous output
                std::copy(m_outputBuffer.begin(), m_outputBuffer.end(), m_colorBuffer.begin());
                data.color = m_colorBuffer.data();
                m_denoiser->update(data);
            }
            
            m_denoiser->exec();
            m_denoiser->getResults();
        }

        // Apply blend factor if specified
        if (m_blendFactor > 0.0f && m_blendFactor < 1.0f) {
            for (size_t i = 0; i < m_outputBuffer.size(); ++i) {
                float original = m_colorBuffer[i];
                float denoised = m_outputBuffer[i];
                m_outputBuffer[i] = denoised * (1.0f - m_blendFactor) + original * m_blendFactor;
            }
        }

        // Write output
        writeOutputPlane(plane);
        
        m_frameCounter++;

    } catch (const std::exception& e) {
        handleDenoiserError(e);
        // Write fallback output
        writeOutputPlane(plane);
        return;
    }
}

static Iop *build(Node *node) { return new OptiXDenoiserIop(node); }
const Iop::Description OptiXDenoiserIop::d("OptiXDenoiser", "Filter/OptiXDenoiser", build);
