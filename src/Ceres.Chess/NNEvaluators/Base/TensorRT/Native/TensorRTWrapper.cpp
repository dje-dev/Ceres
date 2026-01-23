/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#include "TensorRTWrapper.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include <mutex>
#include <fstream>
#include <memory>
#include <functional>
#include <cstring>

// Platform-specific includes for file stat
#ifdef _WIN32
  #include <sys/types.h>
  #include <sys/stat.h>
  #define stat _stat
  #ifndef S_ISREG
    #define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
  #endif
#else
  #include <sys/stat.h>
#endif

namespace
{
  // Simple logger implementation for TensorRT
  class TRTLogger : public nvinfer1::ILogger
  {
  public:
    void log(Severity severity, const char* msg) noexcept override
    {
      if (severity <= Severity::kWARNING)
      {
        const char* severityStr = "";
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: severityStr = "INTERNAL_ERROR"; break;
        case Severity::kERROR:          severityStr = "ERROR"; break;
        case Severity::kWARNING:        severityStr = "WARNING"; break;
        default: break;
        }
        fprintf(stderr, "[TensorRT %s] %s\n", severityStr, msg);
      }
    }
  };

  std::mutex g_mutex;
  TRTLogger g_logger;
  nvinfer1::IRuntime* g_runtime = nullptr;
  std::string g_lastError;
  bool g_initialized = false;

  void SetError(const std::string& error)
  {
    g_lastError = error;
  }

  // Get element size in bytes for a TensorRT data type
  size_t GetElementSize(nvinfer1::DataType dtype)
  {
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT8:  return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL:  return 1;
    case nvinfer1::DataType::kUINT8: return 1;
    case nvinfer1::DataType::kFP8:   return 1;
    case nvinfer1::DataType::kBF16:  return 2;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT4:  return 1; // Packed, but treat as 1
    default: return 2; // Default to FP16
    }
  }

  // Engine wrapper struct
  struct EngineContext
  {
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    int32_t batchSize = 0;
    int32_t deviceId = 0;  // GPU device ID

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<int64_t> inputSizes;  // Total elements per tensor (with batch)
    std::vector<int64_t> outputSizes;
    std::vector<size_t> inputElemSizes;  // Bytes per element for each input
    std::vector<size_t> outputElemSizes; // Bytes per element for each output

    // Pre-allocated GPU buffers for host inference
    std::vector<void*> gpuBuffers;
    int64_t totalInputElements = 0;
    int64_t totalOutputElements = 0;

    // CUDA streams (2 for double-buffering) and graph support
    cudaStream_t streams[2] = { nullptr, nullptr };
    cudaStream_t stream = nullptr;  // Alias to streams[0] for compatibility
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    bool useCudaGraphs = false;
    bool graphCaptured = false;

    // Per-stream CUDA graphs for pipelined inference (2 streams)
    // These are used when external GPU buffers are provided (pipelined mode)
    cudaGraph_t streamGraphs[2] = { nullptr, nullptr };
    cudaGraphExec_t streamGraphExecs[2] = { nullptr, nullptr };
    bool streamGraphsCaptured[2] = { false, false };

    ~EngineContext()
    {
      cudaSetDevice(deviceId);  // Ensure correct device for cleanup
      if (graphExec) cudaGraphExecDestroy(graphExec);
      if (graph) cudaGraphDestroy(graph);
      // Cleanup per-stream graphs
      for (int i = 0; i < 2; ++i)
      {
        if (streamGraphExecs[i]) cudaGraphExecDestroy(streamGraphExecs[i]);
        if (streamGraphs[i]) cudaGraphDestroy(streamGraphs[i]);
      }
      if (streams[0]) cudaStreamDestroy(streams[0]);
      if (streams[1]) cudaStreamDestroy(streams[1]);
      for (void* buf : gpuBuffers)
      {
        if (buf) cudaFree(buf);
      }
      if (context) delete context;
      if (engine) delete engine;
    }
  };
}

extern "C"
{

  TRT_API int32_t TRT_Init()
  {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_initialized)
    {
      SetError("TensorRT already initialized");
      return -1;
    }

    g_runtime = nvinfer1::createInferRuntime(g_logger);
    if (!g_runtime)
    {
      SetError("Failed to create TensorRT runtime");
      return -2;
    }

    g_initialized = true;
    g_lastError.clear();
    return 0;
  }

TRT_API int32_t TRT_Shutdown()
{
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_initialized)
    {
        SetError("TensorRT not initialized");
        return -1;
    }

    if (g_runtime)
    {
        delete g_runtime;
        g_runtime = nullptr;
    }

    g_initialized = false;
    g_lastError.clear();
    return 0;
}

TRT_API int32_t TRT_GetVersion()
{
  return NV_TENSORRT_VERSION;
}

TRT_API const char* TRT_GetLastError()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_lastError.empty() ? nullptr : g_lastError.c_str();
}

TRT_API void TRT_InitBuildOptions(TRT_BuildOptions* options)
{
  if (!options) return;
  options->builderOptimizationLevel = 3;
  options->tilingOptimizationLevel = -1;  // Use TensorRT default
  options->useSpinWait = 1;
  options->useCudaGraphs = 0;
  options->useFP16 = 1;
  options->useBF16 = 0;
  options->useFP8 = 0;
  options->useBest = 0;
  options->minBatchSize = 0;  // 0 = use batchSize parameter
  options->optBatchSize = 0;
  options->maxBatchSize = 0;
  options->fp32PostAttentionNorm = 0;
}

  // Helper: print colored console message
  static void PrintColored(const char* color, const char* message)
  {
    fprintf(stderr, "%s%s\033[0m\n", color, message);
  }

  static void PrintYellow(const char* message)
  {
    PrintColored("\033[33m", message);
  }

  static void PrintGreen(const char* message)
  {
    PrintColored("\033[32m", message);
  }

  static void PrintBlue(const char* message)
  {
    PrintColored("\033[34m", message);
  }

  // Helper: check if layer name is a post-attention normalization layer (ln1)
static bool IsPostAttentionNormLayer(const char* layerName)
{
  std::string name(layerName);
  return name.find("ln1") != std::string::npos;
}

TRT_API TRT_EngineHandle TRT_LoadONNXWithOptions(const char* onnxPath, int32_t batchSize,
  const TRT_BuildOptions* options)
{
  std::lock_guard<std::mutex> lock(g_mutex);

  if (!g_initialized)
  {
    SetError("TensorRT not initialized");
    return nullptr;
  }

  // Use defaults if options is null
  TRT_BuildOptions defaultOptions;
  TRT_InitBuildOptions(&defaultOptions);
  const TRT_BuildOptions* opts = options ? options : &defaultOptions;

  // Create builder
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
  if (!builder)
  {
    SetError("Failed to create builder");
    return nullptr;
  }

  // Create network with explicit batch
  const uint32_t explicitBatch = 1U << static_cast<uint32_t>(
    nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
    builder->createNetworkV2(explicitBatch));
  if (!network)
  {
    SetError("Failed to create network");
    return nullptr;
  }

  // Create ONNX parser
  auto parser = std::unique_ptr<nvonnxparser::IParser>(
    nvonnxparser::createParser(*network, g_logger));
  if (!parser)
  {
    SetError("Failed to create ONNX parser");
    return nullptr;
  }

  // Parse ONNX file
  if (!parser->parseFromFile(onnxPath, static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
  {
    std::string errors;
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
      errors += parser->getError(i)->desc();
      errors += "\n";
    }
    SetError("Failed to parse ONNX: " + errors);
    return nullptr;
  }

  // Create builder config
  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config)
  {
    SetError("Failed to create builder config");
    return nullptr;
  }

  // Apply build options
  config->setBuilderOptimizationLevel(opts->builderOptimizationLevel);

  if (opts->tilingOptimizationLevel >= 0)
  {
    config->setTilingOptimizationLevel(
      static_cast<nvinfer1::TilingOptimizationLevel>(opts->tilingOptimizationLevel));
  }

  // Precision flags
  if (opts->useBest)
  {
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
  }
  if (opts->useFP16)
  {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  if (opts->useBF16)
  {
    config->setFlag(nvinfer1::BuilderFlag::kBF16);
  }
  if (opts->useFP8)
  {
    config->setFlag(nvinfer1::BuilderFlag::kFP8);
  }

  // Enable detailed profiling for layer precision inspection
  config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

  // Force FP32 precision for post-attention normalization (ln1) layers if requested
  if (opts->fp32PostAttentionNorm)
  {
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    int layersMarked = 0;
    int totalLayers = network->getNbLayers();
    for (int32_t i = 0; i < totalLayers; ++i)
    {
      auto* layer = network->getLayer(i);
      const char* layerName = layer->getName();
      if (IsPostAttentionNormLayer(layerName))
      {
        layer->setPrecision(nvinfer1::DataType::kFLOAT);
        for (int32_t j = 0; j < layer->getNbOutputs(); ++j)
        {
          layer->setOutputType(j, nvinfer1::DataType::kFLOAT);
        }
        layersMarked++;
      }
    }
    fprintf(stderr, "[TensorRT] Marked %d/%d layers as FP32 for post-attention norm (ln1)\n", layersMarked, totalLayers);
  }

  // Determine batch sizes for optimization profile
  int32_t minBatch = (opts->minBatchSize > 0) ? opts->minBatchSize : batchSize;
  int32_t optBatch = (opts->optBatchSize > 0) ? opts->optBatchSize : batchSize;
  int32_t maxBatch = (opts->maxBatchSize > 0) ? opts->maxBatchSize : batchSize;

  // Set optimization profile for dynamic batch
  auto profile = builder->createOptimizationProfile();
  for (int32_t i = 0; i < network->getNbInputs(); ++i)
  {
    auto input = network->getInput(i);
    auto dims = input->getDimensions();

    // Replace dynamic dimension (usually first dim) with batch sizes
    nvinfer1::Dims minDims = dims, optDims = dims, maxDims = dims;
    if (dims.d[0] == -1)
    {
      minDims.d[0] = minBatch;
      optDims.d[0] = optBatch;
      maxDims.d[0] = maxBatch;
    }

    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, minDims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, optDims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, maxDims);
  }
  config->addOptimizationProfile(profile);

  // Build engine
  auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(
    builder->buildSerializedNetwork(*network, *config));
  if (!serializedEngine)
  {
    SetError("Failed to build engine");
    return nullptr;
  }

  // Deserialize engine
  nvinfer1::ICudaEngine* engine = g_runtime->deserializeCudaEngine(
    serializedEngine->data(), serializedEngine->size());
  if (!engine)
  {
    SetError("Failed to deserialize engine");
    return nullptr;
  }

  // Create execution context
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();
  if (!context)
  {
    delete engine;
    SetError("Failed to create execution context");
    return nullptr;
  }

  // Apply execution options
  if (opts->useSpinWait)
  {
    context->setEnqueueEmitsProfile(false);
    context->setPersistentCacheLimit(0);  // Minimize latency
  }
  
  // Create engine wrapper
  auto* ec = new EngineContext();
  ec->engine = engine;
  ec->context = context;
  ec->batchSize = batchSize;
  ec->useCudaGraphs = opts->useCudaGraphs != 0;

  // Create CUDA stream
  cudaStreamCreate(&ec->streams[0]);
  cudaStreamCreate(&ec->streams[1]);
  ec->stream = ec->streams[0];  // Alias for compatibility

  // Collect input/output info
  int32_t nbIO = engine->getNbIOTensors();
  for (int32_t i = 0; i < nbIO; ++i)
  {
    const char* name = engine->getIOTensorName(i);
    auto mode = engine->getTensorIOMode(name);
    auto dims = engine->getTensorShape(name);
    auto dtype = engine->getTensorDataType(name);
    size_t elemSize = GetElementSize(dtype);

    // Replace -1 with batch size for size calculation
    int64_t size = 1;
    for (int32_t d = 0; d < dims.nbDims; ++d)
    {
      int64_t dimVal = (dims.d[d] == -1) ? batchSize : dims.d[d];
      size *= dimVal;
    }

    if (mode == nvinfer1::TensorIOMode::kINPUT)
    {
      ec->inputNames.push_back(name);
      ec->inputSizes.push_back(size);
      ec->inputElemSizes.push_back(elemSize);
      ec->totalInputElements += size;
    }
    else
    {
      ec->outputNames.push_back(name);
      ec->outputSizes.push_back(size);
      ec->outputElemSizes.push_back(elemSize);
      ec->totalOutputElements += size;
    }
  }

  // Set input shapes on context (required for dynamic shapes)
  for (size_t i = 0; i < ec->inputNames.size(); ++i)
  {
    const char* name = ec->inputNames[i].c_str();
    auto dims = engine->getTensorShape(name);
    if (dims.d[0] == -1)
    {
      dims.d[0] = batchSize;
    }
    context->setInputShape(name, dims);
  }

  // Pre-allocate GPU buffers for host inference (using actual tensor element sizes)
  for (size_t i = 0; i < ec->inputSizes.size(); ++i)
  {
    void* buf = nullptr;
    cudaMalloc(&buf, ec->inputSizes[i] * ec->inputElemSizes[i]);
    ec->gpuBuffers.push_back(buf);
  }
  for (size_t i = 0; i < ec->outputSizes.size(); ++i)
  {
    void* buf = nullptr;
    cudaMalloc(&buf, ec->outputSizes[i] * ec->outputElemSizes[i]);
    ec->gpuBuffers.push_back(buf);
  }

  g_lastError.clear();
  return ec;
}

TRT_API TRT_EngineHandle TRT_LoadONNX(const char* onnxPath, int32_t batchSize)
{
  return TRT_LoadONNXWithOptions(onnxPath, batchSize, nullptr);
}

TRT_API TRT_EngineHandle TRT_LoadONNXOnDevice(const char* onnxPath, int32_t batchSize,
  const TRT_BuildOptions* options, int32_t deviceId)
{
  // Handle deviceId=-1 as "use current device"
  if (deviceId < 0)
  {
    cudaGetDevice(&deviceId);
  }
  else
  {
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to set CUDA device " + std::to_string(deviceId));
      return nullptr;
    }
  }

  TRT_EngineHandle handle = TRT_LoadONNXWithOptions(onnxPath, batchSize, options);
  if (handle)
  {
    static_cast<EngineContext*>(handle)->deviceId = deviceId;
  }
  return handle;
}

// Helper: check if file exists
static bool FileExists(const char* path)
{
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

// Helper: get base filename without path and extension
static std::string GetBaseName(const char* path)
{
    std::string p(path);
    size_t lastSlash = p.find_last_of("/\\");
    if (lastSlash != std::string::npos)
    {
        p = p.substr(lastSlash + 1);
    }
    size_t lastDot = p.find_last_of('.');
    if (lastDot != std::string::npos)
    {
        p = p.substr(0, lastDot);
    }
    return p;
}

// Helper: compute hash of build options
static uint64_t HashBuildOptions(const TRT_BuildOptions* opts)
{
  uint64_t hash = 0;
  hash ^= std::hash<int32_t>{}(opts->builderOptimizationLevel) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int32_t>{}(opts->tilingOptimizationLevel) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int32_t>{}(opts->useSpinWait) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int32_t>{}(opts->useCudaGraphs) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int32_t>{}(opts->useFP16) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int32_t>{}(opts->useBF16) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int32_t>{}(opts->useFP8) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int32_t>{}(opts->useBest) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int32_t>{}(opts->fp32PostAttentionNorm) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  return hash;
}

// Helper: get GPU identifier string (compute capability + SM count)
static std::string GetGPUIdentifier(int deviceId)
{
  int currentDevice = 0;
  cudaGetDevice(&currentDevice);

  if (deviceId < 0)
  {
    deviceId = currentDevice;
  }

  cudaDeviceProp props;
  cudaError_t err = cudaGetDeviceProperties(&props, deviceId);
  if (err != cudaSuccess)
  {
    return "unknown";
  }

  // Format: sm{major}{minor}_{smCount}sm
  // e.g., sm90_132sm for H100 with 132 SMs
  char buffer[64];
  snprintf(buffer, sizeof(buffer), "sm%d%d_%dsm",
    props.major, props.minor, props.multiProcessorCount);
  return std::string(buffer);
}

// Helper: initialize EngineContext from deserialized engine
static EngineContext* InitializeEngineContext(nvinfer1::ICudaEngine* engine, int32_t batchSize,
  bool useCudaGraphs, bool useSpinWait, int32_t deviceId)
{
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();
  if (!context)
  {
    delete engine;
    return nullptr;
  }

  // Apply execution options
  if (useSpinWait)
  {
    context->setEnqueueEmitsProfile(false);
    context->setPersistentCacheLimit(0);  // Minimize latency
  }

  auto* ec = new EngineContext();
  ec->engine = engine;
  ec->context = context;
  ec->batchSize = batchSize;
  ec->deviceId = deviceId;
  ec->useCudaGraphs = useCudaGraphs;

  cudaStreamCreate(&ec->streams[0]);
  cudaStreamCreate(&ec->streams[1]);
  ec->stream = ec->streams[0];

  // Collect input/output info
  int32_t nbIO = engine->getNbIOTensors();
  for (int32_t i = 0; i < nbIO; ++i)
  {
    const char* name = engine->getIOTensorName(i);
    auto mode = engine->getTensorIOMode(name);
    auto dims = engine->getTensorShape(name);
    auto dtype = engine->getTensorDataType(name);
    size_t elemSize = GetElementSize(dtype);

    int64_t size = 1;
    for (int32_t d = 0; d < dims.nbDims; ++d)
    {
      int64_t dimVal = (dims.d[d] == -1) ? batchSize : dims.d[d];
      size *= dimVal;
    }

    if (mode == nvinfer1::TensorIOMode::kINPUT)
    {
      ec->inputNames.push_back(name);
      ec->inputSizes.push_back(size);
      ec->inputElemSizes.push_back(elemSize);
      ec->totalInputElements += size;
    }
    else
    {
      ec->outputNames.push_back(name);
      ec->outputSizes.push_back(size);
      ec->outputElemSizes.push_back(elemSize);
      ec->totalOutputElements += size;
    }
  }

  // Set input shapes on context
  for (size_t i = 0; i < ec->inputNames.size(); ++i)
  {
    const char* name = ec->inputNames[i].c_str();
    auto dims = engine->getTensorShape(name);
    if (dims.d[0] == -1)
    {
      dims.d[0] = batchSize;
    }
    context->setInputShape(name, dims);
  }

  // Pre-allocate GPU buffers (using actual tensor element sizes)
  for (size_t i = 0; i < ec->inputSizes.size(); ++i)
  {
    void* buf = nullptr;
    cudaMalloc(&buf, ec->inputSizes[i] * ec->inputElemSizes[i]);
    ec->gpuBuffers.push_back(buf);
  }
  for (size_t i = 0; i < ec->outputSizes.size(); ++i)
  {
    void* buf = nullptr;
    cudaMalloc(&buf, ec->outputSizes[i] * ec->outputElemSizes[i]);
    ec->gpuBuffers.push_back(buf);
  }

  return ec;
}

TRT_API TRT_EngineHandle TRT_LoadEngineFile(const char* enginePath, int32_t batchSize, int32_t deviceId)
{
  std::lock_guard<std::mutex> lock(g_mutex);

  if (!g_initialized)
  {
    SetError("TensorRT not initialized");
    return nullptr;
  }

  if (deviceId >= 0)
  {
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess)
    {
      SetError("Failed to set CUDA device " + std::to_string(deviceId));
      return nullptr;
    }
  }
  else
  {
    cudaGetDevice(&deviceId);
  }

  // Read engine file
  std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
  if (!file.is_open())
  {
    SetError("Failed to open engine file: " + std::string(enginePath));
    return nullptr;
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size))
  {
    SetError("Failed to read engine file");
    return nullptr;
  }
  file.close();

  // Deserialize engine
  nvinfer1::ICudaEngine* engine = g_runtime->deserializeCudaEngine(buffer.data(), buffer.size());
  if (!engine)
  {
    SetError("Failed to deserialize engine");
    return nullptr;
  }

  EngineContext* ec = InitializeEngineContext(engine, batchSize, false, false, deviceId);
  if (!ec)
  {
    SetError("Failed to create execution context");
    return nullptr;
  }

  g_lastError.clear();
  return ec;
}

// Internal helper: Load engine file with runtime options (not exposed in public API)
static TRT_EngineHandle LoadEngineFileWithOptions(const char* enginePath, int32_t batchSize, 
  int32_t deviceId, bool useCudaGraphs, bool useSpinWait)
{
  std::lock_guard<std::mutex> lock(g_mutex);

  if (!g_initialized)
  {
    SetError("TensorRT not initialized");
    return nullptr;
  }

  if (deviceId >= 0)
  {
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess)
    {
      SetError("Failed to set CUDA device " + std::to_string(deviceId));
      return nullptr;
    }
  }
  else
  {
    cudaGetDevice(&deviceId);
  }

  // Read engine file
  std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
  if (!file.is_open())
  {
    SetError("Failed to open engine file: " + std::string(enginePath));
    return nullptr;
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size))
  {
    SetError("Failed to read engine file");
    return nullptr;
  }
  file.close();

  // Deserialize engine
  nvinfer1::ICudaEngine* engine = g_runtime->deserializeCudaEngine(buffer.data(), buffer.size());
  if (!engine)
  {
    SetError("Failed to deserialize engine");
    return nullptr;
  }

  EngineContext* ec = InitializeEngineContext(engine, batchSize, useCudaGraphs, useSpinWait, deviceId);
  if (!ec)
  {
    SetError("Failed to create execution context");
    return nullptr;
  }

  g_lastError.clear();
  return ec;
}

TRT_API int32_t TRT_SaveEngine(TRT_EngineHandle handle, const char* enginePath)
{
  if (!handle)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Invalid handle");
    return -1;
  }

  auto* ec = static_cast<EngineContext*>(handle);

  nvinfer1::IHostMemory* serialized = ec->engine->serialize();
  if (!serialized)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Failed to serialize engine");
    return -2;
  }

  std::ofstream file(enginePath, std::ios::binary);
  if (!file.is_open())
  {
    delete serialized;
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Failed to open file for writing: " + std::string(enginePath));
    return -3;
  }

  file.write(static_cast<const char*>(serialized->data()), serialized->size());
  file.close();
  delete serialized;

  return 0;
}

TRT_API char* TRT_GenerateCacheFilename(const char* onnxPath, int32_t batchSize,
  const TRT_BuildOptions* options)
{
  return TRT_GenerateCacheFilenameForDevice(onnxPath, batchSize, options, -1);
}

TRT_API char* TRT_GenerateCacheFilenameForDevice(const char* onnxPath, int32_t batchSize,
  const TRT_BuildOptions* options, int32_t deviceId)
{
  TRT_BuildOptions defaultOpts;
  TRT_InitBuildOptions(&defaultOpts);
  const TRT_BuildOptions* opts = options ? options : &defaultOpts;

  std::string basename = GetBaseName(onnxPath);
  std::string gpuId = GetGPUIdentifier(deviceId);

  int32_t minBatch = (opts->minBatchSize > 0) ? opts->minBatchSize : batchSize;
  int32_t optBatch = (opts->optBatchSize > 0) ? opts->optBatchSize : batchSize;
  int32_t maxBatch = (opts->maxBatchSize > 0) ? opts->maxBatchSize : batchSize;

  uint64_t hash = HashBuildOptions(opts);
  int32_t trtVersion = NV_TENSORRT_VERSION;

  char buffer[512];
  snprintf(buffer, sizeof(buffer), "%s_b%d-%d-%d_%s_trt%d_%016llx.engine",
    basename.c_str(), minBatch, optBatch, maxBatch, gpuId.c_str(), trtVersion,
    static_cast<unsigned long long>(hash));

  return strdup(buffer);
}

TRT_API void TRT_FreeString(char* str)
{
  if (str) free(str);
}

// Helper: format build summary for console output
static std::string FormatBuildSummary(const char* onnxPath, const TRT_BuildOptions* opts, int32_t batchSize, bool isBuilding)
{
  std::string basename = GetBaseName(onnxPath);

  int32_t minBatch = (opts->minBatchSize > 0) ? opts->minBatchSize : batchSize;
  int32_t optBatch = (opts->optBatchSize > 0) ? opts->optBatchSize : batchSize;
  int32_t maxBatch = (opts->maxBatchSize > 0) ? opts->maxBatchSize : batchSize;

  std::string precision;
  if (opts->useFP8) precision = "FP8";
  else if (opts->useBF16) precision = "BF16";
  else if (opts->useFP16) precision = "FP16";
  else precision = "FP32";

  char buffer[512];
  if (minBatch == maxBatch)
  {
    snprintf(buffer, sizeof(buffer), "[TensorRT] %s %s: batch=%d, %s, opt=%d",
      isBuilding ? "Building" : "Loading",
      basename.c_str(), minBatch, precision.c_str(), opts->builderOptimizationLevel);
  }
  else
  {
    snprintf(buffer, sizeof(buffer), "[TensorRT] %s %s: batch=[%d-%d-%d], %s, opt=%d",
      isBuilding ? "Building" : "Loading",
      basename.c_str(), minBatch, optBatch, maxBatch, precision.c_str(), opts->builderOptimizationLevel);
  }
  return std::string(buffer);
}

TRT_API TRT_EngineHandle TRT_LoadONNXCached(const char* onnxPath, int32_t batchSize,
  const TRT_BuildOptions* options, int32_t deviceId,
  const char* cacheDir, int32_t forceRebuild,
  int32_t* outWasCached)
{
  if (outWasCached) *outWasCached = 0;

  TRT_BuildOptions defaultOpts;
  TRT_InitBuildOptions(&defaultOpts);
  const TRT_BuildOptions* opts = options ? options : &defaultOpts;

  // If no cache dir, just build normally
  if (!cacheDir || cacheDir[0] == '\0')
  {
    std::string summary = FormatBuildSummary(onnxPath, opts, batchSize, true);
    PrintYellow(summary.c_str());
    return TRT_LoadONNXOnDevice(onnxPath, batchSize, options, deviceId);
  }

  // Generate cache filename (with GPU identifier)
  char* cacheFilename = TRT_GenerateCacheFilenameForDevice(onnxPath, batchSize, options, deviceId);
  std::string cachePath = std::string(cacheDir) + "/" + cacheFilename;
  TRT_FreeString(cacheFilename);

  // Check if cached engine exists and is valid
  if (!forceRebuild && FileExists(cachePath.c_str()))
  {
    // Check if ONNX file is newer than cache (invalidate cache if so)
    struct stat onnxStat, cacheStat;
    bool cacheValid = true;
    if (stat(onnxPath, &onnxStat) == 0 && stat(cachePath.c_str(), &cacheStat) == 0)
    {
      if (onnxStat.st_mtime > cacheStat.st_mtime)
      {
        cacheValid = false;  // ONNX is newer, rebuild
      }
    }

    if (cacheValid)
    {
      // Use internal helper that applies runtime options (useCudaGraphs, useSpinWait)
      TRT_EngineHandle handle = LoadEngineFileWithOptions(cachePath.c_str(), batchSize, deviceId,
        opts->useCudaGraphs != 0, opts->useSpinWait != 0);
      if (handle)
      {
        std::string summary = FormatBuildSummary(onnxPath, opts, batchSize, false);
        PrintGreen(summary.c_str());
        if (outWasCached) *outWasCached = 1;
        return handle;
      }
      // If loading failed, fall through to rebuild
    }
  }

  // Build from ONNX
  std::string summary = FormatBuildSummary(onnxPath, opts, batchSize, true);
  PrintYellow(summary.c_str());

  TRT_EngineHandle handle = TRT_LoadONNXOnDevice(onnxPath, batchSize, options, deviceId);
  if (!handle)
  {
    return nullptr;
  }

  // Save to cache
  int32_t saveResult = TRT_SaveEngine(handle, cachePath.c_str());
  if (saveResult != 0)
  {
    // Log warning but don't fail - we still have a working engine
    fprintf(stderr, "[TensorRT WARNING] Failed to save engine to cache: %s\n", cachePath.c_str());
  }

  return handle;
}

TRT_API int32_t TRT_GetDevice()
{
  int device = 0;
  cudaGetDevice(&device);
  return device;
}

TRT_API int32_t TRT_SetDevice(int32_t deviceId)
{
  cudaError_t err = cudaSetDevice(deviceId);
  return (err == cudaSuccess) ? 0 : -1;
}

TRT_API int32_t TRT_IsIntegratedGPU(int32_t deviceId)
{
  cudaDeviceProp props;
  cudaError_t err = cudaGetDeviceProperties(&props, deviceId);
  if (err != cudaSuccess)
  {
    return -1;
  }
  // integrated == 1 means unified memory (like Tegra/Jetson/GB10)
  // integrated == 0 means discrete GPU with separate VRAM
  return props.integrated ? 1 : 0;
}

TRT_API void TRT_FreeEngine(TRT_EngineHandle handle)
{
  if (handle)
  {
    delete static_cast<EngineContext*>(handle);
  }
}

TRT_API int32_t TRT_GetNumLayers(TRT_EngineHandle handle)
{
  if (!handle) return -1;
  auto* ec = static_cast<EngineContext*>(handle);
  auto inspector = ec->engine->createEngineInspector();
  if (!inspector) return -1;
  // Get layer count by iterating (no direct API)
  int32_t count = 0;
  while (true)
  {
    const char* info = inspector->getLayerInformation(count, nvinfer1::LayerInformationFormat::kJSON);
    if (!info || info[0] == '\0') break;
    count++;
  }
  delete inspector;
  return count;
}


TRT_API char* TRT_GetEngineLayerInfo(TRT_EngineHandle handle, int32_t layerIndex)
{
  if (!handle) return nullptr;
  auto* ec = static_cast<EngineContext*>(handle);

  auto inspector = ec->engine->createEngineInspector();
  if (!inspector)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Failed to create engine inspector");
    return nullptr;
  }

  // Attach context for more detailed info
  inspector->setExecutionContext(ec->context);

  const char* info = inspector->getLayerInformation(layerIndex, nvinfer1::LayerInformationFormat::kJSON);
  char* result = nullptr;
  if (info && info[0] != '\0')
  {
    result = strdup(info);
  }

  delete inspector;
  return result;
}

TRT_API char* TRT_GetEngineSummary(TRT_EngineHandle handle)
{
  if (!handle) return nullptr;
  auto* ec = static_cast<EngineContext*>(handle);

  auto inspector = ec->engine->createEngineInspector();
  if (!inspector)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Failed to create engine inspector");
    return nullptr;
  }

  inspector->setExecutionContext(ec->context);

  const char* info = inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
  char* result = nullptr;
  if (info && info[0] != '\0')
  {
    result = strdup(info);
  }

  delete inspector;
  return result;
}

TRT_API int32_t TRT_GetNumInputs(TRT_EngineHandle handle)
{
  if (!handle) return 0;
  return static_cast<int32_t>(static_cast<EngineContext*>(handle)->inputNames.size());
}

TRT_API int32_t TRT_GetNumOutputs(TRT_EngineHandle handle)
{
  if (!handle) return 0;
  return static_cast<int32_t>(static_cast<EngineContext*>(handle)->outputNames.size());
}

TRT_API const char* TRT_GetInputName(TRT_EngineHandle handle, int32_t index)
{
    if (!handle) return nullptr;
    auto* ec = static_cast<EngineContext*>(handle);
    if (index < 0 || index >= static_cast<int32_t>(ec->inputNames.size())) return nullptr;
    return ec->inputNames[index].c_str();
}

TRT_API const char* TRT_GetOutputName(TRT_EngineHandle handle, int32_t index)
{
  if (!handle) return nullptr;
  auto* ec = static_cast<EngineContext*>(handle);
  if (index < 0 || index >= static_cast<int32_t>(ec->outputNames.size())) return nullptr;
  return ec->outputNames[index].c_str();
}

TRT_API int64_t TRT_GetInputSize(TRT_EngineHandle handle, int32_t index)
{
  if (!handle) return 0;
  auto* ec = static_cast<EngineContext*>(handle);
  if (index < 0 || index >= static_cast<int32_t>(ec->inputSizes.size())) return 0;
  return ec->inputSizes[index];
}

TRT_API int64_t TRT_GetOutputSize(TRT_EngineHandle handle, int32_t index)
{
  if (!handle) return 0;
  auto* ec = static_cast<EngineContext*>(handle);
  if (index < 0 || index >= static_cast<int32_t>(ec->outputSizes.size())) return 0;
  return ec->outputSizes[index];
}

TRT_API int32_t TRT_GetInputElementSize(TRT_EngineHandle handle, int32_t index)
{
  if (!handle) return 0;
  auto* ec = static_cast<EngineContext*>(handle);
  if (index < 0 || index >= static_cast<int32_t>(ec->inputElemSizes.size())) return 0;
  return static_cast<int32_t>(ec->inputElemSizes[index]);
}

TRT_API int32_t TRT_GetOutputElementSize(TRT_EngineHandle handle, int32_t index)
{
  if (!handle) return 0;
  auto* ec = static_cast<EngineContext*>(handle);
  if (index < 0 || index >= static_cast<int32_t>(ec->outputElemSizes.size())) return 0;
  return static_cast<int32_t>(ec->outputElemSizes[index]);
}

TRT_API int32_t TRT_GetOutputTensorInfo(TRT_EngineHandle handle,
  TRT_OutputTensorInfo* outputInfo,
  int32_t maxOutputs)
{
  if (!handle || !outputInfo)
  {
    return -1;
  }

  auto* ec = static_cast<EngineContext*>(handle);
  int32_t numOutputs = static_cast<int32_t>(ec->outputNames.size());
  int32_t count = (numOutputs < maxOutputs) ? numOutputs : maxOutputs;

  int64_t offset = 0;
  for (int32_t i = 0; i < count; ++i)
  {
    outputInfo[i].name = ec->outputNames[i].c_str();
    outputInfo[i].offset = offset;
    outputInfo[i].size = ec->outputSizes[i];
    offset += ec->outputSizes[i];
  }

  return numOutputs;
}

TRT_API int32_t TRT_Infer(TRT_EngineHandle handle,
  void** inputPtrs, int32_t numInputs,
  void** outputPtrs, int32_t numOutputs)
{
  if (!handle)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Invalid handle");
    return -1;
  }

  auto* ec = static_cast<EngineContext*>(handle);

  // Set tensor addresses
  for (int32_t i = 0; i < numInputs && i < static_cast<int32_t>(ec->inputNames.size()); ++i)
  {
    ec->context->setTensorAddress(ec->inputNames[i].c_str(), inputPtrs[i]);
  }
  for (int32_t i = 0; i < numOutputs && i < static_cast<int32_t>(ec->outputNames.size()); ++i)
  {
    ec->context->setTensorAddress(ec->outputNames[i].c_str(), outputPtrs[i]);
  }

  // Execute on dedicated stream
  bool success = ec->context->enqueueV3(ec->stream);
  if (!success)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Inference failed");
    return -2;
  }

  return 0;
}

TRT_API int32_t TRT_InferHost(TRT_EngineHandle handle,
  void* inputData, int64_t inputSize,
  void* outputData, int64_t outputSize)
{
  if (!handle)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Invalid handle");
    return -1;
  }

  auto* ec = static_cast<EngineContext*>(handle);
  cudaSetDevice(ec->deviceId);  // Ensure correct device
  char* inPtr = static_cast<char*>(inputData);
  char* outPtr = static_cast<char*>(outputData);

  // Verify sizes
  if (inputSize != ec->totalInputElements)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Input size mismatch: expected " + std::to_string(ec->totalInputElements) +
      ", got " + std::to_string(inputSize));
    return -2;
  }
  if (outputSize != ec->totalOutputElements)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Output size mismatch: expected " + std::to_string(ec->totalOutputElements) +
      ", got " + std::to_string(outputSize));
    return -3;
  }

  // Set tensor addresses (must be done before graph capture or execution)
  for (size_t i = 0; i < ec->inputNames.size(); ++i)
  {
    ec->context->setTensorAddress(ec->inputNames[i].c_str(), ec->gpuBuffers[i]);
  }
  size_t outputBufferStart = ec->inputSizes.size();
  for (size_t i = 0; i < ec->outputNames.size(); ++i)
  {
    ec->context->setTensorAddress(ec->outputNames[i].c_str(), ec->gpuBuffers[outputBufferStart + i]);
  }

  if (ec->useCudaGraphs && ec->graphCaptured)
  {
    // Use captured graph for execution
    // Copy inputs to GPU (using actual element sizes)
    int64_t byteOffset = 0;
    for (size_t i = 0; i < ec->inputSizes.size(); ++i)
    {
      size_t bytes = ec->inputSizes[i] * ec->inputElemSizes[i];
      cudaMemcpyAsync(ec->gpuBuffers[i], inPtr + byteOffset,
        bytes, cudaMemcpyHostToDevice, ec->stream);
      byteOffset += bytes;
    }

    // Launch graph
    cudaGraphLaunch(ec->graphExec, ec->stream);

    // Copy outputs from GPU (using actual element sizes)
    byteOffset = 0;
    for (size_t i = 0; i < ec->outputSizes.size(); ++i)
    {
      size_t bytes = ec->outputSizes[i] * ec->outputElemSizes[i];
      cudaMemcpyAsync(outPtr + byteOffset, ec->gpuBuffers[outputBufferStart + i],
        bytes, cudaMemcpyDeviceToHost, ec->stream);
      byteOffset += bytes;
    }

    cudaStreamSynchronize(ec->stream);
  }
  else if (ec->useCudaGraphs && !ec->graphCaptured)
  {
    // First run: capture the graph
    // Copy inputs to GPU (outside graph capture, using actual element sizes)
    int64_t byteOffset = 0;
    for (size_t i = 0; i < ec->inputSizes.size(); ++i)
    {
      size_t bytes = ec->inputSizes[i] * ec->inputElemSizes[i];
      cudaMemcpy(ec->gpuBuffers[i], inPtr + byteOffset,
        bytes, cudaMemcpyHostToDevice);
      byteOffset += bytes;
    }

    // Capture graph
    cudaStreamBeginCapture(ec->stream, cudaStreamCaptureModeGlobal);

    bool success = ec->context->enqueueV3(ec->stream);

    cudaStreamEndCapture(ec->stream, &ec->graph);

    if (!success || !ec->graph)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to capture CUDA graph");
      return -4;
    }

    cudaGraphInstantiate(&ec->graphExec, ec->graph, 0);
    ec->graphCaptured = true;

    // Now execute the graph
    cudaGraphLaunch(ec->graphExec, ec->stream);
    cudaStreamSynchronize(ec->stream);

    // Copy outputs from GPU (using actual element sizes)
    byteOffset = 0;
    for (size_t i = 0; i < ec->outputSizes.size(); ++i)
    {
      size_t bytes = ec->outputSizes[i] * ec->outputElemSizes[i];
      cudaMemcpy(outPtr + byteOffset, ec->gpuBuffers[outputBufferStart + i],
        bytes, cudaMemcpyDeviceToHost);
      byteOffset += bytes;
    }
  }
  else
  {
    // Standard execution (no CUDA graphs)
    // Copy inputs to GPU using async on our stream (using actual element sizes)
    int64_t byteOffset = 0;
    for (size_t i = 0; i < ec->inputSizes.size(); ++i)
    {
      size_t bytes = ec->inputSizes[i] * ec->inputElemSizes[i];
      cudaMemcpyAsync(ec->gpuBuffers[i], inPtr + byteOffset,
        bytes, cudaMemcpyHostToDevice, ec->stream);
      byteOffset += bytes;
    }

    // Execute
    bool success = ec->context->enqueueV3(ec->stream);
    if (!success)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Inference failed");
      return -4;
    }

    // Copy outputs from GPU (using actual element sizes)
    byteOffset = 0;
    for (size_t i = 0; i < ec->outputSizes.size(); ++i)
    {
      size_t bytes = ec->outputSizes[i] * ec->outputElemSizes[i];
      cudaMemcpyAsync(outPtr + byteOffset, ec->gpuBuffers[outputBufferStart + i],
        bytes, cudaMemcpyDeviceToHost, ec->stream);
      byteOffset += bytes;
    }

    // Synchronize our stream
    cudaStreamSynchronize(ec->stream);
  }

  return 0;
}

TRT_API void* TRT_AllocGPU(int64_t bytes)
{
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, bytes);
  if (err != cudaSuccess)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    return nullptr;
  }
  return ptr;
}

TRT_API void TRT_FreeGPU(void* ptr)
{
  if (ptr) cudaFree(ptr);
}

TRT_API int32_t TRT_CopyToGPU(void* dst, const void* src, int64_t bytes)
{
  cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("cudaMemcpy H2D failed: " + std::string(cudaGetErrorString(err)));
    return -1;
  }
  return 0;
}

TRT_API int32_t TRT_CopyFromGPU(void* dst, const void* src, int64_t bytes)
{
  cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("cudaMemcpy D2H failed: " + std::string(cudaGetErrorString(err)));
    return -1;
  }
  return 0;
}

TRT_API int32_t TRT_Synchronize()
{
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("cudaDeviceSynchronize failed: " + std::string(cudaGetErrorString(err)));
    return -1;
  }
  return 0;
}


TRT_API int32_t TRT_SynchronizeDevice(int32_t deviceId)
{
  int prevDevice = 0;
  cudaGetDevice(&prevDevice);
  cudaError_t err = cudaSetDevice(deviceId);
  if (err != cudaSuccess)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("cudaSetDevice failed: " + std::string(cudaGetErrorString(err)));
    return -1;
  }
  err = cudaDeviceSynchronize();
  cudaSetDevice(prevDevice);  // Restore previous device
  if (err != cudaSuccess)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("cudaDeviceSynchronize failed: " + std::string(cudaGetErrorString(err)));
    return -2;
  }
  return 0;
}


TRT_API void* TRT_AllocPinned(int64_t bytes)
{
  void* ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
  if (err != cudaSuccess)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("cudaHostAlloc failed: " + std::string(cudaGetErrorString(err)));
    return nullptr;
  }
  return ptr;
}

TRT_API void TRT_FreePinned(void* ptr)
{
  if (ptr) cudaFreeHost(ptr);
}

TRT_API int32_t TRT_CopyToGPUAsync(TRT_EngineHandle handle, void* dst, const void* src, int64_t bytes)
{
  if (!handle) return -1;
  auto* ec = static_cast<EngineContext*>(handle);
  cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, ec->stream);
  if (err != cudaSuccess)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("cudaMemcpyAsync H2D failed: " + std::string(cudaGetErrorString(err)));
    return -1;
  }
  return 0;
}

TRT_API int32_t TRT_CopyFromGPUAsync(TRT_EngineHandle handle, void* dst, const void* src, int64_t bytes)
{
  if (!handle) return -1;
  auto* ec = static_cast<EngineContext*>(handle);
  cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, ec->stream);
  if (err != cudaSuccess)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("cudaMemcpyAsync D2H failed: " + std::string(cudaGetErrorString(err)));
    return -1;
  }
  return 0;
}

TRT_API int32_t TRT_SyncStream(TRT_EngineHandle handle)
{
  if (!handle) return -1;
  auto* ec = static_cast<EngineContext*>(handle);
  cudaError_t err = cudaStreamSynchronize(ec->stream);
  if (err != cudaSuccess)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("cudaStreamSynchronize failed: " + std::string(cudaGetErrorString(err)));
    return -1;
  }
  return 0;
}

TRT_API int32_t TRT_InferAsync(TRT_EngineHandle handle, void* gpuInput, void* gpuOutput)
{
  constexpr size_t ELEM_SIZE = 2;  // FP16
  if (!handle) return -1;
  auto* ec = static_cast<EngineContext*>(handle);

  // Set tensor addresses
  if (!ec->inputNames.empty())
  {
    ec->context->setTensorAddress(ec->inputNames[0].c_str(), gpuInput);
  }

  size_t outputOffset = 0;
  for (size_t i = 0; i < ec->outputNames.size(); ++i)
  {
    void* outPtr = static_cast<char*>(gpuOutput) + outputOffset * ELEM_SIZE;
    ec->context->setTensorAddress(ec->outputNames[i].c_str(), outPtr);
    outputOffset += ec->outputSizes[i];
  }

  // Execute without sync
  bool success = ec->context->enqueueV3(ec->stream);
  if (!success)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Inference failed");
    return -2;
  }

  return 0;
}

TRT_API void* TRT_GetInputBuffer(TRT_EngineHandle handle, int32_t index)
{
  if (!handle) return nullptr;
  auto* ec = static_cast<EngineContext*>(handle);
  if (index < 0 || index >= static_cast<int32_t>(ec->inputSizes.size())) return nullptr;
  return ec->gpuBuffers[index];
}

TRT_API void* TRT_GetOutputBuffer(TRT_EngineHandle handle, int32_t index)
{
    if (!handle) return nullptr;
    auto* ec = static_cast<EngineContext*>(handle);
    if (index < 0 || index >= static_cast<int32_t>(ec->outputSizes.size())) return nullptr;
    return ec->gpuBuffers[ec->inputSizes.size() + index];
}

TRT_API int32_t TRT_InferOnStream(TRT_EngineHandle handle, int32_t streamIdx,
  void* gpuInput, void* gpuOutput)
{
  constexpr size_t ELEM_SIZE = 2;  // FP16
  if (!handle || streamIdx < 0 || streamIdx > 1) return -1;
  auto* ec = static_cast<EngineContext*>(handle);

  if (!ec->inputNames.empty())
    ec->context->setTensorAddress(ec->inputNames[0].c_str(), gpuInput);

  size_t outputOffset = 0;
  for (size_t i = 0; i < ec->outputNames.size(); ++i)
  {
    void* outPtr = static_cast<char*>(gpuOutput) + outputOffset * ELEM_SIZE;
    ec->context->setTensorAddress(ec->outputNames[i].c_str(), outPtr);
    outputOffset += ec->outputSizes[i];
  }

  bool success = ec->context->enqueueV3(ec->streams[streamIdx]);
  return success ? 0 : -2;
}

TRT_API int32_t TRT_InferOnStreamWithGraph(TRT_EngineHandle handle, int32_t streamIdx,
  void* gpuInput, void* gpuOutput)
{
  constexpr size_t ELEM_SIZE = 2;  // FP16
  if (!handle || streamIdx < 0 || streamIdx > 1) return -1;
  auto* ec = static_cast<EngineContext*>(handle);

  // Set tensor addresses (must be consistent for graph capture and replay)
  if (!ec->inputNames.empty())
    ec->context->setTensorAddress(ec->inputNames[0].c_str(), gpuInput);

  size_t outputOffset = 0;
  for (size_t i = 0; i < ec->outputNames.size(); ++i)
  {
    void* outPtr = static_cast<char*>(gpuOutput) + outputOffset * ELEM_SIZE;
    ec->context->setTensorAddress(ec->outputNames[i].c_str(), outPtr);
    outputOffset += ec->outputSizes[i];
  }

  if (!ec->useCudaGraphs)
  {
    // CUDA graphs disabled - use direct enqueue
    bool success = ec->context->enqueueV3(ec->streams[streamIdx]);
    return success ? 0 : -2;
  }

  if (ec->streamGraphsCaptured[streamIdx])
  {
    // Graph already captured - launch it
    cudaError_t err = cudaGraphLaunch(ec->streamGraphExecs[streamIdx], ec->streams[streamIdx]);
    if (err != cudaSuccess)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("cudaGraphLaunch failed on stream " + std::to_string(streamIdx) + ": " + cudaGetErrorString(err));
      return -3;
    }
    return 0;
  }

  // First call on this stream - capture the graph
  cudaError_t err = cudaStreamBeginCapture(ec->streams[streamIdx], cudaStreamCaptureModeGlobal);
  if (err != cudaSuccess)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("cudaStreamBeginCapture failed: " + std::string(cudaGetErrorString(err)));
    return -4;
  }

  bool success = ec->context->enqueueV3(ec->streams[streamIdx]);

  err = cudaStreamEndCapture(ec->streams[streamIdx], &ec->streamGraphs[streamIdx]);
  if (err != cudaSuccess || !ec->streamGraphs[streamIdx])
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("cudaStreamEndCapture failed: " + std::string(cudaGetErrorString(err)));
    return -5;
  }

  if (!success)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Inference failed during graph capture");
    cudaGraphDestroy(ec->streamGraphs[streamIdx]);
    ec->streamGraphs[streamIdx] = nullptr;
    return -6;
  }

  err = cudaGraphInstantiate(&ec->streamGraphExecs[streamIdx], ec->streamGraphs[streamIdx], 0);
  if (err != cudaSuccess)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("cudaGraphInstantiate failed: " + std::string(cudaGetErrorString(err)));
    cudaGraphDestroy(ec->streamGraphs[streamIdx]);
    ec->streamGraphs[streamIdx] = nullptr;
    return -7;
  }

  ec->streamGraphsCaptured[streamIdx] = true;

  // Launch the newly instantiated graph
  err = cudaGraphLaunch(ec->streamGraphExecs[streamIdx], ec->streams[streamIdx]);
  if (err != cudaSuccess)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("cudaGraphLaunch failed after capture: " + std::string(cudaGetErrorString(err)));
    return -8;
  }

  return 0;
}

TRT_API int32_t TRT_InferOnStreamDynamic(TRT_EngineHandle handle, int32_t streamIdx,
  void* gpuInput, void* gpuOutput, int32_t actualBatchSize)
{
  constexpr size_t ELEM_SIZE = 2;  // FP16
  if (!handle || streamIdx < 0 || streamIdx > 1) return -1;
  auto* ec = static_cast<EngineContext*>(handle);

  // Set dynamic input shapes for actualBatchSize
  for (size_t i = 0; i < ec->inputNames.size(); ++i)
  {
    const char* name = ec->inputNames[i].c_str();
    auto dims = ec->engine->getTensorShape(name);
    if (dims.d[0] == -1)
    {
      dims.d[0] = actualBatchSize;
    }
    if (!ec->context->setInputShape(name, dims))
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to set input shape for " + std::string(name) + " to batch " + std::to_string(actualBatchSize));
      return -3;
    }
  }

  if (!ec->inputNames.empty())
    ec->context->setTensorAddress(ec->inputNames[0].c_str(), gpuInput);

  // Compute per-position output sizes and set tensor addresses
  size_t outputOffset = 0;
  for (size_t i = 0; i < ec->outputNames.size(); ++i)
  {
    void* outPtr = static_cast<char*>(gpuOutput) + outputOffset * ELEM_SIZE;
    ec->context->setTensorAddress(ec->outputNames[i].c_str(), outPtr);
    // Use per-position size * actualBatchSize for the offset calculation
    int64_t perPosSize = ec->outputSizes[i] / ec->batchSize;
    outputOffset += perPosSize * actualBatchSize;
  }

  bool success = ec->context->enqueueV3(ec->streams[streamIdx]);
  return success ? 0 : -2;
}

TRT_API int32_t TRT_CopyToGPUOnStream(TRT_EngineHandle handle, int32_t streamIdx, void* dst, const void* src, int64_t bytes)
{
  if (!handle || streamIdx < 0 || streamIdx > 1) return -1;
  auto* ec = static_cast<EngineContext*>(handle);
  cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, ec->streams[streamIdx]);
  return 0;
}

TRT_API int32_t TRT_CopyFromGPUOnStream(TRT_EngineHandle handle, int32_t streamIdx, void* dst, const void* src, int64_t bytes)
{
  if (!handle || streamIdx < 0 || streamIdx > 1) return -1;
  auto* ec = static_cast<EngineContext*>(handle);
  cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, ec->streams[streamIdx]);
  return 0;
}

TRT_API int32_t TRT_SyncStreamIdx(TRT_EngineHandle handle, int32_t streamIdx)
{
  if (!handle || streamIdx < 0 || streamIdx > 1) return -1;
  auto* ec = static_cast<EngineContext*>(handle);
  cudaStreamSynchronize(ec->streams[streamIdx]);
  return 0;
}

// =========================================================================
// Dynamic Batch Size Inference (for range-mode engines)
// =========================================================================

// Helper: Compute per-position element counts from stored sizes
static void GetPerPositionSizes(EngineContext* ec, int32_t engineBatchSize,
                                 std::vector<int64_t>& inputPerPos,
                                 std::vector<int64_t>& outputPerPos)
{
  inputPerPos.resize(ec->inputSizes.size());
  outputPerPos.resize(ec->outputSizes.size());

  for (size_t i = 0; i < ec->inputSizes.size(); ++i)
  {
    inputPerPos[i] = ec->inputSizes[i] / engineBatchSize;
  }
  for (size_t i = 0; i < ec->outputSizes.size(); ++i)
  {
    outputPerPos[i] = ec->outputSizes[i] / engineBatchSize;
  }
}

TRT_API int32_t TRT_InferHostDynamic(TRT_EngineHandle handle,
  void* inputData, int64_t inputSize,
  void* outputData, int64_t outputSize,
  int32_t actualBatchSize)
{
  if (!handle)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Invalid handle");
    return -1;
  }

  auto* ec = static_cast<EngineContext*>(handle);
  cudaSetDevice(ec->deviceId);

  // Compute per-position sizes
  std::vector<int64_t> inputPerPos, outputPerPos;
  GetPerPositionSizes(ec, ec->batchSize, inputPerPos, outputPerPos);

  // Calculate expected sizes for actualBatchSize
  int64_t expectedInputSize = 0;
  int64_t expectedOutputSize = 0;
  for (size_t i = 0; i < inputPerPos.size(); ++i)
  {
    expectedInputSize += inputPerPos[i] * actualBatchSize;
  }
  for (size_t i = 0; i < outputPerPos.size(); ++i)
  {
    expectedOutputSize += outputPerPos[i] * actualBatchSize;
  }

  // Verify input size
  if (inputSize != expectedInputSize)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Input size mismatch for dynamic batch: expected " + std::to_string(expectedInputSize) +
      " for batch " + std::to_string(actualBatchSize) + ", got " + std::to_string(inputSize));
    return -2;
  }

  // Verify output size
  if (outputSize != expectedOutputSize)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Output size mismatch for dynamic batch: expected " + std::to_string(expectedOutputSize) +
      " for batch " + std::to_string(actualBatchSize) + ", got " + std::to_string(outputSize));
    return -3;
  }

  char* inPtr = static_cast<char*>(inputData);
  char* outPtr = static_cast<char*>(outputData);

  // Set dynamic input shapes for actualBatchSize
  for (size_t i = 0; i < ec->inputNames.size(); ++i)
  {
    const char* name = ec->inputNames[i].c_str();
    auto dims = ec->engine->getTensorShape(name);
    if (dims.d[0] == -1)
    {
      dims.d[0] = actualBatchSize;
    }
    if (!ec->context->setInputShape(name, dims))
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to set input shape for " + std::string(name) + " to batch " + std::to_string(actualBatchSize));
      return -4;
    }
  }

  // Set tensor addresses to pre-allocated GPU buffers
  for (size_t i = 0; i < ec->inputNames.size(); ++i)
  {
    ec->context->setTensorAddress(ec->inputNames[i].c_str(), ec->gpuBuffers[i]);
  }
  size_t outputBufferStart = ec->inputSizes.size();
  for (size_t i = 0; i < ec->outputNames.size(); ++i)
  {
    ec->context->setTensorAddress(ec->outputNames[i].c_str(), ec->gpuBuffers[outputBufferStart + i]);
  }

  // Copy inputs to GPU (only the actual data needed)
  int64_t byteOffset = 0;
  for (size_t i = 0; i < ec->inputSizes.size(); ++i)
  {
    int64_t actualElements = inputPerPos[i] * actualBatchSize;
    size_t bytes = actualElements * ec->inputElemSizes[i];
    cudaMemcpyAsync(ec->gpuBuffers[i], inPtr + byteOffset,
      bytes, cudaMemcpyHostToDevice, ec->stream);
    byteOffset += bytes;
  }

  // Execute inference
  bool success = ec->context->enqueueV3(ec->stream);
  if (!success)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    SetError("Dynamic inference failed for batch " + std::to_string(actualBatchSize));
    return -5;
  }

  // Copy outputs from GPU (only the actual data)
  byteOffset = 0;
  for (size_t i = 0; i < ec->outputSizes.size(); ++i)
  {
    int64_t actualElements = outputPerPos[i] * actualBatchSize;
    size_t bytes = actualElements * ec->outputElemSizes[i];
    cudaMemcpyAsync(outPtr + byteOffset, ec->gpuBuffers[outputBufferStart + i],
      bytes, cudaMemcpyDeviceToHost, ec->stream);
    byteOffset += bytes;
  }

  // Synchronize
  cudaStreamSynchronize(ec->stream);

  return 0;
}

TRT_API int32_t TRT_InferHostBytesDynamic(TRT_EngineHandle handle,
  void* inputData, int64_t inputSize,
  void* outputData, int64_t outputSize,
  int32_t actualBatchSize)
{
  // For byte inputs, the logic is identical - the input element sizes are already
  // stored as 1 byte in ec->inputElemSizes for INT8/UINT8 inputs.
  // The TRT_InferHostDynamic function handles this correctly via inputElemSizes.
  return TRT_InferHostDynamic(handle, inputData, inputSize, outputData, outputSize, actualBatchSize);
}

TRT_API int32_t TRT_GetEngineBatchSize(TRT_EngineHandle handle)
{
  if (!handle) return -1;
  auto* ec = static_cast<EngineContext*>(handle);
  return ec->batchSize;
}

TRT_API int32_t TRT_UsesCudaGraphs(TRT_EngineHandle handle)
{
  if (!handle) return -1;
  auto* ec = static_cast<EngineContext*>(handle);
  return ec->useCudaGraphs ? 1 : 0;
}

TRT_API int64_t TRT_GetInputElementsPerPosition(TRT_EngineHandle handle)
{
  if (!handle) return -1;
  auto* ec = static_cast<EngineContext*>(handle);
  return ec->totalInputElements / ec->batchSize;
}

TRT_API int64_t TRT_GetOutputElementsPerPosition(TRT_EngineHandle handle)
{
  if (!handle) return -1;
  auto* ec = static_cast<EngineContext*>(handle);
  return ec->totalOutputElements / ec->batchSize;
}

}
