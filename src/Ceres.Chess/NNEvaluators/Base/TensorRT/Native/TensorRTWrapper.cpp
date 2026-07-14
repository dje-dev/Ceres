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
#include <cstdlib>
#include <cstdint>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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

// CUDA 12+ uses simplified 3-arg cudaGraphInstantiate; older versions use 5-arg form
#if CUDART_VERSION >= 12000
#define CUDA_GRAPH_INSTANTIATE(exec, graph) cudaGraphInstantiate(exec, graph, 0)
#else
#define CUDA_GRAPH_INSTANTIATE(exec, graph) cudaGraphInstantiate(exec, graph, nullptr, nullptr, 0)
#endif

// Alignment for output tensor addresses in fp16 elements.
// 128 elements * 2 bytes = 256 bytes, satisfying CUDA/TensorRT alignment requirements.
// Without this, small tensors (e.g. 8-element PUNIM outputs) cause misaligned addresses
// that fail cudaStreamBeginCapture during CUDA graph capture.
constexpr size_t OUTPUT_TENSOR_ALIGN_ELEMS = 128;

inline int64_t AlignUp(int64_t value, int64_t alignment)
{
  return ((value + alignment - 1) / alignment) * alignment;
}

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

  // Stream sync wait policy. Default 3="auto" decides per-sync from the per-GPU batch size
  // (the count being synced): SPIN when it is small (<= SYNC_SPIN_MAX_PERGPU, where the GPU
  // burst is short so the block-sync OS wakeup latency is a large fraction of the wait) and
  // BLOCK otherwise (long GPU work => wakeup negligible, and blocking frees the core). This
  // is measured to be the crossover: spin helps fast/small per-GPU batches (e.g. multi-GPU
  // splits and fast nets) while block is free and CPU-cheap on large batches / big nets.
  // Override via env CERES_TRT_SYNC = auto|spin|block|driver.
  //   0 = "driver" (cudaStreamSynchronize), 1 = "spin" (event busy-poll, low latency,
  //   burns a core), 2 = "block" (blocking-sync event, low CPU), 3 = "auto" (the rule).
  int g_syncMode = 3;
  constexpr int SYNC_SPIN_MAX_PERGPU = 64;

  // Thread-local cache for current CUDA device to avoid redundant cudaSetDevice() calls.
  // Value of -1 means no device has been set on this thread yet.
  // This significantly reduces multi-GPU overhead when threads repeatedly call inference
  // on the same device, as cudaSetDevice() has non-trivial driver overhead.
  static thread_local int32_t tls_currentDevice = -1;

  // Ensures the CUDA device is set to the specified deviceId, but only calls
  // cudaSetDevice() if the device has changed from the last call on this thread.
  // Returns true on success, false on failure.
  inline bool EnsureDevice(int32_t deviceId)
  {
    if (tls_currentDevice != deviceId)
    {
      cudaError_t err = cudaSetDevice(deviceId);
      if (err != cudaSuccess)
      {
        return false;
      }
      tls_currentDevice = deviceId;
    }
    return true;
  }

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

  // Ref-counted shared engine for multi-profile support.
  // When multiple EngineContexts share a single ICudaEngine,
  // this struct ensures the engine is only deleted when the last context is freed.
  struct SharedEngine
  {
    nvinfer1::ICudaEngine* engine;
    std::atomic<int> refCount;
    SharedEngine(nvinfer1::ICudaEngine* e, int count) : engine(e), refCount(count) {}
  };

  // Engine wrapper struct
  struct EngineContext
  {
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    // Second execution context for stream 2 concurrent compute.
    // TensorRT execution contexts are NOT thread-safe across streams, so concurrent
    // graph replay on streams 0 and 2 requires separate contexts with independent
    // tensor address bindings.
    nvinfer1::IExecutionContext* context2 = nullptr;
    SharedEngine* sharedOwner = nullptr;  // Non-null when sharing engine via multi-profile
    int32_t batchSize = 0;
    int32_t deviceId = 0;  // GPU device ID
    int32_t profileIndex = 0;  // Optimization profile this context is bound to (0 if single-profile)
    bool useSpinWait = false;  // Retained so a shared-engine clone can reproduce this context's options

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

    // CUDA streams (3: stream 0 = compute A, stream 1 = transfers, stream 2 = compute B)
    cudaStream_t streams[3] = { nullptr, nullptr, nullptr };
    cudaStream_t stream = nullptr;  // Alias to streams[0] for compatibility
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    bool useCudaGraphs = false;
    bool graphCaptured = false;

    // Per-stream CUDA graphs for pipelined inference (3 streams)
    // These are used when external GPU buffers are provided (pipelined mode)
    cudaGraph_t streamGraphs[3] = { nullptr, nullptr, nullptr };
    cudaGraphExec_t streamGraphExecs[3] = { nullptr, nullptr, nullptr };
    bool streamGraphsCaptured[3] = { false, false, false };
    // Track captured buffer addresses to detect when re-capture is needed
    void* streamCapturedInput[3] = { nullptr, nullptr, nullptr };
    void* streamCapturedOutput[3] = { nullptr, nullptr, nullptr };

    // Reusable per-stream events for non-default sync modes (lazily created, see g_syncMode).
    cudaEvent_t syncEvents[3] = { nullptr, nullptr, nullptr };

    ~EngineContext()
    {
      cudaSetDevice(deviceId);  // Ensure correct device for cleanup
      if (graphExec) cudaGraphExecDestroy(graphExec);
      if (graph) cudaGraphDestroy(graph);
      // Cleanup per-stream graphs
      for (int i = 0; i < 3; ++i)
      {
        if (streamGraphExecs[i]) cudaGraphExecDestroy(streamGraphExecs[i]);
        if (streamGraphs[i]) cudaGraphDestroy(streamGraphs[i]);
        if (syncEvents[i]) cudaEventDestroy(syncEvents[i]);
      }
      for (int i = 0; i < 3; ++i)
      {
        if (streams[i]) cudaStreamDestroy(streams[i]);
      }
      for (void* buf : gpuBuffers)
      {
        if (buf) cudaFree(buf);
      }
      if (context2) delete context2;
      if (context) delete context;
      if (sharedOwner)
      {
        // Ref-counted: only delete engine when last context releases it
        if (sharedOwner->refCount.fetch_sub(1) == 1)
        {
          delete sharedOwner->engine;
          delete sharedOwner;
        }
      }
      else
      {
        // Sole owner (backward compat for single-profile code paths)
        if (engine) delete engine;
      }
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
    options->fp32PostAttentionNormStrict = 0;
    options->fp32SmolgenNorm = 0;
    options->fp32Softmax = 0;
    options->fp32AllNorms = 0;
    options->refittable = 0;
  }

} // end extern "C" (temporarily, for C++ helper functions below)

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

// Helper: check if a layer name matches any normalization naming convention.
static bool HasNormName(const char* layerName)
{
  std::string name(layerName);
  return name.find("rms_norm") != std::string::npos
    || name.find("ln1") != std::string::npos
    || name.find("/ln2/") != std::string::npos
    || name.find("qkvLN") != std::string::npos
    || name.find("embedding_norm") != std::string::npos
    || name.find("LayerNorm") != std::string::npos
    || name.find("layer_norm") != std::string::npos
    || name.find("rmsnorm") != std::string::npos;
}

// Check if a layer type performs actual computation (not just data/shape).
static bool IsComputeLayerType(nvinfer1::LayerType type)
{
  return type == nvinfer1::LayerType::kELEMENTWISE
    || type == nvinfer1::LayerType::kREDUCE
    || type == nvinfer1::LayerType::kUNARY
    || type == nvinfer1::LayerType::kNORMALIZATION;
}

// ---------------------------------------------------------------------------
// Structural detection of decomposed RMSNorm chains (naming-independent).
//
// CeresTrain's ONNX exporter (save_model.py, _RMSNormPrimitive) lowers every
// RMSNorm to primitive ops so the graph stays fused-op-free at opset < 23, and
// casts the mean-of-squares reduction to fp32 to avoid fp16 overflow. Exported
// via the TorchDynamo path, the resulting nodes carry GENERIC aten-op names
// (node_pow_*, node_mean*, node_Sqrt_*, node_rsqrt_*, node_mul_*, node__to_copy*)
// rather than module-scoped names -- so the substring tests in HasNormName no
// longer see any norm token, the old name-based detection matches 0 layers, and
// the reduction is left in fp16 (inf/NaN -> garbage accuracy). See the "Marked
// 0/N layers as FP32" symptom.
//
// Rather than chase node names, detect each RMSNorm by its op signature, which is
// stable regardless of naming:
//
//   X --(Cast fp32)--> Pow(2) --> ReduceMean --> Add(eps) --> Sqrt --> Reciprocal
//        --> Mul(x32 * rsqrt) --(Cast fp16)--> Mul(* weight) --> ...
//
// The ReduceMean (kREDUCE) whose producer is the elementwise square is the anchor;
// every RMSNorm has exactly one and the transformer body has no other reductions.
// We collect the fp32-critical math span (square .. Mul(*rsqrt), stopping at the
// fp16 downcast Cast) and classify the norm by walking back through the fp32 Cast
// to the tensor being normalized:
//   producer is a residual/skip Add  -> Residual (post-attention / FFN norm, marked broad)
//   producer is a Reshape (per-head) -> QKV      (small dim, left fp16 unless scoped)
//   producer is another elementwise  -> Smolgen  (attention ln1)
//   anything else (e.g. embedding)   -> Other
enum class DecomposedNormKind { Residual, QKV, Smolgen, Other };

struct DecomposedNormChain
{
  DecomposedNormKind kind;
  std::vector<int32_t> layers;   // compute layers to force to fp32
};

// Layers that are pure fp16<->fp32 conversions inserted around the norm math. The
// backward walk skips these to reach the real producer; the forward collection stops
// at them (the fp16 downcast marks the end of the fp32 span).
static bool IsCastLikeLayer(const nvinfer1::ILayer* layer)
{
  if (layer->getType() == nvinfer1::LayerType::kCAST)
    return true;
  std::string n(layer->getName());
  return n.find("to_copy") != std::string::npos
    || n.find("castHelper") != std::string::npos
    || n.find("ONNXTRT_") != std::string::npos;
}

// True if a name carries any token specific to the decomposed-RMSNorm op cluster.
// Used only to corroborate a reduction as a norm when the producer's layer type is
// unexpected; matching is deliberately narrow to avoid false positives.
static bool ContainsNormOpToken(const char* name)
{
  std::string n(name);
  return n.find("rms") != std::string::npos
    || n.find("norm") != std::string::npos
    || n.find("pow") != std::string::npos
    || n.find("mean") != std::string::npos
    || n.find("Sqrt") != std::string::npos
    || n.find("rsqrt") != std::string::npos;
}

static std::vector<DecomposedNormChain> AnalyzeDecomposedRMSNorms(
  const nvinfer1::INetworkDefinition* network, int32_t totalLayers)
{
  // producer: output tensor name -> producing layer index
  std::unordered_map<std::string, int32_t> tensorProducer;
  // consumers: input tensor name  -> consuming layer indices
  std::unordered_map<std::string, std::vector<int32_t>> tensorConsumers;
  for (int32_t i = 0; i < totalLayers; ++i)
  {
    auto* layer = network->getLayer(i);
    for (int32_t j = 0; j < layer->getNbOutputs(); ++j)
    {
      auto* t = layer->getOutput(j);
      if (t && t->getName())
        tensorProducer[t->getName()] = i;
    }
    for (int32_t j = 0; j < layer->getNbInputs(); ++j)
    {
      auto* t = layer->getInput(j);
      if (t && t->getName())
        tensorConsumers[t->getName()].push_back(i);
    }
  }

  auto producerOf = [&](const nvinfer1::ILayer* layer, int32_t inputIdx) -> int32_t
  {
    if (inputIdx >= layer->getNbInputs())
      return -1;
    auto* t = layer->getInput(inputIdx);
    if (!t || !t->getName())
      return -1;
    auto it = tensorProducer.find(t->getName());
    return it == tensorProducer.end() ? -1 : it->second;
  };
  // Follow input(0) backwards, skipping cast-like layers, to the real producer index.
  auto realProducerOf = [&](int32_t layerIdx) -> int32_t
  {
    int32_t p = producerOf(network->getLayer(layerIdx), 0);
    for (int guard = 0; p >= 0 && guard < 6; ++guard)
    {
      if (!IsCastLikeLayer(network->getLayer(p)))
        break;
      p = producerOf(network->getLayer(p), 0);
    }
    return p;
  };

  std::vector<DecomposedNormChain> chains;

  for (int32_t i = 0; i < totalLayers; ++i)
  {
    auto* reduce = network->getLayer(i);
    if (reduce->getType() != nvinfer1::LayerType::kREDUCE)
      continue;

    // Confirm this reduction is the mean-of-squares of an RMSNorm: its (cast-skipped)
    // producer must be the elementwise square. This excludes any unrelated mean/sum.
    int32_t sqIdx = realProducerOf(i);
    if (sqIdx < 0)
      continue;
    auto* square = network->getLayer(sqIdx);
    bool squareIsElementwise = (square->getType() == nvinfer1::LayerType::kELEMENTWISE);
    if (!squareIsElementwise
      && !ContainsNormOpToken(square->getName())
      && !ContainsNormOpToken(reduce->getName()))
      continue;

    // Classify by the tensor being normalized: the square's (cast-skipped) input.
    DecomposedNormKind kind = DecomposedNormKind::Other;
    int32_t srcIdx = realProducerOf(sqIdx);
    if (srcIdx >= 0)
    {
      auto* src = network->getLayer(srcIdx);
      std::string sn(src->getName());
      const bool srcIsAdd = (src->getType() == nvinfer1::LayerType::kELEMENTWISE)
        && (sn.find("add") != std::string::npos
          || sn.find("Add") != std::string::npos
          || sn.find("skip") != std::string::npos);
      const bool srcIsShuffle = (src->getType() == nvinfer1::LayerType::kSHUFFLE)
        || sn.find("view") != std::string::npos
        || sn.find("reshape") != std::string::npos
        || sn.find("Reshape") != std::string::npos;
      if (srcIsAdd)
        kind = DecomposedNormKind::Residual;         // residual/skip stream norm
      else if (srcIsShuffle)
        kind = DecomposedNormKind::QKV;              // per-head q/k/v norm
      else if (src->getType() == nvinfer1::LayerType::kELEMENTWISE)
        kind = DecomposedNormKind::Smolgen;          // smolgen attention ln1
      // else: embedding norm (fed by activation) etc. -> Other
    }

    // Collect the fp32-critical math span: the square + reduce, then forward through the
    // norm's elementwise/unary chain (Add eps -> Sqrt -> Reciprocal -> Mul(*rsqrt)).
    // Restrict to compute layer types and stop at the fp16 downcast Cast (and never
    // cross a MatMul/Shuffle) so we cannot run into the next block's residual add.
    std::unordered_set<int32_t> chainSet;
    chainSet.insert(sqIdx);
    chainSet.insert(i);
    std::vector<int32_t> frontier = { i };
    for (int depth = 0; depth < 8 && !frontier.empty(); ++depth)
    {
      std::vector<int32_t> next;
      for (int32_t li : frontier)
      {
        auto* L = network->getLayer(li);
        for (int32_t j = 0; j < L->getNbOutputs(); ++j)
        {
          auto* t = L->getOutput(j);
          if (!t || !t->getName())
            continue;
          auto cit = tensorConsumers.find(t->getName());
          if (cit == tensorConsumers.end())
            continue;
          for (int32_t ci : cit->second)
          {
            if (chainSet.count(ci))
              continue;
            auto ct = network->getLayer(ci)->getType();
            if (ct == nvinfer1::LayerType::kELEMENTWISE
              || ct == nvinfer1::LayerType::kUNARY
              || ct == nvinfer1::LayerType::kREDUCE)
            {
              chainSet.insert(ci);
              next.push_back(ci);
            }
            // kCAST / kSHUFFLE / kMATRIX_MULTIPLY / ...: boundary, do not cross.
          }
        }
      }
      frontier.swap(next);
    }

    DecomposedNormChain chain;
    chain.kind = kind;
    chain.layers.assign(chainSet.begin(), chainSet.end());
    chains.push_back(std::move(chain));
  }

  return chains;
}

// Build set of residual-stream normalization layer indices.
// For native kNORMALIZATION layers: mark only those fed by a residual Add.
// For decomposed norms: find entry points on the residual stream, propagate
// through the chain, but only mark compute layers (skip kCONSTANT, etc.).
static std::unordered_set<int32_t> FindResidualStreamNormLayers(
  const nvinfer1::INetworkDefinition* network, int32_t totalLayers)
{
  // Build producer map: tensor name -> producing layer index
  std::unordered_map<std::string, int32_t> tensorProducer;
  for (int32_t i = 0; i < totalLayers; ++i)
  {
    auto* layer = network->getLayer(i);
    for (int32_t j = 0; j < layer->getNbOutputs(); ++j)
    {
      auto* tensor = layer->getOutput(j);
      if (tensor && tensor->getName())
      {
        tensorProducer[tensor->getName()] = i;
      }
    }
  }

  // Count native kNORMALIZATION layers
  int nativeNormCount = 0;
  for (int32_t i = 0; i < totalLayers; ++i)
  {
    if (network->getLayer(i)->getType() == nvinfer1::LayerType::kNORMALIZATION)
      nativeNormCount++;
  }

  std::unordered_set<int32_t> residualNormLayers;

  if (nativeNormCount > 0)
  {
    // Path A: native kNORMALIZATION layers exist - mark only those on residual stream
    for (int32_t i = 0; i < totalLayers; ++i)
    {
      auto* layer = network->getLayer(i);
      if (layer->getType() != nvinfer1::LayerType::kNORMALIZATION)
        continue;

      auto* inputTensor = layer->getInput(0);
      if (!inputTensor || !inputTensor->getName())
        continue;

      auto it = tensorProducer.find(inputTensor->getName());
      if (it == tensorProducer.end())
        continue;

      auto* producer = network->getLayer(it->second);
      for (int depth = 0; depth < 5; ++depth)
      {
        std::string rpName(producer->getName());
        if (rpName.find("ONNXTRT_") != std::string::npos
          || rpName.find("castHelper") != std::string::npos)
        {
          auto* rpInput = producer->getInput(0);
          if (rpInput && rpInput->getName())
          {
            auto rpIt = tensorProducer.find(rpInput->getName());
            if (rpIt != tensorProducer.end())
            {
              producer = network->getLayer(rpIt->second);
              continue;
            }
          }
        }
        break;
      }

      std::string realName(producer->getName());
      bool isResidualAdd = (producer->getType() == nvinfer1::LayerType::kELEMENTWISE)
        && (realName.find("add") != std::string::npos
          || realName.find("Add") != std::string::npos
          || realName.find("skip") != std::string::npos);

      if (isResidualAdd)
      {
        residualNormLayers.insert(i);
      }
    }
    fprintf(stderr, "[TensorRT] Found %d native kNORMALIZATION layers, %d on residual stream\n",
      nativeNormCount, (int)residualNormLayers.size());
  }
  else
  {
    // Path B: decomposed norms. Prefer structural detection (handles the generic
    // aten-op node names emitted by the TorchDynamo exporter, where HasNormName sees
    // nothing). Only the residual-stream norms are marked here; smolgen norms come
    // from the separate fp32AllNorms(scope=3) pass, and per-head QKV norms are left
    // in fp16 (small reduction dim, no overflow).
    auto decomposed = AnalyzeDecomposedRMSNorms(network, totalLayers);
    if (!decomposed.empty())
    {
      int nRes = 0, nQKV = 0, nSmol = 0, nOther = 0;
      for (const auto& c : decomposed)
      {
        switch (c.kind)
        {
        case DecomposedNormKind::Residual: nRes++;   break;
        case DecomposedNormKind::QKV:      nQKV++;    break;
        case DecomposedNormKind::Smolgen:  nSmol++;   break;
        default:                           nOther++;  break;
        }
      }
      // Safety: if classification found no residual norms at all (structure differs
      // from what we expect) fall back to marking every detected norm chain -- upcasting
      // extra norms only costs a little speed, but missing one corrupts accuracy.
      const bool markAll = (nRes == 0);
      for (const auto& c : decomposed)
      {
        if (markAll || c.kind == DecomposedNormKind::Residual)
        {
          for (int32_t li : c.layers)
            residualNormLayers.insert(li);
        }
      }
      fprintf(stderr, "[TensorRT] Decomposed norms (structural): %d residual + %d QKV + %d smolgen "
        "+ %d other norms; marked %d compute layers on residual stream%s\n",
        nRes, nQKV, nSmol, nOther, (int)residualNormLayers.size(),
        markAll ? " (fallback: all norms, no residual class found)" : "");
      return residualNormLayers;
    }

    // Legacy fallback: name-based detection for nets whose norm nodes carry
    // module-scoped names (rms_norm.../ln1/...) recognized by HasNormName.
    // Pass 1: find entry-point norm layers fed by residual Add
    for (int32_t i = 0; i < totalLayers; ++i)
    {
      auto* layer = network->getLayer(i);
      if (!HasNormName(layer->getName()))
        continue;

      auto* inputTensor = layer->getInput(0);
      if (!inputTensor || !inputTensor->getName())
        continue;

      auto it = tensorProducer.find(inputTensor->getName());
      if (it == tensorProducer.end())
        continue;

      auto* producer = network->getLayer(it->second);
      bool producerIsNorm = HasNormName(producer->getName());
      if (producerIsNorm)
        continue;

      auto* realProducer = producer;
      for (int depth = 0; depth < 5; ++depth)
      {
        std::string rpName(realProducer->getName());
        if (rpName.find("ONNXTRT_") != std::string::npos
          || rpName.find("castHelper") != std::string::npos)
        {
          auto* rpInput = realProducer->getInput(0);
          if (rpInput && rpInput->getName())
          {
            auto rpIt = tensorProducer.find(rpInput->getName());
            if (rpIt != tensorProducer.end())
            {
              realProducer = network->getLayer(rpIt->second);
              continue;
            }
          }
        }
        break;
      }

      std::string realName(realProducer->getName());
      bool isResidualAdd = (realProducer->getType() == nvinfer1::LayerType::kELEMENTWISE)
        && (realName.find("add") != std::string::npos
          || realName.find("Add") != std::string::npos
          || realName.find("skip") != std::string::npos);

      if (isResidualAdd && IsComputeLayerType(layer->getType()))
      {
        residualNormLayers.insert(i);
      }
    }

    int entryCount = (int)residualNormLayers.size();

    // Pass 2: propagate through norm chain, only marking compute layers
    bool changed = true;
    while (changed)
    {
      changed = false;
      for (int32_t i = 0; i < totalLayers; ++i)
      {
        if (residualNormLayers.count(i))
          continue;
        auto* layer = network->getLayer(i);
        if (!HasNormName(layer->getName()))
          continue;
        if (!IsComputeLayerType(layer->getType()))
          continue;

        for (int32_t inp = 0; inp < layer->getNbInputs(); ++inp)
        {
          auto* inputTensor = layer->getInput(inp);
          if (!inputTensor || !inputTensor->getName())
            continue;
          auto it = tensorProducer.find(inputTensor->getName());
          if (it != tensorProducer.end() && residualNormLayers.count(it->second))
          {
            residualNormLayers.insert(i);
            changed = true;
            break;
          }
        }
      }
    }

    fprintf(stderr, "[TensorRT] Decomposed norms: %d entry-point + %d chain = %d compute layers on residual stream\n",
      entryCount, (int)residualNormLayers.size() - entryCount, (int)residualNormLayers.size());
  }

  return residualNormLayers;
}


// Build set of ALL normalization layer indices (not just residual stream).
// Includes Q/K/V per-head norms, smolgen norms, and any other norm chains.
// Norm scope values for fp32AllNorms:
//   1 = all norms (Q/K/V + smolgen + residual + embedding)
//   2 = Q/K/V per-head norms only (entry producer is kSHUFFLE from Reshape)
//   3 = smolgen norms only (entry producer is non-residual kELEMENTWISE)
//   4 = Q/K/V + smolgen (non-residual, excludes residual-stream norms)
static std::unordered_set<int32_t> FindAllNormLayers(
  const nvinfer1::INetworkDefinition* network, int32_t totalLayers, int32_t scope = 1)
{
  // Build producer map: tensor name -> producing layer index
  std::unordered_map<std::string, int32_t> tensorProducer;
  for (int32_t i = 0; i < totalLayers; ++i)
  {
    auto* layer = network->getLayer(i);
    for (int32_t j = 0; j < layer->getNbOutputs(); ++j)
    {
      auto* tensor = layer->getOutput(j);
      if (tensor && tensor->getName())
      {
        tensorProducer[tensor->getName()] = i;
      }
    }
  }

  // Count native kNORMALIZATION layers
  int nativeNormCount = 0;
  for (int32_t i = 0; i < totalLayers; ++i)
  {
    if (network->getLayer(i)->getType() == nvinfer1::LayerType::kNORMALIZATION)
      nativeNormCount++;
  }

  std::unordered_set<int32_t> allNormLayers;

  if (nativeNormCount > 0)
  {
    // Path A: native kNORMALIZATION - mark ALL of them (scope filtering not supported)
    for (int32_t i = 0; i < totalLayers; ++i)
    {
      if (network->getLayer(i)->getType() == nvinfer1::LayerType::kNORMALIZATION)
      {
        allNormLayers.insert(i);
      }
    }
    fprintf(stderr, "[TensorRT] AllNorms(scope=%d): %d native kNORMALIZATION layers\n",
      scope, (int)allNormLayers.size());
  }
  else
  {
    // Path B: decomposed norms. Prefer structural detection (naming-independent) so the
    // generic aten-op node names from the TorchDynamo exporter are handled. Select chains
    // by scope, matching the legacy classification semantics below:
    //   1 = all, 2 = QKV, 3 = smolgen, 4 = QKV+smolgen (residual is only ever in scope 1).
    auto decomposed = AnalyzeDecomposedRMSNorms(network, totalLayers);
    if (!decomposed.empty())
    {
      int cQKV = 0, cSmol = 0, cRes = 0, cOther = 0;
      for (const auto& c : decomposed)
      {
        bool include = false;
        switch (c.kind)
        {
        case DecomposedNormKind::QKV:      cQKV++;   include = (scope == 1 || scope == 2 || scope == 4); break;
        case DecomposedNormKind::Smolgen:  cSmol++;  include = (scope == 1 || scope == 3 || scope == 4); break;
        case DecomposedNormKind::Residual: cRes++;   include = (scope == 1); break;
        default:                           cOther++; include = (scope == 1 || scope == 4); break;
        }
        if (include)
        {
          for (int32_t li : c.layers)
            allNormLayers.insert(li);
        }
      }
      fprintf(stderr, "[TensorRT] AllNorms(scope=%d) structural: %d QKV + %d smolgen + %d residual "
        "+ %d other norms -> %d layers\n",
        scope, cQKV, cSmol, cRes, cOther, (int)allNormLayers.size());
      return allNormLayers;
    }

    // Legacy fallback: name/index-based detection for module-scoped rms_norm names.
    // Pre-scan: find max sequential ONNX norm index to distinguish from TRT suffixes.
    // ONNX norms are numbered 0,1,2,...,N-1 sequentially. TRT decomposition creates
    // additional layers with large suffix numbers (typically > N).
    // Collect all numbers from rms_norm layer names, find the longest sequential run from 0.
    std::vector<int> allNormNumbers;
    for (int32_t i = 0; i < totalLayers; ++i)
    {
      auto* layer = network->getLayer(i);
      std::string name(layer->getName());
      auto pos = name.find("rms_norm");
      if (pos == std::string::npos)
        continue;
      size_t numStart = pos + 8;
      if (numStart < name.size() && name[numStart] == '_')
        numStart++;
      std::string numStr;
      while (numStart < name.size() && name[numStart] >= '0' && name[numStart] <= '9')
        numStr += name[numStart++];
      if (!numStr.empty())
        allNormNumbers.push_back(std::stoi(numStr));
      else
        allNormNumbers.push_back(0); // bare "rms_norm" = ONNX index 0
    }
    // Find max ONNX index: largest N such that all of 0..N appear in the set.
    std::unordered_set<int> normNumberSet(allNormNumbers.begin(), allNormNumbers.end());
    int maxOnnxNormIdx = -1;
    for (int n = 0; normNumberSet.count(n); ++n)
      maxOnnxNormIdx = n;

    int countQKV = 0, countSmolgen = 0, countResidual = 0, countOther = 0;

    for (int32_t i = 0; i < totalLayers; ++i)
    {
      auto* layer = network->getLayer(i);
      if (!HasNormName(layer->getName()))
        continue;
      if (!IsComputeLayerType(layer->getType()))
        continue;

      // Check if this is an entry point (producer is not a norm layer)
      auto* inputTensor = layer->getInput(0);
      if (!inputTensor || !inputTensor->getName())
      {
        countOther++;
        if (scope == 1) allNormLayers.insert(i);
        continue;
      }
      auto it = tensorProducer.find(inputTensor->getName());
      if (it == tensorProducer.end())
      {
        countOther++;
        if (scope == 1) allNormLayers.insert(i);
        continue;
      }
      auto* producer = network->getLayer(it->second);
      if (HasNormName(producer->getName()))
        continue; // Not an entry point

      // Classify entry point by producer type and norm index
      auto prodType = producer->getType();
      std::string prodName(producer->getName());
      std::string entryName(layer->getName());

      bool isResidualAdd = (prodType == nvinfer1::LayerType::kELEMENTWISE)
        && (prodName.find("add") != std::string::npos
          || prodName.find("Add") != std::string::npos
          || prodName.find("skip") != std::string::npos);

      // Extract ONNX norm index from TRT layer name.
      // TRT names decomposed layers: "node_rms_norm[_N][_TRT_SUFFIX]"
      // ONNX indices are sequential 0..maxOnnxNormIdx; anything beyond is a TRT suffix.
      // Pattern: norms 0=embedding, then per block: 3 QKV + 2 smolgen + 2 residual
      int normIdx = -1;
      auto pos = entryName.find("rms_norm");
      if (pos != std::string::npos)
      {
        size_t numStart = pos + 8; // skip "rms_norm"
        if (numStart < entryName.size() && entryName[numStart] == '_')
          numStart++;
        std::string numStr;
        while (numStart < entryName.size() && entryName[numStart] >= '0' && entryName[numStart] <= '9')
          numStr += entryName[numStart++];
        if (!numStr.empty())
        {
          int parsed = std::stoi(numStr);
          normIdx = (parsed <= maxOnnxNormIdx) ? parsed : 0;
        }
        else
        {
          normIdx = 0; // "rms_norm" without number = index 0
        }
      }

      // Classify by ONNX norm index within block structure
      // Block 0: norm_0=embedding, norm_1-3=QKV, norm_4-5=smolgen, norm_6-7=residual
      // Block b (b>=1): norm_{1+7b}-{3+7b}=QKV, norm_{4+7b}-{5+7b}=smolgen, norm_{6+7b}-{7+7b}=residual
      bool isQKV = false;
      bool isSmolgen = false;
      if (normIdx >= 0)
      {
        if (normIdx == 0)
        {
          // Embedding norm - treat as "other"
        }
        else
        {
          int blockOffset = (normIdx - 1) % 7; // 0-6 within block
          isQKV = (blockOffset >= 0 && blockOffset <= 2); // first 3 in block
          isSmolgen = (blockOffset >= 3 && blockOffset <= 4); // next 2 in block
          // blockOffset 5-6 are residual
        }
      }

      bool include = false;
      if (isResidualAdd || (!isQKV && !isSmolgen && normIdx >= 0 && ((normIdx - 1) % 7) >= 5))
      {
        countResidual++;
        include = (scope == 1);
      }
      else if (isQKV)
      {
        countQKV++;
        include = (scope == 1 || scope == 2 || scope == 4);
      }
      else if (isSmolgen)
      {
        countSmolgen++;
        include = (scope == 1 || scope == 3 || scope == 4);
      }
      else
      {
        countOther++;
        include = (scope == 1 || scope == 4);
      }

      if (include)
      {
        allNormLayers.insert(i);
      }
    }

    int entryCount = (int)allNormLayers.size();

    // Propagate through norm chains from selected entry points
    bool changed = true;
    while (changed)
    {
      changed = false;
      for (int32_t i = 0; i < totalLayers; ++i)
      {
        if (allNormLayers.count(i))
          continue;
        auto* layer = network->getLayer(i);
        if (!HasNormName(layer->getName()))
          continue;
        if (!IsComputeLayerType(layer->getType()))
          continue;

        for (int32_t inp = 0; inp < layer->getNbInputs(); ++inp)
        {
          auto* inputTensor = layer->getInput(inp);
          if (!inputTensor || !inputTensor->getName())
            continue;
          auto it = tensorProducer.find(inputTensor->getName());
          if (it != tensorProducer.end() && allNormLayers.count(it->second))
          {
            allNormLayers.insert(i);
            changed = true;
            break;
          }
        }
      }
    }

    fprintf(stderr, "[TensorRT] AllNorms(scope=%d): entries: %d QKV + %d smolgen + %d residual + %d other, "
      "%d selected + %d chain = %d layers\n",
      scope, countQKV, countSmolgen, countResidual, countOther,
      entryCount, (int)allNormLayers.size() - entryCount, (int)allNormLayers.size());
  }

  return allNormLayers;
}


// Helper: strict mode - only match main encoder post-attention ln1
// Matches: /transformer_layer.X/ln1/...
// Excludes: /transformer_layer.X/attention/ln1/... (smolgen related)
// NOTE: This is LC0-specific and does NOT cover Ceres-style nets.
static bool IsPostAttentionNormLayerStrict(const char* layerName)
{
  std::string name(layerName);
  return name.find("/ln1/") != std::string::npos &&
    name.find("/attention/ln1/") == std::string::npos;
}

// Helper: smolgen mode - only match smolgen-related ln1 inside attention
// Matches: /transformer_layer.X/attention/ln1/...
// NOTE: This is LC0-specific and does NOT cover Ceres-style nets.
static bool IsSmolgenNormLayer(const char* layerName)
{
  std::string name(layerName);
  return name.find("/attention/ln1/") != std::string::npos;
}

// Read an entire file into memory. Returns false on open/read failure. Used so the ONNX
// bytes are read from disk exactly once and then reused for both strong-typing detection
// and the TensorRT parse (parser->parse) -- nets are single-file with embedded weights,
// so there are no external-data sidecars whose resolution would require parseFromFile.
static bool ReadEntireFile(const char* path, std::vector<char>& out)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return false;
  const std::streamsize size = in.tellg();
  if (size < 0) return false;
  out.resize(static_cast<size_t>(size));
  in.seekg(0);
  if (size > 0 && !in.read(out.data(), size)) return false;
  return true;
}

// Helper: detect explicit quantization (QuantizeLinear) in an in-memory ONNX model buffer.
// TensorRT only honors FP8/FP4 QuantizeLinear/DequantizeLinear in a STRONGLY-TYPED network
// (INT8 works weakly typed, but FP8/FP4 scale layers are silently ignored otherwise). Such
// models are therefore built strongly typed, which also means builder precision flags and
// manual setPrecision norm/softmax marking are skipped -- FP32 norms/softmax must be baked
// into the ONNX instead.
//
// Rather than scanning every byte (which, for the common non-quantized model, means reading
// the whole weight section just to prove a negative), this walks the ONNX protobuf wire
// format and inspects NodeProto.op_type directly: ModelProto.graph (field 7) ->
// GraphProto.node (field 1) -> NodeProto.op_type (field 4). Initializer/weight blobs are
// skipped via their length prefixes, so cost is proportional to graph structure, not weight
// size, and an op_type match cannot be spoofed by a tensor or metadata name.
static bool OnnxNeedsStrongTyping(const void* data, size_t size)
{
  // Explicit override (escape hatch): CERES_TRT_STRONGLY_TYPED=0 forces off, =1 forces on.
  if (const char* e = std::getenv("CERES_TRT_STRONGLY_TYPED"))
  {
    if (e[0] == '0') return false;
    if (e[0] == '1') return true;
  }

  struct Walker
  {
    const uint8_t* const end;
    bool found;

    // Decode a base-128 varint at c (advancing c). False on truncation/overlong.
    bool Varint(const uint8_t*& c, uint64_t& v)
    {
      v = 0;
      for (int shift = 0; shift < 64; shift += 7)
      {
        if (c >= end) return false;
        const uint8_t b = *c++;
        v |= uint64_t(b & 0x7F) << shift;
        if (!(b & 0x80)) return true;
      }
      return false;
    }

    // Decode a field key into (field number, wire type).
    bool Tag(const uint8_t*& c, uint32_t& field, uint32_t& wire)
    {
      uint64_t key;
      if (!Varint(c, key)) return false;
      field = uint32_t(key >> 3);
      wire = uint32_t(key & 7);
      return true;
    }

    // Advance c past one field of the given wire type. False on malformed/overrun.
    bool Skip(const uint8_t*& c, uint32_t wire)
    {
      uint64_t n;
      switch (wire)
      {
        case 0: return Varint(c, n);                                          // varint
        case 1: if (end - c < 8) return false; c += 8; return true;           // 64-bit
        case 5: if (end - c < 4) return false; c += 4; return true;           // 32-bit
        case 2: return Varint(c, n) && uint64_t(end - c) >= n && (c += n, true); // length-delimited
        default: return false;                                               // groups: unsupported
      }
    }

    // NodeProto: look for op_type (field 4) == one of the *QuantizeLinear ops.
    void ScanNode(const uint8_t* c, const uint8_t* nodeEnd)
    {
      while (!found && c < nodeEnd)
      {
        uint32_t field, wire;
        if (!Tag(c, field, wire)) return;
        if (field == 4 && wire == 2)
        {
          uint64_t n;
          if (!Varint(c, n) || uint64_t(nodeEnd - c) < n) return;
          if ((n == 14 && memcmp(c, "QuantizeLinear", 14) == 0) ||
              (n == 16 && memcmp(c, "DequantizeLinear", 16) == 0) ||
              (n == 21 && memcmp(c, "DynamicQuantizeLinear", 21) == 0))
          {
            found = true;
            return;
          }
          c += n;
        }
        else if (!Skip(c, wire)) return;
      }
    }

    // GraphProto: iterate node submessages (field 1); seek past everything else (weights).
    void ScanGraph(const uint8_t* c, const uint8_t* graphEnd)
    {
      while (!found && c < graphEnd)
      {
        uint32_t field, wire;
        if (!Tag(c, field, wire)) return;
        if (field == 1 && wire == 2)
        {
          uint64_t n;
          if (!Varint(c, n) || uint64_t(graphEnd - c) < n) return;
          ScanNode(c, c + n);
          c += n;
        }
        else if (!Skip(c, wire)) return;
      }
    }

    // ModelProto: descend into the graph submessage (field 7).
    void ScanModel(const uint8_t* c)
    {
      while (!found && c < end)
      {
        uint32_t field, wire;
        if (!Tag(c, field, wire)) return;
        if (field == 7 && wire == 2)
        {
          uint64_t n;
          if (!Varint(c, n) || uint64_t(end - c) < n) return;
          ScanGraph(c, c + n);
          c += n;
        }
        else if (!Skip(c, wire)) return;
      }
    }
  };

  const uint8_t* const base = static_cast<const uint8_t*>(data);
  Walker w{ base + size, false };
  w.ScanModel(base);
  return w.found;
}

extern "C"
{

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

    // Read the ONNX bytes once; reused for strong-typing detection and the parse below.
    std::vector<char> onnxBlob;
    if (!ReadEntireFile(onnxPath, onnxBlob))
    {
      SetError("Failed to read ONNX file: " + std::string(onnxPath));
      return nullptr;
    }

    // Create network. Models with explicit QuantizeLinear (FP8/FP4 QDQ) must be STRONGLY TYPED;
    // TensorRT ignores FP8/FP4 QDQ scale layers in a weakly-typed network.
    const bool stronglyTyped = OnnxNeedsStrongTyping(onnxBlob.data(), onnxBlob.size());
    const uint32_t netFlags = stronglyTyped
      ? (1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)) : 0U;
    if (stronglyTyped)
      fprintf(stderr, "[TensorRT] QuantizeLinear detected -> building STRONGLY-TYPED network "
        "(builder precision flags + setPrecision norm/softmax marking skipped; FP32 norms/softmax "
        "must be baked into the ONNX)\n");
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(netFlags));
    if (!network)
    {
      SetError("Failed to create network");
      return nullptr;
    }
    // Strongly-typed builds ignore builder precision flags and forbid setPrecision; neutralize
    // the manual-precision options (FP8 + FP32 norms come from the ONNX itself).
    TRT_BuildOptions optsStrong;
    if (stronglyTyped)
    {
      optsStrong = *opts;
      optsStrong.useBest = optsStrong.useFP16 = optsStrong.useBF16 = optsStrong.useFP8 = 0;
      optsStrong.fp32PostAttentionNorm = optsStrong.fp32PostAttentionNormStrict = 0;
      optsStrong.fp32SmolgenNorm = optsStrong.fp32Softmax = optsStrong.fp32AllNorms = 0;
      opts = &optsStrong;
    }

    // Create ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, g_logger));
    if (!parser)
    {
      SetError("Failed to create ONNX parser");
      return nullptr;
    }

    // Parse ONNX from the already-read bytes (single-file model, weights embedded).
    if (!parser->parse(onnxBlob.data(), onnxBlob.size()))
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

    // Enable refit support if requested
    if (opts->refittable)
    {
      config->setFlag(nvinfer1::BuilderFlag::kREFIT_IDENTICAL);
      fprintf(stderr, "[TensorRT] Building with REFIT_IDENTICAL support enabled\n");
    }

    // Enable detailed profiling for layer precision inspection
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    // Force FP32 precision for normalization layers to prevent FP16 overflow.
    // Three modes: broad (fp32PostAttentionNorm) = all normalization layers (recommended),
    //              strict (fp32PostAttentionNormStrict) = only main encoder ln1 (LC0-specific),
    //              smolgen (fp32SmolgenNorm) = only smolgen attention ln1 (LC0-specific).
    if (opts->fp32PostAttentionNorm || opts->fp32PostAttentionNormStrict || opts->fp32SmolgenNorm)
    {
      config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);

      const char* modeName = "broad";
      int layersMarked = 0;
      int totalLayers = network->getNbLayers();

      // Compute residual-stream norm layer set for the broad mode
      auto residualNormSet = FindResidualStreamNormLayers(network.get(), totalLayers);

      for (int32_t i = 0; i < totalLayers; ++i)
      {
        auto* layer = network->getLayer(i);
        bool shouldMark = false;

        if (opts->fp32SmolgenNorm)
        {
          shouldMark = IsSmolgenNormLayer(layer->getName());
          modeName = "smolgen";
        }
        else if (opts->fp32PostAttentionNormStrict)
        {
          shouldMark = IsPostAttentionNormLayerStrict(layer->getName());
          modeName = "strict";
        }
        else
        {
          shouldMark = residualNormSet.count(i) > 0;
        }

        if (shouldMark)
        {
          layer->setPrecision(nvinfer1::DataType::kFLOAT);
          for (int32_t j = 0; j < layer->getNbOutputs(); ++j)
          {
            layer->setOutputType(j, nvinfer1::DataType::kFLOAT);
          }
          layersMarked++;
        }
      }
      fprintf(stderr, "[TensorRT] Marked %d/%d layers as FP32 for normalization (%s mode)\n",
        layersMarked, totalLayers, modeName);
    }

    // Force FP32 for all Softmax layers (prevents exp() overflow in FP16)
    if (opts->fp32Softmax)
    {
      config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
      int totalLayers = network->getNbLayers();
      int softmaxMarked = 0;
      for (int32_t i = 0; i < totalLayers; ++i)
      {
        auto* layer = network->getLayer(i);
        if (layer->getType() == nvinfer1::LayerType::kSOFTMAX)
        {
          layer->setPrecision(nvinfer1::DataType::kFLOAT);
          for (int32_t j = 0; j < layer->getNbOutputs(); ++j)
          {
            layer->setOutputType(j, nvinfer1::DataType::kFLOAT);
          }
          softmaxMarked++;
        }
      }
      fprintf(stderr, "[TensorRT] Marked %d Softmax layers as FP32\n", softmaxMarked);
    }

    // Force FP32 for normalization chains (scope: 1=all, 2=QKV, 3=smolgen, 4=QKV+smolgen)
    if (opts->fp32AllNorms)
    {
      config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
      int totalLayers = network->getNbLayers();
      auto allNormSet = FindAllNormLayers(network.get(), totalLayers, opts->fp32AllNorms);
      int normsMarked = 0;
      for (int32_t i = 0; i < totalLayers; ++i)
      {
        if (allNormSet.count(i) > 0)
        {
          auto* layer = network->getLayer(i);
          layer->setPrecision(nvinfer1::DataType::kFLOAT);
          for (int32_t j = 0; j < layer->getNbOutputs(); ++j)
          {
            layer->setOutputType(j, nvinfer1::DataType::kFLOAT);
          }
          normsMarked++;
        }
      }
      fprintf(stderr, "[TensorRT] Marked %d/%d layers as FP32 for norms (scope=%d)\n",
        normsMarked, totalLayers, opts->fp32AllNorms);
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

    // Create CUDA streams (0=compute A, 1=transfers, 2=compute B)
    cudaStreamCreate(&ec->streams[0]);
    cudaStreamCreate(&ec->streams[1]);
    cudaStreamCreate(&ec->streams[2]);
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
        ec->totalOutputElements += AlignUp(size, OUTPUT_TENSOR_ALIGN_ELEMS);
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

    // Create context2 for concurrent compute on stream 2
    if (ec->useCudaGraphs)
    {
      ec->context2 = engine->createExecutionContext();
      if (ec->context2)
      {
        for (size_t i = 0; i < ec->inputNames.size(); ++i)
        {
          auto dims = engine->getTensorShape(ec->inputNames[i].c_str());
          if (dims.d[0] == -1) dims.d[0] = batchSize;
          ec->context2->setInputShape(ec->inputNames[i].c_str(), dims);
        }
      }
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
    hash ^= std::hash<int32_t>{}(opts->fp32PostAttentionNormStrict) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int32_t>{}(opts->fp32SmolgenNorm) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    // Only include new fields in hash when non-zero, to avoid invalidating existing cached engines
    if (opts->fp32Softmax)
    {
      hash ^= std::hash<int32_t>{}(opts->fp32Softmax) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    if (opts->fp32AllNorms)
    {
      hash ^= std::hash<int32_t>{}(opts->fp32AllNorms) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    // Only include refittable in hash when true, so existing cached files (with false) remain valid
    if (opts->refittable)
    {
      hash ^= std::hash<int32_t>{}(opts->refittable) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
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
    ec->profileIndex = 0;
    ec->useSpinWait = useSpinWait;

    cudaStreamCreate(&ec->streams[0]);
    cudaStreamCreate(&ec->streams[1]);
    cudaStreamCreate(&ec->streams[2]);
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
        ec->totalOutputElements += AlignUp(size, OUTPUT_TENSOR_ALIGN_ELEMS);
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

    // Create context2 for concurrent compute on stream 2
    if (ec->useCudaGraphs)
    {
      ec->context2 = engine->createExecutionContext();
      if (ec->context2)
      {
        for (size_t i = 0; i < ec->inputNames.size(); ++i)
        {
          auto dims = engine->getTensorShape(ec->inputNames[i].c_str());
          if (dims.d[0] == -1) dims.d[0] = batchSize;
          ec->context2->setInputShape(ec->inputNames[i].c_str(), dims);
        }
      }
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

  TRT_API int32_t TRT_GetMultiProcessorCount(int32_t deviceId)
  {
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, deviceId);
    if (err != cudaSuccess)
    {
      return -1;
    }
    return props.multiProcessorCount;
  }

  TRT_API const char* TRT_GetDeviceName(int32_t deviceId)
  {
    static thread_local char nameBuffer[256];
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, deviceId);
    if (err != cudaSuccess)
    {
      snprintf(nameBuffer, sizeof(nameBuffer), "Unknown");
      return nameBuffer;
    }
    snprintf(nameBuffer, sizeof(nameBuffer), "%s", props.name);
    return nameBuffer;
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
      offset += AlignUp(ec->outputSizes[i], OUTPUT_TENSOR_ALIGN_ELEMS);
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
    if (!EnsureDevice(ec->deviceId))  // Ensure correct device (cached to avoid redundant calls)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to set CUDA device " + std::to_string(ec->deviceId));
      return -1;
    }
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

      // Copy outputs from GPU (using actual element sizes, with alignment gaps)
      byteOffset = 0;
      for (size_t i = 0; i < ec->outputSizes.size(); ++i)
      {
        size_t bytes = ec->outputSizes[i] * ec->outputElemSizes[i];
        cudaMemcpyAsync(outPtr + byteOffset, ec->gpuBuffers[outputBufferStart + i],
          bytes, cudaMemcpyDeviceToHost, ec->stream);
        byteOffset += AlignUp(ec->outputSizes[i], OUTPUT_TENSOR_ALIGN_ELEMS) * ec->outputElemSizes[i];
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

      // Capture graph (ThreadLocal mode avoids blocking other GPU contexts in multi-GPU setups)
      cudaStreamBeginCapture(ec->stream, cudaStreamCaptureModeThreadLocal);

      bool success = ec->context->enqueueV3(ec->stream);

      cudaStreamEndCapture(ec->stream, &ec->graph);

      if (!success || !ec->graph)
      {
        std::lock_guard<std::mutex> lock(g_mutex);
        SetError("Failed to capture CUDA graph");
        return -4;
      }

      CUDA_GRAPH_INSTANTIATE(&ec->graphExec, ec->graph);
      ec->graphCaptured = true;

      // Now execute the graph
      cudaGraphLaunch(ec->graphExec, ec->stream);
      cudaStreamSynchronize(ec->stream);

      // Copy outputs from GPU (using actual element sizes, with alignment gaps)
      byteOffset = 0;
      for (size_t i = 0; i < ec->outputSizes.size(); ++i)
      {
        size_t bytes = ec->outputSizes[i] * ec->outputElemSizes[i];
        cudaMemcpy(outPtr + byteOffset, ec->gpuBuffers[outputBufferStart + i],
          bytes, cudaMemcpyDeviceToHost);
        byteOffset += AlignUp(ec->outputSizes[i], OUTPUT_TENSOR_ALIGN_ELEMS) * ec->outputElemSizes[i];
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

      // Copy outputs from GPU (using actual element sizes, with alignment gaps)
      byteOffset = 0;
      for (size_t i = 0; i < ec->outputSizes.size(); ++i)
      {
        size_t bytes = ec->outputSizes[i] * ec->outputElemSizes[i];
        cudaMemcpyAsync(outPtr + byteOffset, ec->gpuBuffers[outputBufferStart + i],
          bytes, cudaMemcpyDeviceToHost, ec->stream);
        byteOffset += AlignUp(ec->outputSizes[i], OUTPUT_TENSOR_ALIGN_ELEMS) * ec->outputElemSizes[i];
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
    if (!EnsureDevice(ec->deviceId)) return -1;

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
      outputOffset += AlignUp(ec->outputSizes[i], OUTPUT_TENSOR_ALIGN_ELEMS);
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

  // Returns the appropriate execution context for a given stream index.
  // Stream 2 (concurrent compute B) uses context2 to avoid state conflicts
  // with stream 0/1 operations on the primary context.
  static nvinfer1::IExecutionContext* GetContextForStream(EngineContext* ec, int32_t streamIdx)
  {
    return (streamIdx == 2 && ec->context2) ? ec->context2 : ec->context;
  }

  TRT_API int32_t TRT_InferOnStream(TRT_EngineHandle handle, int32_t streamIdx,
    void* gpuInput, void* gpuOutput)
  {
    constexpr size_t ELEM_SIZE = 2;  // FP16
    if (!handle || streamIdx < 0 || streamIdx > 2) return -1;
    auto* ec = static_cast<EngineContext*>(handle);
    if (!EnsureDevice(ec->deviceId)) return -1;

    auto* ctx = GetContextForStream(ec, streamIdx);

    if (!ec->inputNames.empty())
      ctx->setTensorAddress(ec->inputNames[0].c_str(), gpuInput);

    size_t outputOffset = 0;
    for (size_t i = 0; i < ec->outputNames.size(); ++i)
    {
      void* outPtr = static_cast<char*>(gpuOutput) + outputOffset * ELEM_SIZE;
      ctx->setTensorAddress(ec->outputNames[i].c_str(), outPtr);
      outputOffset += AlignUp(ec->outputSizes[i], OUTPUT_TENSOR_ALIGN_ELEMS);
    }

    bool success = ctx->enqueueV3(ec->streams[streamIdx]);
    return success ? 0 : -2;
  }

  TRT_API int32_t TRT_InferOnStreamWithGraph(TRT_EngineHandle handle, int32_t streamIdx,
    void* gpuInput, void* gpuOutput)
  {
    constexpr size_t ELEM_SIZE = 2;  // FP16
    if (!handle || streamIdx < 0 || streamIdx > 2) return -1;
    auto* ec = static_cast<EngineContext*>(handle);
    if (!EnsureDevice(ec->deviceId)) return -1;

    // Select the execution context for this stream.
    // Stream 2 uses a separate context2 so that concurrent graph launches
    // on streams 0 and 2 don't interfere with each other's tensor addresses.
    auto* ctx = GetContextForStream(ec, streamIdx);

    // Set tensor addresses on the appropriate context
    if (!ec->inputNames.empty())
      ctx->setTensorAddress(ec->inputNames[0].c_str(), gpuInput);

    size_t outputOffset = 0;
    for (size_t i = 0; i < ec->outputNames.size(); ++i)
    {
      void* outPtr = static_cast<char*>(gpuOutput) + outputOffset * ELEM_SIZE;
      ctx->setTensorAddress(ec->outputNames[i].c_str(), outPtr);
      outputOffset += AlignUp(ec->outputSizes[i], OUTPUT_TENSOR_ALIGN_ELEMS);
    }

    if (!ec->useCudaGraphs)
    {
      // CUDA graphs disabled - use direct enqueue
      bool success = ctx->enqueueV3(ec->streams[streamIdx]);
      return success ? 0 : -2;
    }

    if (ec->streamGraphsCaptured[streamIdx])
    {
      // Check if buffer addresses match the captured graph
      if (ec->streamCapturedInput[streamIdx] == gpuInput &&
        ec->streamCapturedOutput[streamIdx] == gpuOutput)
      {
        // Addresses match - launch the captured graph
        cudaError_t err = cudaGraphLaunch(ec->streamGraphExecs[streamIdx], ec->streams[streamIdx]);
        if (err != cudaSuccess)
        {
          std::lock_guard<std::mutex> lock(g_mutex);
          SetError("cudaGraphLaunch failed on stream " + std::to_string(streamIdx) + ": " + cudaGetErrorString(err));
          return -3;
        }
        return 0;
      }
      else
      {
        // Addresses changed - fall back to direct enqueue (no graph)
        bool success = ctx->enqueueV3(ec->streams[streamIdx]);
        return success ? 0 : -2;
      }
    }

    // First call on this stream - capture the graph (ThreadLocal mode for multi-GPU compatibility)
    cudaError_t err = cudaStreamBeginCapture(ec->streams[streamIdx], cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("cudaStreamBeginCapture failed: " + std::string(cudaGetErrorString(err)));
      return -4;
    }

    bool success = ctx->enqueueV3(ec->streams[streamIdx]);

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

    err = CUDA_GRAPH_INSTANTIATE(&ec->streamGraphExecs[streamIdx], ec->streamGraphs[streamIdx]);
    if (err != cudaSuccess)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("cudaGraphInstantiate failed: " + std::string(cudaGetErrorString(err)));
      cudaGraphDestroy(ec->streamGraphs[streamIdx]);
      ec->streamGraphs[streamIdx] = nullptr;
      return -7;
    }

    ec->streamGraphsCaptured[streamIdx] = true;
    // Store the buffer addresses used for this capture
    ec->streamCapturedInput[streamIdx] = gpuInput;
    ec->streamCapturedOutput[streamIdx] = gpuOutput;

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
    if (!handle || streamIdx < 0 || streamIdx > 2) return -1;
    auto* ec = static_cast<EngineContext*>(handle);
    if (!EnsureDevice(ec->deviceId)) return -1;

    auto* ctx = GetContextForStream(ec, streamIdx);

    // Set dynamic input shapes for actualBatchSize
    for (size_t i = 0; i < ec->inputNames.size(); ++i)
    {
      const char* name = ec->inputNames[i].c_str();
      auto dims = ec->engine->getTensorShape(name);
      if (dims.d[0] == -1)
      {
        dims.d[0] = actualBatchSize;
      }
      if (!ctx->setInputShape(name, dims))
      {
        std::lock_guard<std::mutex> lock(g_mutex);
        SetError("Failed to set input shape for " + std::string(name) + " to batch " + std::to_string(actualBatchSize));
        return -3;
      }
    }

    if (!ec->inputNames.empty())
      ctx->setTensorAddress(ec->inputNames[0].c_str(), gpuInput);

    // Compute per-position output sizes and set tensor addresses
    size_t outputOffset = 0;
    for (size_t i = 0; i < ec->outputNames.size(); ++i)
    {
      void* outPtr = static_cast<char*>(gpuOutput) + outputOffset * ELEM_SIZE;
      ctx->setTensorAddress(ec->outputNames[i].c_str(), outPtr);
      // Use per-position size * actualBatchSize for the offset calculation
      int64_t perPosSize = ec->outputSizes[i] / ec->batchSize;
      outputOffset += AlignUp(perPosSize * actualBatchSize, OUTPUT_TENSOR_ALIGN_ELEMS);
    }

    bool success = ctx->enqueueV3(ec->streams[streamIdx]);
    return success ? 0 : -2;
  }

  TRT_API int32_t TRT_CopyToGPUOnStream(TRT_EngineHandle handle, int32_t streamIdx, void* dst, const void* src, int64_t bytes)
  {
    if (!handle || streamIdx < 0 || streamIdx > 2) return -1;
    auto* ec = static_cast<EngineContext*>(handle);
    if (!EnsureDevice(ec->deviceId)) return -1;
    // Clear any stale CUDA error from prior unchecked operations
    cudaGetLastError();
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, ec->streams[streamIdx]);
    if (err != cudaSuccess)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("cudaMemcpyAsync H2D failed on stream " + std::to_string(streamIdx) +
        ": " + cudaGetErrorString(err) +
        " (dst=" + std::to_string((uintptr_t)dst) +
        " src=" + std::to_string((uintptr_t)src) +
        " bytes=" + std::to_string(bytes) + ")");
      return -2;
    }
    return 0;
  }

  TRT_API int32_t TRT_CopyFromGPUOnStream(TRT_EngineHandle handle, int32_t streamIdx, void* dst, const void* src, int64_t bytes)
  {
    if (!handle || streamIdx < 0 || streamIdx > 2) return -1;
    auto* ec = static_cast<EngineContext*>(handle);
    if (!EnsureDevice(ec->deviceId)) return -1;
    cudaGetLastError();
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, ec->streams[streamIdx]);
    if (err != cudaSuccess)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("cudaMemcpyAsync D2H failed on stream " + std::to_string(streamIdx) +
        ": " + cudaGetErrorString(err));
      return -2;
    }
    return 0;
  }

  // count = number of positions in the batch being synced on this stream (i.e. the per-GPU
  // batch size). Used to resolve "auto" mode (3): spin for small batches, block for large.
  TRT_API int32_t TRT_SyncStreamIdx(TRT_EngineHandle handle, int32_t streamIdx, int32_t count)
  {
    if (!handle || streamIdx < 0 || streamIdx > 2) return -1;
    auto* ec = static_cast<EngineContext*>(handle);
    if (!EnsureDevice(ec->deviceId)) return -1;

    cudaStream_t stream = ec->streams[streamIdx];

    // Resolve the wait policy. "auto" picks spin vs block per-sync from the per-GPU batch
    // size: short GPU bursts (small batch) spin to avoid the OS wakeup latency; long bursts
    // block to free the core.
    int mode = g_syncMode;
    if (mode == 3)
    {
      mode = (count <= SYNC_SPIN_MAX_PERGPU) ? 1 : 2;
    }

    if (mode == 0)
    {
      // Driver policy (typically spins when CUDA contexts <= cores, else blocks).
      cudaStreamSynchronize(stream);
      return 0;
    }

    // Reusable per-stream event. Created with the blocking flag so the BLOCK path actually
    // blocks; the SPIN path busy-polls cudaEventQuery, which ignores the flag. A single
    // stream may use either policy across batches (auto mode), hence the fixed flag.
    if (!ec->syncEvents[streamIdx])
    {
      cudaEventCreateWithFlags(&ec->syncEvents[streamIdx], cudaEventBlockingSync | cudaEventDisableTiming);
    }
    cudaEventRecord(ec->syncEvents[streamIdx], stream);
    if (mode == 1)
    {
      while (cudaEventQuery(ec->syncEvents[streamIdx]) == cudaErrorNotReady) { }
    }
    else
    {
      cudaEventSynchronize(ec->syncEvents[streamIdx]);
    }
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
    if (!EnsureDevice(ec->deviceId))
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to set CUDA device " + std::to_string(ec->deviceId));
      return -1;
    }

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
      expectedOutputSize += AlignUp(outputPerPos[i] * actualBatchSize, OUTPUT_TENSOR_ALIGN_ELEMS);
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

    // Copy outputs from GPU (only the actual data, with alignment gaps)
    byteOffset = 0;
    for (size_t i = 0; i < ec->outputSizes.size(); ++i)
    {
      int64_t actualElements = outputPerPos[i] * actualBatchSize;
      size_t bytes = actualElements * ec->outputElemSizes[i];
      cudaMemcpyAsync(outPtr + byteOffset, ec->gpuBuffers[outputBufferStart + i],
        bytes, cudaMemcpyDeviceToHost, ec->stream);
      byteOffset += AlignUp(actualElements, OUTPUT_TENSOR_ALIGN_ELEMS) * ec->outputElemSizes[i];
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

  TRT_API void TRT_DisableCudaGraphs(TRT_EngineHandle handle)
  {
    if (!handle) return;
    auto* ec = static_cast<EngineContext*>(handle);
    ec->useCudaGraphs = false;
  }

  TRT_API int32_t TRT_IsStreamGraphCaptured(TRT_EngineHandle handle, int32_t streamIdx)
  {
    if (!handle || streamIdx < 0 || streamIdx > 2) return -1;
    auto* ec = static_cast<EngineContext*>(handle);
    return ec->streamGraphsCaptured[streamIdx] ? 1 : 0;
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

  // =========================================================================
  // Weight Refitting (for refittable engines)
  // =========================================================================

  TRT_API int32_t TRT_SetNamedWeights(TRT_EngineHandle handle, const char* weightTensorName,
    const void* weights, int64_t numElements)
  {
    if (!handle)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Invalid handle");
      return -1;
    }
    if (!weightTensorName || !weights)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Invalid weight tensor name or weights pointer");
      return -2;
    }

    auto* ec = static_cast<EngineContext*>(handle);
    if (!EnsureDevice(ec->deviceId))
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to set CUDA device " + std::to_string(ec->deviceId));
      return -3;
    }

    // Create refitter if not exists
    nvinfer1::IRefitter* refitter = nvinfer1::createInferRefitter(*ec->engine, g_logger);
    if (!refitter)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to create refitter - engine may not have been built with refittable=1");
      return -4;
    }

    // Set the named weights
    // Weights are expected to be FP16 (Half precision)
    nvinfer1::Weights trtWeights;
    trtWeights.type = nvinfer1::DataType::kHALF;
    trtWeights.values = weights;
    trtWeights.count = numElements;

    bool success = refitter->setNamedWeights(weightTensorName, trtWeights);
    if (!success)
    {
      delete refitter;
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to set weights for tensor: " + std::string(weightTensorName));
      return -5;
    }

    // Refit the engine immediately
    success = refitter->refitCudaEngine();
    delete refitter;

    if (!success)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to refit CUDA engine after setting weights");
      return -6;
    }

    return 0;
  }

  TRT_API int32_t TRT_RefitEngine(TRT_EngineHandle handle)
  {
    if (!handle)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Invalid handle");
      return -1;
    }

    auto* ec = static_cast<EngineContext*>(handle);
    if (!EnsureDevice(ec->deviceId))
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to set CUDA device " + std::to_string(ec->deviceId));
      return -2;
    }

    // Create refitter
    nvinfer1::IRefitter* refitter = nvinfer1::createInferRefitter(*ec->engine, g_logger);
    if (!refitter)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to create refitter - engine may not have been built with refittable=1");
      return -3;
    }

    // Refit the engine
    bool success = refitter->refitCudaEngine();
    delete refitter;

    if (!success)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to refit CUDA engine");
      return -4;
    }

    return 0;
  }

  // =========================================================================
  // Multi-Profile Engine (shared weights, N execution contexts)
  // =========================================================================

  // Helper: initialize EngineContext for a specific optimization profile of a shared engine
  static EngineContext* InitializeEngineContextForProfile(nvinfer1::ICudaEngine* engine,
    SharedEngine* shared, int32_t profileIndex, int32_t batchSize,
    bool useCudaGraphs, bool useSpinWait, int32_t deviceId)
  {
    // Ensure correct device for stream/buffer creation (keeps tls_currentDevice in sync)
    if (!EnsureDevice(deviceId))
    {
      return nullptr;
    }

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context)
    {
      return nullptr;
    }

    auto* ec = new EngineContext();
    ec->engine = engine;
    ec->context = context;
    ec->sharedOwner = shared;
    ec->batchSize = batchSize;
    ec->deviceId = deviceId;
    ec->useCudaGraphs = useCudaGraphs;
    ec->profileIndex = profileIndex;
    ec->useSpinWait = useSpinWait;

    // Create streams first (needed for setOptimizationProfileAsync)
    for (int i = 0; i < 3; ++i)
    {
      cudaError_t serr = cudaStreamCreate(&ec->streams[i]);
      if (serr != cudaSuccess)
      {
        // Clean up already-created streams, then the partial context. Detach the (shared) engine
        // and owner first so ~EngineContext neither double-frees `context` nor touches the
        // engine/refcount on this failure path (the caller owns refcount rollback).
        for (int j = 0; j < i; ++j)
        {
          cudaStreamDestroy(ec->streams[j]);
          ec->streams[j] = nullptr;
        }
        ec->stream = nullptr;
        ec->context = nullptr;
        ec->sharedOwner = nullptr;
        ec->engine = nullptr;
        delete context;
        delete ec;
        return nullptr;
      }
    }
    ec->stream = ec->streams[0];

    // Select the optimization profile for this context (TRT 10+ API)
    if (profileIndex > 0)
    {
      if (!context->setOptimizationProfileAsync(profileIndex, ec->stream))
      {
        // Detach engine/owner so ~EngineContext does not touch the engine/refcount here
        // (caller owns refcount rollback); the streams and context are real and freed by the dtor.
        ec->sharedOwner = nullptr;
        ec->engine = nullptr;
        delete ec;
        return nullptr;
      }
      cudaStreamSynchronize(ec->stream);
    }

    // Apply execution options
    if (useSpinWait)
    {
      context->setEnqueueEmitsProfile(false);
      context->setPersistentCacheLimit(0);
    }

    // First pass: collect input names and set input shapes on context.
    // This must happen BEFORE querying output shapes, because output dims
    // depend on input shapes (especially the batch dimension).
    int32_t nbIO = engine->getNbIOTensors();
    for (int32_t i = 0; i < nbIO; ++i)
    {
      const char* name = engine->getIOTensorName(i);
      auto mode = engine->getTensorIOMode(name);
      if (mode != nvinfer1::TensorIOMode::kINPUT)
      {
        continue;
      }

      auto dims = engine->getTensorShape(name);
      auto dtype = engine->getTensorDataType(name);
      size_t elemSize = GetElementSize(dtype);

      // Set input shape on context with our batch size
      nvinfer1::Dims inputDims = dims;
      if (inputDims.d[0] == -1)
      {
        inputDims.d[0] = batchSize;
      }
      context->setInputShape(name, inputDims);

      // Compute input size using resolved dims
      int64_t size = 1;
      for (int32_t d = 0; d < inputDims.nbDims; ++d)
      {
        size *= inputDims.d[d];
      }

      ec->inputNames.push_back(name);
      ec->inputSizes.push_back(size);
      ec->inputElemSizes.push_back(elemSize);
      ec->totalInputElements += size;
    }

    // Second pass: collect output info from the CONTEXT (not engine).
    // After input shapes are set, the context resolves output shapes
    // for the active profile's batch size.
    for (int32_t i = 0; i < nbIO; ++i)
    {
      const char* name = engine->getIOTensorName(i);
      auto mode = engine->getTensorIOMode(name);
      if (mode != nvinfer1::TensorIOMode::kOUTPUT)
      {
        continue;
      }

      auto dtype = engine->getTensorDataType(name);
      size_t elemSize = GetElementSize(dtype);

      // Query output shape from context (resolved for this profile's batch size)
      auto dims = context->getTensorShape(name);

      int64_t size = 1;
      for (int32_t d = 0; d < dims.nbDims; ++d)
      {
        int64_t dimVal = (dims.d[d] == -1) ? batchSize : dims.d[d];
        size *= dimVal;
      }

      ec->outputNames.push_back(name);
      ec->outputSizes.push_back(size);
      ec->outputElemSizes.push_back(elemSize);
      ec->totalOutputElements += AlignUp(size, OUTPUT_TENSOR_ALIGN_ELEMS);
    }

    // Pre-allocate GPU buffers
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

    // Create second execution context for stream 2 (concurrent compute).
    // TensorRT execution contexts are not thread-safe, so concurrent graph replay
    // on streams 0 and 2 requires separate contexts with independent tensor bindings.
    if (useCudaGraphs)
    {
      ec->context2 = engine->createExecutionContext();
      if (ec->context2)
      {
        if (profileIndex > 0)
        {
          ec->context2->setOptimizationProfileAsync(profileIndex, ec->streams[2]);
          cudaStreamSynchronize(ec->streams[2]);
        }
        if (useSpinWait)
        {
          ec->context2->setEnqueueEmitsProfile(false);
          ec->context2->setPersistentCacheLimit(0);
        }
        // Set input shapes on context2 (must match context)
        for (size_t i = 0; i < ec->inputNames.size(); ++i)
        {
          auto dims = engine->getTensorShape(ec->inputNames[i].c_str());
          if (dims.d[0] == -1)
          {
            dims.d[0] = batchSize;
          }
          ec->context2->setInputShape(ec->inputNames[i].c_str(), dims);
        }
      }
    }

    return ec;
  }


  TRT_API int32_t TRT_LoadONNXMultiProfile(const char* onnxPath,
    const int32_t* batchSizes, int32_t numProfiles,
    const TRT_BuildOptions* options, int32_t deviceId,
    TRT_EngineHandle* outHandles)
  {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_initialized)
    {
      SetError("TensorRT not initialized");
      return -1;
    }

    if (!batchSizes || numProfiles <= 0 || !outHandles)
    {
      SetError("Invalid arguments for multi-profile load");
      return -2;
    }

    // Handle deviceId
    if (deviceId < 0)
    {
      cudaGetDevice(&deviceId);
    }
    else
    {
      cudaError_t err = cudaSetDevice(deviceId);
      if (err != cudaSuccess)
      {
        SetError("Failed to set CUDA device " + std::to_string(deviceId));
        return -3;
      }
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
      return -4;
    }

    // Read the ONNX bytes once; reused for strong-typing detection and the parse below.
    std::vector<char> onnxBlob;
    if (!ReadEntireFile(onnxPath, onnxBlob))
    {
      SetError("Failed to read ONNX file: " + std::string(onnxPath));
      return -8;
    }

    // Create network. Models with explicit QuantizeLinear (FP8/FP4 QDQ) must be STRONGLY TYPED;
    // TensorRT ignores FP8/FP4 QDQ scale layers in a weakly-typed network.
    const bool stronglyTyped = OnnxNeedsStrongTyping(onnxBlob.data(), onnxBlob.size());
    const uint32_t netFlags = stronglyTyped
      ? (1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)) : 0U;
    if (stronglyTyped)
      fprintf(stderr, "[TensorRT] QuantizeLinear detected -> building STRONGLY-TYPED network "
        "(builder precision flags + setPrecision norm/softmax marking skipped; FP32 norms/softmax "
        "must be baked into the ONNX)\n");
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(netFlags));
    if (!network)
    {
      SetError("Failed to create network");
      return -5;
    }
    // Strongly-typed builds ignore builder precision flags and forbid setPrecision; neutralize
    // the manual-precision options (FP8 + FP32 norms come from the ONNX itself).
    TRT_BuildOptions optsStrong;
    if (stronglyTyped)
    {
      optsStrong = *opts;
      optsStrong.useBest = optsStrong.useFP16 = optsStrong.useBF16 = optsStrong.useFP8 = 0;
      optsStrong.fp32PostAttentionNorm = optsStrong.fp32PostAttentionNormStrict = 0;
      optsStrong.fp32SmolgenNorm = optsStrong.fp32Softmax = optsStrong.fp32AllNorms = 0;
      opts = &optsStrong;
    }

    // Parse ONNX
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, g_logger));
    if (!parser)
    {
      SetError("Failed to create ONNX parser");
      return -6;
    }

    // Parse ONNX from the already-read bytes (single-file model, weights embedded).
    if (!parser->parse(onnxBlob.data(), onnxBlob.size()))
    {
      std::string errors;
      for (int32_t i = 0; i < parser->getNbErrors(); ++i)
      {
        errors += parser->getError(i)->desc();
        errors += "\n";
      }
      SetError("Failed to parse ONNX: " + errors);
      return -7;
    }

    // Reject float32 ONNX models early (before expensive engine build).
    // The managed Half[] buffers assume 2-byte elements; float32 I/O would cause crashes.
    for (int32_t i = 0; i < network->getNbInputs(); ++i)
    {
      if (network->getInput(i)->getType() == nvinfer1::DataType::kFLOAT)
      {
        SetError("Unsupported ONNX model: input '" + std::string(network->getInput(i)->getName()) +
          "' uses float32. Only float16/bfloat16 ONNX models are supported by the TensorRT native evaluator. "
          "Convert the model to float16 first.");
        return -20;
      }
    }

    // Create builder config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
      SetError("Failed to create builder config");
      return -8;
    }

    // Apply build options
    config->setBuilderOptimizationLevel(opts->builderOptimizationLevel);

    if (opts->tilingOptimizationLevel >= 0)
    {
      config->setTilingOptimizationLevel(
        static_cast<nvinfer1::TilingOptimizationLevel>(opts->tilingOptimizationLevel));
    }

    if (opts->useBest) config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    if (opts->useFP16) config->setFlag(nvinfer1::BuilderFlag::kFP16);
    if (opts->useBF16) config->setFlag(nvinfer1::BuilderFlag::kBF16);
    if (opts->useFP8) config->setFlag(nvinfer1::BuilderFlag::kFP8);

    if (opts->refittable)
    {
      config->setFlag(nvinfer1::BuilderFlag::kREFIT_IDENTICAL);
      fprintf(stderr, "[TensorRT] Building multi-profile with REFIT_IDENTICAL support enabled\n");
    }

    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    // Force FP32 precision for normalization layers to prevent FP16 overflow.
    if (opts->fp32PostAttentionNorm || opts->fp32PostAttentionNormStrict || opts->fp32SmolgenNorm)
    {
      config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);

      const char* modeName = "broad";
      int layersMarked = 0;
      int totalLayers = network->getNbLayers();

      // Compute residual-stream norm layer set for the broad mode
      auto residualNormSet = FindResidualStreamNormLayers(network.get(), totalLayers);

      for (int32_t i = 0; i < totalLayers; ++i)
      {
        auto* layer = network->getLayer(i);
        bool shouldMark = false;

        if (opts->fp32SmolgenNorm)
        {
          shouldMark = IsSmolgenNormLayer(layer->getName());
          modeName = "smolgen";
        }
        else if (opts->fp32PostAttentionNormStrict)
        {
          shouldMark = IsPostAttentionNormLayerStrict(layer->getName());
          modeName = "strict";
        }
        else
        {
          shouldMark = residualNormSet.count(i) > 0;
        }

        if (shouldMark)
        {
          layer->setPrecision(nvinfer1::DataType::kFLOAT);
          for (int32_t j = 0; j < layer->getNbOutputs(); ++j)
          {
            layer->setOutputType(j, nvinfer1::DataType::kFLOAT);
          }
          layersMarked++;
        }
      }
      fprintf(stderr, "[TensorRT] Marked %d/%d layers as FP32 for normalization (%s mode)\n",
        layersMarked, totalLayers, modeName);
    }

    // Force FP32 for all Softmax layers (prevents exp() overflow in FP16)
    if (opts->fp32Softmax)
    {
      config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
      int totalLayers = network->getNbLayers();
      int softmaxMarked = 0;
      for (int32_t i = 0; i < totalLayers; ++i)
      {
        auto* layer = network->getLayer(i);
        if (layer->getType() == nvinfer1::LayerType::kSOFTMAX)
        {
          layer->setPrecision(nvinfer1::DataType::kFLOAT);
          for (int32_t j = 0; j < layer->getNbOutputs(); ++j)
          {
            layer->setOutputType(j, nvinfer1::DataType::kFLOAT);
          }
          softmaxMarked++;
        }
      }
      fprintf(stderr, "[TensorRT] Marked %d Softmax layers as FP32\n", softmaxMarked);
    }

    // Force FP32 for normalization chains (scope: 1=all, 2=QKV, 3=smolgen, 4=QKV+smolgen)
    if (opts->fp32AllNorms)
    {
      config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
      int totalLayers = network->getNbLayers();
      auto allNormSet = FindAllNormLayers(network.get(), totalLayers, opts->fp32AllNorms);
      int normsMarked = 0;
      for (int32_t i = 0; i < totalLayers; ++i)
      {
        if (allNormSet.count(i) > 0)
        {
          auto* layer = network->getLayer(i);
          layer->setPrecision(nvinfer1::DataType::kFLOAT);
          for (int32_t j = 0; j < layer->getNbOutputs(); ++j)
          {
            layer->setOutputType(j, nvinfer1::DataType::kFLOAT);
          }
          normsMarked++;
        }
      }
      fprintf(stderr, "[TensorRT] Marked %d/%d layers as FP32 for norms (scope=%d)\n",
        normsMarked, totalLayers, opts->fp32AllNorms);
    }

    // Create N optimization profiles (one per batch size, Exact mode: min=opt=max)
    for (int32_t p = 0; p < numProfiles; ++p)
    {
      auto profile = builder->createOptimizationProfile();
      for (int32_t i = 0; i < network->getNbInputs(); ++i)
      {
        auto input = network->getInput(i);
        auto dims = input->getDimensions();

        nvinfer1::Dims minDims = dims, optDims = dims, maxDims = dims;
        if (dims.d[0] == -1)
        {
          minDims.d[0] = batchSizes[p];
          optDims.d[0] = batchSizes[p];
          maxDims.d[0] = batchSizes[p];
        }

        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, minDims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, optDims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, maxDims);
      }
      config->addOptimizationProfile(profile);
    }

    // Build a batch sizes description for logging
    std::string batchDesc;
    for (int32_t p = 0; p < numProfiles; ++p)
    {
      if (p > 0) batchDesc += ",";
      batchDesc += std::to_string(batchSizes[p]);
    }

    std::string basename = GetBaseName(onnxPath);
    char msg[512];
    snprintf(msg, sizeof(msg), "[TensorRT] Building multi-profile %s: batches=[%s], %d profiles",
      basename.c_str(), batchDesc.c_str(), numProfiles);
    PrintYellow(msg);

    // Build serialized engine
    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine)
    {
      SetError("Failed to build multi-profile engine");
      return -9;
    }

    // Deserialize engine
    nvinfer1::ICudaEngine* engine = g_runtime->deserializeCudaEngine(
      serializedEngine->data(), serializedEngine->size());
    if (!engine)
    {
      SetError("Failed to deserialize multi-profile engine");
      return -10;
    }

    // Create shared ownership
    auto* shared = new SharedEngine(engine, numProfiles);

    // Create N execution contexts, one per profile
    for (int32_t p = 0; p < numProfiles; ++p)
    {
      EngineContext* ec = InitializeEngineContextForProfile(engine, shared, p, batchSizes[p],
        opts->useCudaGraphs != 0, opts->useSpinWait != 0, deviceId);
      if (!ec)
      {
        // Cleanup already-created contexts (their destructors decrement refcount)
        for (int32_t j = 0; j < p; ++j)
        {
          delete static_cast<EngineContext*>(outHandles[j]);
          outHandles[j] = nullptr;
        }
        // After deleting p contexts, refCount = numProfiles - p (for uncreated contexts).
        // If engine still alive, force cleanup.
        if (shared->refCount.load() > 0)
        {
          delete engine;
          delete shared;
        }
        SetError("Failed to create execution context for profile " + std::to_string(p));
        return -11;
      }
      outHandles[p] = ec;
    }

    g_lastError.clear();
    return 0;
  }


  // Helper: create N EngineContexts from an already-deserialized multi-profile engine
  static int32_t CreateContextsFromEngine(nvinfer1::ICudaEngine* engine,
    const int32_t* batchSizes, int32_t numProfiles,
    bool useCudaGraphs, bool useSpinWait, int32_t deviceId,
    TRT_EngineHandle* outHandles)
  {
    auto* shared = new SharedEngine(engine, numProfiles);

    for (int32_t p = 0; p < numProfiles; ++p)
    {
      EngineContext* ec = InitializeEngineContextForProfile(engine, shared, p, batchSizes[p],
        useCudaGraphs, useSpinWait, deviceId);
      if (!ec)
      {
        for (int32_t j = 0; j < p; ++j)
        {
          delete static_cast<EngineContext*>(outHandles[j]);
          outHandles[j] = nullptr;
        }
        if (shared->refCount.load() > 0)
        {
          delete engine;
          delete shared;
        }
        return -1;
      }
      outHandles[p] = ec;
    }
    return 0;
  }


  // Create a new EngineContext that SHARES the already-deserialized ICudaEngine owned by an
  // existing context (referenceHandle), rather than deserializing/allocating the weights again.
  // The clone gets its OWN IExecutionContext, streams, and GPU buffers (so it can run concurrently
  // with the reference), but points at the same engine via the same ref-counted SharedEngine.
  // Used so a second (overlap) evaluator can reuse the primary evaluator's engine weights.
  // Returns 0 on success and writes the new handle to *outHandle; negative on error.
  TRT_API int32_t TRT_CloneContextSharingEngine(TRT_EngineHandle referenceHandle,
    int32_t deviceId, TRT_EngineHandle* outHandle)
  {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_initialized)
    {
      SetError("TensorRT not initialized");
      return -1;
    }
    if (!referenceHandle || !outHandle)
    {
      SetError("Invalid arguments for clone-context");
      return -2;
    }

    auto* ref = static_cast<EngineContext*>(referenceHandle);
    if (!ref->engine)
    {
      SetError("Reference handle has no engine to share");
      return -3;
    }

    if (deviceId < 0)
    {
      deviceId = ref->deviceId;
    }

    // Reuse the reference's shared owner so the reference and all clones share one refcount.
    // If the reference is a sole-owner (single-profile) context, promote it to ref-counted
    // ownership first: count = 1 (the existing reference) + 1 (this clone, added below).
    SharedEngine* shared = ref->sharedOwner;
    bool promoted = false;
    if (!shared)
    {
      shared = new SharedEngine(ref->engine, 1);
      ref->sharedOwner = shared;
      promoted = true;
    }

    // Reserve the clone's reference before creating it. InitializeEngineContextForProfile is
    // refcount-neutral on failure, so on error we simply undo this reservation.
    shared->refCount.fetch_add(1);

    EngineContext* ec = InitializeEngineContextForProfile(ref->engine, shared,
      ref->profileIndex, ref->batchSize, ref->useCudaGraphs, ref->useSpinWait, deviceId);
    if (!ec)
    {
      shared->refCount.fetch_sub(1);
      if (promoted)
      {
        // Revert the reference to sole-owner; do NOT delete the engine (reference still owns it).
        ref->sharedOwner = nullptr;
        delete shared;
      }
      SetError("Failed to initialize cloned execution context");
      return -4;
    }

    *outHandle = ec;
    g_lastError.clear();

#if NOT
    char msg[256];
    snprintf(msg, sizeof(msg),
      "[TensorRT] Sharing engine (cloned context, batch=%d, profile=%d) - no deserialize",
      ref->batchSize, ref->profileIndex);
    PrintGreen(msg);

#endif
    return 0;
  }


  TRT_API char* TRT_GenerateMultiProfileCacheFilename(const char* onnxPath,
    const int32_t* batchSizes, int32_t numProfiles,
    const TRT_BuildOptions* options, int32_t deviceId)
  {
    TRT_BuildOptions defaultOpts;
    TRT_InitBuildOptions(&defaultOpts);
    const TRT_BuildOptions* opts = options ? options : &defaultOpts;

    std::string basename = GetBaseName(onnxPath);
    std::string gpuId = GetGPUIdentifier(deviceId);
    uint64_t hash = HashBuildOptions(opts);
    int32_t trtVersion = NV_TENSORRT_VERSION;

    // Build batch sizes string: "b1-b2-...-bN"
    std::string batchStr;
    for (int32_t i = 0; i < numProfiles; ++i)
    {
      if (i > 0) batchStr += "-";
      batchStr += std::to_string(batchSizes[i]);
    }

    char buffer[1024];
    snprintf(buffer, sizeof(buffer), "%s_mp%s_%s_trt%d_%016llx.engine",
      basename.c_str(), batchStr.c_str(), gpuId.c_str(), trtVersion,
      static_cast<unsigned long long>(hash));

    return strdup(buffer);
  }


  TRT_API int32_t TRT_LoadONNXMultiProfileCached(const char* onnxPath,
    const int32_t* batchSizes, int32_t numProfiles,
    const TRT_BuildOptions* options, int32_t deviceId,
    const char* cacheDir, int32_t forceRebuild,
    int32_t* outWasCached, TRT_EngineHandle* outHandles)
  {
    if (outWasCached) *outWasCached = 0;

    TRT_BuildOptions defaultOpts;
    TRT_InitBuildOptions(&defaultOpts);
    const TRT_BuildOptions* opts = options ? options : &defaultOpts;

    // If no cache dir, just build normally
    if (!cacheDir || cacheDir[0] == '\0')
    {
      return TRT_LoadONNXMultiProfile(onnxPath, batchSizes, numProfiles, options, deviceId, outHandles);
    }

    // Generate cache filename
    char* cacheFilename = TRT_GenerateMultiProfileCacheFilename(onnxPath, batchSizes, numProfiles, options, deviceId);
    std::string cachePath = std::string(cacheDir) + "/" + cacheFilename;
    TRT_FreeString(cacheFilename);

    // Handle deviceId
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
        return -1;
      }
    }

    // Try loading from cache.
    // NOTE: every branch that ends up rebuilding logs an explicit reason to avoid a silent rebuilds
    if (forceRebuild)
    {
      fprintf(stderr, "[TensorRT] Cache rebuild forced; ignoring any cached engine: %s\n", cachePath.c_str());
    }
    else if (!FileExists(cachePath.c_str()))
    {
      fprintf(stderr, "[TensorRT] Cache miss (no cached engine file found): %s\n", cachePath.c_str());
    }
    else
    {
      // Check if ONNX file is newer than cache
      struct stat onnxStat, cacheStat;
      bool cacheValid = true;
      if (stat(onnxPath, &onnxStat) == 0 && stat(cachePath.c_str(), &cacheStat) == 0)
      {
        if (onnxStat.st_mtime > cacheStat.st_mtime)
        {
          cacheValid = false;
          fprintf(stderr, "[TensorRT] Cache stale (ONNX is newer than cached engine), rebuilding: %s\n", cachePath.c_str());
        }
      }

      if (cacheValid)
      {
        // Read engine file
        std::ifstream file(cachePath, std::ios::binary | std::ios::ate);
        if (file.is_open())
        {
          std::streamsize size = file.tellg();
          file.seekg(0, std::ios::beg);
          std::vector<char> buffer(size);
          if (file.read(buffer.data(), size))
          {
            file.close();

            // Lock-free deserialize: g_runtime is a single shared IRuntime created once in TRT_Init
            // and never mutated; IRuntime::deserializeCudaEngine is thread-safe, and each concurrent
            // call here targets a distinct deviceId (bound per-thread above) allocating only
            // device-local resources. Dropping g_mutex here lets homogeneous multi-GPU loads
            // deserialize in parallel rather than serializing on the global lock. Only the
            // g_lastError accesses below are briefly locked.
            nvinfer1::ICudaEngine* engine = g_runtime->deserializeCudaEngine(buffer.data(), buffer.size());
            if (engine)
            {
              int32_t result = CreateContextsFromEngine(engine, batchSizes, numProfiles,
                opts->useCudaGraphs != 0, opts->useSpinWait != 0, deviceId, outHandles);
              if (result == 0)
              {
                // Build batch sizes string for logging
                std::string batchDesc;
                for (int32_t p = 0; p < numProfiles; ++p)
                {
                  if (p > 0) batchDesc += ",";
                  batchDesc += std::to_string(batchSizes[p]);
                }
                std::string basename = GetBaseName(onnxPath);
                char msg[512];
                snprintf(msg, sizeof(msg), "[TensorRT] Loading multi-profile %s: batches=[%s], %d profiles (%lld bytes)",
                  basename.c_str(), batchDesc.c_str(), numProfiles, (long long)buffer.size());
                PrintGreen(msg);

                if (outWasCached) *outWasCached = 1;
                { std::lock_guard<std::mutex> lk(g_mutex); g_lastError.clear(); }
                return 0;
              }
              else
              {
                // CreateContextsFromEngine deleted engine on failure
                fprintf(stderr, "[TensorRT WARNING] Cache load FAILED: CreateContextsFromEngine returned %d (%lld bytes), rebuilding: %s\n",
                  result, (long long)buffer.size(), cachePath.c_str());
              }
            }
            else
            {
              fprintf(stderr, "[TensorRT WARNING] Cache load FAILED: deserializeCudaEngine returned null "
                "(cached engine file likely corrupt, truncated, or built by an incompatible TensorRT/GPU), "
                "size=%lld bytes, rebuilding: %s\n", (long long)buffer.size(), cachePath.c_str());
            }
          }
          else
          {
            fprintf(stderr, "[TensorRT WARNING] Cache load FAILED: short read of engine file (got %lld of %lld bytes), rebuilding: %s\n",
              (long long)file.gcount(), (long long)size, cachePath.c_str());
            file.close();
          }
        }
        else
        {
          fprintf(stderr, "[TensorRT WARNING] Cache load FAILED: could not open cached engine file for reading, rebuilding: %s\n",
            cachePath.c_str());
        }
        // Fall through to rebuild if cache load failed
      }
    }

    // Build from ONNX
    int32_t result = TRT_LoadONNXMultiProfile(onnxPath, batchSizes, numProfiles, options, deviceId, outHandles);
    if (result != 0)
    {
      return result;
    }

    // Save to cache (serialize engine from first context)
    auto* firstEc = static_cast<EngineContext*>(outHandles[0]);
    nvinfer1::IHostMemory* serialized = firstEc->engine->serialize();
    if (serialized)
    {
      std::ofstream file(cachePath, std::ios::binary);
      if (file.is_open())
      {
        file.write(static_cast<const char*>(serialized->data()), serialized->size());
        file.close();
      }
      else
      {
        fprintf(stderr, "[TensorRT WARNING] Failed to save multi-profile engine to cache: %s\n", cachePath.c_str());
      }
      delete serialized;
    }

    return 0;
  }


  // Read the cached multi-profile serialized engine (.plan) blob for the given parameters into a
  // heap buffer, WITHOUT deserializing or creating any contexts. Uses the same arch-keyed cache
  // filename as TRT_LoadONNXMultiProfileCached, so homogeneous GPUs resolve to the same file.
  // On a valid, non-stale cache hit: returns 0 and sets *outBuffer (caller frees via TRT_FreeBlob)
  // and *outSize. If there is no cache dir, the file is absent, or it is stale (ONNX newer):
  // returns 1 with *outBuffer null (caller should route through the normal build path). Negative
  // on error.
  TRT_API int32_t TRT_ReadMultiProfileBlobFromCache(const char* onnxPath,
    const int32_t* batchSizes, int32_t numProfiles,
    const TRT_BuildOptions* options, int32_t deviceId,
    const char* cacheDir, char** outBuffer, int64_t* outSize)
  {
    if (outBuffer) *outBuffer = nullptr;
    if (outSize) *outSize = 0;

    if (!onnxPath || !batchSizes || numProfiles <= 0 || !outBuffer || !outSize)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Invalid arguments for ReadMultiProfileBlobFromCache");
      return -1;
    }

    if (!cacheDir || cacheDir[0] == '\0')
    {
      return 1;  // No cache dir: treat as not present
    }

    // Generate cache filename (arch-keyed: identical for homogeneous GPUs).
    char* cacheFilename = TRT_GenerateMultiProfileCacheFilename(onnxPath, batchSizes, numProfiles, options, deviceId);
    std::string cachePath = std::string(cacheDir) + "/" + cacheFilename;
    TRT_FreeString(cacheFilename);

    if (!FileExists(cachePath.c_str()))
    {
      return 1;  // Not present
    }

    // Stale if the ONNX source is newer than the cached engine.
    struct stat onnxStat, cacheStat;
    if (stat(onnxPath, &onnxStat) == 0 && stat(cachePath.c_str(), &cacheStat) == 0)
    {
      if (onnxStat.st_mtime > cacheStat.st_mtime)
      {
        return 1;  // Stale
      }
    }

    std::ifstream file(cachePath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to open cached engine file: " + cachePath);
      return -2;
    }
    std::streamsize size = file.tellg();
    if (size <= 0)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Cached engine file empty: " + cachePath);
      return -3;
    }
    file.seekg(0, std::ios::beg);

    char* buf = static_cast<char*>(malloc(static_cast<size_t>(size)));
    if (!buf)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Out of memory reading cached engine file");
      return -4;
    }
    if (!file.read(buf, size))
    {
      free(buf);
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Short read of cached engine file: " + cachePath);
      return -5;
    }
    file.close();

    *outBuffer = buf;
    *outSize = static_cast<int64_t>(size);
    { std::lock_guard<std::mutex> lock(g_mutex); g_lastError.clear(); }
    return 0;
  }


  // Deserialize an already-in-memory serialized multi-profile engine blob onto a specific device
  // and create N execution contexts (one per profile). Does NOT touch disk and does NOT hold the
  // global lock across the deserialize, so it is safe to call concurrently for distinct deviceIds
  // (the keystone of parallel homogeneous multi-GPU loads). The blob must be the serialized engine
  // for this device's architecture (caller guarantees homogeneity). Returns 0 on success; negative
  // on error. On success, outHandles[0..numProfiles-1] receive the new contexts.
  TRT_API int32_t TRT_DeserializeMultiProfileFromBuffer(const char* buffer, int64_t bufferSize,
    const int32_t* batchSizes, int32_t numProfiles,
    const TRT_BuildOptions* options, int32_t deviceId,
    TRT_EngineHandle* outHandles)
  {
    if (!g_initialized)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("TensorRT not initialized");
      return -1;
    }
    if (!buffer || bufferSize <= 0 || !batchSizes || numProfiles <= 0 || !outHandles)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Invalid arguments for DeserializeMultiProfileFromBuffer");
      return -2;
    }

    TRT_BuildOptions defaultOpts;
    TRT_InitBuildOptions(&defaultOpts);
    const TRT_BuildOptions* opts = options ? options : &defaultOpts;

    // Bind this thread to the target device (per-thread; no global lock needed).
    if (deviceId < 0)
    {
      cudaGetDevice(&deviceId);
    }
    else if (!EnsureDevice(deviceId))
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to set CUDA device " + std::to_string(deviceId));
      return -3;
    }

    // Lock-free deserialize (see rationale in TRT_LoadONNXMultiProfileCached).
    nvinfer1::ICudaEngine* engine = g_runtime->deserializeCudaEngine(buffer, static_cast<size_t>(bufferSize));
    if (!engine)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to deserialize multi-profile engine from buffer");
      return -4;
    }

    int32_t result = CreateContextsFromEngine(engine, batchSizes, numProfiles,
      opts->useCudaGraphs != 0, opts->useSpinWait != 0, deviceId, outHandles);
    if (result != 0)
    {
      // CreateContextsFromEngine deletes the engine on failure.
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to create contexts from buffer-deserialized engine");
      return -5;
    }

    { std::lock_guard<std::mutex> lock(g_mutex); g_lastError.clear(); }
    return 0;
  }


  // Free a buffer returned by TRT_ReadMultiProfileBlobFromCache.
  TRT_API void TRT_FreeBlob(char* buffer)
  {
    if (buffer) free(buffer);
  }


  TRT_API int32_t TRT_LoadMultiProfileEngineFile(const char* enginePath,
    const int32_t* batchSizes, int32_t numProfiles,
    int32_t useCudaGraphs, int32_t useSpinWait, int32_t deviceId,
    TRT_EngineHandle* outHandles)
  {
    if (!enginePath || !batchSizes || numProfiles <= 0 || !outHandles)
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Invalid arguments for LoadMultiProfileEngineFile");
      return -1;
    }

    // Handle deviceId
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
        return -2;
      }
    }

    // Read engine file
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to open engine file: " + std::string(enginePath));
      return -3;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
      file.close();
      std::lock_guard<std::mutex> lock(g_mutex);
      SetError("Failed to read engine file: " + std::string(enginePath));
      return -4;
    }
    file.close();

    // Deserialize engine
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_initialized)
    {
      SetError("TensorRT not initialized");
      return -5;
    }

    nvinfer1::ICudaEngine* engine = g_runtime->deserializeCudaEngine(buffer.data(), buffer.size());
    if (!engine)
    {
      SetError("Failed to deserialize engine from file: " + std::string(enginePath));
      return -6;
    }

    int32_t result = CreateContextsFromEngine(engine, batchSizes, numProfiles,
      useCudaGraphs != 0, useSpinWait != 0, deviceId, outHandles);
    if (result != 0)
    {
      SetError("Failed to create execution contexts from engine file");
      return -7;
    }

    // Build batch sizes string for logging
    std::string batchDesc;
    for (int32_t p = 0; p < numProfiles; ++p)
    {
      if (p > 0) batchDesc += ",";
      batchDesc += std::to_string(batchSizes[p]);
    }
    std::string basename = GetBaseName(enginePath);
    char msg[512];
    snprintf(msg, sizeof(msg), "[TensorRT] Loaded multi-profile engine from file %s: batches=[%s], %d profiles",
      basename.c_str(), batchDesc.c_str(), numProfiles);
    PrintGreen(msg);

    g_lastError.clear();
    return 0;
  }

}
