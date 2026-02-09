/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <cstdint>

#ifdef _WIN32
#define TRT_API __declspec(dllexport)
#else
#define TRT_API __attribute__((visibility("default")))
#endif

extern "C"
{
  // Initialize TensorRT runtime. Returns 0 on success, negative on error.
  TRT_API int32_t TRT_Init();

  // Shutdown TensorRT runtime and free resources. Returns 0 on success.
  TRT_API int32_t TRT_Shutdown();

  // Get TensorRT version as (major * 10000 + minor * 100 + patch).
  TRT_API int32_t TRT_GetVersion();

  // Get last error message (nullptr if no error).
  TRT_API const char* TRT_GetLastError();

  // =========================================================================
  // Engine Management
  // =========================================================================

  // Opaque handle to a TensorRT engine + context pair
  typedef void* TRT_EngineHandle;

  // Build options for TensorRT engine
  struct TRT_BuildOptions
  {
    int32_t builderOptimizationLevel;  // 0-5, default 3
    int32_t tilingOptimizationLevel;   // -1 to use default, 0-5 otherwise
    int32_t useSpinWait;               // 1 = true (default), 0 = false
    int32_t useCudaGraphs;             // 1 = true, 0 = false (default)
    int32_t useFP16;                   // 1 = true (default), 0 = false
    int32_t useBF16;                   // 1 = true, 0 = false (default)
    int32_t useFP8;                    // 1 = true, 0 = false (default)
    int32_t useBest;                   // 1 = true, 0 = false (default) - use best precision
    int32_t minBatchSize;              // Min batch size for optimization profile (0 = use batchSize)
    int32_t optBatchSize;              // Optimal batch size for optimization profile (0 = use batchSize)
    int32_t maxBatchSize;              // Max batch size for optimization profile (0 = use batchSize)
    int32_t fp32PostAttentionNorm;     // 1 = force FP32 for post-attention norm (ln1) layers, 0 = false (default)
    int32_t fp32PostAttentionNormStrict; // 1 = stricter filter: only main encoder ln1, exclude smolgen ln1
    int32_t fp32SmolgenNorm;           // 1 = only smolgen-related ln1 inside attention (the critical layers)
    int32_t refittable;                // 1 = enable refit support (kREFIT_IDENTICAL), 0 = false (default)
  };

  // Initialize build options with defaults
  TRT_API void TRT_InitBuildOptions(TRT_BuildOptions* options);

  // Load ONNX model and build engine with fixed batch size and options.
  // Returns engine handle on success, nullptr on failure.
  TRT_API TRT_EngineHandle TRT_LoadONNXWithOptions(const char* onnxPath, int32_t batchSize,
    const TRT_BuildOptions* options);

  // Load ONNX model on a specific GPU device.
  TRT_API TRT_EngineHandle TRT_LoadONNXOnDevice(const char* onnxPath, int32_t batchSize,
    const TRT_BuildOptions* options, int32_t deviceId);

  // Load ONNX model with default options (legacy, calls TRT_LoadONNXWithOptions internally).
  TRT_API TRT_EngineHandle TRT_LoadONNX(const char* onnxPath, int32_t batchSize);

  // Load a pre-built serialized engine file (.engine).
  // Returns engine handle on success, nullptr on failure.
  TRT_API TRT_EngineHandle TRT_LoadEngineFile(const char* enginePath, int32_t batchSize, int32_t deviceId);

  // Load ONNX with caching support.
  // cacheDir: directory to store/load cached engines (nullptr to disable caching)
  // forceRebuild: if non-zero, rebuild even if cache exists
  // outWasCached: if non-null, set to 1 if loaded from cache, 0 if built fresh
  // Returns engine handle on success, nullptr on failure.
  TRT_API TRT_EngineHandle TRT_LoadONNXCached(const char* onnxPath, int32_t batchSize,
    const TRT_BuildOptions* options, int32_t deviceId,
    const char* cacheDir, int32_t forceRebuild,
    int32_t* outWasCached);

  // Save engine to file. Returns 0 on success, negative on error.
  TRT_API int32_t TRT_SaveEngine(TRT_EngineHandle handle, const char* enginePath);

  // Generate cache filename for given parameters. Caller must free returned string with TRT_FreeString.
  // Format: {basename}_b{min}-{opt}-{max}_{gpuid}_trt{version}_{optionshash}.engine
  // gpuid format: sm{major}{minor}_{smCount}sm (e.g., sm90_132sm for H100)
  TRT_API char* TRT_GenerateCacheFilename(const char* onnxPath, int32_t batchSize,
    const TRT_BuildOptions* options);

  // Generate cache filename for a specific device
  TRT_API char* TRT_GenerateCacheFilenameForDevice(const char* onnxPath, int32_t batchSize,
    const TRT_BuildOptions* options, int32_t deviceId);

  // Free a string returned by TRT_GenerateCacheFilename.
  TRT_API void TRT_FreeString(char* str);

  // Get/set current CUDA device
  TRT_API int32_t TRT_GetDevice();
  TRT_API int32_t TRT_SetDevice(int32_t deviceId);

  // Check if GPU has integrated memory (unified memory like Tegra/Jetson/GB10).
  // Returns 1 if integrated, 0 if discrete, -1 on error.
  TRT_API int32_t TRT_IsIntegratedGPU(int32_t deviceId);

  // Get the number of streaming multiprocessors (SMs) on a GPU device.
  // Returns SM count on success, -1 on error.
  TRT_API int32_t TRT_GetMultiProcessorCount(int32_t deviceId);

  // Engine inspection - get layer information as JSON string
  // Returns allocated string that must be freed with TRT_FreeString, or nullptr on error
  TRT_API char* TRT_GetEngineLayerInfo(TRT_EngineHandle handle, int32_t layerIndex);

  // Get number of layers in engine
  TRT_API int32_t TRT_GetNumLayers(TRT_EngineHandle handle);

  // Get engine info summary (includes precision info)
  TRT_API char* TRT_GetEngineSummary(TRT_EngineHandle handle);

  // Free engine and associated resources.
  TRT_API void TRT_FreeEngine(TRT_EngineHandle handle);

  // =========================================================================
  // Tensor Info
  // =========================================================================

  // Get number of input/output tensors.
  TRT_API int32_t TRT_GetNumInputs(TRT_EngineHandle handle);
  TRT_API int32_t TRT_GetNumOutputs(TRT_EngineHandle handle);

  // Get tensor name by index. Returns nullptr if invalid.
  TRT_API const char* TRT_GetInputName(TRT_EngineHandle handle, int32_t index);
  TRT_API const char* TRT_GetOutputName(TRT_EngineHandle handle, int32_t index);

  // Get total element count for a tensor (includes batch dimension).
  TRT_API int64_t TRT_GetInputSize(TRT_EngineHandle handle, int32_t index);
  TRT_API int64_t TRT_GetOutputSize(TRT_EngineHandle handle, int32_t index);

  // Get element size in bytes for a tensor (1=INT8/UINT8, 2=FP16/BF16, 4=FP32/INT32, 8=INT64).
  TRT_API int32_t TRT_GetInputElementSize(TRT_EngineHandle handle, int32_t index);
  TRT_API int32_t TRT_GetOutputElementSize(TRT_EngineHandle handle, int32_t index);

  // Output tensor info struct for retrieving multiple outputs
  struct TRT_OutputTensorInfo
  {
    const char* name;      // Tensor name (pointer to internal string, do not free)
    int64_t offset;        // Offset in elements from start of output buffer
    int64_t size;          // Size in elements
  };

  // Get info for all output tensors. Fills outputInfo array (up to maxOutputs).
  // Returns actual number of outputs, or -1 on error.
  TRT_API int32_t TRT_GetOutputTensorInfo(TRT_EngineHandle handle,
    TRT_OutputTensorInfo* outputInfo,
    int32_t maxOutputs);

  // =========================================================================
  // Inference
  // =========================================================================

  // Run inference. Input/output arrays are device pointers (GPU memory).
  // inputPtrs and outputPtrs are arrays of device pointers.
  // Returns 0 on success.
  TRT_API int32_t TRT_Infer(TRT_EngineHandle handle,
    void** inputPtrs, int32_t numInputs,
    void** outputPtrs, int32_t numOutputs);

  // Run inference with managed memory (host pointers, copies handled internally).
  // This is simpler but slower due to host<->device copies each call.
  // Data is assumed to be FP16 (Half precision, 2 bytes per element).
  // Returns 0 on success.
  TRT_API int32_t TRT_InferHost(TRT_EngineHandle handle,
    void* inputData, int64_t inputSize,
    void* outputData, int64_t outputSize);

  // =========================================================================
  // GPU Memory Helpers
  // =========================================================================

  // Allocate GPU memory. Returns device pointer, nullptr on failure.
  TRT_API void* TRT_AllocGPU(int64_t bytes);

  // Free GPU memory.
  TRT_API void TRT_FreeGPU(void* ptr);

  // Allocate pinned (page-locked) host memory for faster transfers.
  TRT_API void* TRT_AllocPinned(int64_t bytes);

  // Free pinned host memory.
  TRT_API void TRT_FreePinned(void* ptr);

  // Copy data between host and device.
  TRT_API int32_t TRT_CopyToGPU(void* dst, const void* src, int64_t bytes);
  TRT_API int32_t TRT_CopyFromGPU(void* dst, const void* src, int64_t bytes);

  // Async copy using engine's stream (for double-buffering).
  TRT_API int32_t TRT_CopyToGPUAsync(TRT_EngineHandle handle, void* dst, const void* src, int64_t bytes);
  TRT_API int32_t TRT_CopyFromGPUAsync(TRT_EngineHandle handle, void* dst, const void* src, int64_t bytes);

  // Synchronize engine's CUDA stream.
  TRT_API int32_t TRT_SyncStream(TRT_EngineHandle handle);

  // Synchronize all CUDA operations (device-wide).
  TRT_API int32_t TRT_Synchronize();

  // Synchronize all operations on a specific device.
  TRT_API int32_t TRT_SynchronizeDevice(int32_t deviceId);

  // =========================================================================
  // Advanced Inference (for double-buffering)
  // =========================================================================

  // Run inference without sync - caller must sync manually.
  // streamIdx: 0 or 1 for two-stream double-buffering
  TRT_API int32_t TRT_InferOnStream(TRT_EngineHandle handle, int32_t streamIdx,
    void* gpuInput, void* gpuOutput);

  // Run inference on stream with CUDA graph support (for Exact mode engines).
  // On first call per stream, captures a CUDA graph. Subsequent calls replay the graph.
  // For engines with useCudaGraphs=false, this behaves like TRT_InferOnStream.
  TRT_API int32_t TRT_InferOnStreamWithGraph(TRT_EngineHandle handle, int32_t streamIdx,
    void* gpuInput, void* gpuOutput);

  // Run inference on stream with dynamic batch size (for range-mode engines).
  // Sets input shape to actualBatchSize before inference.
  TRT_API int32_t TRT_InferOnStreamDynamic(TRT_EngineHandle handle, int32_t streamIdx,
    void* gpuInput, void* gpuOutput,
    int32_t actualBatchSize);

  // Async copy on specified stream
  TRT_API int32_t TRT_CopyToGPUOnStream(TRT_EngineHandle handle, int32_t streamIdx,
    void* dst, const void* src, int64_t bytes);
  TRT_API int32_t TRT_CopyFromGPUOnStream(TRT_EngineHandle handle, int32_t streamIdx,
    void* dst, const void* src, int64_t bytes);

  // Sync specific stream
  TRT_API int32_t TRT_SyncStreamIdx(TRT_EngineHandle handle, int32_t streamIdx);

  // Legacy single-stream versions
  TRT_API int32_t TRT_InferAsync(TRT_EngineHandle handle, void* gpuInput, void* gpuOutput);
  TRT_API void* TRT_GetInputBuffer(TRT_EngineHandle handle, int32_t index);
  TRT_API void* TRT_GetOutputBuffer(TRT_EngineHandle handle, int32_t index);

  // =========================================================================
  // Dynamic Batch Size Inference (for range-mode engines)
  // =========================================================================

  // Run inference with dynamic batch size (for engines built with min/max batch range).
  // actualBatchSize: the actual number of positions to process (must be <= engine's max batch)
  // inputSize/outputSize: sizes in ELEMENTS for the actual batch (not the engine's max batch)
  // Returns 0 on success.
  TRT_API int32_t TRT_InferHostDynamic(TRT_EngineHandle handle,
    void* inputData, int64_t inputSize,
    void* outputData, int64_t outputSize,
    int32_t actualBatchSize);

  // Same as TRT_InferHostDynamic but for byte (INT8/UINT8) inputs.
  TRT_API int32_t TRT_InferHostBytesDynamic(TRT_EngineHandle handle,
    void* inputData, int64_t inputSize,
    void* outputData, int64_t outputSize,
    int32_t actualBatchSize);

  // Get the batch size the engine was built with (max batch for range engines).
  TRT_API int32_t TRT_GetEngineBatchSize(TRT_EngineHandle handle);

  // =========================================================================
  // Multi-Profile Engine (shared weights, N execution contexts)
  // =========================================================================

  // Build a single engine with N optimization profiles (one per batch size),
  // then create N independent execution contexts sharing that engine.
  // outHandles must point to an array of numProfiles TRT_EngineHandle slots.
  // Returns 0 on success, negative on error.
  TRT_API int32_t TRT_LoadONNXMultiProfile(const char* onnxPath,
    const int32_t* batchSizes, int32_t numProfiles,
    const TRT_BuildOptions* options, int32_t deviceId,
    TRT_EngineHandle* outHandles);

  // Multi-profile engine with caching support.
  // Serializes/deserializes a single engine file containing all profiles.
  TRT_API int32_t TRT_LoadONNXMultiProfileCached(const char* onnxPath,
    const int32_t* batchSizes, int32_t numProfiles,
    const TRT_BuildOptions* options, int32_t deviceId,
    const char* cacheDir, int32_t forceRebuild,
    int32_t* outWasCached, TRT_EngineHandle* outHandles);

  // Generate cache filename for a multi-profile engine.
  // Caller must free returned string with TRT_FreeString.
  TRT_API char* TRT_GenerateMultiProfileCacheFilename(const char* onnxPath,
    const int32_t* batchSizes, int32_t numProfiles,
    const TRT_BuildOptions* options, int32_t deviceId);

  // Check if this engine uses CUDA graphs for inference.
  // Returns 1 if enabled, 0 if disabled, -1 on error.
  TRT_API int32_t TRT_UsesCudaGraphs(TRT_EngineHandle handle);

  // Check if a stream graph has already been captured.
  // Returns 1 if captured, 0 if not captured, -1 on error.
  TRT_API int32_t TRT_IsStreamGraphCaptured(TRT_EngineHandle handle, int32_t streamIdx);

  // Get input elements per position (total input size / batch size).
  TRT_API int64_t TRT_GetInputElementsPerPosition(TRT_EngineHandle handle);

  // Get output elements per position (total output size / batch size).
  TRT_API int64_t TRT_GetOutputElementsPerPosition(TRT_EngineHandle handle);

  // =========================================================================
  // Weight Refitting (for refittable engines)
  // =========================================================================

  // Set weights for a named tensor in the engine. Requires engine built with refittable=1.
  // weightTensorName: name of the weight tensor to update (e.g., "conv1.weight")
  // weights: pointer to FP16 weight data
  // numElements: number of elements in the weights array
  // Returns 0 on success, negative on error.
  TRT_API int32_t TRT_SetNamedWeights(TRT_EngineHandle handle, const char* weightTensorName,
    const void* weights, int64_t numElements);

  // Refit the engine after setting weights. Must be called after TRT_SetNamedWeights
  // to apply the weight changes. Returns 0 on success, negative on error.
  TRT_API int32_t TRT_RefitEngine(TRT_EngineHandle handle);
}
