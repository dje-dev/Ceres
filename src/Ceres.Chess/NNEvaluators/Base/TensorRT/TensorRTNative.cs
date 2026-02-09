#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives

using System;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess.NNEvaluators.TensorRT;

/// <summary>
/// P/Invoke declarations for the TensorRT native wrapper library.
/// </summary>
internal static partial class TensorRTNative
{
  // Platform-agnostic library name - .NET will resolve to libTensorRTWrapper.so on Linux
  // and TensorRTWrapper.dll on Windows via the runtimes/{rid}/native/ directory structure.
  private const string LibraryName = "TensorRTWrapper";

  // =========================================================================
  // Core functions
  // =========================================================================

  [LibraryImport(LibraryName, EntryPoint = "TRT_Init")]
  internal static partial int Init();

  [LibraryImport(LibraryName, EntryPoint = "TRT_Shutdown")]
  internal static partial int Shutdown();

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetVersion")]
  internal static partial int GetVersion();

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetLastError")]
  internal static partial IntPtr GetLastError();

  // =========================================================================
  // Build options
  // =========================================================================

  [LibraryImport(LibraryName, EntryPoint = "TRT_InitBuildOptions")]
  internal static partial void InitBuildOptions(ref TensorRTBuildOptions options);

  // =========================================================================
  // Engine management
  // =========================================================================

  [LibraryImport(LibraryName, EntryPoint = "TRT_LoadONNX", StringMarshalling = StringMarshalling.Utf8)]
  internal static partial IntPtr LoadONNX(string onnxPath, int batchSize);

  [LibraryImport(LibraryName, EntryPoint = "TRT_LoadONNXWithOptions", StringMarshalling = StringMarshalling.Utf8)]
  internal static partial IntPtr LoadONNXWithOptions(string onnxPath, int batchSize, ref TensorRTBuildOptions options);

  [LibraryImport(LibraryName, EntryPoint = "TRT_LoadONNXOnDevice", StringMarshalling = StringMarshalling.Utf8)]
  internal static partial IntPtr LoadONNXOnDevice(string onnxPath, int batchSize, ref TensorRTBuildOptions options, int deviceId);

  [LibraryImport(LibraryName, EntryPoint = "TRT_LoadEngineFile", StringMarshalling = StringMarshalling.Utf8)]
  internal static partial IntPtr LoadEngineFile(string enginePath, int batchSize, int deviceId);

  [LibraryImport(LibraryName, EntryPoint = "TRT_LoadONNXCached", StringMarshalling = StringMarshalling.Utf8)]
  internal static partial IntPtr LoadONNXCached(string onnxPath, int batchSize, ref TensorRTBuildOptions options,
                                                 int deviceId, string cacheDir, int forceRebuild, out int wasCached);

  [LibraryImport(LibraryName, EntryPoint = "TRT_SaveEngine", StringMarshalling = StringMarshalling.Utf8)]
  internal static partial int SaveEngine(IntPtr handle, string enginePath);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GenerateCacheFilename", StringMarshalling = StringMarshalling.Utf8)]
  internal static partial IntPtr GenerateCacheFilename(string onnxPath, int batchSize, ref TensorRTBuildOptions options);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GenerateCacheFilenameForDevice", StringMarshalling = StringMarshalling.Utf8)]
  internal static partial IntPtr GenerateCacheFilenameForDevice(string onnxPath, int batchSize, ref TensorRTBuildOptions options, int deviceId);

  [LibraryImport(LibraryName, EntryPoint = "TRT_FreeString")]
  internal static partial void FreeString(IntPtr str);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetDevice")]
  internal static partial int GetDevice();

  [LibraryImport(LibraryName, EntryPoint = "TRT_SetDevice")]
  internal static partial int SetDevice(int deviceId);

  /// <summary>
  /// Check if GPU has integrated memory (unified memory like Tegra/Jetson/GB10).
  /// Returns 1 if integrated, 0 if discrete, -1 on error.
  /// </summary>
  [LibraryImport(LibraryName, EntryPoint = "TRT_IsIntegratedGPU")]
  internal static partial int IsIntegratedGPU(int deviceId);

  /// <summary>
  /// Get the number of streaming multiprocessors (SMs) on a GPU device.
  /// Returns SM count on success, -1 on error.
  /// </summary>
  [LibraryImport(LibraryName, EntryPoint = "TRT_GetMultiProcessorCount")]
  internal static partial int GetMultiProcessorCount(int deviceId);

  /// <summary>
  /// Get the device name string for a GPU device (e.g., "NVIDIA RTX PRO 6000 Blackwell").
  /// </summary>
  [LibraryImport(LibraryName, EntryPoint = "TRT_GetDeviceName")]
  internal static partial IntPtr GetDeviceNamePtr(int deviceId);

  [LibraryImport(LibraryName, EntryPoint = "TRT_FreeEngine")]
  internal static partial void FreeEngine(IntPtr handle);

  // =========================================================================
  // Engine inspection
  // =========================================================================

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetNumLayers")]
  internal static partial int GetNumLayers(IntPtr handle);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetEngineLayerInfo")]
  internal static partial IntPtr GetEngineLayerInfo(IntPtr handle, int layerIndex);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetEngineSummary")]
  internal static partial IntPtr GetEngineSummary(IntPtr handle);

  // =========================================================================
  // Tensor info
  // =========================================================================

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetNumInputs")]
  internal static partial int GetNumInputs(IntPtr handle);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetNumOutputs")]
  internal static partial int GetNumOutputs(IntPtr handle);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetInputName")]
  internal static partial IntPtr GetInputName(IntPtr handle, int index);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetOutputName")]
  internal static partial IntPtr GetOutputName(IntPtr handle, int index);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetInputSize")]
  internal static partial long GetInputSize(IntPtr handle, int index);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetOutputSize")]
  internal static partial long GetOutputSize(IntPtr handle, int index);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetInputElementSize")]
  internal static partial int GetInputElementSize(IntPtr handle, int index);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetOutputElementSize")]
  internal static partial int GetOutputElementSize(IntPtr handle, int index);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetOutputTensorInfo")]
  internal static unsafe partial int GetOutputTensorInfo(IntPtr handle, NativeOutputTensorInfo* outputInfo, int maxOutputs);

  // =========================================================================
  // Inference (FP16 - Half precision)
  // =========================================================================

  [LibraryImport(LibraryName, EntryPoint = "TRT_InferHost")]
  internal static unsafe partial int InferHost(IntPtr handle, Half* inputData, long inputSize,
                                                Half* outputData, long outputSize);

  [LibraryImport(LibraryName, EntryPoint = "TRT_InferHost")]
  internal static unsafe partial int InferHostBytes(IntPtr handle, byte* inputData, long inputSize,
                                                     Half* outputData, long outputSize);

  // =========================================================================
  // GPU memory helpers
  // =========================================================================

  [LibraryImport(LibraryName, EntryPoint = "TRT_AllocGPU")]
  internal static partial IntPtr AllocGPU(long bytes);

  [LibraryImport(LibraryName, EntryPoint = "TRT_FreeGPU")]
  internal static partial void FreeGPU(IntPtr ptr);

  [LibraryImport(LibraryName, EntryPoint = "TRT_CopyToGPU")]
  internal static unsafe partial int CopyToGPU(IntPtr dst, float* src, long bytes);

  [LibraryImport(LibraryName, EntryPoint = "TRT_CopyFromGPU")]
  internal static unsafe partial int CopyFromGPU(float* dst, IntPtr src, long bytes);

  [LibraryImport(LibraryName, EntryPoint = "TRT_Infer")]
  internal static unsafe partial int Infer(IntPtr handle, IntPtr* inputPtrs, int numInputs,
                                            IntPtr* outputPtrs, int numOutputs);

  [LibraryImport(LibraryName, EntryPoint = "TRT_Synchronize")]
  internal static partial int Synchronize();

  [LibraryImport(LibraryName, EntryPoint = "TRT_SynchronizeDevice")]
  internal static partial int SynchronizeDevice(int deviceId);

  // =========================================================================
  // Pinned memory
  // =========================================================================

  [LibraryImport(LibraryName, EntryPoint = "TRT_AllocPinned")]
  internal static partial IntPtr AllocPinned(long bytes);

  [LibraryImport(LibraryName, EntryPoint = "TRT_FreePinned")]
  internal static partial void FreePinned(IntPtr ptr);

  // =========================================================================
  // Async operations
  // =========================================================================

  [LibraryImport(LibraryName, EntryPoint = "TRT_CopyToGPUAsync")]
  internal static partial int CopyToGPUAsync(IntPtr handle, IntPtr dst, IntPtr src, long bytes);

  [LibraryImport(LibraryName, EntryPoint = "TRT_CopyFromGPUAsync")]
  internal static partial int CopyFromGPUAsync(IntPtr handle, IntPtr dst, IntPtr src, long bytes);

  [LibraryImport(LibraryName, EntryPoint = "TRT_SyncStream")]
  internal static partial int SyncStream(IntPtr handle);

  [LibraryImport(LibraryName, EntryPoint = "TRT_InferAsync")]
  internal static partial int InferAsync(IntPtr handle, IntPtr gpuInput, IntPtr gpuOutput);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetInputBuffer")]
  internal static partial IntPtr GetInputBuffer(IntPtr handle, int index);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetOutputBuffer")]
  internal static partial IntPtr GetOutputBuffer(IntPtr handle, int index);

  // =========================================================================
  // Two-stream double-buffering
  // =========================================================================

  [LibraryImport(LibraryName, EntryPoint = "TRT_InferOnStream")]
  internal static partial int InferOnStream(IntPtr handle, int streamIdx, IntPtr gpuInput, IntPtr gpuOutput);

  [LibraryImport(LibraryName, EntryPoint = "TRT_InferOnStreamWithGraph")]
  internal static partial int InferOnStreamWithGraph(IntPtr handle, int streamIdx, IntPtr gpuInput, IntPtr gpuOutput);

  [LibraryImport(LibraryName, EntryPoint = "TRT_InferOnStreamDynamic")]
  internal static partial int InferOnStreamDynamic(IntPtr handle, int streamIdx, IntPtr gpuInput, IntPtr gpuOutput, int actualBatchSize);

  [LibraryImport(LibraryName, EntryPoint = "TRT_CopyToGPUOnStream")]
  internal static partial int CopyToGPUOnStream(IntPtr handle, int streamIdx, IntPtr dst, IntPtr src, long bytes);

  [LibraryImport(LibraryName, EntryPoint = "TRT_CopyFromGPUOnStream")]
  internal static partial int CopyFromGPUOnStream(IntPtr handle, int streamIdx, IntPtr dst, IntPtr src, long bytes);

  [LibraryImport(LibraryName, EntryPoint = "TRT_SyncStreamIdx")]
  internal static partial int SyncStreamIdx(IntPtr handle, int streamIdx);

  // =========================================================================
  // Dynamic Batch Size Inference (for range-mode engines)
  // =========================================================================

  [LibraryImport(LibraryName, EntryPoint = "TRT_InferHostDynamic")]
  internal static unsafe partial int InferHostDynamic(IntPtr handle, Half* inputData, long inputSize,
                                                       Half* outputData, long outputSize, int actualBatchSize);

  [LibraryImport(LibraryName, EntryPoint = "TRT_InferHostBytesDynamic")]
  internal static unsafe partial int InferHostBytesDynamic(IntPtr handle, byte* inputData, long inputSize,
                                                            Half* outputData, long outputSize, int actualBatchSize);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetEngineBatchSize")]
  internal static partial int GetEngineBatchSize(IntPtr handle);

  [LibraryImport(LibraryName, EntryPoint = "TRT_UsesCudaGraphs")]
  internal static partial int UsesCudaGraphs(IntPtr handle);

  [LibraryImport(LibraryName, EntryPoint = "TRT_IsStreamGraphCaptured")]
  internal static partial int IsStreamGraphCaptured(IntPtr handle, int streamIdx);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetInputElementsPerPosition")]
  internal static partial long GetInputElementsPerPosition(IntPtr handle);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GetOutputElementsPerPosition")]
  internal static partial long GetOutputElementsPerPosition(IntPtr handle);

  // =========================================================================
  // Multi-Profile Engine (shared weights, N execution contexts)
  // =========================================================================

  [LibraryImport(LibraryName, EntryPoint = "TRT_LoadONNXMultiProfile", StringMarshalling = StringMarshalling.Utf8)]
  internal static unsafe partial int LoadONNXMultiProfile(string onnxPath,
      int* batchSizes, int numProfiles,
      ref TensorRTBuildOptions options, int deviceId,
      IntPtr* outHandles);

  [LibraryImport(LibraryName, EntryPoint = "TRT_LoadONNXMultiProfileCached", StringMarshalling = StringMarshalling.Utf8)]
  internal static unsafe partial int LoadONNXMultiProfileCached(string onnxPath,
      int* batchSizes, int numProfiles,
      ref TensorRTBuildOptions options, int deviceId,
      string cacheDir, int forceRebuild,
      out int wasCached, IntPtr* outHandles);

  [LibraryImport(LibraryName, EntryPoint = "TRT_GenerateMultiProfileCacheFilename", StringMarshalling = StringMarshalling.Utf8)]
  internal static unsafe partial IntPtr GenerateMultiProfileCacheFilename(string onnxPath,
      int* batchSizes, int numProfiles,
      ref TensorRTBuildOptions options, int deviceId);

  // =========================================================================
  // Weight Refitting (for refittable engines)
  // =========================================================================

  [LibraryImport(LibraryName, EntryPoint = "TRT_SetNamedWeights", StringMarshalling = StringMarshalling.Utf8)]
  internal static unsafe partial int SetNamedWeights(IntPtr handle, string weightTensorName,
                                                      Half* weights, long numElements);

  [LibraryImport(LibraryName, EntryPoint = "TRT_RefitEngine")]
  internal static partial int RefitEngine(IntPtr handle);

  // =========================================================================
  // Helper methods
  // =========================================================================

  internal static string GetLastErrorString()
  {
    IntPtr ptr = GetLastError();
    return ptr == IntPtr.Zero ? null : Marshal.PtrToStringAnsi(ptr);
  }

  internal static string PtrToString(IntPtr ptr)
  {
    return ptr == IntPtr.Zero ? null : Marshal.PtrToStringAnsi(ptr);
  }

  /// <summary>
  /// Get the device name string for a GPU device.
  /// </summary>
  internal static string GetDeviceName(int deviceId)
  {
    IntPtr ptr = GetDeviceNamePtr(deviceId);
    return ptr == IntPtr.Zero ? "Unknown" : Marshal.PtrToStringAnsi(ptr);
  }
}
