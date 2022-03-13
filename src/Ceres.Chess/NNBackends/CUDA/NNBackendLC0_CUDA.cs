﻿#region License notice

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
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;

using ManagedCuda;
using ManagedCuda.CudaBlas;
using Pblczero;

using Ceres.Base.Benchmarking;
using Ceres.Base.CUDA;
using Ceres.Base.DataTypes;
using System.Threading;
using System.Diagnostics;
using Math = System.Math;

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  // Performance benefits of using graph API on high-end hardware (A100):
  //   - for T60 (384 filters) circa 5% improvement at small batch sizes (1 to 16)
  //   - for T75 (192 filters) circa 15% improvement at small/medium batch sizes (1 to 64)
  //   - for T74 (128 filters) circa 20% improvement at small/medium batch sizes (1 to 128)
  // These benefits may be most important with training games, and when testing engines.
  // Also smaller batches may then be 
  // Gains from:
  //   (1) copy less data back from GPU (new kernel extracts only the legal moves from policy array)
  //   (2) async copies with pinned memory
  //   (3) use Graph API (with a cache of different sizes)
  //   (4) for special case of batch size less than 4, always use use 4 or 5 instead (dramatic speedup)
  //   (5) parallel layer initialization for faster load
  //   (6) true parallelism of executors working on single query
  //       (via "dual overlapping executors" with separate CUDA streams)
  //   (7) allow extraction of activations of intermediate layers for research or transparency purposes
  //   (8) capture CUDA kernel timing of all layers and dump if requested for diagnostics
  //   (9) share layers/weights across different evaluator instances for reduced memory


  /// <summary>
  /// Neural network backend for LC0 weights files 
  /// (via CUDA directly in C#).
  /// </summary>
  public partial class NNBackendLC0_CUDA : IDisposable
  {
    #region Internal options

    const bool DIAGNOSTIC = false;

    /// <summary>
    /// If the NVIDIA BLAS LT library should be used (experimental).
    /// </summary>
    public const bool BLASLT = false;
    public const bool BLASLT_USE_LOOP = false;
    public static int BLASLT_N = 0;

    /// <summary>
    /// Currently only FUSED is supported.
    /// </summary>
    public const bool USE_FUSED = true;

    #endregion


    /// <summary>
    /// If the layer-wise characteristics and timing statistics
    /// should be output after each batch is evaluated.
    /// </summary>
    public readonly bool DumpTiming = false;


    /// <summary>
    /// If CUDA graph features enabled.
    /// </summary>
    public bool UseGraphs => EnableCUDAGraphs && !DumpTiming;


    /// <summary>
    /// ID of GPU to which this backend is attached.
    /// </summary>
    public readonly int GPUID;

    /// <summary>
    /// If certain hidden layer activations should be recorded.
    /// </summary>
    public readonly bool SaveActivations;

    /// <summary>
    /// Maximum number of positions per batch supported.
    /// </summary>
    public readonly int MaxBatchSize;

    /// <summary>
    /// Optionally another NNBackendCUDA already initialized
    /// which shares same parameters (from which weights can be reused).
    /// </summary>
    NNBackendLC0_CUDA ReferenceBackend;

    #region Optional concurrency management for results buffer

    bool asyncMode = false;
    ManualResetEventSlim eventOkToFillResults = null;
    public ManualResetEventSlim InputsCopyToDeviceFinished;

    public bool AsyncMode
    {
      get => asyncMode;
      set
      {
        if (value)
        {
          if (eventOkToFillResults == null)
          {
            eventOkToFillResults = new ManualResetEventSlim(false);
            InputsCopyToDeviceFinished = new ManualResetEventSlim(false);
          }

          eventOkToFillResults.Reset();
          asyncMode = true;
        }
        else
        {
          asyncMode = false;
        }
      }
    }

    public void SetOkToFillResults() => eventOkToFillResults.Set();

    #endregion

    internal NNBackendInputOutput inputOutput;

    public float LargestLayerWeightsScale { private set; get; }


    LC0LegacyWeights weights;

    /// <summary>
    /// The set of underlying weights used for the network.
    /// </summary>
    public LC0LegacyWeights Weights
    {
      get
      {
        if (!SaveActivations)
        {
          throw new Exception("Weights are only available when saveActivation is set true in constructor.");
        }
        else
        {
          return weights;
        }
      }
    }


    #region Cuda context

    NNInputCudaVariables networkInputs;
    NNOutputCudaVariables networkOutputs;

    NNBackendExecContext ExecContext;

    CudaDeviceProperties deviceProperties;

    NNBackendCUDAGraphSet graphSet;

    // For unknown reasons very small batches (e.g. 1)
    // run much more slowly (e.g. 1/2 speed or worse) on the device than larger ones.
    // Therefore we always run network with a certain minimum
    // (some of which will be padded entries which are subsequently ignored).
    const int MIN_BATCH_SIZE = 5;

    #endregion

    #region Cuda memory

    long scratchSizeBytes;

    internal CUDAPinnedMemory<FP16> mlhOutputBuffer;
    internal CUDAPinnedMemory<FP16> wdlOutputBuffer;
    internal CUDAPinnedMemory<FP16> valueHeadFC2OutputBuffer;

    #endregion

    /// <summary>
    /// If win/draw/loss head present.
    /// </summary>
    public bool HasWDL;

    /// <summary>
    /// If moves left head present.
    /// </summary>
    public bool HasMLH;

    /// <summary>
    /// Number of residual blocks.
    /// </summary>
    public int NumBlocks;

    /// <summary>
    /// Number of convolutional filters.
    /// </summary>
    public int NumFilters;

    /// <summary>
    /// The graph of network layers.
    /// </summary>
    NNBackendCUDALayers Layers;


    /// <summary>
    /// If the CUDA graphs feature should be used.
    /// 
    /// Enabling improves performance considerably (sometimes 10% to 20%)
    /// on small to medium-sized batches (e.g. less than 100)
    /// especially when using more recent hardware (e.g. Ampere)
    /// and more recent CUDA releases (e.g. at least version 11.3).
    /// </summary>
    /// </summary>
    public readonly bool EnableCUDAGraphs;

    public const int DEFAULT_MAX_BATCH_SIZE = 1024;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="gpuID"></param>
    /// <param name="net"></param>
    /// <param name="saveActivations"></param>
    /// <param name="maxBatchSize"></param>
    /// <param name="dumpTiming"></param>
    /// <param name="enableCUDAGraphs"></param>
    /// <param name="graphBatchSizeDivisor"></param>
    /// <param name="referenceBackend"></param>
    public NNBackendLC0_CUDA(int gpuID, Net net, bool saveActivations = false, 
                             int maxBatchSize = DEFAULT_MAX_BATCH_SIZE,
                             bool dumpTiming = false,
                             bool enableCUDAGraphs = true,
                             int graphBatchSizeDivisor = 1,
                             NNBackendLC0_CUDA referenceBackend = null)
    {
      GPUID = gpuID;
      SaveActivations = saveActivations;
      MaxBatchSize = maxBatchSize;
      ReferenceBackend = referenceBackend;
      DumpTiming = dumpTiming;
      EnableCUDAGraphs = enableCUDAGraphs;

      try
      {
        InitCUDAContextAndTakeWriteLock();

        InitGPUVars();

        InitNetwork(net);
      }
      catch (Exception e) 
      {
        Console.WriteLine("Error when initializing CUDA. Did you install NVidia's CUDA? https://developer.nvidia.com/cuda-zone");
        Console.WriteLine(e);
        Console.WriteLine(e.StackTrace);
      }
      finally
      {
        ExecContext.Device.GraphCaptureRWLock.ExitWriteLock();
      }

      if (UseGraphs)
      {
        BuildGraphSet(graphBatchSizeDivisor);
      }
    }

    public int DeviceComputeCapabilityMajor => deviceProperties.ComputeCapability.Major;
    
    private void InitCUDAContextAndTakeWriteLock()
    {
      CUDADevice deviceContext = null;
      using (new TimingBlock("InitCUDA", DIAGNOSTIC ? TimingBlock.LoggingType.ConsoleWithMemoryTracking : TimingBlock.LoggingType.None))
      {
        deviceContext = CUDADevice.GetContext(GPUID);
        deviceContext.GraphCaptureRWLock.EnterWriteLock();
        deviceContext.SetCurrent();
      }

      // Check if the GPU support FP16.
      deviceProperties = deviceContext.Context.GetDeviceInfo();
      int minor = deviceProperties.DriverVersion.Minor;
      int major = deviceProperties.DriverVersion.Major;
      if (major < 7 && !(major == 6 && minor == 0) && !(major == 5 && minor == 3))
      {
        throw new Exception("Selected GPU does not natively support FP16, Ceres CUDA backend not compatible.");
      }
      else
      {
        Console.WriteLine($"CUDA device {GPUID}:  { deviceProperties.DeviceName}" 
                        + $" Compute: {deviceProperties.ComputeCapability.Major}.{deviceProperties.ComputeCapability.Minor}" 
                        + $" SMs: { deviceProperties.MultiProcessorCount } Mem: { deviceProperties.TotalGlobalMemory / (1024 * 1024 * 1024) }gb");
      }
      // showInfo();

      CudaBlas cuBlas = null;
      CudaBlasLTHandle ltHandle = default;

      CudaStream stream = new CudaStream();
      CudaStream stream2 = new CudaStream();

      using (new TimingBlock("Init BLAS", DIAGNOSTIC ? TimingBlock.LoggingType.ConsoleWithMemoryTracking : TimingBlock.LoggingType.None))
      {
        cuBlas = new CudaBlas(stream.Stream);
        cuBlas.MathMode = ManagedCuda.CudaBlas.Math.TensorOpMath;

        if (BLASLT)
        {
          throw new NotImplementedException();
          //CUDABlasLTNativeMethods.cublasLtCreate(ref ltHandle);
        }
      }

      ExecContext = new (deviceContext, stream, stream2, cuBlas, ltHandle, GetType().Assembly, DumpTiming, deviceProperties.MaxSharedMemoryPerBlockOptin);
    }

    void SetCUDAState()
    {
      // Inherit existing (typically default) context
      ExecContext.Device.SetCurrent();
      ExecContext.CuBlas.Stream = ExecContext.Stream.Stream;
    }

    public void InitNetwork(Net net)
    {
      weights = new LC0LegacyWeights(net.Weights);
      LargestLayerWeightsScale = weights.LargestScaleSeen;

      ExecContext.ReferenceLayers = ReferenceBackend?.Layers;
      Layers = new NNBackendCUDALayers(ExecContext, DeviceComputeCapabilityMajor, 
                                       net, weights, SaveActivations, ReferenceBackend?.Layers);

      CheckNetworkCompatible(net);

      PrepareForNetwork(net, weights);

      inputOutput = new NNBackendInputOutput(MaxBatchSize, HasWDL, HasMLH);

      AllocateGPUMemory(net, weights);

      Layers.BuildNetworkAndLoadWeights(ExecContext, weights, NNBackendInputOutput.NUM_INPUT_PLANES);

      if (!SaveActivations)
      {
        // Release weights to reduce memory usage.
        weights = null;
      }
    }


    private void PrepareForNetwork(Net net, LC0LegacyWeights weights)
    {
      NumFilters = weights.input.biases.Length;
      NumBlocks = weights.residual.Length;

      if (net.Format.NetworkFormat == null)
      {
        HasWDL = false;
        HasMLH = false;
      }
      else
      {
        HasWDL = net.Format.NetworkFormat.Value == NetworkFormat.ValueFormat.ValueWdl;
        HasMLH = net.Format.NetworkFormat.MovesLeft == NetworkFormat.MovesLeftFormat.MovesLeftV1;
      }
    }


    static int MaxAttentionSize(Net net, LC0LegacyWeights weights, int n)
    {
      int embedding_op_size = weights.ip_pol_b == null ? 0 : weights.ip_pol_b.Length;
      int policy_d_model = weights.ip2_pol_b == null ? 0 : weights.ip2_pol_b.Length;
      Debug.Assert(policy_d_model == (weights.ip3_pol_b == null ? 0 : weights.ip3_pol_b.Length));

      int encoder_d_model = 0;
      int encoder_dff = 0;

      int encoder_heads = weights.numPolicyEncoderHeads;// weights.pol_encoder_head_count;

      //  if (weights.pol_encoder.size() > 0) {
      if (encoder_heads > 0)
      {
        throw new NotImplementedException("Encoders not yet supported");
#if NOT
        encoder_d_model = weights.pol_encoder[0].mha.q_b.size();
          encoder_dff = weights.pol_encoder[0].ffn.dense1_b.size();

          Debug.Assert(encoder_d_model ==  weights.pol_encoder[0].mha.k_b.size());
          Debug.Assert(encoder_d_model == weights.pol_encoder[0].mha.v_b.size());
          Debug.Assert(embedding_op_size == weights.pol_encoder[0].ffn.dense2_b.size());
#endif
      }


      int size = n * 64 * Math.Max(Math.Max(embedding_op_size, encoder_dff),
                                   Math.Max(policy_d_model, encoder_d_model));

      // size of matmul_qk matrix = encoder_heads_ * Batch * 64 * 64
      int matmul_qk_size = encoder_heads * n * 64 * 64;
      int output_size = n * (64 * 64 + 8 * 24);

      size = Math.Max(size, Math.Max(matmul_qk_size, output_size));
      return size;
    }



    private void AllocateGPUMemory(Net net, LC0LegacyWeights weights)
    {
      // The original implementation turned off custom Winograd algorithm if device memory was low.
      // But this implmenetation always requries custon Winograd.
      long _residual_single_layer_weight_size = 3 * 3 * NumFilters * NumFilters * Marshal.SizeOf<FP16>();
      long _residual_weight_size = _residual_single_layer_weight_size * NumBlocks * 2;
      long _transformed_residual_weight_size = _residual_weight_size * 4;
      if ((float)_transformed_residual_weight_size > 0.4 * (float)(long)deviceProperties.TotalGlobalMemory)
      {
        Console.WriteLine("WARNING: Low GPU video memory. You may run into OOM errors. Try using a smaller network.");
      }

      // Have some minumum as we also use this for transforming weights.
      long _max_weight_size = 128 * 1024 * 1024;

      if (_max_weight_size < 3 * _residual_single_layer_weight_size)
      {
        _max_weight_size = 3 * _residual_single_layer_weight_size;
      }

      scratchSizeBytes = _max_weight_size;

      // Need additional space for transformed input/outputs which are 36/16
      // times size (4x4 block transformed into 6x6).
      // These scratch sizes are not huge (e.g. about 225MB for a 384x30 network).
      long transformed_tensor_size = (long)(MaxBatchSize * NumFilters * 64 * (36.0 / 16.0) * Marshal.SizeOf<FP16>());
      scratchSizeBytes = System.Math.Max(scratchSizeBytes, 2 * transformed_tensor_size);

      // Attention policy head may need more memory
      // (We also split the allocations into two parts, so need 2x)
      int attentionSize = MaxAttentionSize(net, weights, MaxBatchSize);
      scratchSizeBytes = System.Math.Max(scratchSizeBytes, 2 * attentionSize);

      long scratchSizeElements = scratchSizeBytes / Marshal.SizeOf<FP16>();

      // =========================================================================
      // 3. Allocate GPU memory for running the network:
      //    - three buffers of max size are enough (one to hold input, second to
      //      hold output and third to hold skip connection's input).

      // size of input to the network
      long maxSizeElements = MaxBatchSize * NNBackendInputOutput.NUM_INPUT_PLANES * 64;

      // take max size of all layers
      foreach (BaseLayerCUDA layer in Layers.Layers)
      {
        maxSizeElements = System.Math.Max(maxSizeElements, layer.GetOutputSize(MaxBatchSize));
      }

      if (USE_FUSED && scratchSizeElements > maxSizeElements)
      {
        maxSizeElements = scratchSizeElements;
      }

      networkInputs = new NNInputCudaVariables(scratchSizeElements, maxSizeElements);
      networkOutputs = new NNOutputCudaVariables(MaxBatchSize, HasWDL, HasMLH);


      if (mlhOutputBuffer == null)
      {
        mlhOutputBuffer = new CUDAPinnedMemory<FP16>(MaxBatchSize);
        wdlOutputBuffer = new CUDAPinnedMemory<FP16>(MaxBatchSize * (HasWDL ? 3 : 1));

        if (SaveActivations)
        {
          valueHeadFC2OutputBuffer = new CUDAPinnedMemory<FP16>(MaxBatchSize * 128);
        }
      }
    }


    void InitGPUVars()
    {
      moveMasksGPU = new CudaDeviceVariable<short>(96 * MaxBatchSize);
      outputMaskedPoliciesGPU = new CudaDeviceVariable<float>(96 * MaxBatchSize);
    }

    CudaDeviceVariable<short> moveMasksGPU;
    CudaDeviceVariable<float> outputMaskedPoliciesGPU;
    private bool disposedValue;

    public void ExtractMaskedPolicies(CudaStream stream, CudaDeviceVariable<FP16> policiesRaw, int batchSize)
    {
      // Invoke kernel
      // Set kernel dimensions
      int threads = batchSize * 96;
      const int blockSize = 256;
      int blocks = CUDAUtils.DivUp(threads, blockSize);
      Layers.maskedMovesKernel.GridDimensions = blocks;
      Layers.maskedMovesKernel.BlockDimensions = blockSize;
      Layers.maskedMovesKernel.RunAsync(stream.Stream,
                                 policiesRaw.DevicePointer,
                                 moveMasksGPU.DevicePointer,
                                 outputMaskedPoliciesGPU.DevicePointer,
                                 batchSize);

      inputOutput.OutputPolicyHeadMasked.CopyToHostAsync(outputMaskedPoliciesGPU, 96 * batchSize, stream);
    }



    public void EvaluateNN(int batchSize)
    {
      lock (ExecContext.Device.ExecLockObj)
      {
        bool needWriterLock = UseGraphs && graphSet.GraphForBatchSizeNeedsConstruction(batchSize);

        if (needWriterLock)
        {
          ExecContext.Device.GraphCaptureRWLock.EnterUpgradeableReadLock();
        }
        else
        {
          ExecContext.Device.GraphCaptureRWLock.EnterReadLock();
        }

        try
        {
          DoEvaluateNN(batchSize);
        }
        finally
        {
          if (needWriterLock)
          {
            ExecContext.Device.GraphCaptureRWLock.ExitUpgradeableReadLock();
          }
          else
          {
            ExecContext.Device.GraphCaptureRWLock.ExitReadLock();
          }
        }
      }
    }

    void DoEvaluateNN(int batchSize)
    {
      //  Console.WriteLine($" #{INSTANCE}: evaluate {batchSize} on thread={ Thread.CurrentThread.ManagedThreadId} context: {Context.Context.Pointer}");

      SetCUDAState();

      NNBackendInputOutput io = inputOutput; // shorter alias

      PrepareInputs(batchSize);

      if (AsyncMode)
      {
        // Complete work of copying inputs to device,
        // making input variables subsequently safe to modify.
        ExecContext.Stream.Synchronize();

        InputsCopyToDeviceFinished.Set();
      }

      int batchSizeForNetwork;
      if (batchSize < MIN_BATCH_SIZE && MIN_BATCH_SIZE < MaxBatchSize)
      {
        batchSizeForNetwork = MIN_BATCH_SIZE;
      }
      else
      {
        batchSizeForNetwork = batchSize;
      }

      RunNetwork(batchSizeForNetwork);

      // Possibly wait for signal that ok to fill buffers.
      if (AsyncMode)
      {
        eventOkToFillResults?.Wait();
        eventOkToFillResults?.Reset();
      }

      RetrieveResultsFromGPU(batchSize);

      // Retrieve results
      ExecContext.Stream.Synchronize();

      if (DumpTiming)
      {
        Console.WriteLine("CUDA DETAIL - Batch size " + batchSize);
        Layers.DumpTimings();
      }
    }


    private void RetrieveResultsFromGPU(int batchSize)
    {
      // TODO: the use of async copies with pinned memory probably not worthwhile
      ExtractMaskedPolicies(ExecContext.Stream, networkOutputs.PolicyOut, batchSize);
      wdlOutputBuffer.CopyToHostAsync(networkOutputs.ValueOut, batchSize * (HasWDL ? 3 : 1), ExecContext.Stream);
      if (SaveActivations)
      {
        valueHeadFC2OutputBuffer.CopyToHostAsync(networkOutputs.ValueHeadFC2Out, batchSize * 128, ExecContext.Stream);
      }

      if (HasMLH)
      {
        mlhOutputBuffer.CopyToHostAsync(networkOutputs.MLHOut, batchSize, ExecContext.Stream);
      }
    }

    internal void ExtractActivations(int batchSize)
    {
      // Improve speed here.
      // Better (vectorized) FP32 to FP16 here: https://gist.github.com/rygorous/2156668
      // Alternate or in addition,the versions in ILGPU seems much faster (non-vectorized)

      unsafe
      {
        // TODO: Experimental only, eventually remove
        if (SaveActivations)
        {
          throw new NotImplementedException();
#if NOT
          if (NNBackendInputOutput.OutputValueHeadRaw == null)
          {
            NNBackendInputOutput.OutputValueHeadRaw = new float[inputOutput.OutputValueHead.GetLength(0), inputOutput.OutputValueHead.GetLength(1)];
          }
          Array.Copy(inputOutput.OutputValueHead, NNBackendInputOutput.OutputValueHeadRaw, batchSize * 3);
#endif
        }

        if (SaveActivations)
        {
          throw new NotImplementedException();
#if NOT
          Span<FP16> valueFC2Activations = valueHeadOutputBuffer.AsSpan();
          int layerWidth = valueFC2Activations.Length / MaxBatchSize;
          inputOutput.OutputValueHeadFC2 = FP16.ToFloat(valueFC2Activations, batchSize, layerWidth);
#endif
        }
      }
    }

    private void PrepareInputs(int batchSize)
    {
      if (batchSize == 0)
      {
        throw new ArgumentException("Invalid batch size of 0.");
      }

      NNBackendInputOutput io = inputOutput; // shorter alias

      int threads = batchSize * 8 * 8 * 112;  // each thread writes a single element
      const int blockSize = 256;
      int blocks = CUDAUtils.DivUp(threads, blockSize);
      Layers.expandPlanesKernel.GridDimensions = blocks;
      Layers.expandPlanesKernel.BlockDimensions = blockSize;

      // TODO: Use cudaHostAllocWriteCombined for 40% faster transfer? ****

      // Copy move indices.
      io.InputMoveIndices.CopyToDeviceAsync(moveMasksGPU, batchSize * 96, ExecContext.Stream);

      // Expand packed planes to full planes.
      // TODO: someday use more encoded masks, encode only squares with pieces for about 2x improvement
      io.InputBoardMasks.CopyToDeviceAsync(io.input_masks_gpu_, 112 * batchSize, ExecContext.Stream);
      io.InputBoardValues.CopyToDeviceAsync(io.input_val_gpu_, 112 * batchSize, ExecContext.Stream);
      Layers.expandPlanesKernel.RunAsync(ExecContext.Stream.Stream, networkInputs.Tensors[0].DevicePointer,
                                         io.input_masks_gpu_.DevicePointer,
                                         io.input_val_gpu_.DevicePointer,
                                         batchSize * 112);
    }

    /// <summary>
    /// Gives a hint to the executor that a specified batch size
    /// is commonly used and should possibly be optimized for.
    /// </summary>
    /// <param name="batchSize"></param>
    public void SetCommonBatchSize(int? batchSize)
    {
      if (batchSize.HasValue && UseGraphs)
      {
        graphSet.EnsureGraphOfSizeCreated(ExecContext, batchSize.Value);
      }
    }


    private void RunNetwork(int batchSize)
    {
      NNBackendCUDAGraph thisGraph = null;

      if (UseGraphs)
      {
        thisGraph = graphSet.GetGraphForBatchSize(ExecContext, batchSize);
      }

      if (thisGraph == null)
      {
        Layers.RunLayers(ExecContext.Stream, batchSize,
                         networkInputs,
                         networkOutputs);
      }
      else
      {
        thisGraph.RunGraph(ExecContext.Stream, batchSize);
      }
    }


    static int[] Set(int batchSizeDivisor,
                     int start1, int skip1, int max1,
                     int skip2, int max2,
                     int? firstValue = null)
    {
      List<int> ints = new();
      if (firstValue.HasValue) ints.Add(firstValue.Value);

      for (int i = start1; i < max1; i += skip1 / batchSizeDivisor)
      {
        ints.Add(i);
      }

      for (int i = max1; i < max2; i += skip2/ batchSizeDivisor)
      {
        ints.Add(i);
      }
      
      return ints.ToArray();
    }


    private void BuildGraphSet(int batchSizeDivisor = 1)
    {
      int[] breaks;

      // Values are tuned to find a balance between
      // speed (more frequent cutpoints) and 
      // CUDA memory use (which increases with more cutpoints).
      // Using default settings the incremental GPU memory usage appears to be about 500mb per instance.
      switch (NumFilters)
      {
        case > 256:
          breaks = Set(batchSizeDivisor, 
                       8, 4, 32, 
                       4, 160, MIN_BATCH_SIZE);
          break;

        case >= 192:
          breaks = Set(batchSizeDivisor, 
                       8, 4, 96, 
                       6, 320, MIN_BATCH_SIZE);
          break;

        default:
          breaks = Set(batchSizeDivisor, 
                       12, 4, 48, 
                       12, 384, MIN_BATCH_SIZE);
          break;
      }

      void GraphBuilder(CudaStream stream, int batchSize)
      {
        Layers.RunLayers(stream, batchSize,
                         networkInputs,
                         networkOutputs);
      }

      graphSet = new NNBackendCUDAGraphSet(GraphBuilder, breaks);
    }

    private void CheckNetworkCompatible(Net net)
    {
      if (net.Format.NetworkFormat.Network != NetworkFormat.NetworkStructure.NetworkClassicalWithHeadformat &&
          net.Format.NetworkFormat.Network != NetworkFormat.NetworkStructure.NetworkSeWithHeadformat &&
          net.Format.NetworkFormat.Network != NetworkFormat.NetworkStructure.NetworkSe)
      {
        throw new Exception($"Network format {net.Format.NetworkFormat.Network} not supported.");
      }

      if (net.Format.NetworkFormat.Policy != NetworkFormat.PolicyFormat.PolicyUnknown &&  // T40
          net.Format.NetworkFormat.Policy != NetworkFormat.PolicyFormat.PolicyConvolution &&
          net.Format.NetworkFormat.Policy != NetworkFormat.PolicyFormat.PolicyAttention &&
          net.Format.NetworkFormat.Policy != NetworkFormat.PolicyFormat.PolicyClassical)
      {
        throw new Exception($"Policy format {net.Format.NetworkFormat.Policy} not supported.");
      }

      if (net.Format.NetworkFormat.Value != NetworkFormat.ValueFormat.ValueUnknown &&  // T40
          net.Format.NetworkFormat.Value != NetworkFormat.ValueFormat.ValueClassical &&
          net.Format.NetworkFormat.Value != NetworkFormat.ValueFormat.ValueWdl)
      {
        throw new Exception($"Value format {net.Format.NetworkFormat.Value} not supported.");
      }

      if (net.Format.NetworkFormat.MovesLeft != NetworkFormat.MovesLeftFormat.MovesLeftNone &&
          net.Format.NetworkFormat.MovesLeft != NetworkFormat.MovesLeftFormat.MovesLeftV1)
      {
        throw new Exception($"Moves left head format {net.Format.NetworkFormat.MovesLeft} not supported.");
      }
    }

#region Disposal

    protected virtual void Dispose(bool disposing)
    {
      if (!disposedValue)
      {
        if (disposing)
        {
// may be reused?  Layers = new NNBackendCUDALayers(ExecContext, net, weights, SaveActivations, ReferenceBackend?.Layers);

          inputOutput?.Dispose();
          mlhOutputBuffer?.Dispose();
          wdlOutputBuffer?.Dispose();
          valueHeadFC2OutputBuffer?.Dispose();

          networkInputs?.Dispose();
          networkOutputs?.Dispose();

          graphSet?.Dispose();

          moveMasksGPU?.Dispose();
          outputMaskedPoliciesGPU?.Dispose();

          ExecContext.Dispose();
        }


        ReferenceBackend = null;
        disposedValue = true;
      }
    }

    public void Dispose()
    {
      Dispose(disposing: true);
    }

#endregion
  }


}
