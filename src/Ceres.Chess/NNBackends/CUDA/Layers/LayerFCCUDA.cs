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
using Ceres.Base.CUDA;
using Ceres.Base.DataTypes;

using ManagedCuda;
using ManagedCuda.CudaBlas;

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  /// <summary>
  /// General fully connected layer.
  /// </summary>
  public class LayerFCCUDA : BaseLayerCUDA
  {
    /// <summary>
    /// If RELU acivtion used.
    /// </summary>
    bool UseRELU;

    /// <summary>
    /// If bias weights used.
    /// </summary>
    bool UseBias;

    /// <summary>
    /// If hyperbolic tangent activation used.
    /// </summary>
    bool UseTanH;

    /// <summary>
    /// If sigmoid activation used.
    /// </summary>
    bool UseSigmoid;

    CudaDeviceVariable<FP16> biases;
    CudaDeviceVariable<FP16> weights;


   public LayerFCCUDA(NNBackendExecContext parent, string name, int layerIndex, 
                      BaseLayerCUDA ip, int c, int h, int w,
                      bool relu = false, bool bias = false, bool tanh = false, bool sigmoid = false)
      : base(parent, name, layerIndex, c, h, w, ip)
    {
      UseRELU = relu;
      UseBias = bias;
      UseTanH = tanh;
      UseSigmoid = sigmoid;
    }

    CudaKernel kernelAddVectors;

    public override void LoadKernels()
    {
      const string kn = "_ZN6lczero13cudnn_backend17addVectors_kernelI6__halfEEvPT_S4_S4_iiibbb";
      kernelAddVectors = Parent.Device.GetKernel(Parent.PTXAssembly, @"common_kernels.ptx", kn);
    }


    public void LoadWeights(float[] cpuWeight, float[] cpuBias)
    {
      if (Parent.ReferenceLayers != null)
      {
        LayerFCCUDA refLayer = Parent.ReferenceLayers.Layers[LayerIndex] as LayerFCCUDA;
        this.weights = refLayer.weights;
        this.biases = refLayer.biases;
      }
      else
      {
        weights = LoadedWeights(cpuWeight);
        if (UseBias)
        {
          int num_biases = C * GetH * W;
          biases = LoadedWeights(cpuBias, num_biases);
        }
      }
    }

    protected override void DoEval(CudaStream stream, int N, CudaDeviceVariable<FP16> output, CudaDeviceVariable<FP16> input,
                                   CudaDeviceVariable<FP16> scratch, long scratch_size,
                                   CudaDeviceVariable<FP16> scratchSecondHalf)
    {
      int num_outputs = C * GetH * W;
      int num_inputs = input_.C * input_.GetH * input_.W;
      Forward1D(stream, N, num_inputs, num_outputs, input, weights, biases, UseRELU, UseTanH, output);
    }

    internal void Forward1D(CudaStream stream, int batch_size, int input_size, int output_size,
                               CudaDeviceVariable<FP16> inputs, 
                               CudaDeviceVariable<FP16> weights,
                               CudaDeviceVariable<FP16> biases,
                               bool apply_relu, bool apply_tanh, 
                               CudaDeviceVariable<FP16> outputs)
    {
      CublasStatus err = CudaBlasNativeMethods.cublasHgemm(
                           Parent.CuBlas.CublasHandle, 
                           Operation.Transpose, Operation.NonTranspose,
                           (int)output_size,      // M
                           (int)batch_size,       // N
                           (int)input_size,       // K
                           ref CUDAUtils.halfOne, // alpha
                           weights.DevicePointer, // A
                           (int)input_size,       // lda, leading rank of A
                           inputs.DevicePointer,  // B
                           (int)input_size,       // ldb, leading rank of B
                           ref CUDAUtils.halfZero,// beta
                           outputs.DevicePointer, // C
                           (int)output_size);     // ldc, leading rank of C

      if (err != CublasStatus.Success)
      {
        throw new Exception("LayerFCCUDA failure " + err);
      }
#if NOT

      CUDA.BLAS.GemmEx(
                             Operation.Transpose, Operation.NonTranspose,
                             (int)output_size,      // M
                             (int)batch_size,       // N
                             (int)input_size,       // K
                           CUDA.halfOneDevice.DevicePointer, // alpha
//                             ref CUDA.halfOne,        // alpha
//1.0f,
                             weights.DevicePointer, // A
                             DataType.CUDA_R_16F,
                             (int)input_size,       // lda, leading rank of A
                             inputs.DevicePointer,  // B
                             DataType.CUDA_R_16F,
                             (int)input_size,       // ldb, leading rank of B
                           CUDA.halfZeroDevice.DevicePointer,// beta
//                            ref CUDA.halfZero,             // beta
//0.0f,
                             outputs.DevicePointer, // C
                             DataType.CUDA_R_16F,
                             (int)output_size);     // ldc, leading rank of C
#endif
      if (UseBias || UseRELU || UseTanH || UseSigmoid)
      {
        int num_outputs = C * GetH * W;
        int num_inputs = input_.C * input_.GetH * input_.W;

        const int BLOCK_DIM = 256;
        kernelAddVectors.BlockDimensions = BLOCK_DIM;
        kernelAddVectors.GridDimensions = CUDAUtils.DivUp(num_outputs * batch_size, BLOCK_DIM);

        // Adds two vectors (possibly of different sizes), also do optional activation (relu, tanh or sigmoid).
        LaunchKernel(stream, kernelAddVectors, outputs.DevicePointer, biases.DevicePointer, outputs.DevicePointer, 
                   num_outputs * batch_size,
                   num_outputs, 
                   num_outputs * batch_size, 
                   UseRELU? 1:0, UseTanH?1:0, UseSigmoid?1:0);

      }
    }

  }

}
