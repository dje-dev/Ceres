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

using ManagedCuda;
using Ceres.Base.DataTypes;
using System;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using System.Diagnostics;

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  public class LayerEmbedding : BaseLayerCUDA, IDisposable
  {
    CudaDeviceVariable<FP16> weights;
    CudaDeviceVariable<FP16> biases;
    int num_inputs;

    public LayerEmbedding(NNBackendExecContext parent, 
                          string name, int layerIndex,
                          BaseLayerCUDA ip, float[] weights, float[] biases, ActivationFunction activation,
                          int numInputs)
             : base(parent, name, layerIndex, biases.Length, 8, 8, ip, activation)
    {
      this.weights = LoadedWeights(weights);
      this.biases = LoadedWeights(biases);
      this.num_inputs = numInputs;
    }

    public override void LoadKernels()
    {
    }

    
    protected override void DoEval(CudaStream stream, int N, 
                                   CudaDeviceVariable<FP16> output, 
                                   CudaDeviceVariable<FP16> input, CudaDeviceVariable<FP16> input2, 
                                   CudaDeviceVariable<FP16> scratch, long scratchSizeBytes, CudaDeviceVariable<FP16> scratchSecondHalf = null)
    {
      int num_outputs = C;

      int batch = N * 64;
      unsafe
      {
        half halfZero = new half(0);
        half halfOne = new half(1);
        void* ptrHalfZero = &halfZero;
        void* ptrHalfOne = &halfOne;
        IntPtr ipHalfZero = (IntPtr)ptrHalfZero;
        IntPtr ipHalfOne = (IntPtr)ptrHalfOne;

        CudaBlasNativeMethods.cublasGemmEx(Parent.CuBlas.CublasHandle, Operation.Transpose, Operation.NonTranspose,
                                         num_outputs, batch, num_inputs,
                                         ipHalfOne, weights.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs,
                                         input.DevicePointer, cudaDataType.CUDA_R_16F, num_inputs,
                                         ipHalfZero, output.DevicePointer, cudaDataType.CUDA_R_16F, num_outputs,
                                         ComputeType.Compute16F, GemmAlgo.Default);
      }

      AddBiasBatched(GetAddBiasBatchedKernel(Activation), output, output, biases, 1, batch, num_outputs, stream);
    }


    public override void Dispose()
    {
      weights?.Dispose();
      biases?.Dispose();
    }
  }
}
