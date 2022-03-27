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

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  public class LayerEncoder : BaseLayerCUDA
  {
    internal EncoderWeights encoderWeights;

    float encoder_heads;
    float embedding_op_size_;
    float alpha;

    public LayerEncoder(NNBackendExecContext parent, EncoderWeights encoderWeights,
                        string name, LC0LegacyWeights weights, int layerIndex,
                        BaseLayerCUDA ip, int heads, int size, float alpha)
       : base(parent, name, layerIndex, 0, 0, 0, ip, ActivationFunction.NONE)
    {
      encoder_heads = heads;
      embedding_op_size_ = size;
      this.alpha = alpha;

      this.encoderWeights = encoderWeights;
    }


    public override void LoadKernels()
    {
//      string kn = "_ZN6lczero13cudnn_backend16policyMap_kernelI6__halfEEvPT_PKS3_PKsiiii";
//      kernelPolicyMap = Parent.Device.GetKernel(Parent.PTXAssembly, COMMON_KERNELS_PTX_NAME, kn);
    }


    public void LoadWeights(short[] cpuWeight)
    {
    }

    protected override void DoEval(CudaStream stream, int N, CudaDeviceVariable<FP16> output,
                                   CudaDeviceVariable<FP16> input, CudaDeviceVariable<FP16> input2,
                                   CudaDeviceVariable<FP16> scratch, long scratch_size, CudaDeviceVariable<FP16> scratchSecondHalf)
    {
    }
  }
}
