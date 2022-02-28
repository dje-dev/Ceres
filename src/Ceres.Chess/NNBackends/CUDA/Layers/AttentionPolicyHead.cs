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
using ManagedCuda;
using Ceres.Base.CUDA;
using Ceres.Base.DataTypes;

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  public class AttentionPolicyHead : BaseLayerCUDA
  {
    public AttentionPolicyHead(NNBackendExecContext parent, string name, int layerIndex,
                               int c, int h, int w, BaseLayerCUDA inputLayer, ActivationFunction activation)//, CudaDeviceVariable<FP16> scratch) 
      : base(parent, name, layerIndex, c, h, w, inputLayer, activation)
    {
    }


    public override void LoadKernels()
    {
      throw new NotImplementedException();
    }

    protected override void DoEval(CudaStream stream, int N, CudaDeviceVariable<FP16> output, CudaDeviceVariable<FP16> input, CudaDeviceVariable<FP16> scratch, long scratchSizeBytes, CudaDeviceVariable<FP16> scratchSecondHalf = null)
    {
      throw new NotImplementedException();
    }
  }
}
