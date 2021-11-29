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

using Ceres.Base.DataTypes;
using ManagedCuda;

#endregion

namespace Ceres.Chess.NNBackends.CUDA
{
  /// <summary>
  /// Base class for residual layers (either Winograd or not).
  /// </summary>
  public abstract class ResidualBlockBaseCUDA : BlockWithWinogradCUDA
  {
    public ResidualBlockBaseCUDA(NNBackendExecContext parent, string name, int layerIndex,
                                 int c, int h, int w, 
                                 BaseLayerCUDA inputLayer, bool hasSE, int seK)
  : base(parent, name, layerIndex, c, h, w, inputLayer)
    {

      HasSE = hasSE;
      SEK = seK;
    }

    /// <summary>
    /// If SE (squeeze and excitation layers) are used.
    /// </summary>
    protected readonly bool HasSE;

    /// <summary>
    /// SE channel count
    /// </summary>
    protected readonly int SEK;

    /// <summary>
    /// SE weights 1
    /// </summary>
    protected CudaDeviceVariable<FP16> Weights1;

    /// <summary>
    /// SE weights 2
    /// </summary>
    protected CudaDeviceVariable<FP16> Weights2;

    /// <summary>
    /// SE biases 1
    /// </summary>
    protected CudaDeviceVariable<FP16> Biases1;

    /// <summary>
    /// SE biases 2
    /// </summary>
    protected CudaDeviceVariable<FP16> Biases2;


    public abstract void LoadWeights0(CudaStream stream, float[] weights, float[] bias);
    
    public abstract void LoadWeights1(CudaStream stream, float[] weights, float[] bias);


    #region Protected helpers

    public static CudaDeviceVariable<FP16> CudaHalf(float[] floats)
    {
      FP16[] buffer = new FP16[floats.Length];
      for (int i = 0; i < floats.Length; i++)
      {
        buffer[i] = (FP16)floats[i];
      }
      return (CudaDeviceVariable<FP16>)buffer;
    }

    void CPUTranspose(float[] op, float[] ip, int rows, int cols)
    {
      for (int i = 0; i < rows; i++)
      {
        for (int j = 0; j < cols; j++)
        {
          op[j * rows + i] = ip[i * cols + j];
        }
      }
    }


    public void LoadSEWeights(float[] weights1, float[] biases1,
                              float[] weights2, float[] biases2)
    {
      if (Parent.ReferenceLayers != null)
      {
        ResidualBlockBaseCUDA refLayer = Parent.ReferenceLayers.Layers[LayerIndex] as ResidualBlockBaseCUDA;
        Biases1 = refLayer.Biases1;
        Biases2 = refLayer.Biases2;
        Weights1 = refLayer.Weights1;
        Weights2 = refLayer.Weights2;
      }
      else
      {
        Biases1 = CudaHalf(biases1);
        Biases2 = CudaHalf(biases2);

        float[] temp1 = new float[weights1.Length];
        CPUTranspose(temp1, weights1, SEK, C);
        Weights1 = CudaHalf(temp1);

        float[] temp2 = new float[weights2.Length];
        CPUTranspose(temp2, weights2, 2 * C, SEK);
        Weights2 = CudaHalf(temp2);
      }
    }

    #endregion

  }
}
