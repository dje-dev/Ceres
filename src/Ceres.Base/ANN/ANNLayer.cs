#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directive

using System;
using System.Numerics;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Base.DataTypes.Aligned;
using Ceres.Base.Math;
using Ceres.Base.Math.MKL;

#endregion

namespace Ceres.Base.ANN
{
  [Serializable]
  public class ANNLayerDef
  {
    public enum ActivationType { Linear=0, RELU=1, LeakyRELU=2, ELU=3, SELU=4, Swish = 5, Tanh = 6 }; // N.B. in some C++ code these constants are assumed
                                                                                                      
    public enum LayerSubtype { Normal, SparseBinaryInput, CustomFunc };

    public readonly ANNDef Parent;
    public readonly string Name;
    public LayerSubtype Subtype;
    public readonly ActivationType Activation;
    public readonly int WidthIn;
    public int InputLayerMaxCountNonzeroPerRow; // only if sparse binary input
    public readonly int WidthOut;
    public readonly float[,] Weights;
    public readonly float[] Weights1D;
    public readonly float[] Weights1DTr;
    public readonly float[] Weights1DTrBlocked8;

    public readonly float[,] WeightsTr;
    public readonly FP16[,] WeightsFP16Tr;
    public readonly float[] Biases;
    public readonly Func<float[], object, float[]> CustomFunc;

    // Statistics saved for batch normalization
    public readonly float[] MovingMeans;
    public readonly float[] MovingStdDevs;
    public  float[] Betas;
    public readonly float[] Gammas;

    public readonly float ClipValueMax;

   
    const float EPSILON = 0.001f;


    public ANNLayerDef(ANNDef parent, string name, int widthIn, int widthOut, Func<float[], object, float[]> customFunc)
    {
      Parent = parent;
      Name = name;
      WidthIn = widthIn;
      WidthOut = widthOut;

      Subtype = LayerSubtype.CustomFunc;
      CustomFunc = customFunc;
    }


    public ANNLayerDef(ANNDef parent, string name, int widthIn, int widthOut, 
                       float[,] weights, float[] biases,
                       ActivationType activation, LayerSubtype subtype = LayerSubtype.Normal,
                       float[] movingMeans = null, float[] movingVariances = null,
                       float[] betas = null, float[] gammas = null,
                       float clipValue = float.NaN, 
                       int sparseBinaryInputMaxNonzerPerRow = 0)
    {
      Parent = parent;
      Name = name;
      Activation = activation;
      WidthIn = widthIn;
      WidthOut = widthOut;
      Weights = weights;
      WeightsTr = new float[weights.GetLength(1), weights.GetLength(0)];
      WeightsFP16Tr = new FP16[weights.GetLength(1), weights.GetLength(0)];
      InputLayerMaxCountNonzeroPerRow = sparseBinaryInputMaxNonzerPerRow;

      if (weights.GetLength(0) != widthIn || weights.GetLength(1) != widthOut)
        throw new Exception("incorrect weight shape");

      //Weights1D = new float[WidthIn * WidthOut];
      Weights1D = new AlignedFloatArray(WidthIn * WidthOut, 128).GetManagedArray();
      Weights1DTr = new AlignedFloatArray(WidthIn * WidthOut, 128).GetManagedArray();
      int offset = 0;

      for (int i = 0; i < WidthIn; i++)
      {
        for (int j = 0; j < WidthOut; j++)
        {
          WeightsTr[j, i] = weights[i, j];
          WeightsFP16Tr[j, i] =  (FP16) weights[i, j];
          Weights1D[offset++] = weights[i, j];
          Weights1DTr[j * WidthIn + i] = weights[i, j];
        }
      }

      // Prepare copy of weights for AVX2 (8 floats at a time) where stride is 8
      if (WidthOut >= 8)
      {
        Weights1DTrBlocked8 = MathUtils.BlockedMatrix(Weights1DTr, WidthOut, WidthIn, 8, 8);
      }

      Biases = new AlignedFloatArray(biases.Length, 128, biases).GetManagedArray();
      Subtype = subtype;
      MovingMeans = movingMeans;
      float[] MovingVariances = movingVariances;
      Betas = betas;
      Gammas = gammas;
      ClipValueMax = clipValue;

      if (MovingVariances != null)
      {
        // Precompute standard deviations for speed
        MovingStdDevs = new AlignedFloatArray(MovingVariances.Length, 128, biases).GetManagedArray(); 
        for (int i = 0; i < MovingStdDevs.Length; i++)
          MovingStdDevs[i] = (float)System.Math.Sqrt(MovingVariances[i] + EPSILON);
      }
    }


    public void ComputeMulti(int numInputs, float[,] input, float[,] outBufferMulti)
    {
      if (outBufferMulti == null) throw new ArgumentException();

      if (Subtype == LayerSubtype.CustomFunc)
      {
        throw new Exception("CustomFunc not yet supported");
      }
      else if (Subtype == LayerSubtype.SparseBinaryInput)
      {
        int MAX_ENTRIES = input.GetLength(0) * InputLayerMaxCountNonzeroPerRow;
//        using (new TimingBlock("sparse " + input.GetLength(0)))
          MKLUtils.SparseMatrixMatrixMult(input, Weights, outBufferMulti, MAX_ENTRIES); 
      }
      else if (Subtype == LayerSubtype.Normal)
      {
        if (WidthIn >= 0 && WidthOut >= 0)
        {
//          using (new TimingBlock("not sparse " + input.GetLength(0)))
            MKLUtils.FloatMultGEMM(input, Weights, outBufferMulti, numInputs);
        }         
      }
      else
        throw new Exception("internal error");

//          SGEMMEngineGPU.fastActivationBNMulti(outBufferMulti, numInputs, WidthOut, Biases, MovingMeans, MovingStdDevs, Betas, Gammas, (int)Activation);

      if (!float.IsNaN(ClipValueMax))
      {
        if (outBufferMulti.GetLength(1) != 1) throw new Exception("Clipping not supported");
        for (int i = 0; i < outBufferMulti.GetLength(0); i++)
           outBufferMulti[i, 0] = MathF.Min(ClipValueMax, outBufferMulti[i, 0]);
      }

      //Parallel.For(0, numInputs, new ParallelOptions() { MaxDegreeOfParallelism = NUM_PROC }, delegate (int item)
      for (int item = 0; item < numInputs; item++)
      {
        // Add biases and activation
        for (int i = 0; i < WidthOut; i++)
        {
          float val = outBufferMulti[item, i] + Biases[i];

          if (Activation == ActivationType.ELU && val < 0)
          {
            const float ALPHA = 1.0f;
            val = ALPHA * ((float)MathUtils.FastExp(val) - 1.0f);
          }

          else if (Activation == ActivationType.Tanh)
          {
            // NOTE: Tanh is not yet working in C++ (seems to crash in the math::tanh call, maybe because runtime not properly initialized by DLL)
            // So we do it "manually" here
            if (outBufferMulti.GetLength(1) != 1) throw new Exception("Not supported");
            for (int ix = 0; ix < outBufferMulti.GetLength(0); ix++)
              outBufferMulti[ix, 0] = (float)System.Math.Tanh(outBufferMulti[ix, 0] + Biases[0]);
          }
          else if (Activation == ActivationType.RELU && val < 0)
          {
            val = 0;
          }
          else if (Activation == ActivationType.SELU)
          {
            const float ALPHA = 1.6732632423543772848170429916717f;
            const float SCALE = 1.0507009873554804934193349852946f;

            if (val < 0) val = ALPHA * ((float)MathUtils.FastExp(val) - 1.0f);
            val *= SCALE;
          }
          else if (Activation == ActivationType.LeakyRELU && val < 0)
          {
            const float ALPHA = 0.1f; // Tensorflow default

            float x = val < 0 ? 0 : val;
            float m_x = -val < 0 ? 0 : -val;
            x -= ALPHA * m_x;
            val = x;
          }

          if (Betas != null)
          {
            float newVal = (val - MovingMeans[i]) / MovingStdDevs[i];
            val = Gammas[i] * newVal + Betas[i];
          }

          outBufferMulti[item, i] = val;
        }
      }

    }



    public override string ToString()
    {
      return "<Layer " + WidthIn + "x" + WidthOut + " " + Activation + " " + Subtype + " " + (MovingMeans != null ? "BN" : "") + ">";
    }

  }

}
