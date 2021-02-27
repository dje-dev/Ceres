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


#endregion

using Ceres.Base.DataTypes.Aligned;
using System;

namespace Ceres.Base.ANN
{
  /// <summary>
  /// Contains the calculation result from applying a specified to ANN to an input batch.
  /// </summary>
  public class ANNCalcResult
  {
    public readonly ANNDef Parent;
    public readonly int MultiCount;
    public readonly float[][] LayerOutputBuffersSingle;

    float[][,] layerOutputBuffersMulti;

    
    /// <summary>
    /// Array of output buffers (lazy created)
    /// </summary>
    public float[][,] LayerOutputBuffersMulti
    {
      get
      {
        if (layerOutputBuffersMulti == null)
        {
          layerOutputBuffersMulti = new float[Parent.Layers.Count][,];

          for (int i = 0; i < Parent.Layers.Count; i++)
            layerOutputBuffersMulti[i] = new float[MultiCount, Parent.Layers[i].WidthOut];  // ** TO DO: ALIGN ****

        }
        return layerOutputBuffersMulti;
      }
    }


    /// <summary>
    /// Constructor
    /// </summary>
    /// <param name="def"></param>
    /// <param name="multiCount"></param>
    public ANNCalcResult(ANNDef def, int multiCount)
    {
      if (multiCount <= 0) throw new ArgumentException("multiCount must be positive");

      Parent = def;
      MultiCount = multiCount;
      LayerOutputBuffersSingle = new float[def.Layers.Count][];

      for (int i=0; i<def.Layers.Count; i++)
        LayerOutputBuffersSingle[i] = new AlignedFloatArray(def.Layers[i].WidthOut, 128).GetManagedArray();
    }

  }

}
