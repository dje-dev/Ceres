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

#endregion

namespace Ceres.Chess.NNEvaluators.Defs
{
  public enum NNDeviceType { CPU, GPU, RemoteServer };

  /// <summary>
  /// Defines a device used for NN evaluation.
  /// </summary>
  [Serializable]
  public record NNEvaluatorDeviceDef
  {
    /// <summary>
    /// Type of device (e.g. CPU or GPU).
    /// </summary>
    public readonly NNDeviceType Type;

    /// <summary>
    /// Index of the device.
    /// </summary>
    public  int DeviceIndex;

    /// <summary>
    /// If evaluations on this device should be done with low priority.
    /// </summary>
    public bool LowPriority = false;

    /// <summary>
    /// An optional directive indicating the maximum batch size to be sent to the device.
    /// </summary>
    public int? MaxBatchSize;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="type"></param>
    /// <param name="deviceNum"></param>
    /// <param name="lowPriority"></param>
    /// <param name="maxBatchSize"></param>
    public NNEvaluatorDeviceDef(NNDeviceType type, int deviceNum, bool lowPriority = false, int? maxBatchSize = null)
    {
      Type = type;
      DeviceIndex = deviceNum;
      LowPriority = lowPriority;
      MaxBatchSize = maxBatchSize; 
    }


    public static (NNEvaluatorDeviceDef, float)[] DevicesInRange(NNDeviceType type, int minDeviceIndex, int maxDeviceIndex)
    {
      int numDevices = maxDeviceIndex - minDeviceIndex + 1;
      (NNEvaluatorDeviceDef, float)[] ret = new (NNEvaluatorDeviceDef, float)[numDevices];
      for (int i = minDeviceIndex; i <= maxDeviceIndex; i++)
      {
        ret[i - minDeviceIndex] = (new NNEvaluatorDeviceDef(type, i), 1.0f / numDevices);
      }
      return ret;
    }


    /// <summary>
    /// Returns string representation.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      string maxStr = MaxBatchSize.HasValue ? $" Max={MaxBatchSize}" : "";
      return $"<NNEvaluatorDeviceDef {Type} #{DeviceIndex} {(LowPriority ? "Low Priority" : "")}{maxStr}>";
    }

  }
}
