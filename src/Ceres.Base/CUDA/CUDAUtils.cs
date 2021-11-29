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
using Ceres.Base.DataTypes;

using ManagedCuda;
using ManagedCuda.BasicTypes;

#endregion

namespace Ceres.Base.CUDA
{
  /// <summary>
  /// Set of static helper class related to CUDA.
  /// 
  /// NOTE: the following is very helpful to track down timing/synchronization issues:
  //          Environment.SetEnvironmentVariable("CUDA_LAUNCH_BLOCKING", "1");
  /// </summary>
  public static class CUDAUtils
  {
    public static half halfZero = default; 
    public static half halfOne = default;

    public static int DivUp(int size, int mod) => ((size % mod) != 0) ? (size / mod + 1) : (size / mod);


    static CUDAUtils()
    {
      // Hack needed to initialize half because field is private
      half[] ones = new half[1];
      unsafe
      {
        fixed (half* xx = &ones[0])
        {
          *((short*)xx) = 0x3C00;
        }
      }
      halfOne = ones[0];
    }


    /// <summary>
    /// Checks CUDA last error result and throws if not success.
    /// </summary>
    /// <param name="result"></param>
    public static void Check(CUResult result)
    {
      if (result != CUResult.Success)
      {
        throw new Exception("CUDA Error: " + result.ToString() + " ");
      }
    }


    /// <summary>
    /// Dumps the FP16 contents of a CUDA variable to console.
    /// </summary>
    /// <param name="data"></param>
    /// <param name="length"></param>
    /// <param name="verbose"></param>
    /// <returns></returns>
    public static FP16[] Dump(CudaDeviceVariable<FP16> data, int length, bool verbose = false)
    {
      FP16[] scr = new FP16[length];
      data.CopyToHost(scr, 0, 0, scr.Length * Marshal.SizeOf<FP16>());
      float acc = 0;
      for (int i = 0; i < scr.Length; i++)
      {
        if (verbose && scr[i] != 0) Console.WriteLine(i + " " + scr[i]);
        acc += scr[i];
      }
      Console.WriteLine("first " + scr[0] + "last " + scr[^1] + " acc: " + acc);
      return scr;
    }

    public static CUDADevice Context(int gpuID)
    {
      return CUDADevice.GetContext(gpuID);
    }
  }

}
