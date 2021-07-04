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
using ManagedCuda.CudaBlas;

#endregion

namespace Ceres.Base.CUDA
{
  /// <summary>
  /// Wrapper for a CUBLAS handle.
  /// </summary>
  public struct CudaBlasLTHandle
  {
    public CudaBlasLTHandle(CudaBlasHandle handle)
    {
      Handle = handle.Pointer;
    }
    public readonly IntPtr Handle;
  }
}
