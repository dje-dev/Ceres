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

#endregion

namespace Ceres.Chess.NNEvaluators.TensorRT;

/// <summary>
/// Native struct for marshaling output tensor info from the TensorRT wrapper.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal record struct NativeOutputTensorInfo
{
  /// <summary>
  /// Pointer to tensor name (const char* - pointer to internal string, do not free)
  /// </summary>
  public IntPtr Name;

  /// <summary>
  /// Offset in elements from start of output buffer
  /// </summary>
  public long Offset;

  /// <summary>
  /// Size in elements
  /// </summary>
  public long Size;
}
