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

namespace Ceres.Base.Threading
{
  /// <summary>
  /// An Int32 isolated on its own cache line, for counters that are updated
  /// (typically via Interlocked) by many threads while neighboring fields of the
  /// enclosing object are read-mostly and hot on those same threads.
  /// 
  /// The value sits at offset 64 of a 128-byte block, so the 64-byte cache line
  /// containing it lies entirely inside this struct regardless of where the CLR
  /// places the field within the enclosing object (objects are only 8-byte aligned,
  /// and the CLR reorders class fields, so declaration order alone cannot isolate a line).
  /// 
  /// Fields of this type must not be declared readonly (Interlocked requires a writable ref).
  /// </summary>
  [Serializable]
  [StructLayout(LayoutKind.Explicit, Size = 128)]
  public struct PaddedInt32
  {
    /// <summary>
    /// Size in bytes of a cache line (also the offset of Value within the struct).
    /// </summary>
    public const int CACHE_LINE_SIZE = 64;

    /// <summary>
    /// The counter value.
    /// </summary>
    [FieldOffset(CACHE_LINE_SIZE)] public int Value;
  }
}
