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
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Base.DataType
{
  /// <summary>
  /// Miscellaneous extension methods relating to Spans.
  /// </summary>
  public static class SpanUtils
  {
    // Get index of element within its Span.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int IndexOfRef<T>(this Span<T> span, ref T value)
    {
      ref T r0 = ref MemoryMarshal.GetReference(span);
      IntPtr byteOffset = Unsafe.ByteOffset(ref r0, ref value);

      nint elementOffset = byteOffset / (nint)(uint)Unsafe.SizeOf<T>();
      if ((long)elementOffset >= (uint)span.Length)
      {
        throw new Exception("Internal error: element does not fall within specified Span.");
      }

      return (int)elementOffset;
    }

    // Get index of element within its Span.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int IndexOf<T>(this Span<T> span, in T value) where T : unmanaged
    {
      ref T r0 = ref MemoryMarshal.GetReference(span);
      IntPtr byteOffset = Unsafe.ByteOffset(ref r0, ref Unsafe.AsRef(in value));

      nint elementOffset = byteOffset / (nint)(uint)Unsafe.SizeOf<T>();
      if ((long)elementOffset >= (uint)span.Length)
      {
        throw new Exception("Internal error: element does not fall within specified Span.");
      }

      return (int)elementOffset;
    }

  }
}
