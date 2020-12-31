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

namespace Ceres.Base.DataType
{
  /// <summary>
  /// Spans of unmanaged types where the span first address is 
  /// guaranteed to be aligned to a specified boundary.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  public class SpanAligned<T> : IDisposable where T : unmanaged
  {
    /// <summary>
    /// Number of elements in the Span.
    /// </summary>
    public readonly int Length;

    /// <summary>
    /// Raw buffer for data.
    /// </summary>
    private byte[] buffer;

    /// <summary>
    /// Fixed handle backing memory.
    /// </summary>
    private GCHandle bufferHandle;

    /// <summary>
    /// Pointer to start of Span.
    /// </summary>
    private IntPtr bufferPointer;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="length">number of elements to allocate</param>
    /// <param name="alignmentBytes">number which starting address must divide</param>
    public SpanAligned(int length, int alignmentBytes)
    {
      this.Length = length;
      buffer = new byte[length * Marshal.SizeOf<T>() + alignmentBytes];
      bufferHandle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
      long ptr = bufferHandle.AddrOfPinnedObject().ToInt64();

      // round up ptr to nearest 'byteAlignment' boundary
      ptr = (ptr + alignmentBytes - 1) & ~(alignmentBytes - 1);
      bufferPointer = new IntPtr(ptr);
    }

    /// <summary>
    /// Returns the allocated span.
    /// </summary>
    public unsafe Span<T> Span => new Span<T>(bufferPointer.ToPointer(), Length);

    /// <summary>
    /// Destructor to release underlying span.
    /// </summary>
    ~SpanAligned() => Dispose(false);


    #region IDisposable Members

    protected void Dispose(bool disposing)
    {
      if (bufferHandle.IsAllocated)
      {
        bufferHandle.Free();
        buffer = null;
      }
    }

    public void Dispose()
    {
      Dispose(true);
    }

    #endregion

  }
}
  
