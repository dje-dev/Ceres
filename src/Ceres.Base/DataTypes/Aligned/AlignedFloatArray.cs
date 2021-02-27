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
using System.Text;
using System.Runtime.InteropServices;
using Ceres.Base.DataType;


#endregion

namespace Ceres.Base.DataTypes.Aligned
{
  /// <summary>
  /// Array of floats aligned to specified memory divisibility boundary.
  /// </summary>
  public class AlignedFloatArray : IDisposable
  {
    public readonly int Length;

    #region Internal data

    private byte[] buffer;
    private GCHandle bufferHandle;
    private IntPtr bufferPointer;

    #endregion

    /// <summary>
    /// Constructor (with specified length, aligment and optional initial values).
    /// </summary>
    /// <param name="length"></param>
    /// <param name="byteAlignment"></param>
    /// <param name="initialValues"></param>
    public AlignedFloatArray(int length, int byteAlignment, float[] initialValues = null)
    {
      this.Length = length;
      buffer = new byte[length * sizeof(float) + byteAlignment];
      bufferHandle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
      long ptr = bufferHandle.AddrOfPinnedObject().ToInt64();

      // round up ptr to nearest 'byteAlignment' boundary
      ptr = (ptr + byteAlignment - 1) & ~(byteAlignment - 1);
      bufferPointer = new IntPtr(ptr);

      if (initialValues != null)
      {
        Write(0, initialValues, 0, length);
      }
    }


    /// <summary>
    /// Gets or sets element at specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public unsafe float this[int index]
    {
      get => GetPointer()[index];
      set => GetPointer()[index] = value;
    }


    /// <summary>
    /// Writes values into array at specified starting index.
    /// </summary>
    /// <param name="index"></param>
    /// <param name="src"></param>
    /// <param name="srcIndex"></param>
    /// <param name="count"></param>
    public void Write(int index, float[] src, int srcIndex, int count)
    {
      if (index < 0 || index >= Length) throw new IndexOutOfRangeException();
      if ((index + count) > Length) count = System.Math.Max(0, Length - index);

      Marshal.Copy(
        src,
        srcIndex,
        new IntPtr(bufferPointer.ToInt64() + index * sizeof(float)),
        count);
    }


    /// <summary>
    /// Reads values from array at specified stratin gindex.
    /// </summary>
    /// <param name="index"></param>
    /// <param name="dest"></param>
    /// <param name="dstIndex"></param>
    /// <param name="count"></param>
    public void Read(int index, float[] dest, int dstIndex, int count)
    {
      if (index < 0 || index >= Length) throw new IndexOutOfRangeException();
      if ((index + count) > Length) count = System.Math.Max(0, Length - index);

      Marshal.Copy(new IntPtr(bufferPointer.ToInt64() + index * sizeof(float)),
                   dest, dstIndex, count);
    }


    /// <summary>
    /// Returns a new managed array containing contents of aligned array.
    /// </summary>
    /// <returns></returns>
    public float[] GetManagedArray()
    {
      return GetManagedArray(0, Length);
    }


    /// <summary>
    /// Returns a new managed array containing subset of contents of aligned array.
    /// </summary>
    /// <param name="index"></param>
    /// <param name="count"></param>
    /// <returns></returns>
    public float[] GetManagedArray(int index, int count)
    {
      float[] result = new float[count];
      Read(index, result, 0, count);
      return result;
    }


    /// <summary>
    /// Returns string representation.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      StringBuilder sb = new StringBuilder();
      sb.Append('[');
      for (int t = 0; t < Length; t++)
      {
        sb.Append(this[t].ToString());
        if (t < (Length - 1)) sb.Append(',');
      }
      sb.Append(']');
      return sb.ToString();
    }


    /// <summary>
    /// Returns pointer to element of array at specified starting index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public unsafe float* GetPointer(int index) => GetPointer() + index;


    /// <summary>
    /// Returns pointer to beginning of aligned array.
    /// </summary>
    /// <returns></returns>
    public unsafe float* GetPointer() => ((float*)bufferPointer.ToPointer());


    #region Destructor and IDisposable Members

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

    ~AlignedFloatArray()
    {
      Dispose(false);
    }


    #endregion

  }
}

