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

using Ceres.Base.Misc;
using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;

#endregion

namespace Ceres.Base.DataTypes
{
  /// <summary>
  /// Class that acts like List<T> but with fixed maximum size (not resizable). 
  /// The actual storage is in an array (T[]) which can be any of:
  ///   - allocated by this object for its exclusive use
  ///   - rented by this object from the ArrayPool for its use
  ///   - passed into constructor if already extant
  ///   
  /// Possible performance benefits:
  ///   - no logic needed to test for resizing, since this is not supported (oversized use would result in array out of bounds Exception)
  ///   - renting from ArrayPool option can speed up
  ///   - supports returning references to items rather than items themselves
  ///     (impossible with List<T> because possibility of resize makes this not supported)
  ///   
  /// The behavior is similar to that of List<T> but more efficient
  /// (primarily because it never resizes, can use a pooled array, and can return references to items).
  /// </summary>
  /// <typeparam name="T"></typeparam>
  public partial class ListBounded<T> : IDisposable, IEnumerable<T>, IList<T> where T : IComparable<T>
  {
    bool useArrayPool;
    public int MaxLength { get; private set; }
    int length;
    T[] array;
    int version;

    public enum StorageMode {  AllocatedArray, RentedArrayPool };
    public enum CopyMode {  ReferencePassedMembers, CopyPassedMembers };


    /// <summary>
    /// Constructor from array.
    /// </summary>
    /// <param name="members"></param>
    /// <param name="createMode"></param>
    /// <param name="storageMode"></param>
    public ListBounded(T[] members, CopyMode createMode, StorageMode storageMode = StorageMode.AllocatedArray)
    {
      Debug.Assert(!(createMode == CopyMode.ReferencePassedMembers && storageMode == StorageMode.RentedArrayPool)); // inconsistent

      MaxLength = length = members.Length;

      if (createMode == CopyMode.ReferencePassedMembers)
      {
        array = members;
      }
      else
      {
        DoCreate(members.Length, storageMode);
        Array.Copy(members, array, members.Length);
      }
    }


    public ListBounded(ListBounded<T> other, StorageMode storageMode = StorageMode.AllocatedArray)
    {
      DoCreate(other.Count, storageMode);
      length = other.Count;
      Array.Copy(other.array, array, other.Count);
    }


    /// <summary>
    /// 
    /// </summary>
    /// <param name="maxLength"></param>
    /// <param name="useArrayPool">if the storage should come from a shared pool (if so, must call Dispose when done)</param>
    public ListBounded(int maxLength, StorageMode storageMode = StorageMode.AllocatedArray)
    {
      DoCreate(maxLength, storageMode);
    }

    void DoCreate(int maxLength, StorageMode storageMode = StorageMode.AllocatedArray)
    {
      useArrayPool = storageMode == StorageMode.AllocatedArray;
      MaxLength = maxLength;
      array = useArrayPool ? ArrayPool<T>.Shared.Rent(maxLength) : new T[maxLength];
    }

    #region Access

    /// <summary>
    /// Number of items added to the ListBounded.
    /// </summary>
    public int Count => length;

    /// <summary>
    /// A view of the ListBounded as a span.
    /// </summary>
    public Span<T> AsSpan => new Span<T>(array).Slice(0, length);

    bool ICollection<T>.IsReadOnly => false;


    /// <summary>
    /// Indexer
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public T this[int index]
    {
      get { Debug.Assert(index < length); return array[index]; }
      set { Debug.Assert(index < length); array[index] = value; }
    }

    /// <summary>
    /// Returns reference to item at specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public ref T ItemRef(int index)
    {
      Debug.Assert(index < length); 
      return ref array[index];
    }

    public T[] ToArray()
    {
      T[] ret = new T[length];
      Array.Copy(array, ret, length);
      return ret;
    }

    #endregion


    #region Updating

    /// <summary>
    /// Adds a specified item to the ListBounded.
    /// </summary>
    /// <param name="t"></param>
    public void Add(T t)
    {
      version++;
      array[length++] = t;
    }

    /// <summary>
    /// Adds a specified array of items to the ListBounded.
    /// </summary>
    /// <param name="t"></param>
    public void Add(T[] t, int maxElements = int.MaxValue)
    {
      version++;
      Array.Copy(t, 0, array, length, System.Math.Min(t.Length, maxElements));
      length += t.Length;
    }

    /// <summary>
    /// Adds a specified ListBounded of items to the ListBounded.
    /// </summary>
    /// <param name="t"></param>
    public void Add(ListBounded<T> t, int maxElements = int.MaxValue)
    {
      version++;
      Array.Copy(t.array, 0, array, length, System.Math.Min(t.Count, maxElements));
      length += t.Count;
    }

    /// <summary>
    /// Adds a specified item (by its reference) to the ListBounded.
    /// </summary>
    /// <param name="t"></param>
    public void Add(ref T t)
    {
      version++;
      array[length++] = t;
    }


    /// <summary>
    /// Clears (removes) all elements from the ListBounded.
    /// </summary>
    /// <param name="zeroItems"></param>
    public void Clear(bool zeroItems)
    {
      if (zeroItems)
      {
        Array.Clear(array, 0, length);
      }

      length = 0;
      version++;
    }


    public void Set(ListBounded<T> items)
    {
      length = items.Count;
      Array.Copy(items.array, array, items.Count);
      version++;
    }


    #endregion

    #region Dispose

    public void Dispose()
    {
      if (array != null && useArrayPool)
      {
        T[] copy = array;
        array = null;
        ArrayPool<T>.Shared.Return(copy);
      }
      else
        array = null;
    }

    #endregion

    #region Overrides


    public override string ToString()
    {
      return $"<ListBounded[{MaxLength}] with {Count} elements {(Count > 0 ? ("first:" + this[0]) : "")}>";
    }

    #endregion


    #region IEnumerator

    public Enumerator GetEnumerator() => new Enumerator(this);    

    IEnumerator<T> IEnumerable<T>.GetEnumerator() => new Enumerator(this);

    IEnumerator IEnumerable.GetEnumerator() => new Enumerator(this);

    public int IndexOf(T item) => Array.IndexOf(array, item, 0, length);

    // TO DO: implement this and other methods on the List interface
    public void Insert(int index, T item) =>  throw new NotImplementedException();
    
    public void RemoveAt(int index)
    {
      if (index == Count - 1)
      {
        // currently we support only this simple special case
        this[length - 1] = default;
        length--;
      }
      else
        throw new NotImplementedException();
    }

    public void Clear() => Clear(true);

    public bool Contains(T item) => throw new NotImplementedException();

    public void CopyTo(T[] array, int arrayIndex) => throw new NotImplementedException();

    public bool Remove(T item) => throw new NotImplementedException();

    #endregion

    [Serializable]
    public struct Enumerator : IEnumerator<T>, System.Collections.IEnumerator
    {
      private ListBounded<T> list;
      private int index;
      private int version;
      private T current;

      internal Enumerator(ListBounded<T> list)
      {
        this.list = list;
        index = 0;
        version = list.version;
        current = default(T);
      }

      public void Dispose()
      {
      }

      public bool MoveNext()
      {
        ListBounded<T> localArray = list;

        if (version == localArray.version && ((uint)index < (uint)localArray.length))
        {
          current = localArray.array[index];
          index++;
          return true;
        }
        return MoveNextRare();
      }

      private bool MoveNextRare()
      {
        if (version != list.version)
        {
          ThrowHelper.ThrowWrongVersion();
        }

        index = list.length + 1;
        current = default(T);
        return false;
      }

      public T Current => current;

      Object IEnumerator.Current => Current;


      void IEnumerator.Reset()
      {
        if (version != list.version)
        {
          ThrowHelper.ThrowWrongVersion();
        }

        index = 0;
        current = default(T);
      }
    }

    internal static class ThrowHelper
    {
      internal static void ThrowWrongVersion() => throw new Exception("Enumerable changed during enumeration");
    }
  }
 
}
