
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
  public unsafe class StructArrayRefMapper<S, T> where T : struct
  {
    #region Data structures

    const int MAX_ENTRIES = 256;

    StructLookupEntry<S>[] entries = new StructLookupEntry<S>[MAX_ENTRIES];

    readonly object lockObj = new();

    #endregion

    /// <summary>
    /// Records the fact that an array of structs (of specified size) was created.
    /// </summary>
    public void Register(S arrayObj, ref T firstT, int count)
    {
      int sizeBytes = Marshal.SizeOf<T>();

      // Remove if already previously registered.
      Unregister(ref firstT);

      // Determine first and last addresses.
      IntPtr ax = new IntPtr(Unsafe.AsPointer(ref firstT));
      ulong firstAddr = (uint)ax.ToInt64();
      ulong lastAddr = firstAddr + (ulong)sizeBytes * (ulong)count;

      // Find an empty slot in the table and add entry there.
      lock (lockObj)
      {
        for (int i = 0; i < MAX_ENTRIES; i++)
        {
          if (entries[i].Obj == null)
          {
            entries[i] = new StructLookupEntry<S>(arrayObj, firstAddr, lastAddr);
            return;
          }
        }

        throw new Exception("Table overflow in ArrayStructRefMapper");
      }
    }


    /// <summary>
    /// Returns the object associated with the struct array
    /// containing struct referenced by argument.
    /// </summary>
    public S Lookup(ref T t)
    {
      return entries[LookupIndex(ref t)].Obj;
    }


    /// <summary>
    /// Unregisters specified array (specified by reference to an element).
    /// </summary>
    void Unregister(ref T t)
    {
      int index = LookupIndex(ref t, false);
      if (index != -1)
      {
        Unregister(entries[index].Obj);
      }
    }


    /// <summary>
    /// Unregisters specified array.
    /// </summary>
    public void Unregister(S arrayObj)
    {
      lock (lockObj)
      {
        for (int i = 0; i < MAX_ENTRIES; i++)
        {
          if (object.ReferenceEquals(arrayObj, entries[i].Obj))
          {
            entries[i] = default;
            return;
          }
        }
      }
    }

    #region Private helpers

    int LookupIndex(ref T t, bool throwIfNotFound = true)
    {
      IntPtr ax = new IntPtr(Unsafe.AsPointer(ref t));
      ulong addr = (uint)ax.ToInt64();
      for (int i = 0; i < MAX_ENTRIES; i++)
      {
        if (addr >= entries[i].Start && addr < entries[i].End)
        {
          return i;
        }
      }

      if (throwIfNotFound)
      {
        throw new Exception("Entry not found");
      }
      else
      {
        return -1;
      }
    }

    #endregion
  }




  readonly struct StructLookupEntry<S>
  {
    public readonly S Obj;
    public readonly ulong Start;
    public readonly ulong End;

    public StructLookupEntry(S obj, ulong start, ulong end)
    {
      Obj = obj;
      Start = start;
      End = end;
    }

  }



}
