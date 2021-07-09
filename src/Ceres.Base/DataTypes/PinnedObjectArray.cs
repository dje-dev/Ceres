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

namespace Ceres.Base.DataTypes
{
  /// <summary>
  /// Array of objects on the pinned object heap.
  /// </summary>
  public class PinnedObjectArray : IDisposable
  {
    #region Private data

    IntPtr[] pointers;
    GCHandle[] gcHandleList;

    #endregion

    public PinnedObjectArray(params object[] parameters)
    {
      int paramCount = parameters.Length;
      pointers = new IntPtr[paramCount];
      GCHandle[] gcHandleList = new GCHandle[paramCount];

      for (int i = 0; i < paramCount; i++)
      {
        gcHandleList[i] = GCHandle.Alloc(parameters[i], GCHandleType.Pinned);
        pointers[i] = gcHandleList[i].AddrOfPinnedObject();
      }
    }

    public IntPtr[] Pointers => pointers;

    public ref T ObjRef<T>(int objIndex)
    {
      unsafe
      {
        return ref Unsafe.AsRef<T>((void*)pointers[objIndex]);
      }
    }


    #region Shutdown

    public void Dispose()
    {
      if (gcHandleList != null)
      {
        for (int i = 0; i < pointers.Length; i++)
        {
          gcHandleList[i].Free();
        }

        gcHandleList = null;
        pointers = null;
      }
    }

    #endregion

  }

}
