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
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Ceres.MCTS.MTCSNodes.Storage;

#endregion

namespace Ceres.MCTS.MTCSNodes.Struct
{
  /// <summary>
  /// Wrapper around a pointer which is a gateway to associated context, either:
  ///   - the StoreID of associated MCTSNodeStore, or
  ///   - pointer to MCTSNodeInfo if associated node info is currently in the cache
  /// </summary>
  [Serializable]
  public unsafe struct MCTSNodeStructContextPtr
  {
    /// <summary>
    /// Arbitrary value guaranteed to be less than 
    /// the smallest possible address for an object.
    /// </summary>
    const int MIN_INFOPTR_ADDRESS = 1024 * 1024;

    void* rawPtr;

    public void* RawPtr => rawPtr;

    //public static implicit operator void*(MCTSNodeStructContextPtr d) => d.rawPtr; // ** TO DO: remove
    public void SetAsStoreID(int storeID)
    {
      Debug.Assert(storeID < MIN_INFOPTR_ADDRESS);
      rawPtr = (void*)storeID;
    }

    public int StoreID => IsCached ? Info.Store.StoreID : (int)rawPtr;

    public ref MCTSNodeInfo Info
    {
      get
      {
        if (IsCached)
        {
          ref MCTSNodeInfo infoPtr = ref Unsafe.AsRef<MCTSNodeInfo>(rawPtr);
          return ref infoPtr;
        }
        else
        {
          throw new NotImplementedException();
          //return MCTSNodeStore.StoresByID[(int)rawPtr];
        }
      }
    }

    public MCTSNodeStore Store
    {
      get
      {
        if (IsCached)
        {
          ref MCTSNodeInfo infoPtr = ref Unsafe.AsRef<MCTSNodeInfo>(rawPtr);
          MCTSNodeStore store = infoPtr.Store;
          Debug.Assert(store is not null);
          return store;
        }
        else
        {
          Debug.Assert(rawPtr != null); // Zero is reserved, not valid store ID
          MCTSNodeStore store = MCTSNodeStore.StoresByID[(int)rawPtr];
          Debug.Assert(store is not null);
          return store;
        }
      }
    }

    public void SetAsCachePtr(void* ptr)
    {
      Debug.Assert(Unsafe.AsRef<MCTSNodeInfo>(ptr).Store != null);
      rawPtr = ptr;
    }

    public bool IsCached => (long)rawPtr > MIN_INFOPTR_ADDRESS;

    public override string ToString()
    {
      string typeStr = IsCached ? $"Cached {Unsafe.AsRef<object>(rawPtr)}" : $"Parent Store {(long)rawPtr}";
      return $"<MCTSNodeStructContextPtr {typeStr}>";
    }
  }

}