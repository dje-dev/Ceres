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
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Base.OperatingSystem;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.NodeCache
{
  /// <summary>
  /// Node cache which maintains an array of MCTSNode and stash the index of the node 
  /// corresponding to a given structure into the structure, if is in cache.
  /// 
  /// Advantages:
  ///   - only takes 4 (or possibly 3) bytes from each MCTSNodeStruct
  ///     plus a fixed array of (say 300,000) MCTSNode
  ///   - pruning only has to scan over lineary array cached nodes
  /// </summary>
  public unsafe class MCTSNodeCacheArrayPurgeable : IMCTSNodeCache
  {
    internal const int THRESHOLD_PCT_DO_PRUNE = 75;
    internal const int THRESHOLD_PCT_PRUNE_TO = 50;

    /// <summary>
    /// Parent MCTSNodeStore to which the MCTSNode belong.
    /// </summary>
    public MCTSNodeStore ParentStore;


    public readonly int MaxCacheSize;

    #region Cache data

    MCTSNodeInfo[] nodes;

    // Each slot of cachedNodesIndices contains the index of the
    // node currently occupying the slot (also corresponding slot in nodes),
    // or 0 if empty (available).
    // These entries are used for efficient scanning for an available entry
    // (when adding node) in a threadsafe way (using Interlocked).   
    int[] cachedNodesIndices;

    MemoryBufferOS<MCTSNodeStruct> nodesStore;

    readonly object lockObj = new object();

    int numInUse = 0;
    int searchFreeEntryNextIndex = 0;

    void* ptrFirstItem;
    int lengthItem;

    #endregion

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="parentStore"></param>
    /// <param name="maxCacheSize"></param>
    public MCTSNodeCacheArrayPurgeable(MCTSNodeStore parentStore, int maxCacheSize)
    {
      const int LOH_THRESHOLD_SIZE_BYTES = 110_000;
      int minCacheSize = (LOH_THRESHOLD_SIZE_BYTES / Marshal.SizeOf<MCTSNodeInfo>());

      maxCacheSize = Math.Max(minCacheSize, maxCacheSize);
      MaxCacheSize = maxCacheSize;

      SetNodeStore(parentStore);

      // N.B. The elements of the nodes array must remain 
      //      at fixed location in memory (since raw pointers refer to them).

      // Using GC.AllocateArray with pinned is not possible because:
      //   "Only value types without pointers or references are supported."

      // Therefore allocate using ordinary new operator and rely upon fact that
      // by default objects on LOH are not compacted/moved.

      nodes = new MCTSNodeInfo[maxCacheSize];
      
      cachedNodesIndices = new int[maxCacheSize];

      pruneSequenceNums = GC.AllocateUninitializedArray<int>(maxCacheSize);

      ptrFirstItem = Unsafe.AsPointer(ref nodes[0]);
      lengthItem = (int) (new IntPtr(Unsafe.AsPointer(ref nodes[1])).ToInt64() 
                        - new IntPtr(Unsafe.AsPointer(ref nodes[0])).ToInt64());
    }

    /// <summary>
    /// Sets/resets the node store to which the cached items below.
    /// </summary>
    /// <param name="parentStore"></param>
    public void SetNodeStore(MCTSNodeStore parentStore)
    {
      ParentStore = parentStore;
      nodesStore = parentStore.Nodes.nodes;
    }

    public void Clear() => throw new NotImplementedException();


    /// <summary>
    /// store from which the MCTSNode objects originated.
    /// </summary>
    MCTSNodeStore IMCTSNodeCache.ParentStore => ParentStore;

    /// <summary>
    /// Returns the number of nodes currently present in the cache.
    /// </summary>
    public int NumInUse => numInUse;


    bool TryTake(MCTSNodeStructIndex nodeIndex, int thisTryIndex)
    {
      // Try to make the swap, but only if prior entry was null.
      int exchangeTry = Interlocked.CompareExchange(ref cachedNodesIndices[thisTryIndex], (int)nodeIndex.Index, (int)0);
      
      if (exchangeTry == 0)
      {
        // Success
        searchFreeEntryNextIndex = thisTryIndex + 1;
        Interlocked.Increment(ref numInUse);
        //nodes[thisTryIndex].StructRef.CacheIndex = thisTryIndex;

        return true;
      }
      else
      {
        return false;
      }
    }

    /// <summary>
    /// Returns the MCTSNode having the specified index and stored in the cache
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns>pointer to space allocated for this node</returns>
    /// <returns></returns>
    public void* Add(MCTSNodeStructIndex node)
    {
      int numTries = 0;

      // Make local copy of the entry we are going to try
      int thisTryIndex = searchFreeEntryNextIndex;

      // Repeatedly try to grab a slot.
      while (true)
      {
        // Wrap around if necessary.
        if (thisTryIndex >= nodes.Length - 1)
        {
          thisTryIndex = 0;
        }

        if (TryTake(node, thisTryIndex))
        {
          // Make cache index which contains both set index and index within this array
          return nodesStore[node.Index].CachedInfoPtr = Unsafe.AsPointer(ref nodes[thisTryIndex]);
        }

        // Make sure we haven't overflowed.
        numTries++;
        if (numTries > nodes.Length * 2)
        {
          throw new Exception("MCTSNodeCache overflow");
        }

        // Move to next possible slot.
        thisTryIndex++;
      };

    }


    /// <summary>
    /// Possibly prunes the cache to remove some of the least recently accessed nodes.
    /// </summary>
    /// <param name="store"></param>
    public void PossiblyPruneCache(MCTSNodeStore store)
    {
      bool almostFull = numInUse > (nodes.Length * THRESHOLD_PCT_DO_PRUNE) / 100;
      if (almostFull)
      {
        int targetSize = (nodes.Length * THRESHOLD_PCT_PRUNE_TO) / 100;
        Prune(store, targetSize);
      }

    }

    int pruneCount = 0;

    int[] pruneSequenceNums;


    /// <summary>
    /// Prunes cache down to approximately specified target size.
    /// </summary>
    /// <param name="store"></param>
    /// <param name="targetSize">target numer of nodes, or -1 to use default sizing</param>
    /// <returns></returns>
    internal int Prune(MCTSNodeStore store, int targetSize)
    {
      int startNumInUse = numInUse;

      if (targetSize == -1) targetSize = (nodes.Length * THRESHOLD_PCT_PRUNE_TO) / 100;
      if (numInUse <= targetSize) return 0;

      lock (lockObj)
      {
        int count = 0;
        for (int i = 0; i < nodes.Length; i++)
        {
          // TODO: the long is cast to int, could we possibly overflow? make long?
          if (nodes[i].IsInitialized)
          {
            pruneSequenceNums[count++] = (int)nodes[i].LastAccessedSequenceCounter;
          }
        }

        Span<int> slice = new Span<int>(pruneSequenceNums).Slice(0, count);

        // Compute the minimum sequence number an entry must have
        // to be retained (to enforce LRU eviction)
        //float cutoff = KthSmallestValue.CalcKthSmallestValue(keyPrioritiesForSorting, numToPrune);
        int cutoff;

        cutoff = KthSmallestValueInt.CalcKthSmallestValue(slice, numInUse - targetSize);
        //Console.WriteLine(slice.Length + " " + (numInUse-targetSize) + " --> " 
        //                 + cutoff + " correct " + slice[numInUse-targetSize] + " avg " + slice[numInUse/2]);

        int maxEntries = pruneCount == 0 ? (numInUse + 1) : nodes.Length;
        for (int i = 1; i < maxEntries; i++)
        {
          ref MCTSNodeInfo node = ref nodes[i];
          if (node.IsInitialized && node.LastAccessedSequenceCounter < cutoff
             && !node.IsInFlight // never prune active node in flight
             )
          {
            MCTSNodeStructIndex nodeIndex = new MCTSNodeStructIndex(node.Index);
            nodesStore[nodeIndex.Index].CachedInfoPtr = null;
            cachedNodesIndices[i] = 0;
            nodes[i].SetUninitialized();

            numInUse--;
          }
        }
        pruneCount++;
      }

      return startNumInUse - numInUse;
    }
   

    /// <summary>
    /// Returns the MCTSNode having the specified index and stored in the cache
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    public void* Lookup(MCTSNodeStructIndex nodeIndex) => nodesStore[nodeIndex.Index].CachedInfoPtr;


    /// <summary>
    /// Returns the MCTSNode stored in the cache 
    /// corresponding to specified MCTSNodeStruct
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    public void* Lookup(in MCTSNodeStruct nodeRef) => nodeRef.CachedInfoPtr;


    /// <summary>
    /// Clears table entries and possibly resets back to null the CacheIndex for every node.
    /// </summary>
    /// <param name="resetNodeCacheIndex"></param>
    public void ResetCache(bool resetNodeCacheIndex)
    {
      if (resetNodeCacheIndex)
      {
        for (int i = 0; i < nodes.Length; i++)
        {
          if (cachedNodesIndices[i] != 0)
          {
            nodesStore[nodes[i].index.Index].CachedInfoPtr = null;
          }
        }
      }

      Array.Clear(cachedNodesIndices, 0, nodes.Length);
      numInUse = 0;
      searchFreeEntryNextIndex = 0;
    }


    /// <summary>
    /// Nodes in node cache are stamped with the sequence number
    /// of the last batch in which they were accessed to faciltate LRU determination.
    /// </summary>
    public int NextBatchSequenceNumber { get; set; }


    /// <summary>
    /// Removes a specified node from the cache, if present.
    /// </summary>
    /// <param name="nodeIndex"></param>
    public void Remove(MCTSNodeStructIndex nodeIndex)
    {
      ref MCTSNodeStruct storeItem = ref ParentStore.Nodes.nodes[nodeIndex.Index];
      if (storeItem.CachedInfoPtr != null)
      {
        // Release the cache item from the nodes array
        long bytesDiff = new IntPtr(storeItem.CachedInfoPtr).ToInt64() - new IntPtr(ptrFirstItem).ToInt64();
        int indexItem = (int)(bytesDiff / lengthItem);

        Debug.Assert(cachedNodesIndices[indexItem] == nodeIndex.Index);
        cachedNodesIndices[indexItem] = 0;

        // Clear the cache item in the node struct
        storeItem.CachedInfoPtr = null;

 
        numInUse--;
      }
    }


    /// <summary>
    /// Returns string summary.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<MCTSNodeCacheArrayPurgeable MaxSize={MaxCacheSize} NumInUse={NumInUse}>";
    }


    #region Diagnostic methods

    /// <summary>
    /// Verifies that the numInUse field correctly reflects the state of the cache.
    /// </summary>
    void ValidateState()
    {
      int numUsed = 0;
      for (int i = 0; i < nodes.Length; i++)
      {
        if (nodes[i].IsInitialized)
        {
          numUsed++;
        }
      }

      if (numUsed != numInUse)
      {
        throw new Exception("Internal error: numInUse incorrect.");
      }
    }

    #endregion

  }
}
