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
using System.Threading;
using System.Threading.Tasks;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
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
  public class MCTSNodeCacheArrayPurgeable : IMCTSNodeCache
  {
    internal const int THRESHOLD_PCT_DO_PRUNE = 75;
    internal const int THRESHOLD_PCT_PRUNE_TO = 55;

    public MCTSTree ParentTree;

    public readonly int MaxCacheSize;

    public static long NUM_HITS = 0;
    public static long NUM_MISSES = 0;
    public static float HitRate => 100.0f * ((float)NUM_HITS / (float)(NUM_HITS + NUM_MISSES));

    #region Cache data

    MCTSNode[] nodes;

    readonly object lockObj = new object();

    int numInUse = 0;
    int searchFreeEntryNextIndex = 0;
    int numCachePrunesInProgress = 0;

    #endregion

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="parentTree"></param>
    /// <param name="maxCacheSize"></param>
    public MCTSNodeCacheArrayPurgeable(MCTSTree parentTree, int maxCacheSize)
    {
      maxCacheSize = Math.Max(3_000, maxCacheSize);

      ParentTree = parentTree;
      MaxCacheSize = maxCacheSize;

      nodes = new MCTSNode[maxCacheSize];
      pruneSequenceNums = GC.AllocateUninitializedArray<int>(maxCacheSize);
    }

    public void Clear() => throw new NotImplementedException();


    /// <summary>
    /// Tree from which the MCTSNode objects originated.
    /// </summary>
    MCTSTree IMCTSNodeCache.ParentTree => ParentTree;

    /// <summary>
    /// Returns the number of nodes currently present in the cache.
    /// </summary>
    public int NumInUse => numInUse;

    bool firstPassAllocateSequential = true;

    int nextIndexPreWrap = 0;

    /// <summary>
    /// Returns the MCTSNode having the specified index and stored in the cache
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    public void Add(MCTSNode node)
    {
      lock (lockObj)
      {
        int thisIndex;

        if (firstPassAllocateSequential)
        {
          // If we have never wrapped, we can just allocate mostly sequentially.
          // However initially the table is empty so each batch of nodes
          // is allocated (and typically purged) all at once, leading to "clumps"
          // which then can cause longer search times to find a free entry.
          // Therefore we leave a few spaces in the table on the first pass.
          thisIndex = nextIndexPreWrap;

          int increment = numInUse % 20 == 0 ? 2 : 1;
          nextIndexPreWrap = thisIndex + increment;
          if (nextIndexPreWrap >  MaxCacheSize - 2)
          {
            firstPassAllocateSequential = false;
            nextIndexPreWrap = 0;
          }
        }
        else
        {
          thisIndex = NextSearchFreeEntry();
        }

        nodes[thisIndex] = node;
        node.Ref.CacheIndex = thisIndex;

        numInUse++;
      }
    }

    /// <summary>
    /// Determines the index of the next unused node.
    /// </summary>
    /// <param name="recursionDepth"></param>
    /// <returns></returns>
    int NextSearchFreeEntry(int recursionDepth = 0)
    {
      if (numInUse >= nodes.Length - 2 || recursionDepth > 1)
      {
        throw new Exception("Internal table: MCTSNodeCache overflow");
      }

      int foundIndex = -1;
      for (int i = searchFreeEntryNextIndex; i < nodes.Length; i++)
      {
        if (nodes[i] == null)
        {
          foundIndex = i;
          break;
        }
      }

      if (foundIndex == -1)
      {
        searchFreeEntryNextIndex = 1;

        // Skip this null position, restart search from here.
        return NextSearchFreeEntry(++recursionDepth); 
      }
      else
      {
        searchFreeEntryNextIndex = foundIndex + 1;
        return foundIndex;
      }

    }


    /// <summary>
    /// Possibly prunes the cache to remove some of the least recently accessed nodes.
    /// </summary>
    /// <param name="store"></param>
    public void PossiblyPruneCache(MCTSNodeStore store)
    {
      bool almostFull = numInUse > (nodes.Length * THRESHOLD_PCT_DO_PRUNE) / 100;
      if (numCachePrunesInProgress == 0 && almostFull)
        {
          Task.Run(() =>
          {
            //using (new TimingBlock("Prune"))
            {
              Interlocked.Increment(ref numCachePrunesInProgress);
              int targetSize = (nodes.Length * THRESHOLD_PCT_PRUNE_TO) / 100;
              Prune(store, targetSize);
              Interlocked.Decrement(ref numCachePrunesInProgress);
            };
          });
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
          if (nodes[i] != null)
            pruneSequenceNums[count++] = (int)nodes[i].LastAccessedSequenceCounter;
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
          MCTSNode node = nodes[i];
          if (node != null && node.LastAccessedSequenceCounter < cutoff)
          {
            MCTSNodeStructIndex nodeIndex = new MCTSNodeStructIndex(node.Index);
            nodes[i] = null;

            ref MCTSNodeStruct refNode = ref store.Nodes.nodes[nodeIndex.Index];
            refNode.CacheIndex = 0;

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
    public MCTSNode Lookup(MCTSNodeStructIndex nodeIndex)
    {
      ref MCTSNodeStruct nodeRef = ref nodeIndex.Ref;

      if (nodeRef.CacheIndex == 0)
      {
        NUM_MISSES++;
        return null;
      }
      else
      {
        MCTSNode node = nodes[nodeRef.CacheIndex];

        NUM_HITS++;
        return node;
      }
    }

    /// <summary>
    /// Clears table entries and resets back to null the CacheIndex for every node.
    /// </summary>
    public void ResetCache()
    {
      for (int i = 1; i <= nodes.Length; i++)
       {
        if (nodes[i] != null)
        {
          nodes[i].Ref.CacheIndex = 0;
        }
       }

      Array.Clear(nodes, 0, nodes.Length);
      numInUse = 0;
      searchFreeEntryNextIndex = 0;
      firstPassAllocateSequential = true;
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
        if (nodes[i] != null)
          numUsed++;

      if (numUsed != numInUse)
        throw new Exception("Internal error: numInUse incorrect.");
    }

    #endregion

  }
}
