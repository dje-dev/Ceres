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
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Ceres.Base.Math;
using Ceres.Base.OperatingSystem;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.NodeCache
{
  /// <summary>
  /// Node cache which distributes the cache entires over a 
  /// set of MCTSNodeCacheArrayPurgeable objects, where
  /// simple modulo hashing on the node index is used to 
  /// determine which member of the set will store a given node.
  /// 
  /// This approach combines the benefits of MCTSNodeCacheArrayPurgeable with also:
  ///   - improved concurrency since locks are taken only on
  ///     one member of the set for a given node, and
  ///   - opportunity to prune in parallel (across each of the sets)
  ///   
  /// The only drawback is somewhat increased code complexity, 
  /// and also the LRU purging becomes slightly approximate 
  /// because it is not coordinated across set members.
  /// </summary>
  public unsafe class MCTSNodeCacheArrayPurgeableSet : IMCTSNodeCache
  {
    internal const int MAX_SETS = 256;

    public MCTSNodeStore ParentStore;

    public readonly int MaxCacheSize;

    #region Private data

    public readonly int NumSubcaches;

    MCTSNodeCacheArrayPurgeable[] subCaches;


    MemoryBufferOS<MCTSNodeStruct> nodes;

    #endregion

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="parentStore"></param>
    /// <param name="cacheSize"></param>
    /// <param name="estimatedNumNodesInSearch"></param>
    public MCTSNodeCacheArrayPurgeableSet(MCTSNodeStore parentStore,
                                          int cacheSize,
                                          int estimatedNumNodesInSearch)
    {
      ParentStore = parentStore;
      MaxCacheSize = cacheSize;

      nodes = parentStore.Nodes.nodes;

      // Compute number of subcaches, increasing as a function of estimates search size
      // (because degree of concurrency rises with size of batches and search.
      if (cacheSize >1_000_000)
      {
        NumSubcaches = 32;
      }
      else if (cacheSize > 200_000)
      {
        NumSubcaches = 16;
      }
      else if (cacheSize > 50_000)
      {
        NumSubcaches = 4;
      }
      else
      {
        NumSubcaches = 2;
      }

      // Initialize the sub-caches
      subCaches = new MCTSNodeCacheArrayPurgeable[NumSubcaches];
      Parallel.For(0, NumSubcaches, i => subCaches[i] = new MCTSNodeCacheArrayPurgeable(parentStore, cacheSize / NumSubcaches));
    }


    /// <summary>
    /// Store from which the MCTSNode objects originated.
    /// </summary>
    MCTSNodeStore IMCTSNodeCache.ParentStore => ParentStore;


    /// <summary>
    /// Sets/resets the node context to which the cached items belong.
    /// </summary>
    /// <param name="context"></param>
    public void SetContext(MCTSIterator context)
    {
      ParentStore = context.Tree.Store;
      nodes = ParentStore.Nodes.nodes;

      Parallel.ForEach(subCaches, subCache => subCache.SetContext(context));
    }


    /// <summary>
    /// Adds a specified node to the cache.
    /// </summary>
    /// <param name="node"></param>
    /// <returns>pointer to space allocated for this node</returns>
    public void* Add(MCTSNodeStructIndex node) => subCaches[node.Index % NumSubcaches].Add(node);


    public void* Lookup(MCTSNodeStructIndex nodeIndex) => nodes[nodeIndex.Index].CachedInfoPtr;


    /// <summary>
    /// Returns the MCTSNode stored in the cache 
    /// corresponding to specified MCTSNodeStruct
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    public void* Lookup(in MCTSNodeStruct nodeRef) => nodeRef.CachedInfoPtr;


    /// <summary>
    /// Returns the number of nodes currently present in the cache.
    /// </summary>
    int NumInUse
    {
      get
      {
        int numInUse = 0;
        {
          for (int i = 0; i < subCaches.Length; i++)
          {
            numInUse += subCaches[i].NumInUse;
          }
        }
        return numInUse;
      }
    }

    /// <summary>
    /// Returns the maximum number of nodes in use across all subcaches.
    /// </summary>
    int MaxNumInUse
    {
      get
      {
        int maxNumInUse = 0;
        {
          for (int i = 0; i < subCaches.Length; i++)
          {
            if (subCaches[i].NumInUse > maxNumInUse)
            {
              maxNumInUse = subCaches[i].NumInUse;
            }
          }
        }

        return maxNumInUse;
      }
    }

    static readonly object lockObj = new();

    /// <summary>
    /// Possibly prunes the cache to remove some of the least recently accessed nodes.
    /// </summary>
    /// <param name="store"></param>
    public void PossiblyPruneCache(MCTSNodeStore store)
    {
      // Determine if any of the subcaches is nearly full.
      int numItemsPerSubcache = MaxCacheSize / NumSubcaches;
      bool almostFull = MaxNumInUse > (numItemsPerSubcache * MCTSNodeCacheArrayPurgeable.THRESHOLD_PCT_DO_PRUNE) / 100;
      if (almostFull)
      {
        lock (lockObj)
        {
          int countPurged = 0;
          Parallel.ForEach(Enumerable.Range(0, NumSubcaches),
                           i => Interlocked.Add(ref countPurged, subCaches[i].Prune(store, -1)));

        }
      }
    }

    /// <summary>
    /// Removes a specified node from the cache, if present.
    /// </summary>
    /// <param name="nodeIndex"></param>
    public void Remove(MCTSNodeStructIndex nodeIndex)
    {
      subCaches[nodeIndex.Index % NumSubcaches].Remove(nodeIndex);
    }


    /// <summary>
    /// Clears table entries and possibly resets back to null the CacheIndex for every node.
    /// </summary>
    /// <param name="resetNodeCacheIndex"></param>
    public void ResetCache(bool resetNodeCacheIndex)
    {
      Parallel.For(0, subCaches.Length, i => subCaches[i].ResetCache(resetNodeCacheIndex));
    }


    /// <summary>
    /// Nodes in node cache are stamped with the sequence number
    /// of the last batch in which they were accessed to faciltate LRU determination.
    /// </summary>
    public int NextBatchSequenceNumber { get; set; }


    public override string ToString()
    {
      return $"<MCTSNodeCacheArrayPurgeableSet MaxSize={MaxCacheSize} NumInUse={NumInUse}>";
    }

  }

}





