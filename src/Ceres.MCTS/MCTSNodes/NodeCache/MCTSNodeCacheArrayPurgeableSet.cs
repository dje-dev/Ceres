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
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Ceres.Base.Math;
using Ceres.Base.OperatingSystem;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes;
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
  public class MCTSNodeCacheArrayPurgeableSet : IMCTSNodeCache
  {
    internal const int MAX_SETS = 256;

    public MCTSTree ParentTree;

    public readonly int MaxCacheSize;

    #region Private data

    public readonly int NumSubcaches;

    MCTSNodeCacheArrayPurgeable[] subCaches;


    MemoryBufferOS<MCTSNodeStruct> nodes;

    #endregion

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="parentTree"></param>
    /// <param name="definitiveMaxCacheSize"></param>
    /// <param name="estimatedNumNodesInSearch"></param>
    public MCTSNodeCacheArrayPurgeableSet(MCTSTree parentTree,
                                          int definitiveMaxCacheSize,
                                          int estimatedNumNodesInSearch)
    {
      ParentTree = parentTree;
      MaxCacheSize = definitiveMaxCacheSize;

      nodes = parentTree.Store.Nodes.nodes;

      // Compute number of subcaches, increasing as a function of estimates search size
      // (because degree of concurrency rises with size of batches and search.
      NumSubcaches = (int)StatUtils.Bounded(4 * MathF.Log2((float)estimatedNumNodesInSearch / 1000), 4, 32);

      // Initialize the sub-caches
      subCaches = new MCTSNodeCacheArrayPurgeable[NumSubcaches];
      for (int i = 0; i < NumSubcaches; i++)
      {
        subCaches[i] = new MCTSNodeCacheArrayPurgeable(parentTree, definitiveMaxCacheSize / NumSubcaches);
      }
    }


    /// <summary>
    /// Tree from which the MCTSNode objects originated.
    /// </summary>
    MCTSTree IMCTSNodeCache.ParentTree => ParentTree;


    /// <summary>
    /// Adds a specified node to the cache.
    /// </summary>
    /// <param name="node"></param>
    /// <returns>the internal index number assigned to this node</returns>
    public int Add(MCTSNode node)
    {
      int subcacheIndex = node.Index % NumSubcaches;
      int indexInSubcache = subCaches[subcacheIndex].Add(node);
      int cacheID = 1 + subcacheIndex + (indexInSubcache * MAX_SETS);
      node.Ref.CacheIndex = cacheID;
      
      Debug.Assert(cacheID != 0); // reserved for null
      //Console.WriteLine($" assigned cacheID={cacheID} from nodeIndex={node.Index} via {subcacheIndex} {indexInSubcache} ");
      return cacheID;
    }


    public MCTSNode Lookup(MCTSNodeStructIndex nodeIndex)
    {
      int cacheIndex = nodes[nodeIndex.Index].CacheIndex;
      return cacheIndex == 0 ? null 
                             : LookupByCacheID(cacheIndex);
    }


    /// <summary>
    /// Returns the MCTSNode stored in the cache 
    /// corresponding to specified MCTSNodeStruct
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    public MCTSNode Lookup(in MCTSNodeStruct nodeRef)
    {
      int cacheID = nodeRef.CacheIndex;
      return cacheID == 0 ? null : LookupByCacheID(cacheID);
    }


    /// <summary>
    /// Returns the MCTSNode at a specified cache index.
    /// </summary>
    /// <param name="cacheID"></param>
    /// <returns></returns>
    public MCTSNode LookupByCacheID(int cacheID)
    {
      int indexInCache = Math.DivRem(cacheID - 1, MAX_SETS, out int cacheSetNum);
      MCTSNode node = subCaches[cacheSetNum].AtIndex(indexInCache);
      Debug.Assert(node == null || node.Ref.CacheIndex == cacheID);
      return node;
    }


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
    /// Clears table entries and possibly resets back to null the CacheIndex for every node.
    /// </summary>
    /// <param name="resetNodeCacheIndex"></param>
    public void ResetCache(bool resetNodeCacheIndex)
    {
      Parallel.For(0, subCaches.Length, i => subCaches[i].ResetCache(resetNodeCacheIndex));
    }


    public override string ToString()
    {
      return $"<MCTSNodeCacheArrayPurgeableSet MaxSize={MaxCacheSize} NumInUse={NumInUse}>";
    }

  }

}





