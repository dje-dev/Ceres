
#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#if NOT_USED

#region Using directives

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Ceres.Base.DataTypes;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.NodeCache
{
  /// <summary>
  /// Uses a ConcurrentDictionary to maintain a cache of nodes.
  /// 
  /// Disadvantages of this approach include:
  ///   - the ConcurrentDictionary stresses the GC by creating many objects
  ///   - pruning the cache requires a lot of pointer-chasing
  /// Advantages include:
  ///   - the memory used by the CacheIndex field in MCTSNodeStruct is used/needed
  ///   
  /// </summary>
  public class MCTSNodeCacheDict : IMCTSNodeCache
  {
    public MCTSTree ParentTree;

    public readonly int MaxCacheSize;

    #region Private data

    const int EST_NUM_CONCURRENT = 50;
    internal ConcurrentDictionary<int, MCTSNode> nodeCache;
    int numCachePrunesInProgress = 0;

    #endregion

    /// <summary>
    /// Tree from which the MCTSNode objects originated.
    /// </summary>
    MCTSTree IMCTSNodeCache.ParentTree => ParentTree;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="parentTree"></param>
    /// <param name="initialCacheSize"></param>
    /// <param name="maxCacheSize"></param>
    public MCTSNodeCacheDict(MCTSTree parentTree, int initialCacheSize, int maxCacheSize)
    {
      ParentTree = parentTree;
      MaxCacheSize = maxCacheSize;
      nodeCache = new ConcurrentDictionary<int, MCTSNode>(EST_NUM_CONCURRENT, initialCacheSize);
    }

    /// <summary>
    /// Adds a specified node to the cache.
    /// </summary>
    /// <param name="node"></param>
    /// <returns>the internal index number assigned to this node</returns>
    public int Add(MCTSNode node) => nodeCache.TryAdd(node.Index, node) ? 1 : 0;


    /// <summary>
    /// Clears table entries and possibly resets back to null the CacheIndex for every node.
    /// </summary>
    /// <param name="resetNodeCacheIndex"></param>
    public void ResetCache(bool resetNodeCacheIndex)
    {
      if (resetNodeCacheIndex)
      {
        foreach (KeyValuePair<int, MCTSNode> kvp in nodeCache)
        {
          kvp.Value.StructRef.CacheIndex = 0;
        }
      }

      nodeCache.Clear();
    }

    /// <summary>
    /// Possibly prunes the cache to remove some of the least recently accessed nodes.
    /// </summary>
    /// <param name="store"></param>
    public void PossiblyPruneCache(MCTSNodeStore store)
    {
      if (numCachePrunesInProgress == 0 && nodeCache.Count > MaxCacheSize)
      {
        // Reduce to 80% of prior size
        Task.Run(() =>
        {
          //using (new TimingBlock("Prune"))
          {
            Interlocked.Increment(ref numCachePrunesInProgress);
            DictionaryUtils.PruneDictionary(nodeCache, a => a.LastAccessedSequenceCounter,
                                                (MaxCacheSize * 8) / 10);
            Interlocked.Decrement(ref numCachePrunesInProgress);
          };
        });
      }
    }

    /// <summary>
    /// Returns the MCTSNode stored in the cache 
    /// corresponding to specified MCTSNodeStruct
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    public MCTSNode Lookup(in MCTSNodeStruct nodeRef) => Lookup(nodeRef.Index);


    /// <summary>
    /// Returns the MCTSNode having the specified index and stored in the cache
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    public MCTSNode Lookup(MCTSNodeStructIndex nodeIndex)
    {
      bool alreadyInCache = false;

      MCTSNode cachedItem = default;
      alreadyInCache = nodeCache.TryGetValue(nodeIndex.Index, out cachedItem);

      return cachedItem;
    }
  }


}

#endif