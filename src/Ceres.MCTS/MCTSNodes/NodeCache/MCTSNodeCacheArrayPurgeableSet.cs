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
    public MCTSTree ParentTree;

    public readonly int MaxCacheSize;

    #region Private data

    const int MAX_SUBCACHES = 16;

    MCTSNodeCacheArrayPurgeable[] subCaches;

    int numCachePrunesInProgress = 0;

    #endregion

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="parentTree"></param>
    /// <param name="maxCacheSize"></param>
    public MCTSNodeCacheArrayPurgeableSet(MCTSTree parentTree, int maxCacheSize)
    {
      ParentTree = parentTree;
      MaxCacheSize = maxCacheSize;

      // Initialize the sub-caches
      subCaches = new MCTSNodeCacheArrayPurgeable[MAX_SUBCACHES];
      for (int i = 0; i < MAX_SUBCACHES; i++)
        subCaches[i] = new MCTSNodeCacheArrayPurgeable(parentTree, maxCacheSize / MAX_SUBCACHES);
    }

    public void Clear() => throw new NotImplementedException();


    /// <summary>
    /// Tree from which the MCTSNode objects originated.
    /// </summary>
    MCTSTree IMCTSNodeCache.ParentTree => ParentTree;


    /// <summary>
    /// Returns the MCTSNode having the specified index and stored in the cache
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    public void Add(MCTSNode node) => subCaches[node.Index % MAX_SUBCACHES].Add(node);
    

    /// <summary>
    /// Returns the number of nodes currently present in the cache.
    /// </summary>
    int NumInUse
    {
      get
      {
        int numInUse = 0;
        for (int i = 0; i < subCaches.Length; i++)
          numInUse += subCaches[i].NumInUse;
        return numInUse;
      }
    }


    /// <summary>
    /// Possibly prunes the cache to remove some of the least recently accessed nodes.
    /// </summary>
    /// <param name="store"></param>
    public void PossiblyPruneCache(MCTSNodeStore store)
    {
      bool almostFull = NumInUse > (MaxCacheSize * 85) / 100;
      if (numCachePrunesInProgress == 0 && almostFull)
      {
        Interlocked.Increment(ref numCachePrunesInProgress);

        int countPurged = 0;
        Parallel.ForEach(Enumerable.Range(0, MAX_SUBCACHES),
                         new ParallelOptions() { MaxDegreeOfParallelism = 4 }, // memory access already saturated at 4
          i =>
          {
            Interlocked.Add(ref countPurged, subCaches[i].Prune(store, -1));
          });

        Interlocked.Decrement(ref numCachePrunesInProgress);
      }
    }


    /// <summary>
    /// Returns the MCTSNode having the specified index and stored in the cache
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    public MCTSNode Lookup(MCTSNodeStructIndex nodeIndex)
      => subCaches[nodeIndex.Index % MAX_SUBCACHES].Lookup(nodeIndex);


    /// <summary>
    /// Resets back to null (zero) the CacheIndex for every node currently in the cache.
    /// </summary>
    public void ClearMCTSNodeStructValues()
    {
      for (int i = 0; i < subCaches.Length; i++)
        subCaches[i].ClearMCTSNodeStructValues();
    }

    public override string ToString()
    {
      return $"<MCTSNodeCacheArrayPurgeableSet MaxSize={MaxCacheSize} NumInUse={NumInUse}>";
    }

  }

}





