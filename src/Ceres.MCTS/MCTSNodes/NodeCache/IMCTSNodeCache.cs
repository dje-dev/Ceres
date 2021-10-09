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


using Ceres.MCTS.Iteration;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.NodeCache
{
  /// <summary>
  /// Interface implemented by objects that provide object node caching
  /// by maintaining a working set of MCTSNode objects indexed by node index.
  /// 
  /// A least recently used (LRU) strategy may be used to
  /// evict nodes when the cache approaches maximum capacity.
  /// </summary>
  public unsafe interface IMCTSNodeCache
  {
    /// <summary>
    /// Store from which the MCTSNode objects originated.
    /// </summary>
    MCTSNodeStore ParentStore { get; }

    /// <summary>
    /// Sets/resets the node context to which the cached items belong.
    /// </summary>
    /// <param name="context"></param>
    void SetContext(MCTSIterator context);

    /// <summary>
    /// Adds a specified node to the cache.
    /// </summary>
    /// <param name="node"></param>
    /// <returns>pointer to space allocated for MCTSNodeInfo</returns>
    void* Add(MCTSNodeStructIndex node);

    /// <summary>
    /// Returns the MCTSNode having the specified index and stored in the cache
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    void* Lookup(MCTSNodeStructIndex nodeIndex);

    /// <summary>
    /// Returns the MCTSNode stored in the cache 
    /// corresponding to specified MCTSNodeStruct
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    void* Lookup(in MCTSNodeStruct nodeRef);

    /// <summary>
    /// Possibly prunes the cache to remove some of the least recently accessed nodes.
    /// </summary>
    /// <param name="store"></param>
    void PossiblyPruneCache(MCTSNodeStore store);

    /// <summary>
    /// Removes a specified node from the cache, if present.
    /// </summary>
    /// <param name="nodeIndex"></param>
    void Remove(MCTSNodeStructIndex nodeIndex);

    /// <summary>
    /// Clears table entries and possibly resets back to null the CacheIndex for every node.
    /// </summary>
    /// <param name="resetNodeCacheIndex"></param>
    void ResetCache(bool resetNodeCacheIndex);

    /// <summary>
    /// Nodes in node cache are stamped with the sequence number
    /// of the last batch in which they were accessed to faciltate LRU determination.
    /// </summary>
    int NextBatchSequenceNumber { get; set; }
  }
}
