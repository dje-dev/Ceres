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


using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.NodeCache;
using System;
using System.Threading;

#endregion

namespace Ceres.MCTS.LeafExpansion
{
  /// <summary>
  /// Implementor of IMCTSNodeCache interface for the simple case
  /// where the maximum number of nodes accessed can be guaranteed
  /// to not exceed some fixed size.
  /// 
  /// In this case the implementation is very simple and efficient,
  /// with direct indexing into a table of MCTSNode in same order as 
  /// underlying MCTSNodeStructs.
  /// </summary>
  public class MCTSNodeCacheArrayFixed : IMCTSNodeCache
  {
    public readonly int MaxNodes;

    public readonly MCTSTree ParentTree;

    #region Private data

    MCTSNode[] nodes;

    #endregion

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="parentTree"></param>
    /// <param name="maxNodes"></param>
    public MCTSNodeCacheArrayFixed(MCTSTree parentTree, int maxNodes)
    {
      ParentTree = parentTree;
      MaxNodes = maxNodes;
      nodes = new MCTSNode[maxNodes + 1];
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
      int nodeIndex = node.Index;

      nodes[nodeIndex] = node;
      node.StructRef.CacheIndex = nodeIndex;
      return nodeIndex;
    }


    /// <summary>
    /// Returns the MCTSNode having the specified index and stored in the cache
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    public MCTSNode Lookup(MCTSNodeStructIndex nodeIndex) => nodes[nodeIndex.Index];


    /// <summary>
    /// Returns the MCTSNode stored in the cache 
    /// corresponding to specified MCTSNodeStruct
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    public MCTSNode Lookup(in MCTSNodeStruct nodeRef) => nodes[nodeRef.CacheIndex];


    /// <summary>
    /// Possibly prunes the cache to remove some of the least recently accessed nodes.
    /// </summary>
    /// <param name="store"></param>
    public void PossiblyPruneCache(MCTSNodeStore store)
    {
      // Nothing to do since fixed caches do not purge.
    }


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
          if (nodes[i] != null)
          {
            nodes[i].StructRef.CacheIndex = 0;
          }
        }
      }

      Array.Clear(nodes, 0, nodes.Length);
    }
  }
}

#endif