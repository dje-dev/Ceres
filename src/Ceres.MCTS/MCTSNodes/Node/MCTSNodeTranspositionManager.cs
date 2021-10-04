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
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

[assembly: InternalsVisibleTo("Ceres.EngineMCTS.Test")] // TODO: move or remove me.

namespace Ceres.MCTS.MTCSNodes
{
  /// <summary>
  /// Static helper class which manages 
  /// the handling of transpositions including tracking
  /// or resolving reference between nodes that are pending
  /// transposition copies.
  /// </summary>
  public static class MCTSNodeTranspositionManager
  {
    /// <summary>
    /// Returns the node which is the root of the cluster (possibly same as node).
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    public static int CheckAddToCluster(MCTSNode node)
    {
      int rootIndex;
      if (!node.Tree.TranspositionRoots.TryGetValue(node.StructRef.ZobristHash, out rootIndex))
      {
        throw new Exception("Internal error");
#if NOT
        // We are the new root, just add to roots table and exit
        bool added = node.Context.TranspositionRoots.TryAdd(node.Annotation.PositionHashForCaching, node.Index);

        // If we failed to add, this means this node was already added in the interim
        // Therefore recursively call ourself so that we can get our self added to the end of the list
        if (!added)
          return CheckAddToCluster(node);
        else
          return node.Index;
#endif
      }
      else
      {
        // Cluster already exists. Apppend ourself
        ref MCTSNodeStruct traverseRef = ref node.Store.Nodes.nodes[rootIndex];
        while (true)
        {
          if (traverseRef.NextTranspositionLinked == 0) 
            break;

          traverseRef = ref node.Store.Nodes.nodes[traverseRef.NextTranspositionLinked];
        }

        // Tack ourself onto the end
        // TODO: could we more efficiently put ourself at beginning?
        // TODO: concurrency?
        if (traverseRef.Index.Index != node.Index)
          traverseRef.NextTranspositionLinked = node.Index;

        return rootIndex;
      }

    }

    public static ref MCTSNodeStruct GetNodeWithMaxNInCluster(MCTSNode node)
    {
      int rootIndex;
      if (!node.Tree.TranspositionRoots.TryGetValue(node.StructRef.ZobristHash, out rootIndex))
        return ref node.Store.Nodes.nodes[0]; // TODO clean up 

      int maxN = 0;
      ref MCTSNodeStruct bestNodeRef = ref node.StructRef;

      ref MCTSNodeStruct traverseNodeRef = ref node.Store.Nodes.nodes[rootIndex];
      int count = 0;
      while (true)
      {
        count++;
        if (count > 100_000) throw new Exception("Internal error: apparent cycle in transposition cluster linked list");

        if (traverseNodeRef.N >= maxN)
        {
          maxN = traverseNodeRef.N;
          bestNodeRef = ref traverseNodeRef;
        }

        if (traverseNodeRef.NextTranspositionLinked == 0)
          break;
        else
          traverseNodeRef = ref node.Store.Nodes.nodes[traverseNodeRef.NextTranspositionLinked];
      }

      return ref bestNodeRef;
    }
  }
}

