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
using System.Collections;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Tasks;
using Ceres.Base.Benchmarking;
using Ceres.Base.OperatingSystem;
using Ceres.Base.Threading;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.PositionEvalCaching;
using Ceres.MCTS.Environment;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.MTCSNodes.Storage
{
  /// <summary>
  /// Manages extraction of some positions in a MCTSNodeStore
  /// into a cache object (repacking into alternate storage format).
  /// </summary>
  public static class MCTSNodeStorePositionExtractorToCache
  {
    public enum ExtractMode { ExtractNonRetained, ExtractRetained };

    /// <summary>
    /// Extracts nodes from store based on specified bitmap
    /// into a PositionEvalCache.
    /// </summary>
    /// <param name="store"></param>
    /// <param name="policySoftmax"></param>
    /// <param name="includedNodes"></param>
    /// <param name="newRoot"></param>
    /// <param name="extractMode"></param>
    /// <param name="cacheNodes"></param>
    /// <param name="transpositionRoots"></param>
    /// <exception cref="Exception"></exception>
    public static void ExtractPositionCacheNodes(MCTSNodeStore store, float policySoftmax,
                                                 BitArray includedNodes, in MCTSNodeStruct newRoot,
                                                 ExtractMode extractMode,
                                                 PositionEvalCache cacheNodes,
                                                 TranspositionRootsDict transpositionRoots)
    {
      // Parallel.For is used for efficiency.
      Debug.Assert(cacheNodes.SupportsConcurrency);

      int nonReachable = 0;

      int countGT1 = 0;
      int skipRootPresent = 0;

      MCTSTree tree = MCTSManager.ThreadSearchContext.Tree;
      int newRootIndex = newRoot.Index.Index;
      MemoryBufferOS<MCTSNodeStruct> nodes = store.Nodes.nodes;

      // TODO: can efficiency by improved? Refs can't be used in local functions.
      bool Reachable(int nodeIndex) => nodes[nodeIndex].IsPossiblyReachableFrom(in nodes[newRootIndex]);

      //using (new TimingBlock("Extract " + store.Nodes.NumTotalNodes))
      {
        Parallel.ForEach(Partitioner.Create(1, store.Nodes.NumTotalNodes),
          ParallelUtils.ParallelOptions(store.Nodes.NumTotalNodes, 1024),
          (range) =>
          {
            using (new MCTSNodeStoreContext(store))
            {
              for (int nodeIndex = range.Item1; nodeIndex < range.Item2; nodeIndex++)
              {
                ref MCTSNodeStruct nodeRef = ref nodes[nodeIndex];

                if ((includedNodes[nodeIndex] == (extractMode == ExtractMode.ExtractNonRetained)))
                /*|| nodeRef.IsOldGeneration*/
                {
                  continue;
                }

                if (nodeRef.ZobristHash == 0)
                {
                  // TODO: understand why a small number of uninitialized nodes
                  //       sometimes appear when swap root is enabled.
                  continue;
                }

                if (nodeRef.IsTranspositionLinked)
                {
                  continue;
                }

                // Filter out positions which are obviously impossible to be reached from new root
                // (considering pieces on board, pawns pushed, etc.).
                if (extractMode == ExtractMode.ExtractNonRetained 
                && !Reachable(nodeIndex))
                {
                  nonReachable++;
                  continue;
                }

                // Disable check of transposition since it is not worth the time taken,
                // only a small fraction of nodes are found and disqualified in this way.
                const bool CHECK_TRANSPOSITIONS = false;
                if (CHECK_TRANSPOSITIONS)
                {
                  if (transpositionRoots.TryGetValue(nodeRef.ZobristHash, out int transpositionRootIndex)
                    && includedNodes[transpositionRootIndex])
                  {
                    // The node in in the same transposition equivalence class as a node being retained, so no need to save it.
                    skipRootPresent++;
                    continue;
                  }
                }

                if (nodeRef.Terminal == Chess.GameResult.Unknown)
                {
                  CompressedPolicyVector policy = default;
                  MCTSNodeStructUtils.ExtractPolicyVector(policySoftmax, in nodeRef, ref policy);
                  cacheNodes.Store(nodeRef.ZobristHash, nodeRef.Terminal, nodeRef.WinP, nodeRef.LossP, nodeRef.MPosition, in policy);
                  if (nodeRef.N > 1) countGT1++;
                  //MCTSEventSource.TestCounter1++;
                }

              }
            }
          });


        //Console.WriteLine($"ExtractPositionCacheNodes store {store.RootNode.N} extracted {cacheNodes.Count} root_present {skipRootPresent}  nonr. {nonReachable} new root {newRoot.N} count>1 {countGT1}");
      }

    }
  }

}
