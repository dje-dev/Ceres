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
using Ceres.Base.Threading;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.PositionEvalCaching;
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

//      using (new TimingBlock("Extract " + store.Nodes.NumTotalNodes))
      {
        Parallel.ForEach(Partitioner.Create(1, store.Nodes.NumTotalNodes),
//          new ParallelOptions() { MaxDegreeOfParallelism = 1},
          (range) =>
          {
            using (new MCTSNodeStoreContext(store))
            {
              for (int nodeIndex = range.Item1; nodeIndex < range.Item2; nodeIndex++)
              {
                ref MCTSNodeStruct nodeRef = ref store.Nodes.nodes[nodeIndex];

                if ((includedNodes[nodeIndex] == (extractMode == ExtractMode.ExtractNonRetained))
                  /*|| nodeRef.IsOldGeneration*/
                  )
                {
                  continue;
                }
                {
                  if (nodeRef.IsTranspositionLinked)
                  {
                    continue;
                  }

                  // TODO: someday filter out impossible positions given new root
                  // (consider pieces on board, pawns pushed castling rights, etc.).
                  // The idea of using MGPositionReachability fails because:
                  //   -- probably too slow run the test, and
                  //   -- requires a call to GetNode which overflows node cache

                  // TODO: possibly try alternate method of just comparing piece counts
                  //       (use 5 bits to store in MCTSNodeStructMiscFields)
                  //MGPosition thisPos = tree.GetNode(new MCTSNodeStructIndex(nodeIndex)).Annotation.PosMG;
                  bool reachable = true;// MGPositionReachability.IsProbablyReachable(in newRootPos, in thisPos);
                  if (!reachable)
                  {
                    nonReachable++;
                    continue;
                  }

                  CompressedPolicyVector policy = default;
                  MCTSNodeStructUtils.ExtractPolicyVector(policySoftmax, in nodeRef, ref policy);

                  if (nodeRef.ZobristHash == 0)
                  {
                    throw new Exception("Internal error: node encountered without hash");
                  }
                  // TODO: could the cast to FP16 below lose useful precision? Perhaps save as float instead

                  if (transpositionRoots.TryGetValue(nodeRef.ZobristHash, out int transpositionRootIndex)
                    && includedNodes[transpositionRootIndex])
                  {
                    // The node in in the same transposition equivalence class as a node being retained, so no need to save it.
                    skipRootPresent++;
                    continue;
                  }

                  if (nodeRef.Terminal == Chess.GameResult.Unknown)
                  {
                    cacheNodes.Store(nodeRef.ZobristHash, nodeRef.Terminal, nodeRef.WinP, nodeRef.LossP, nodeRef.MPosition, in policy);
                    if (nodeRef.N > 1) countGT1++;
                  }
                }
              }
            }
          });


//        Console.WriteLine($"ExtractPositionCacheNodes store {store.RootNode.N} extracted {cacheNodes.Count} root_present {skipRootPresent}  nonr. {nonReachable} new root {newRoot.N} count>1 {countGT1}");
      }

    }
  }

}
