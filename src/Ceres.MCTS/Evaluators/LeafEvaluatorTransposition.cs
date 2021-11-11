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
using System.Runtime.CompilerServices;
using System.Threading;
using Ceres.Base.Environment;
using Ceres.Base.Threading;
using Ceres.Chess;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Annotation;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Evaluators
{
  /// <summary>
  /// Node evaluator which can yield evaluations by other nodes
  /// representing the same position which are already extant elsewhere in the search tree.
  /// </summary>
  public sealed partial class LeafEvaluatorTransposition : LeafEvaluatorBase
  {
    // If TestMetric and TestCounter should track the number of
    // virtual visits whih obviate node allocation (TestCounter1) 
    // and also the number of nodes which subsequently had to be materialized (TestMetric)
    // and were therefore not ultimately actual memory savings.
    // Disabled by default due to performance impact (especially due to false sharing).
    internal const bool TRACK_VIRTUAL_VISITS = false;


    /// <summary>
    /// Maintain data structure to map between position hash codes and 
    /// indices of nodes already extant in the search tree store.
    /// </summary>
    public readonly TranspositionRootsDict TranspositionRoots;

    /// <summary>
    /// Record new pending transposition roots accumulated during a batch
    /// so they can be added to the TranspositionRoots dictionary 
    /// all at once at end of batch collection.
    /// </summary>
    int nextIndexPendingTranspositionRoots;
    (ulong, int)[] pendingTranspositionRoots;

    const bool VERBOSE = false;
    static int WARN_COUNT = 0;

    MCTSTree tree;

    #region Statistics tracking

    internal static AccumulatorMultithreaded NumHits;
    internal static AccumulatorMultithreaded NumMisses;
    internal static AccumulatorMultithreaded NumHitsOldGeneration;

    public static float HitRatePct => 100.0f * (float)NumHits.Value / (float)(NumHits.Value + NumMisses.Value);
    public static float HitRateOldGenerationPct => 100.0f * (float)NumHitsOldGeneration.Value / (float)(NumHits.Value + NumMisses.Value);

    #endregion


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="treeRoots"></param>
    /// <param name="transpositionRoots"></param>
    public LeafEvaluatorTransposition(MCTSTree tree, TranspositionRootsDict transpositionRoots)
    {
      TranspositionRoots = transpositionRoots;
      this.tree = tree;

      // TODO: currently hardcoded at 2048, potentially
      // dynamically detemrine possibly smaller necessary value
      pendingTranspositionRoots = new (ulong, int)[2048];
    }



  /// <summary>
  /// Called when a node pending evaluation is found to be the same position
  /// (for the purposes of transposition equivalence classes)
  /// as another node in the tree.
  /// </summary>
  /// <param name="node"></param>
  /// <param name="transpositionRootNodeIndex"></param>
  /// <param name="transpositionRootNode"></param>
  /// <returns></returns>
  LeafEvaluationResult ProcessFirstLinkage(MCTSNode node, MCTSNodeStructIndex transpositionRootNodeIndex,
                                             in MCTSNodeStruct transpositionRootNode)
    {
      Debug.Assert(transpositionRootNodeIndex.Index != 0);

      TranspositionMode transpositionMode = node.Context.ParamsSearch.Execution.TranspositionMode;
      if (transpositionMode == TranspositionMode.SingleNodeCopy
       || transpositionRootNode.Terminal.IsTerminal())
      {
        if (CeresEnvironment.MONITORING_METRICS) NumHits.Add(1, node.Index);
        node.StructRef.CopyUnexpandedChildrenFromOtherNode(node.Tree, transpositionRootNodeIndex);
        node.InfoRef.TranspositionRootNodeIndex = transpositionRootNodeIndex;
      }
      else if (transpositionMode == TranspositionMode.SingleNodeDeferredCopy
            || transpositionMode == TranspositionMode.SharedSubtree)
      {
        // Note that no need to increment NumHits here since this happens elsewhere (when pending value is used).
        SetTranspositionRootReuseFields(node, transpositionRootNodeIndex, in transpositionRootNode);
        node.InfoRef.TranspositionRootNodeIndex = transpositionRootNodeIndex;
      }
      else
      {
        throw new NotImplementedException();
      }


      if (/*node.Context.ParamsSearch.TranspositionUseTransposedQ &&*/ transpositionMode != TranspositionMode.SharedSubtree)
      {
#if NOT
        // We deviate from pure MCTS and 
        // set the backed up node for this leaf node to be the 
        // overall score (Q) from the complete linked tranposition subtree
        if (node.Context.ParamsSearch.TranspositionUseCluster)
        {
          // Take the Q from the largest subtree (presumably the most accurate value)
          ref MCTSNodeStruct biggestTranspositionClusterNode = ref MCTSNodeTranspositionManager.GetNodeWithMaxNInCluster(node);
          node.OverrideVToApplyFromTransposition = (FP16)biggestTranspositionClusterNode.Q;
          node.OverrideMPositionToApplyFromTransposition = (FP16)biggestTranspositionClusterNode.MPosition;
        }
        else
        {
          node.OverrideVToApplyFromTransposition = (FP16)transpositionRootNode.Q;
          node.OverrideMPositionToApplyFromTransposition = (FP16)transpositionRootNode.MPosition;
        }

#endif
      }

      return new LeafEvaluationResult(transpositionRootNode.Terminal, transpositionRootNode.WinP,
                                      transpositionRootNode.LossP, transpositionRootNode.MPosition);
    }


    /// <summary>
    /// Implements virtual method to try to resolve evaluation of a specified node
    /// from the transposition data structures.
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    protected override LeafEvaluationResult DoTryEvaluate(MCTSNode node)
    {
      Debug.Assert(float.IsNaN(node.V));

      if (node.StructRef.HasRepetitions)
      {
        // Avoid linking if node already has transpositions
        // because of possibility linked node will differ in history of tranpositions
        return default;
      }

      if (node.Context.ParamsSearch.Execution.TranspositionMode == TranspositionMode.SingleNodeDeferredCopy
       && node.IsTranspositionLinked)
      {
        // Node is already being used for transpositions.
        return default;
      }

      if (node.Context.ParamsSearch.Execution.TranspositionMode == TranspositionMode.SharedSubtree)
      //&& node.Context.ParamsSearch.Execution.TRANSPOSITION_SINGLE_TREE
      //       && !float.IsNaN(node.OverrideVToApplyFromTransposition))
      {
        throw new NotImplementedException();

        // If the selector already stopped descent here 
        // due to a tranpsosition cluster virtual visit,
        // return a NodeEvaluationResult to 
        // indicate this can be processed without NN
        // (and use an NodeEvaluationResult which leaves unchanged)
        //return new LeafEvaluationResult(node.Terminal, node.WinP, node.LossP, node.MPosition);
      }

      // Check if this position already exists in the tree
      if (TranspositionRoots.TryGetValue(node.StructRef.ZobristHash, out int transpositionNodeIndex))
      {
        // Found already existing node
        ref readonly MCTSNodeStruct transpositionNode = ref node.Store.Nodes.nodes[transpositionNodeIndex];

        if (transpositionNodeIndex == 1)
        {
          return default;
        }

        if (!transpositionNode.IsValidTranspositionLinkedSource)
        {
          return default;
        }

        if (CeresEnvironment.MONITORING_METRICS)
        {
          if (transpositionNode.IsOldGeneration)
          {
            NumHitsOldGeneration.Add(1, node.Index);
          }
        }

        ref readonly MCTSNodeAnnotation annotationThisNode = ref node.Annotation;
#if NOT
        MCTSNode transpositionNodeItem = tree.GetNode(new MCTSNodeStructIndex(transpositionNodeIndex));
        transpositionNodeItem.Annotate();
        ref readonly MCTSNodeAnnotation annotationTranspositionNode = ref transpositionNodeItem.annotation;

        bool posIdentical = transpositionNodeItem.Annotation.Pos.PiecesShortHash == node.Annotation.Pos.PiecesShortHash
                         && transpositionNodeItem.Annotation.PosMG.A == annotationThisNode.PosMG.A
                         && transpositionNodeItem.Annotation.PosMG.B == annotationThisNode.PosMG.B
                         && transpositionNodeItem.Annotation.PosMG.C == annotationThisNode.PosMG.C
                         && transpositionNodeItem.Annotation.PosMG.D == annotationThisNode.PosMG.D;

        if (!posIdentical)
        {
          Console.WriteLine("no p" + annotationThisNode.Pos.FEN + " ");
        }
#endif

#if NOT
        // Make sure the linked position is actually the same.
        // In theory due to collisions perahps this would not be the case
        // (and we might need to use the HashCrosscheck to avoid)
        // but in practice does not seem to happen.
        Span<EncodedMove> moves = stackalloc EncodedMove[annotationThisNode.Moves.NumMovesUsed];
        for (int i = 0; i < annotationThisNode.Moves.NumMovesUsed; i++)
        {
          moves[i] = transpositionNode.ChildAtIndex(i).Move;
        }
        for (int i = 0; i < annotationThisNode.Moves.NumMovesUsed; i++)
        {
          EncodedMove moveThisPos = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(annotationThisNode.Moves.MovesArray[i]);
          if (MemoryExtensions.IndexOf(moves, moveThisPos) == -1)
          {
            Console.WriteLine("Warning: transposition link is to different position! fen= " + annotationThisNode.Pos.FEN);
            Console.WriteLine(moveThisPos.RawValue + " "
              + annotationThisNode.Moves.MovesArray[i] + " "
              + annotationThisNode.Moves.MovesArray[i].FromSquareIndex + " "
              + annotationThisNode.Moves.MovesArray[i].ToSquareIndex + " "
              + annotationThisNode.Moves.MovesArray[i].Flags + " ");
          }
        }
#endif        


        if (node.Context.ParamsSearch.TranspositionUseCluster)
        {
          MCTSNodeTranspositionManager.CheckAddToCluster(node);
        }

        return ProcessFirstLinkage(node, new MCTSNodeStructIndex(transpositionNodeIndex), in transpositionNode);
      }
      else
      {
        // Never allow root to be transposition root.
        if (node.Index != 1)
        {
          int pendingIndex = Interlocked.Increment(ref nextIndexPendingTranspositionRoots) - 1;
          pendingTranspositionRoots[pendingIndex] = (node.StructRef.ZobristHash, node.Index);
        }

        if (CeresEnvironment.MONITORING_METRICS)
        {
          NumMisses.Add(1, node.Index);
        }

        return default;
      }
    }


    /// <summary>
    /// BatchPostprocess is called at the end of gathering a batch of leaf nodes.
    /// It is guaranteed that no other operations are concurrently active at this time.
    /// </summary>
    public override void BatchPostprocess()
    {
      if (nextIndexPendingTranspositionRoots > 0)
      {
        Span<MCTSNodeStruct> nodes = tree.Store.Nodes.Span;

        // Add in all the accumulated transposition roots to the dictionary.
        // This is done in postprocessing for efficiency because it obviates any locking.
        for (int i = 0; i < nextIndexPendingTranspositionRoots - 1; i++)
        {
          (ulong, int) rootItem = pendingTranspositionRoots[i];
          TranspositionRoots.TryAdd(rootItem.Item1, rootItem.Item2, rootItem.Item2, nodes);
        }

        nextIndexPendingTranspositionRoots = 0;
      }
    }

    [ModuleInitializer]
    internal static void ModuleInitialize()
    {
      NumHits.Initialize();
      NumMisses.Initialize();
      NumHitsOldGeneration.Initialize();
    }

  }

  }
