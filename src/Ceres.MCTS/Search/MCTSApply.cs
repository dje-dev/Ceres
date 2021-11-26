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
using System.Threading.Tasks;
using System.Diagnostics;
using System.Collections.Generic;
using System.Collections.Concurrent;

using Ceres.Base.DataTypes;
using Ceres.Base.Math.Probability;
using Ceres.Base.Threading;

using Ceres.Chess;
using Ceres.Chess.PositionEvalCaching;

using Ceres.MCTS.Evaluators;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.Params;
using System.Reflection;
using Ceres.MCTS.Environment;
using Ceres.MCTS.LeafExpansion;

#endregion

namespace Ceres.MCTS.Search
{
  /// <summary>
  /// Manages updating of tree nodes to reflect evaluation results 
  /// that have been returned by  neural network or other evalluators
  /// (and have been stored in the node's LeafEvaluationResult field).
  /// </summary>
  public class MCTSApply
  {
    public MultinomialBayesianThompsonSampler FirstMoveSampler;

    /// <summary>
    /// Cumulative number of nodes applied.
    /// </summary>
    public int NumNodesApplied;

    /// <summary>
    /// Cumulative number of batches of nodes applied.
    /// </summary>
    public int NumBatchesApplied;


    /// <summary>
    /// Global cumulative number of nodes applied.
    /// </summary>
    internal static long TotalNumNodesApplied;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="firstMoveSampler"></param>
    public MCTSApply(MultinomialBayesianThompsonSampler firstMoveSampler)
    {
      FirstMoveSampler = firstMoveSampler;
    }


    /// <summary>
    /// Applies the results for all nodes (originating from a specified selector).
    /// </summary>
    /// <param name="selectorID"></param>
    /// <param name="nodes"></param>
    internal void Apply(int selectorID, ListBounded<MCTSNode> nodes)
    {
      if (nodes.Count > 0)
      {
        DoApply(selectorID, nodes);

        NumNodesApplied += nodes.Count;
        NumBatchesApplied++;
        TotalNumNodesApplied += nodes.Count;
      }
    }


    /// <summary>
    /// Diagnostic method to verify that there are no nodes duplicated in a batch,
    /// and that all of the nodes have NInFlight and NInFlight2 equal to zero.
    /// </summary>
    /// <param name="nodes"></param>
    [Conditional("DEBUG")]
    public static void DebugVerifyNoDuplicatesAndInFlight(ListBounded<MCTSNode> nodes)
    {
      HashSet<MCTSNode> nodesSet = new HashSet<MCTSNode>(nodes.Count);
      foreach (MCTSNode node in nodes)
      {
        if (node.ActionType == MCTSNodeInfo.NodeActionType.MCTSApply
          && !node.Terminal.IsTerminal()
          && node.NInFlight == 0
          && node.NInFlight2 == 0)
          throw new Exception($"Internal error: node was generated but not marked in flight");
        if (nodesSet.Contains(node))
          throw new Exception($"Internal error: duplicate node found in Apply  { node }");
        else
          nodesSet.Add(node);
      }

    }


    /// <summary>
    /// Coordinates (possibly parallelized) application of 
    /// evauation results for all nodes in a specified batch.
    /// </summary>
    /// <param name="selectorID"></param>
    /// <param name="batchlet"></param>
    void DoApply(int selectorID, ListBounded<MCTSNode> batchlet)
    {
      DebugVerifyNoDuplicatesAndInFlight(batchlet);

      if (batchlet.Count == 0)
      {
        return;
      }

      MCTSIterator context = batchlet[0].Context;

      if (batchlet.Count > context.ParamsSearch.Execution.SetPoliciesNumPoliciesPerThread)
      {
        Parallel.Invoke(
          () => { DoApplySetPolicies(batchlet); },
          () => { using (new SearchContextExecutionBlock(context)) DoApplyBackup(selectorID, batchlet); });
      }
      else
      {
        DoApplySetPolicies(batchlet);
        DoApplyBackup(selectorID, batchlet);
      }
#if CRASHES
      // The main two operations to be performed are independently and
      // can possibly be performed in parallel
      const int PARALLEL_THRESHOLD = MCTSParamsFixed.APPLY_NUM_POLICIES_PER_THREAD + (MCTSParamsFixed.APPLY_NUM_POLICIES_PER_THREAD / 2);
      if (false && MCTSParamsFixed.APPLY_PARALLEL_ENABLED && batchlet.Count > PARALLEL_THRESHOLD)
      {
        Parallel.Invoke
          (
            () => DoApplySetPolicies(batchlet),
            () => DoApplyBackup(batchlet)
          );
      }
      else
      {
        DoApplySetPolicies(batchlet); // must go first
        DoApplyBackup(batchlet);
      }
      //foreach (var node in batchlet.Nodes) node.EvalResult = null;
#endif

    }


    /// <summary>
    /// Implements apply logic for nodes in a batch.
    /// </summary>
    /// <param name="nodes"></param>
    /// <param name="selectorID"></param>
    /// <param name="node"></param>
    /// <param name="evalResult"></param>
    void ApplyResult(Span<MCTSNodeStruct> nodes, int selectorID, MCTSNode node, in LeafEvaluationResult evalResult)
    {
      ref MCTSNodeStruct nodeRef = ref node.StructRef;

      MCTSNodeStructIndex indexOfChildDescendentFromRoot = default;

      nodeRef.VSecondary = FP16.NaN;

      int numUpdateSelector1 = selectorID == 0 ? nodeRef.NInFlight : 0;
      int numUpdateSelector2 = selectorID == 1 ? nodeRef.NInFlight2 : 0;
      int numInFlight = numUpdateSelector1 + numUpdateSelector2;

      if (node.ActionType == MCTSNodeInfo.NodeActionType.None)
      {
        nodeRef.BackupDecrementInFlight(numUpdateSelector1, numUpdateSelector2);
      }
      else if (node.ActionType == MCTSNodeInfo.NodeActionType.CacheOnly)
      {
        // TODO: needs remediation
        throw new NotImplementedException(); // need to set value on node ? (V, WinP, LossP, M).
      }
      else if (node.ActionType == MCTSNodeInfo.NodeActionType.MCTSApply)
      {
        // If first time visiting this node then update values on the node itself.
        if (nodeRef.N == 0)
        {
          SetNodeEvaluationValues(node);
        }

        bool wasTerminal = nodeRef.Terminal.IsTerminal();

        // If we are revisiting a terminal or transposition linked node,
        // just reiterate the prior evaluation.
        if (nodeRef.N > 0 && node.EvalResult.IsNull)
        {
          Debug.Assert(wasTerminal || node.NumVisitsPendingTranspositionRootExtraction > 0);
          node.EvalResult = new LeafEvaluationResult(nodeRef.Terminal, nodeRef.WinP, nodeRef.LossP, nodeRef.MPosition);
        }

        // First visit by any selector, set evaluation result and backup in tree
        bool allowMultivisits = wasTerminal || node.NumVisitsPendingTranspositionRootExtraction > 1;
        int numToApply = allowMultivisits ? numInFlight : 1;

        float dToApply;
        float vToApply;
        float mToApply;

        if (node.NumVisitsPendingTranspositionRootExtraction == 0)
        {
          vToApply = evalResult.V;
          dToApply = evalResult.DrawP;
          mToApply = evalResult.M;
        }
        else
        {
          Debug.Assert(nodeRef.N > 0 || !float.IsNaN(node.EvalResult.V));
          Debug.Assert(!FP16.IsNaN(node.PendingTranspositionV));
          Debug.Assert(nodeRef.TranspositionRootIndex != 0);

          numToApply = Math.Min(nodeRef.NumVisitsPendingTranspositionRootExtraction, numToApply);

          // Increment count of number of "extra" (beyond 1) values used without tree replication.
          if (LeafEvaluatorTransposition.TRACK_VIRTUAL_VISITS && nodeRef.NumVisitsPendingTranspositionRootExtraction > 1)
          {
            MCTSEventSource.TestCounter1++;
          }

          // Switch to propagate this "pseudo V" for this node and all nodes above
          vToApply = node.PendingTranspositionV;
          mToApply = node.PendingTranspositionM;
          dToApply = node.PendingTranspositionD;

          // Now we've used up more visits, decrement count.
          nodeRef.NumVisitsPendingTranspositionRootExtraction -= numToApply;
        }

        float vToApplyFirst = vToApply;
        float vToApplyNonFirst;
        float dToApplyFirst = dToApply;
        float dToApplyNonFirst;

        // Check if a sibling eval was previously calculated.
        if (node.SiblingEval is not null)
        {
          (vToApplyNonFirst, dToApplyNonFirst) = node.SiblingEval.Value.BackupValueForNode(node, vToApply, dToApply);
          node.SiblingEval = null;
        }
        else
        {
          vToApplyNonFirst = vToApply;
          dToApplyNonFirst = dToApply;
        }

        float contemptAdjustment = node.Context.CurrentContempt;
        if (contemptAdjustment != 0 && !float.IsNaN(dToApply))
        {
          contemptAdjustment *= dToApply * (node.IsOurMove ? -1 : 1);
          vToApply += contemptAdjustment;
        }

        Debug.Assert(!float.IsNaN(vToApply));

        // Substitute auxilliary value if present.
        if (!float.IsNaN(node.InfoRef.EvalResultAuxilliary))
        {
          if (node.InfoRef.EvalResultAuxilliary == 0)
          {
            vToApplyNonFirst = vToApplyFirst = 0;
            dToApplyNonFirst = dToApplyFirst = 1;
          }
          else
          {
            // Use both the tablebase evaluation (which is correct)
            // but also some from neural network evaluation which provides
            // some gradient toward the actual winning sequence.
            vToApplyNonFirst = 0.5f * (vToApplyNonFirst + node.InfoRef.EvalResultAuxilliary);
            vToApplyFirst = 0.5f * (vToApplyFirst + node.InfoRef.EvalResultAuxilliary);
            dToApplyNonFirst = dToApplyFirst = 0;
          }
        }

        nodeRef.BackupApply(nodes, numToApply, 
                            vToApplyFirst, vToApplyNonFirst, mToApply, 
                            dToApplyFirst, dToApplyNonFirst, 
                            numUpdateSelector1, numUpdateSelector2, out indexOfChildDescendentFromRoot);

        PossiblyUpdateFirstMoveSampler(node, indexOfChildDescendentFromRoot, numUpdateSelector1, numUpdateSelector2, wasTerminal, vToApply);

        // Update depth statistic
        int depth = node.Depth;
        if (numToApply == 1)
        {
          node.Context.CumulativeSelectedLeafDepths.Add(depth, depth + node.PriorMove.RawValue);
        }
        else
        {
          node.Context.CumulativeSelectedLeafDepths.Add(depth * numToApply, depth + node.PriorMove.RawValue);
        }
      }
      else
      {
        throw new Exception("Internal error, unknown NodeActionType");
      }


      if (FirstMoveSampler != null)
      {
        // this mismatch can/will happen with tree reuse
        //        if (node.Context.Root.N > 1 && node.Context.Root.ChildAtIndex(0).N != FirstMoveSampler.childrenDistributions[0].NumSamples)
        //          Console.WriteLine("first mismatch on zero");
      }
      node.Context.RecordVisitToTopLevelMove(node, indexOfChildDescendentFromRoot, evalResult);
    }


    private static void SetNodeEvaluationValues(MCTSNode node)
    {
      Debug.Assert(!node.EvalResult.IsNull);

      ref MCTSNodeStruct nodeRef = ref node.StructRef;
      ref readonly LeafEvaluationResult nodeEvalResult = ref node.EvalResult;

      nodeRef.Terminal = node.EvalResult.TerminalStatus;
      nodeRef.HasRepetitions = node.Annotation.Pos.MiscInfo.RepetitionCount > 0;

      if (ParamsSelect.VIsForcedLoss(node.EvalResult.V))
      {
        nodeRef.SetProvenLossAndPropagateToParent(nodeEvalResult.LossP, node.EvalResult.M);
      }
      else
      {
        nodeRef.WinP = nodeEvalResult.WinP;
        nodeRef.LossP = nodeEvalResult.LossP;
        nodeRef.MPosition = (byte)MathF.Round(nodeEvalResult.M, 0);
      }

      if (!node.IsRoot && nodeRef.Terminal == GameResult.Draw)
      {
        nodeRef.ParentRef.DrawKnownToExistAmongChildren = true;
        //Chess.NNEvaluators.LC0DLL.LC0DLLSyzygyEvaluator.NumTablebaseMisses++;
      }
    }


    private void PossiblyUpdateFirstMoveSampler(MCTSNode node, MCTSNodeStructIndex indexOfChildDescendentFromRoot, int numUpdateSelector1, int numUpdateSelector2, bool wasTerminal, float vToApply)
    {
      if (FirstMoveSampler != null && indexOfChildDescendentFromRoot != default)
      {
        MCTSNode root = node.Context.Root;
        int childIndex = -1;

        // Find the child of the root having this index
        for (int i = 0; i < root.NumChildrenExpanded; i++)
        {
          if (root.ChildAtIndex(i).Index == indexOfChildDescendentFromRoot.Index)
          {
            childIndex = i;
            break;
          }
        }

        Debug.Assert(childIndex != -1);
        int numTimesToApply = wasTerminal ? numUpdateSelector1 + numUpdateSelector2 : 1; // only terminals applied mulitple times
        FirstMoveSampler.AddSample(childIndex, node.IsOurMove ? -vToApply : vToApply, numTimesToApply);
      }
    }


    /// <summary>
    /// Calls ApplyResult for every node in a specified list of nodes.
    /// </summary>
    /// <param name="selectorID"></param>
    /// <param name="nodes"></param>
    void DoApplyBackup(int selectorID, ListBounded<MCTSNode> nodes)
    {
      if (nodes.Count == 0) return;

      Span<MCTSNodeStruct> nodesSpan = nodes[0].Store.Nodes.nodes.Span;

      bool refreshTranspositionRoots = nodes[0].Context.ParamsSearch.TranspositionRootMaxN;
      MCTSTree tree = nodes[0].Tree;

      // Note that this is not parallelized to avoid updates
      // unsynchronized updates to fields in nodes higher up in the tree
      for (int i = 0; i < nodes.Count; i++)
      {
        MCTSNode node = nodes[i];
        ApplyResult(nodesSpan, selectorID, node, in node.EvalResult);

        // Possibly refresh transposition table if node now has greater N than current root
        const int MIN_N = 1; // for efficiency, possibly don't bother if very small
        if (refreshTranspositionRoots 
         && node.N > MIN_N 
         && !node.StructRef.IsTranspositionRoot)
        {
          bool updated = tree.TranspositionRoots.PossiblyUpdateIfNBigger(nodesSpan, node.StructRef.ZobristHash, node.Index, node.N);
          //if (updated) MCTSEventSource.TestMetric1++;          
        }

        // Note that we cannot clear the EvalResult
        // because possibly the next overlapped batch
        // points back to this node and will need to copy the value for itself
      }
    }


    /// <summary>
    /// Calls SetPolicy for all nodes in a list of nodes (possibly with parallelism).
    /// </summary>
    /// <param name="nodes"></param>
    void DoApplySetPolicies(ListBounded<MCTSNode> nodes)
    {
      if (nodes.Count == 0) return;

      MCTSIterator context = nodes[0].Context;
      bool cachingInUse = context.EvaluatorDef.CacheMode > PositionEvalCache.CacheMode.None;
      bool movesSameOrderMoveList = context.NNEvaluators.PolicyReturnedSameOrderMoveList;

      float policySoftmax = context.ParamsSelect.PolicySoftmax;

      if (context.ParamsSearch.Execution.SetPoliciesParallelEnabled)
      {
        //        Parallel.ForEach(nodes, ParallelUtils.ParallelOptions(nodes.Count, context.ParamsSearch.Execution.SetPoliciesNumPoliciesPerThread),
        Parallel.ForEach(Partitioner.Create(0, nodes.Count), ParallelUtils.ParallelOptions(nodes.Count, context.ParamsSearch.Execution.SetPoliciesNumPoliciesPerThread),
        (range) =>
        {
          using (new SearchContextExecutionBlock(context))
          {
            for (int i = range.Item1; i < range.Item2; i++)
            {
              SetPolicy(nodes[i], policySoftmax, cachingInUse, movesSameOrderMoveList);
            }
          }
        });
      }
      else
      {
        foreach (MCTSNode node in nodes)
        {
          SetPolicy(node, policySoftmax, cachingInUse, movesSameOrderMoveList);
        }
      }
    }


    /// <summary>
    /// Sets the policy for a specified node using pending EvalResult.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="policySoftmax"></param>
    /// <param name="cachingInUse"></param>
    void SetPolicy(MCTSNode node, float policySoftmax, bool cachingInUse,
                   bool returnedMovesAreInSameOrderAsMGMoveList)
    {
      // If already set by another overlapped/concurrent selector we don't need to repeat this
      if (node.ActionType == MCTSNodeInfo.NodeActionType.CacheOnly && cachingInUse)
      {
        // We don't put cached node in tree at all (it will have been stored in cache)
        return;
      }
      else if (node.ActionType == MCTSNodeInfo.NodeActionType.None)
      {
        // Nothing to do!
      }
      else if (!node.IsTranspositionLinked)
      {
        if (!node.PolicyHasAlreadyBeenInitialized
         && !node.Terminal.IsTerminal()
         && !node.EvalResult.TerminalStatus.IsTerminal()
         )
        {
          if (node.EvalResult.PolicyIsReleased || node.EvalResult.IsNull)
          {
            Console.WriteLine("Warning: lost policy " + node.Annotation.Pos.FEN + " " + node.Terminal + " " + node + " " + node.Index);
            node.StructRef.Terminal = GameResult.Draw; //Best alternative
          }
          else
          {
            node.SetPolicy(policySoftmax, ParamsSelect.MinPolicyProbability, in node.Annotation.PosMG, node.Annotation.Moves,
                           in node.EvalResult.PolicyRef, returnedMovesAreInSameOrderAsMGMoveList);

            if (node.Context.ParamsSearch.Execution.InFlightOtherBatchLinkageEnabled)
            {
              throw new Exception("Unsupported. We would have to disable the ReleasePolicyValue here but unsure if that would create serious memory leak.");
              // TODO: we shouldn't release policy because in flight transpositions requires them to stick around
              //       but make sure that the memory is eventually released, not hung onto by annotation entries
            }
          }
        }

        node.EvalResult.ReleasePolicyValue();
      }
    }


  }
}

#if NOT
  // Sample code from abandoned subleaf feature (reverted on 27 July 2020)
  // which shows how to create nodes that can be appended to tree 
  // in Apply method:
        ListBounded<MCTSNode> subleafNodes = GatherSubleafs(nodes);
        DoApply(selectorID, subleafNodes, false);

 // --------------------------------------------------------------------------------------------
    ListBounded<MCTSNode> GatherSubleafs(ListBounded<MCTSNode> nodes)
    {
      ListBounded<MCTSNode> subleafNodes = new ListBounded<MCTSNode>(nodes.Count);

      foreach (MCTSNode node in nodes)
      {
        if (!node.EvalResultSubleaf.IsNull && !node.EvalResultSubleaf.PolicyIsReleased
          && node.NumPolicyMoves > 0
          && node.NumChildrenExpanded == 0)
        {
          MCTSNode subleafNode = node.CreateChild(0);
          subleafNodes.Add(subleafNode);

          subleafNode.ActionType = MCTSNode.NodeActionType.CacheOnly;

          Debug.Assert(!node.EvalResultSubleaf.PolicyIsReleased);

          subleafNode.EvalResult = node.EvalResultSubleaf;

          subleafNode.Annotate();

        }
      }
      return subleafNodes;
    }

#endif