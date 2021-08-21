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
        if (node.ActionType == MCTSNode.NodeActionType.MCTSApply
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
    void ApplyResult(Span<MCTSNodeStruct> nodes, int selectorID, MCTSNode node, LeafEvaluationResult evalResult)
    {
      ref MCTSNodeStruct nodeRef = ref node.Ref;

      MCTSNodeStructIndex indexOfChildDescendentFromRoot = default;

      nodeRef.VSecondary = FP16.NaN;

      int numUpdateSelector1 = selectorID == 0 ? nodeRef.NInFlight : 0;
      int numUpdateSelector2 = selectorID == 1 ? nodeRef.NInFlight2 : 0;
      int numInFlight = numUpdateSelector1 + numUpdateSelector2;


      if (node.ActionType == MCTSNode.NodeActionType.CacheOnly ||
          node.ActionType == MCTSNode.NodeActionType.None)
      {
        nodeRef.BackupDecrementInFlight(numUpdateSelector1, numUpdateSelector2);
      }
      else if (node.ActionType == MCTSNode.NodeActionType.MCTSApply)
      {
        bool wasTerminal = node.Terminal.IsTerminal();
        float dToApply = evalResult.DrawP;

        float contemptAdjustment = 0;
        if (!float.IsNaN(dToApply))
        {
          contemptAdjustment = (node.IsOurMove ? -1 : 1) * (dToApply * node.Context.CurrentContempt);
        }

        // If we are revisiting a terminal node, just reiterate the prior evaluation
        if (wasTerminal && node.EvalResult.IsNull)
          node.EvalResult = new LeafEvaluationResult(nodeRef.Terminal, nodeRef.WinP, nodeRef.LossP, nodeRef.MPosition);

        // Update statistics
        float vToApply;
        int numToApply;

        if (nodeRef.N > 0)
        {
          if (wasTerminal)
          {
            // Repeat visit to a terminal, even "collisions" are applied multiple times
            numToApply = numInFlight;
            vToApply = nodeRef.V + contemptAdjustment;

            nodeRef.BackupApply(nodes, numToApply, vToApply, 0, dToApply, wasTerminal, numUpdateSelector1, numUpdateSelector2, out indexOfChildDescendentFromRoot);
          }
          else
          {
            throw new Exception("Internal error: Unexpected N > 0 on non-terminal");
          }
        }
        else
        {
          // First visit by any selector, set evaluation result and backup in tree
          numToApply = 1;
          nodeRef.Terminal = node.EvalResult.TerminalStatus;

          if (!node.IsRoot && node.Terminal == GameResult.Draw)
          {
            node.Parent.Ref.DrawKnownToExistAmongChildren = true;
            //Chess.NNEvaluators.LC0DLL.LC0DLLSyzygyEvaluator.NumTablebaseMisses++;
          }

          if (ParamsSelect.VIsForcedLoss(node.EvalResult.V))
          {
            SetProvenLossAndPropagateToParent(node, node.EvalResult.LossP, node.EvalResult.M);
          }
          else
          {
            nodeRef.WinP = evalResult.WinP;
            nodeRef.LossP = evalResult.LossP;
            nodeRef.MPosition = (byte)MathF.Round(evalResult.M, 0);
          }

          vToApply = nodeRef.V;
          float mToApply = nodeRef.MPosition;

          if (!FP16.IsNaN(node.OverrideVToApplyFromTransposition))
          {
            // Switch to propagate this "pseudo V" for this node and all nodes above
            vToApply = node.OverrideVToApplyFromTransposition;
            mToApply = node.OverrideMPositionToApplyFromTransposition;
          }

          vToApply += contemptAdjustment;

          nodeRef.BackupApply(nodes, numToApply, vToApply, mToApply, dToApply, wasTerminal, numUpdateSelector1, numUpdateSelector2, out indexOfChildDescendentFromRoot);
        }

        PossiblyUpateFirstMoveSampler(node, indexOfChildDescendentFromRoot, numUpdateSelector1, numUpdateSelector2, wasTerminal, vToApply);

        // Update depth statistic
        node.Context.CumulativeSelectedLeafDepths += node.Depth * numToApply;
      }
      else
        throw new Exception("Internal error, unknown NodeActionType");


      if (FirstMoveSampler != null)
      {
// this mismatch can/will happen with tree reuse
//        if (node.Context.Root.N > 1 && node.Context.Root.ChildAtIndex(0).N != FirstMoveSampler.childrenDistributions[0].NumSamples)
//          Console.WriteLine("first mismatch on zero");
      }
      node.Context.RecordVisitToTopLevelMove(node, indexOfChildDescendentFromRoot, evalResult);
    }


    private void PossiblyUpateFirstMoveSampler(MCTSNode node, MCTSNodeStructIndex indexOfChildDescendentFromRoot, int numUpdateSelector1, int numUpdateSelector2, bool wasTerminal, float vToApply)
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

      Span<MCTSNodeStruct> nodesSpan = nodes[0].Context.Tree.Store.Nodes.nodes.Span;

      // Note that this is not parallelized to avoid updates
      // unsynchronized updates to fields in nodes higher up in the tree
      for (int i = 0; i < nodes.Count; i++)
      {
        MCTSNode node = nodes[i];
        ApplyResult(nodesSpan, selectorID, node, node.EvalResult);

        if (node.EvalResult.ExtraResults != null)
        {
          foreach (LeafEvaluationResult result in node.EvalResult.ExtraResults)
          {
            ApplyResult(nodesSpan, selectorID, node, result);
          }
        }

        // Note that we cannot clear the EvalResult
        // because possibly the next overlapped batch
        // points back to this node and will need to copy the value for itself
      }
    }


    /// <summary>
    /// Processes a node which has been determined to be a proven loss.
    /// Propagates this upward to the parent since parent's best move 
    /// is now obviously this one.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="lossP"></param>
    /// <param name="m"></param>
    private static void SetProvenLossAndPropagateToParent(MCTSNode node, float lossP, float m)
    {
      ref MCTSNodeStruct nodeRef = ref node.Ref;

      nodeRef.WinP = 0;
      nodeRef.LossP = (FP16)lossP;
      nodeRef.MPosition = (byte)MathF.Round(m, 0);

      if (!node.IsRoot)
      {
        // This checkmate will obviously be chosen by the opponent
        // Therefore propagate the result up to the opponent as a victory,
        // overriding such that the Q for that node reflects the certain loss
        MCTSNode parent = node.Parent;
        ref MCTSNodeStruct parentRef = ref parent.Ref;
        parentRef.WinP = (FP16)lossP;
        parentRef.LossP = 0;
        parentRef.W = parentRef.V * parentRef.N; // Make Q come out to be same as V (which has already been set to the sure win)
        parentRef.MPosition = (byte)MathF.Round(m + 1, 0);
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
      if (node.ActionType == MCTSNode.NodeActionType.CacheOnly && cachingInUse)
      {
        // We don't put cached node in tree at all (it will have been stored in cache)
        return;
      }
      else if (node.ActionType == MCTSNode.NodeActionType.None)
      {
        // Nothing to do!
      }
      else if (!node.IsTranspositionLinked)
      {
        if (!node.PolicyHasAlreadyBeenInitialized)
        {
          if (!node.EvalResult.TerminalStatus.IsTerminal())
          {
            node.SetPolicy(policySoftmax, ParamsSelect.MinPolicyProbability, in node.Annotation.PosMG, node.Annotation.Moves, 
                           in node.EvalResult.PolicyRef, returnedMovesAreInSameOrderAsMGMoveList);
          }

          if (node.Context.ParamsSearch.Execution.InFlightOtherBatchLinkageEnabled)
          {
            throw new Exception("Unsupported. We would have to disable the ReleasePolicyValue here but unsure if that woudld create serious memory leak.");
            // TODO: we shouldn't release policy because in flight transpositions requires them to stick around
            //       but make sure that the memory is eventually released, not hung onto by annotation entries
          }
          node.EvalResult.ReleasePolicyValue();
        }
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