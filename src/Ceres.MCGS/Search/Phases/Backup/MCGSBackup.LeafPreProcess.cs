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
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Tasks;

using Ceres.Base.DataTypes;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Paths;

#endregion

namespace Ceres.MCGS.Search.Phases.Backup;

/// <summary>
/// Container for the values to be backed up during the backup phase.
/// </summary>
/// <param name="V">Value (win minus loss probability) from the leaf's perspective.</param>
/// <param name="D">Draw probability.</param>
public record struct BackupValue(double V, double D);


public partial class MCGSBackup
{
  /// <summary>
  /// Cached options for the leaf preprocessing parallel loop (avoids per-batch allocations).
  /// </summary>
  private readonly ParallelOptions cachedLeafPrepParallelOptions = new();

  /// <summary>
  /// Cached delegate for ApplyLeafNodeUpdates (avoids per-batch allocations).
  /// </summary>
  private readonly Action<MCGSPath> applyLeafNodeUpdatesAction;


  /// <summary>
  /// Prepares collection of MCGSPath paths to begin backup from their leaves.
  /// </summary>
  internal void PreparePathsForBackup(ConcurrentQueue<MCGSPath> paths)
  {
    const int PATHS_PER_THREAD = 32;

    int numPaths = paths.Count;
    if (numPaths <= 2 * PATHS_PER_THREAD)
    {
      // Below this size the Parallel.ForEach machinery costs more than it saves.
      foreach (MCGSPath path in paths)
      {
        ApplyLeafNodeUpdates(path);
      }
      return;
    }

    cachedLeafPrepParallelOptions.MaxDegreeOfParallelism = Math.Min(1 + numPaths / PATHS_PER_THREAD,
                                                                    Math.Min(System.Environment.ProcessorCount, MAX_BACKUP_THREADS));
    Parallel.ForEach(paths, cachedLeafPrepParallelOptions, applyLeafNodeUpdatesAction);
  }


  /// <summary>
  /// Applies any updates to leaf node which are appropriate based on the termination reason.
  /// </summary>
  /// <param name="path"></param>
  internal void ApplyLeafNodeUpdates(MCGSPath path)
  {
    int numToApply = (int)path.LeafVisitRef.NumVisitsAccepted;
    GNode leafNode = path.LeafNode;

    switch (path.TerminationReason)
    {
      case MCGSPathTerminationReason.PendingNeuralNetEval:
        Debug.Assert(numToApply == 1);
        // deltaD = DrawP * numToApply for running average accumulation
        Engine.Strategy.BackupToNode(leafNode, numToApply, path.TerminationInfo.V,
          numToApply * path.TerminationInfo.DrawP);
        break;

      case MCGSPathTerminationReason.TranspositionCopyValues:
        // Select accepts exactly one visit for a non-terminal newly created node
        // (see MCGSSelect.DoNewlyCreatedNode); the unmultiplied V below relies on this.
        Debug.Assert(numToApply == 1);
        // deltaD = DrawP * numToApply for running average accumulation
        Engine.Strategy.BackupToNode(leafNode, numToApply, leafNode.V,
          numToApply * leafNode.DrawP);
        break;

      case MCGSPathTerminationReason.TranspositionCopyValuesAutoExtended:
        // The node was already fully installed during the select phase
        // (own evaluation visit plus one auto-extension visit into its top-policy child,
        // applied via the same BackupToNode/BackupToEdge primitives).
        // Applying the usual leaf update here would double-count.
        Debug.Assert(leafNode.IsEvaluated && leafNode.N >= 1);
        break;

      case MCGSPathTerminationReason.Terminal:
        // Update search statistics (all visits are absorbed).
        // deltaD = DrawP * numToApply for running average accumulation
        Engine.Strategy.BackupToNode(leafNode, numToApply, numToApply * leafNode.V,
          numToApply * leafNode.DrawP);
        break;

      case MCGSPathTerminationReason.PiggybackPendingNNEval:
        // The BackupToNode will be performed by another path which was in flight
        if (!path.LeafNode.IsEvaluated)
        {
          // We hoped to piggyback.
          // But apparently we were piggybacking on the other iterator
          // and that iterator has not yet finished setting the evaluation.
          // Therefore convert to abort.
          path.TerminationReason = MCGSPathTerminationReason.Abort;
          path.LeafVisitRef.NumVisitsAccepted = 0;
        }

        break;

      case MCGSPathTerminationReason.AlreadyNNEvaluated:
        // Already evaluated and applied to node.
        Debug.Assert(leafNode.IsEvaluated);
        break;

      case MCGSPathTerminationReason.TerminalEdge:
        break;

      case MCGSPathTerminationReason.DrawByRepetitionInCoalesceMode:
        Debug.Assert(Engine.Manager.ParamsSearch.PathTranspositionMode == Params.PathMode.PositionEquivalence);
        Debug.Assert(leafNode.N > 0 || FP16.IsNaN(leafNode.WinP));
        // No impact on child node, the draw by repetition is associated with edge only.
        break;

      case MCGSPathTerminationReason.TranspositionLinkNodeSufficientN:
        break;

      case MCGSPathTerminationReason.NotYetTerminated:
        throw new NotImplementedException();

      case MCGSPathTerminationReason.Abort:
        break;

      default:
        throw new ArgumentOutOfRangeException($"Unknown termination reason {path.TerminationReason}");
    }
  }



  /// <summary>
  /// Returns the values to be used for the initial backup of a path
  /// (based on the termination reason).
  /// </summary>
  /// <param name="path"></param>
  public BackupValue InitialBackupValueForPath(MCGSPath path)
  {
    ref readonly MCGSPathVisit leafPathVisit = ref path.LeafVisitRef;
    GEdge leafEdge = leafPathVisit.ParentChildEdge;

    // The EnablePVAutoExtend feature was found ineffective and its backup-side
    // support has been removed; the assert guards against it being re-enabled.
    Debug.Assert(!path.Engine.Manager.ParamsSearch.EnablePVAutoExtend);

#if NOT
    // EnablePVAutoExtend support:
    //   check if this was a first explicit visit to a node 
    //   which was created by PV auto extensions.
    //   TODO: can this be extended to other cases, especially TranspositionLinkNodeSufficientN?
    if ((path.TerminationReason == MCGSPathTerminationReason.PendingNeuralNetEval
       || path.TerminationReason == MCGSPathTerminationReason.PiggybackPendingNNEval
       || path.TerminationReason == MCGSPathTerminationReason.AlreadyNNEvaluated)
       && leafEdge.ChildNode.N == 1
       && path.Engine.Manager.ParamsSearch.EnablePVAutoExtend
       && !leafEdge.Type.IsTerminal()
       && leafEdge.ChildNode.IsEvaluated // is this necessary?
       && leafEdge.ChildNode.NumPolicyMoves > 0
       && !leafEdge.ChildNode.IsPendingPolicyCopy) // TODO: someday if pending policy copy, link thru instead of giving up
    {
      GEdgeHeaderStruct topMoveInfo;
      using (new NodeLockBlock(leafEdge.ChildNode))
      {
        topMoveInfo = path.LeafNode.EdgeHeadersSpan[0];
      }

      if (!topMoveInfo.IsExpanded // has not been converted concurrently to edge
        && topMoveInfo.P > MCGSParamsFixed.AUTOEXTEND_MIN_TOP_MOVE_P)
      {
        // Generate position after top policy move.
        MGMove topPolicyMoveMG = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(topMoveInfo.Move, in leafPathVisit.ChildPosition);
        MGPosition childPosMG = leafPathVisit.ChildPosition;
        childPosMG.MakeMove(topPolicyMoveMG);

        PosHash64WithMove50AndReps hash64WithMoveAndReps;
        hash64WithMoveAndReps = MGPositionHashing.Hash64WithMove50AndRepsAdded(MGPositionHashing.Hash64(in childPosMG),
                                                                                childPosMG.RepetitionCount,
                                                                                childPosMG.Move50Category);

        (double childSiblingSetAvgQ, double childSiblingSetAvgD) = NodeIndexSet.AvgSqrtNWeightedStats(Engine.Graph, hash64WithMoveAndReps);
        if (!double.IsNaN(childSiblingSetAvgQ))
        {
          // Return the average of the actually visited chlid node and its PV continuation.
          const float FRAC_CHILD = 0.5f;
          float v = (1.0f - FRAC_CHILD) * (float)(leafEdge.ChildNode.V) + FRAC_CHILD * (float)(-childSiblingSetAvgQ);
          float d = (1.0f - FRAC_CHILD) * (float)(leafEdge.ChildNode.D) + FRAC_CHILD * (float)(childSiblingSetAvgD);
          return new BackupValue(v, d);
        }
      }
    }
#endif

    if (leafEdge.Type == GEdgeStruct.EdgeType.ChildEdge)
    {
      leafEdge.SetUncertaintyValues(leafEdge.ChildNode.UncertaintyValue,
                                    leafEdge.ChildNode.UncertaintyPolicy);
    }

    switch (path.TerminationReason)
    {
      case MCGSPathTerminationReason.PendingNeuralNetEval:
        Debug.Assert(!float.IsNaN(path.TerminationInfo.V));
        return new BackupValue(leafEdge.ChildNode.Q, path.LeafNode.D);

      case MCGSPathTerminationReason.AlreadyNNEvaluated:
      case MCGSPathTerminationReason.PiggybackPendingNNEval:
        Debug.Assert(!double.IsNaN(path.LeafNode.Q));
        return new BackupValue(leafEdge.ChildNode.Q, path.LeafNode.D);

      case MCGSPathTerminationReason.Terminal:
        Debug.Assert(path.LeafNode.IsEvaluated);
        return new BackupValue(leafEdge.ChildNode.Q, path.LeafNode.D);

      case MCGSPathTerminationReason.TerminalEdge:
        Debug.Assert(!(path.LeafVisitRef.ParentChildEdge.Q == 0
                  && path.LeafVisitRef.ParentChildEdge.Type != GEdgeStruct.EdgeType.TerminalEdgeDrawn));
        return new BackupValue(leafEdge.Q,
                               leafEdge.Type == GEdgeStruct.EdgeType.TerminalEdgeDrawn ? 1 : 0);

      case MCGSPathTerminationReason.TranspositionLinkNodeSufficientN:
        return new BackupValue(leafEdge.ChildNode.Q, leafEdge.ChildNode.D);

      case MCGSPathTerminationReason.TranspositionCopyValues:
      case MCGSPathTerminationReason.TranspositionCopyValuesAutoExtended:
        Debug.Assert(leafEdge.ChildNode.IsEvaluated);
        return new BackupValue(leafEdge.ChildNode.Q, leafEdge.ChildNode.D);

      case MCGSPathTerminationReason.DrawByRepetitionInCoalesceMode:
        return new BackupValue(0f, 1f);

      case MCGSPathTerminationReason.Abort:
        return new BackupValue(double.NaN, double.NaN);

      default:
        throw new ArgumentOutOfRangeException();
    }
  }
}
