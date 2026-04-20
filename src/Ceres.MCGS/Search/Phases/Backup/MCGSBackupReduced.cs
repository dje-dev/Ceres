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
using System.Threading;

using Ceres.Base.DataTypes;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Paths;
using Ceres.MCGS.Search.Phases;
using Ceres.MCGS.Search.Strategies;

#endregion

namespace Ceres.MCGS.Search.Phases.Backup;


/// <summary>
/// Applies updates to graph (Q, N, etc. for both edges and nodes) arising from selected visits.
/// A "reduction" algorithm is used to coalesce values from multiple paths in a graph.
/// </summary>
public partial class MCGSBackup
{
  public void BackupReduced(MCGSSelectBackupStrategyBase strategy, MCGSPath path, int iteratorID)
  {
    double rpoLambda = Engine.Manager.ParamsSelect.RPOBackupLambda;
    bool enablePseudoTranspositionBlending = Engine.Manager.ParamsSearch.EnablePseudoTranspositionBlending;

    ref MCGSPathVisit leafVisitRef = ref path.LeafVisitRef;

    int numVisitsAttempted = leafVisitRef.NumVisitsAttempted;
    int numVisitsAccepted = leafVisitRef.NumVisitsAccepted.Value;

    BackupValue initialBackupValue = (numVisitsAccepted > 0) ? initialBackupValue = InitialBackupValueForPath(path)
                                                             : default;

    bool haveProcessedLeaf = false;

    // Perform backup, starting from the leaf and working toward root.
    foreach (MCGSPathVisitMember visitPair in path.PathVisitsLastBackedUpToRoot)
    {
      ref MCGSPathVisit visitPathRef = ref visitPair.PathVisitRef;
      GEdge visitEdge = visitPathRef.ParentChildEdge;

      // Prefetch the edge structure to reduce memory latency when accessed later
      visitEdge.Prefetch();

      GNode parentNode = visitEdge.ParentNode;

      using (new NodeLockBlock(parentNode))
      {
        //          using (new NodeLockBlock(visitPathRef.ChildNode, true)) // Child might be null (if terminal) hence allow null lock
        {
          // Check if this is a synchronization edge where multiple paths converge.
          // If we are not last to be processed, update accumulator and exit to leave for others to finish.
          // Otherwise, bring the values previously stored in the accumulator into this update.
          if (!haveProcessedLeaf)
          {
            Interlocked.Add(ref visitPathRef.NumVisitsAttemptedPendingBackup, -numVisitsAttempted);
          }
          else
          {
            if (rpoLambda > 0 && visitEdge.ChildNode.NumEdgesExpanded > 1)
            {
              visitEdge.ChildNode.RPOUpdateQ(enablePseudoTranspositionBlending, rpoLambda);
            }

            int newNumPendingBackup = Interlocked.Add(ref visitPathRef.NumVisitsAttemptedPendingBackup, -numVisitsAttempted);
            Debug.Assert(numVisitsAccepted >= 0);
            if (newNumPendingBackup > 0)
            {
              // There are more paths yet to be processed that will pass thru this visit.
              // We'll exit and let them eventually finish the work.
              // But first we have to update the accumulator for the child
              visitPathRef.Accumulator.DoAdd(numVisitsAttempted, numVisitsAccepted);
              return;
            }
            else
            {
              if (visitPathRef.Accumulator.NumVisitsAttempted > 0)
              {
                // Other paths already visited this child and accumulated updates.
                // We first apply our own update and then reload from accumulator
                // so we see all accumulated values.
                ref MCGSBackupAccumulator inFlightAccumulator = ref visitPathRef.Accumulator;
                inFlightAccumulator.DoAdd(numVisitsAttempted, numVisitsAccepted);
                Debug.Assert(inFlightAccumulator.NumVisitsAttempted > 0);

                numVisitsAttempted = inFlightAccumulator.NumVisitsAttempted;
                numVisitsAccepted = inFlightAccumulator.NumVisitsAccepted;
              }
            }

          }

          // Possibly initiate prefetch of our parent (for speed)
          if (!parentNode.IsSearchRoot)
          {
            PrefetchChild(parentNode, visitPathRef.IndexOfChildInParent);
          }

          // STEP 1: capture the W previously contributed upward by this edge before this update.
          int visitEdgeN = visitEdge.N;
          double priorEdgeW = visitEdgeN == 0 ? 0 : visitEdgeN * visitEdge.Q;
          visitPathRef.NumVisitsAccepted = (short)numVisitsAccepted;
          if (numVisitsAccepted > 0)
          {
            // If the child was selected "off-policy" then we disconnect
            // disconnect the path starting at this level (and above)
            // so that the backed up value is not used and we only backout the in flight values.
            if (visitPathRef.DisconnectFromEdgeNStartingThisVisit)
            {
              numVisitsAccepted = 0;
            }

            // STEP 2: update the edge: (a) increase N by numVisitsAccepted, and (b) reset Q to be same as child.
            if (haveProcessedLeaf)
            {
              strategy.BackupToEdge(visitEdge, numVisitsAccepted, visitEdge.ChildNode.Q, visitEdge.ChildNode.D, visitEdge.ChildNodeHasDrawKnownToExist);
            }
            else
            {
              // TODO: enable this fix after more testing, likely appropriate (but testing was not clearly positive).
              const bool BUGFIX_INITIAL_BACKUP = false; 
              if (BUGFIX_INITIAL_BACKUP)
              {
                // Passing initialBackupValue.V (=0) as newQChild on a rep-draw would clobber
                // edge.QChild and make edge.Q = 0 until the next non-rep visit, spuriously
                // inflating the Δ sent to the parent. Use ChildNode.Q instead (matching the
                // internal-edge branch above).
                bool isRepDraw = path.TerminationReason == MCGSPathTerminationReason.DrawByRepetitionInCoalesceMode;
                double newQChildForLeaf = isRepDraw ? visitEdge.ChildNode.Q : initialBackupValue.V;
                strategy.BackupToEdge(visitEdge, numVisitsAccepted, newQChildForLeaf, initialBackupValue.D, false);
                if (isRepDraw)
                {
                  // The NDrawByRepeition for the edge immediately leading to the draw
                  // is incremented in tandem with the primary (but is not backed up).
                  visitEdge.IncrementNDrawRepetition(numVisitsAccepted);
                }
              }
              else
              {
                strategy.BackupToEdge(visitEdge, numVisitsAccepted, initialBackupValue.V, initialBackupValue.D, false);
                if (path.TerminationReason == MCGSPathTerminationReason.DrawByRepetitionInCoalesceMode)
                {
                  // The NDrawByRepeition for the edge immediately leading to the draw
                  // is incremented in tandem with the primary (but is not backed up).
                  visitEdge.IncrementNDrawRepetition(numVisitsAccepted);
                }
              }
            }
          }

          // Decrement NInFlight
          GNodeStruct.UpdateEdgeNInFlightForIterator(visitEdge, iteratorID, (short)-numVisitsAttempted);

          // Capture the W contributed by this edge after the update.
          // Use the difference from prior W contribution to determine update magnitude,
          // thereby capturing the impact of all changes to child Q since this path last visited
          // (it may have happened that other transposition paths had caused interim updates to child Q).
          double newEdgeW = visitEdge.N == 0 ? 0 : visitEdge.N * visitEdge.Q;

          double newEdgeDeltaW = priorEdgeW - newEdgeW;

          // STEP 5: update parent node fields
          if (numVisitsAccepted > 0)
          {
            if (!haveProcessedLeaf && visitEdge.N > 0 && visitEdge.Q <= -1)
            {
              if (!parentNode.CheckmateKnownToExistAmongChildren) // only do first time
              {
                parentNode.UpdateNodeForProvenChildLoss((FP16)(float)-visitEdge.Q, 0);
              }
              parentNode.NodeRef.N += numVisitsAccepted;

              //                UpdateParentNodeForProvenChildLoss(numVisitsAccepted, visitEdge);
            }
            else
            {
              // Compute deltaD for the parent node backup.
              // childD is already available from the child node (already read for BackupToEdge).
              // For leaf edge: use initialBackupValue.D
              // For internal edges: use visitEdge.ChildNode.D
              double childD = haveProcessedLeaf ? visitEdge.ChildNode.D : initialBackupValue.D;
              double newParentDeltaD = childD * numVisitsAccepted;
              strategy.BackupToNode(parentNode, numVisitsAccepted, newEdgeDeltaW, newParentDeltaD);
            }
          }
        }

        haveProcessedLeaf = true;
      }
    }
  }
}
