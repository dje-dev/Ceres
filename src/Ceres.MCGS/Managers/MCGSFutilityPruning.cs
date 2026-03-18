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
using System.Collections.Generic;
using System.Linq;
using System.Threading;

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Search;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Managers;

/// <summary>
/// Manages identification of moves at root which can have
/// further search suspended due to futility - impossibility
/// or improbability of further visits changing the final selected best move.
/// 
/// Pruning of primary and/or secondary moves is possible
/// (where pruning of primary moves effectively shuts down the whole search).
/// </summary>
public class MCGSFutilityPruning
{
  /// <summary>
  /// Associated context.
  /// </summary>
  public readonly MCGSManager Manager;

  /// <summary>
  /// Helper method which returns root node of search.
  /// </summary>
  public GNode SearchRoot => Manager.Engine.SearchRootNode;

  public MGPosition SearchRootPosMG => Manager.Engine.SearchRootPosMG;


  /// <summary>
  /// Optional List of moves to which the top-level search is to be restricted.
  /// </summary>
  public List<Move> SearchMoves;

  public List<MGMove> SearchMovesTablebaseRestricted = null;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="manager"></param>
  public MCGSFutilityPruning(MCGSManager manager, List<Move> searchMoves, List<MGMove> searchMovesTablebaseRestricted)
  {
    Manager = manager;
    SearchMoves = searchMoves;
    SearchMovesTablebaseRestricted = searchMovesTablebaseRestricted;
  }


  public bool HaveAppliedSearchMoves { private set; get; }

  private readonly Lock applySearchMovesLock = new();


  internal void ApplySearchMovesIfNeeded()
  {
    if (HaveAppliedSearchMoves)
    {
      return;
    }

    lock (applySearchMovesLock)
    {
      if (HaveAppliedSearchMoves)
      {
        return;
      }

      if (Manager.RootMovesPruningStatus == null && Manager.Engine.SearchRootNode.N > 1)
      {
        Manager.RootMovesPruningStatus = new Managers.MCGSFutilityPruningStatus[SearchRoot.NumPolicyMoves];
      }

      if (SearchMoves != null)
      {
        // Start by assuming all are pruned.
        Array.Fill(Manager.RootMovesPruningStatus, Managers.MCGSFutilityPruningStatus.PrunedDueToSearchMoves);

        // Specifically un-prune any which are specified as valid search moves.
        bool whiteToMove = SearchRoot.IsWhite;
        foreach (Move move in SearchMoves)
        {
          bool found = false;
          for (int i = 0; i < SearchRoot.NumPolicyMoves; i++)
          {
            MGMove moveMG = SearchRoot.ChildEdgeAtIndex(i).MoveMGFromPos(SearchRootPosMG);
            string moveSAN = MGMoveConverter.ToMove(moveMG).ToSAN(SearchRootPosMG.ToPosition);
            if (MGMoveConverter.ToMove(moveMG) == move)
            {
              Manager.RootMovesPruningStatus[i] = Managers.MCGSFutilityPruningStatus.NotPruned;
              found = true;
              break;
            }
          }

          if (!found)
          {
            Console.WriteLine($"Internal error: specified search move not found {move}");
          }
        }
      }

      if (SearchMovesTablebaseRestricted != null)
      {
        for (int i = 0; i < SearchRoot.NumPolicyMoves; i++)
        {
          MGMove moveMG = SearchRoot.ChildEdgeAtIndex(i).MoveMGFromPos(SearchRootPosMG);
          if (!SearchMovesTablebaseRestricted.Contains(moveMG))
          {
            Manager.RootMovesPruningStatus[i] = Managers.MCGSFutilityPruningStatus.PrunedDueToTablebaseNotWinning;
          }

        }
      }

      HaveAppliedSearchMoves = true;
    }
  }


  /// <summary>
  /// Updates internal pruning statistics based on current state of search tree.
  /// </summary>
  internal void UpdatePruningFlags()
  {
    // Exit if early stopping not enabled
    if (Manager.ParamsSearch.MoveFutilityPruningAggressiveness == 0f)
    {
      return;
    }
    // Because estimates of nodes remaining are noisy for searces with time limits,
    // conservatively decline to set MinNToVisit unless we are 
    // reasonably close to the end of the search (and thus had sufficient time to get accurate statistics)
    if (Manager.NumNodesVisitedThisSearch < 100)
    {
      return;
    }

    // Potentially Q still could change significantly with a large fraction of search remaining.
    // Therefore never allow search abort unless at least 40% complete.
    // Even if the extra visits almost all go to a dominant move, they 
    // are likely to be benefical subsequently due to tree reuse.
    if (Manager.FractionSearchRemaining > 0.40)
    {
      return;
    }

    if (Manager.RootMovesPruningStatus == null && Manager.Engine.SearchRootNode.N > 1)
    {
      Manager.RootMovesPruningStatus = new Managers.MCGSFutilityPruningStatus[SearchRoot.NumPolicyMoves];
    }

    DoSetEarlyStopMoveSecondaryFlags();
  }


  /// <summary>
  /// Worker method that actually computes and updates the statistic.
  /// </summary>
  private void DoSetEarlyStopMoveSecondaryFlags()
  {
    if (Manager.Engine.SearchRootNode.NumEdgesExpanded == 0)
    {
      return;
    }

    // Don't prune any more if we disallow stop search from pruning
    // and we are already down to 2 remaining unpruned moves.
    bool okToPrune = true;
    int numNotPruned = Manager.RootMovesPruningStatus.Sum(p => p == Managers.MCGSFutilityPruningStatus.NotPruned ? 1 : 0);
    if (!Manager.ParamsSearch.FutilityPruningStopSearchEnabled && numNotPruned <= 2)
    {
      okToPrune = false;
    }
    float aggressiveness = Manager.ParamsSearch.MoveFutilityPruningAggressiveness;
    if (aggressiveness >= 1.5f)
    {
      throw new Exception("Maximum value of EarlyStopMoveSecondaryAggressiveness is 1.5.");
    }

    // Calibrate aggressiveness such that :
    //   - at maximum value of 1.0 we assume 50% visits go to second best move(s)
    //   - at reasonable default value 0.5 we assume 75% of visits to go second best move(s) 
    float aggressivenessMultiplier = 1.0f / (1.0f - aggressiveness * 0.5f);

    int? numRemainingSteps = Manager.EstimatedNumVisitsRemaining();

    // Can't make any determiniation if we can't estimate how many steps left
    if (numRemainingSteps is null) return;

    int numStepsTotal = (int)(numRemainingSteps / Manager.FractionSearchRemaining);

    GEdge bestNEdge = SearchRoot.EdgeWithMaxValue(n => n.N);
    GEdge bestQEdge = SearchRoot.EdgeWithMaxValue(n => -n.Q);
    float bestQ = (float)bestQEdge.Q;
    float qOfBestN = (float)bestNEdge.Q;

    ManagerChooseBestMoveMCGS bestMoveChooser = new(Manager, Manager.Engine.SearchRootNode, false, default, false);

    float MIN_BEST_N_FRAC_REQUIRED = ManagerChooseBestMoveMCGS.MIN_FRAC_N_REQUIRED_MIN;


    int numNewlyShutdown = 0;
    foreach ((GEdge edge, int indexChild) in SearchRoot.ChildEdgesExpandedWithIndex)
    {
      // Never shut down second best move unless the whole search is eligible to shut down
      if (SearchRoot.NumEdgesExpanded > 1)
      {
        double thisQ = edge.Q;

        // Unprune any this move if its Q has become better than that of the top N move.
        if ((thisQ <= qOfBestN || thisQ <= bestQ)
         && Manager.RootMovesPruningStatus[indexChild] == Managers.MCGSFutilityPruningStatus.PrunedDueToFutility)
        {
          Manager.RootMovesPruningStatus[indexChild] = Managers.MCGSFutilityPruningStatus.NotPruned;
          continue;
        }

        // Do not ever shut down a node with a better Q than that of the best N.
        // Even if it seems unreachable with current search limits,
        // there is a possibility the search will be extended.
        if (thisQ <= qOfBestN)
        {
          continue;
        }
      }

      if (Manager.RootMovesPruningStatus[indexChild] != Managers.MCGSFutilityPruningStatus.NotPruned)
      {
        continue;
      }

      float earlyStopGapRaw;
      if (Manager.ParamsSearch.BestMoveMode == ParamsSearch.BestMoveModeEnum.TopN)
      {
        earlyStopGapRaw = bestNEdge.N - edge.N;
      }
      else
      {
        int minNRequired = (int)(bestQEdge.N * MIN_BEST_N_FRAC_REQUIRED);
        earlyStopGapRaw = minNRequired - edge.N;
      }

      float earlyStopGapAdjusted = earlyStopGapRaw * aggressivenessMultiplier;
      bool earlyStop = earlyStopGapAdjusted > numRemainingSteps;

#if FEATURE_NO_SHUTDOWN
      // Do not shutdown nodes which have Q close to best Q
      // because they may still have a chance to get best Q (or close, triggering a search extension)
      // and also because the extra visits may not be wasted if the opponent 
      // happens to choose this move which seems to us near as good.
      const float SHUTDOWN_Q_MIN_SUBOPTIMALTIY = 0.02f;

      bool nearlyBestQ = Math.Abs(childRef.Q - nodesSortedQ[0].Q) < SHUTDOWN_Q_MIN_SUBOPTIMALTIY;

      if (okToPrune && childRef.N > 100 && Context.ParamsSearch.TestFlag2 && Context.RootMoveTracker.RunningVValues != null)
      {
        float runningQ = Context.RootMoveTracker.RunningVValues[childRef.IndexInParent];
        if (runningQ < bestQ)
        {
//            Console.WriteLine(Context.Manager.Search.SearchRootNode.N + " " + childRef.N + " [" + childRef.IndexInParent 
//                              + "] disable prune " + runningQ + " " + bestQ);
          earlyStop = false;
        }
      }

#endif


      if (okToPrune
       && earlyStop)
      {
#if NOT_HELPFUL
        // Never shutdown nodes getting large fraction of all visits
        float fracVisitsThisMoveRunningAverage = Context.RootMoveTracker != null ? Context.RootMoveTracker.RunningFractionVisits[i] : 0;

        const float THRESHOLD_VISIT_FRACTION_DO_NOT_SHUTDOWN_CHILD = 0.20f;
        bool shouldVeto = (fracVisitsThisMoveRunningAverage > THRESHOLD_VISIT_FRACTION_DO_NOT_SHUTDOWN_CHILD);

        if (Context.ParamsSearch.TestFlag 
          && nodesSortedN[0] != Context.Root.ChildAtIndex(i)
         && NumberOfNotShutdownChildren() > 1
         && shouldVeto)
        {
          MCTSEventSource.TestCounter1++;
          continue;
        }
        else
#endif

        bool shouldDump = MCGSDiagnostics.DumpSearchFutilityShutdown && Manager.RootMovesPruningStatus[indexChild] == Managers.MCGSFutilityPruningStatus.NotPruned;
        numNewlyShutdown++;
        //shouldDump = true;
        if (shouldDump)
        {
          Console.WriteLine();
          Console.WriteLine($"\r\nShutdown {edge.MoveMG} [{edge.N}] at root N  {Manager.Engine.SearchRootNode.N} with remaning {numRemainingSteps}"
                          + $" due to raw gapN {earlyStopGapRaw} adusted to {earlyStopGapAdjusted} in mode {Manager.ParamsSearch.BestMoveMode} aggmult {aggressivenessMultiplier}");
          //            DumpDiagnosticsMoveShutdown();
        }
        Manager.RootMovesPruningStatus[indexChild] = Managers.MCGSFutilityPruningStatus.PrunedDueToFutility;
      }
      // Console.WriteLine(i + $" EarlyStopMoveSecondary(simple) gap={gapToBest} adjustedGap={inflatedGap} remaining={numRemainingSteps} ");
    }

    if (numNewlyShutdown > 0)
    {
      // Once any node is pruned, no unexpanded nodes could ever become best. 
      for (int i = SearchRoot.NumEdgesExpanded; i < SearchRoot.NumPolicyMoves; i++)
      {
        Manager.RootMovesPruningStatus[i] = Managers.MCGSFutilityPruningStatus.PrunedDueToFutility;
      }
    }


    // TODO: log this
    //Console.WriteLine($"{Context.RemainingTime,5:F2}sec remains at N={Root.N}, setting  minN to {minN} number still considered {count} " +
    //                  $"using EstimatedNPS {Context.EstimatedNPS} with nodes remaining {Context.EstimatedNumStepsRemaining()} " +
    //                  statsStr);
  }


  /// <summary>
  /// Dumps diagnostic information related to the futility statistics.
  /// </summary>
  public void DumpDiagnosticsMoveShutdown()
  {
    Manager.DumpTimeInfo(null /*, Search.SearchRootNode*/);
    Manager.Engine.Graph.DumpNodesStructure(1, 0); // TODO: replace with below line instead (remediated)
    //Manager.Engine.RootNode.Dump(1, 1);
    Console.WriteLine();
  }


  /// <summary>
  /// Returns if the move at a specified index (from root)
  /// is a valid search move (not filtered out by possible limited SearchMoves).
  /// </summary>
  /// <param name="childIndex"></param>
  /// <returns></returns>
  public bool MoveAtIndexAllowed(int childIndex)
  {
    if (SearchMoves == null)
    {
      return true;
    }

    if (childIndex >= SearchRoot.NumEdgesExpanded)
    {
      return false;
    }

    Position rootPos = Manager.Engine.SearchRootNode.CalcPosition().ToPosition;

    // Get this child node and extract associated Move.
    GEdge edge = SearchRoot.ChildEdgeAtIndex(childIndex);

    string moveSAN = MGMoveConverter.ToMove(edge.MoveMG).ToSAN(rootPos);

    foreach (Move testMove in SearchMoves)
    {
      // TODO: Comparision via SAN is ugly, clean up.
      if (testMove.ToSAN(in rootPos).Equals(moveSAN, StringComparison.OrdinalIgnoreCase))
      {
        return true;
      }
    }

    return false;
  }


  /// <summary>
  /// Returns the number of children at root which are not in a shutdown state
  /// (and visited at least once).
  /// </summary>
  /// <returns></returns>
  public int NumberOfNotShutdownChildren()
  {
    GNode root = Manager.Engine.SearchRootNode;

    if (Manager.RootMovesPruningStatus == null)
    {
      return Manager.Engine.SearchRootNode.NumPolicyMoves;
    }

    int count = 0;
    for (int i = 0; i < root.NumPolicyMoves; i++)
    {
      int n = i < root.NumEdgesExpanded ? root.ChildEdgeAtIndex(i).N : 0;
      if (Manager.RootMovesPruningStatus[i] == Managers.MCGSFutilityPruningStatus.NotPruned
       || n == 0)
      {
        count++;
      }
    }

    return count;

  }
}
