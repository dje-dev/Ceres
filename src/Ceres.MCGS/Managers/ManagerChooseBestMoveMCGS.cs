
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

using Ceres.Base.Misc;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators.LC0DLL;

using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.RPO;

#endregion


namespace Ceres.MCGS.Managers;

/// <summary>
/// Manager that selects which move at the root of the search is best to play.
/// </summary>
public class ManagerChooseBestMoveMCGS
{

  internal const float MIN_FRAC_N_REQUIRED_MIN = 0.325f;

  public readonly GNode Node;
  public readonly MCGSManager Manager;

  public readonly bool UpdateStatistics;
  public readonly MGMove ForcedMove;
  public readonly bool IsFinalBestMoveCalc;

  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="manager"></param>
  /// <param name="node"></param>
  /// <param name="updateStatistics"></param>
  /// <param name="forcedMove"></param>
  public ManagerChooseBestMoveMCGS(MCGSManager manager, GNode node,
                                   bool updateStatistics, MGMove forcedMove,
                                   bool isFinalBestMoveCalc)
  {
    Manager = manager;
    Node = node;
    UpdateStatistics = updateStatistics;
    ForcedMove = forcedMove;
    IsFinalBestMoveCalc = isFinalBestMoveCalc;
  }


  /// <summary>
  /// Calculates the best move to play from root 
  /// given the current state of the search.
  /// </summary>
  public BestMoveInfoMCGS BestMoveCalc
  {
    get
    {
      MGPosition position = Node.IsSearchRoot ? this.Manager.Engine.SearchRootPosMG : Node.CalcPosition();

      if (ForcedMove != default)
      {
        return new BestMoveInfoMCGS(BestMoveInfoMCGS.BestMoveReason.UserForcedMove, ForcedMove, (float)Node.Q);
      }

      if (Manager.TopVForcedMove != default)
      {
        return new BestMoveInfoMCGS(BestMoveInfoMCGS.BestMoveReason.TopVMove, Manager.TopVForcedMove, (float)Node.Q);
      }

      if (Node.N <= 1 && Manager.CheckTablebaseBestNextMove != null)
      {
        // TODO: Improve this: in some situations, the best move coming back might not be actually best
        //       (for example, if falls into draw by repetition or if winningMoveListOrderedByDTM is false indicating search required).
        // TODO: Centralize this logic, appears also elsewhere.
        MGMove tablebaseMove = Manager.CheckTablebaseBestNextMove(position.ToPosition,
                                                                  out WDLResult result,
                                                                  out List<(MGMove, short)> otherWinningMoves,
                                                                  out bool winningMoveListOrderedByDTM);
        if (tablebaseMove != default && winningMoveListOrderedByDTM)
        {
          return new BestMoveInfoMCGS(BestMoveInfoMCGS.BestMoveReason.TablebaseImmediateMove, tablebaseMove, Node.V);
        }
      }

      if (Node.N <= 1)
      {
        MGMoveList mgMoves = new();
        MGMoveGen.GenerateMoves(position, mgMoves);
        if (mgMoves.NumMovesUsed == 1)
        {
          return new BestMoveInfoMCGS(BestMoveInfoMCGS.BestMoveReason.OneLegalMove, mgMoves.MovesArray[0], 0);
        }
      }


      if (Node.NumPolicyMoves == 0)
      {
        return new BestMoveInfoMCGS(BestMoveInfoMCGS.BestMoveReason.NoLegalMoves, default, Node.V);
      }
      else if (Node.NumEdgesExpanded == 0)
      {
        // No visits, create a node for the first child (which will be move with highest prior)
        return new BestMoveInfoMCGS(position, Node, BestMoveInfoMCGS.BestMoveReason.ImmediateNoSearchPolicyMove, Node.V);
      }
      else if (Node.NumEdgesExpanded == 1)
      {
        GEdge onlyEdge = Node.ChildEdgeAtIndex(0);
        BestMoveInfoMCGS.BestMoveReason reason = Node.NumPolicyMoves == 1 ? BestMoveInfoMCGS.BestMoveReason.OneLegalMove
                                                                          : BestMoveInfoMCGS.BestMoveReason.SearchResult;
        return new BestMoveInfoMCGS(reason, position, onlyEdge, (float)-onlyEdge.Q, onlyEdge.N, BestNSecond);
      }

      BestMoveInfoMCGS baselineBestMoveInfo = DoCalcBestMove();

#if MCGS_TEST_TERMINAL_PLAYOUTS
      ////////////////////////////////////////////////////////////////
      if (false && Manager.ParamsSearch.TestFlag
       && IsFinalBestMoveCalc
       && baselineBestMoveInfo.Reason == BestMoveInfoMCGS.BestMoveReason.SearchResult)
      {
        const float MIN_VISITS_MULTIPLIER = 0.02f;
        (BestMoveInfoMCGS bestMoveWithPlayouts, bool foundClose) =
          MCGSTest.ChooseBestMoveIncludingTerminalPlayouts(baselineBestMoveInfo, Manager, (int)(Node.N * MIN_VISITS_MULTIPLIER));
        bool bestChanged = bestMoveWithPlayouts.BestMove != baselineBestMoveInfo.BestMove;

        const bool VERBOSE = false;
        if (VERBOSE && foundClose)
        {
          Console.WriteLine($"foundClose={foundClose}, bestChanged={bestChanged}, baselineBest={baselineBestMoveInfo.BestMove}, playoutBest={bestMoveWithPlayouts.BestMove}");
        }
      }
      ////////////////////////////////////////////////////////////////
#endif

      // If we are winning, check if the best move
      // leads to opponent having a draw by repetition available.
      // If so, attempt to substitute another move which preserves the win.
      //
      // This can happen due to graph reuse if in Position mode, where the
      // children are contaminated by the effect of earlier visits which 
      // saw a draw by repetition pattern which is very different from
      // the one now at hand.
      //
      // In theory the opposite situation could also arise
      // (we fail to take advantage of a draw-by-repetition at hand in a losing position)
      // but this is very rare because a strong opponent would not have let this happen.
      const float THRESHOLD_SWITCH_AVOID_DRP = 0.05f;
      const bool DUMP_TO_CONSOLE = false;

      if (IsFinalBestMoveCalc
//       && Manager.ParamsSearch.TestFlag
       && Node.Q > THRESHOLD_SWITCH_AVOID_DRP
       && Manager.ParamsSearch.PathTranspositionMode == PathMode.PositionEquivalence
       && baselineBestMoveInfo.BestMoveEdge != default
       && !baselineBestMoveInfo.BestMoveEdge.ChildNode.IsNull)
      {
        GNode childNodeAfterBestMove = baselineBestMoveInfo.BestMoveEdge.ChildNode;
        ReadOnlySpan<GraphRootToSearchRootNodeInfo> nodesGraphToSearchRoot = Manager.Engine.SearchRootPathFromGraphRoot;

        bool opponentHasDrawByRepetition = GNode.DrawByRepetitionExistsAtChildEdgeAmongExpandedChildren(
          childNodeAfterBestMove,
          baselineBestMoveInfo.BestMoveEdge,
          nodesGraphToSearchRoot,
          pathHashes: null);

        if (opponentHasDrawByRepetition)
        {
          // Find second-best move (the move that would be chosen if best move was disqualified)
          GEdge? secondBestEdge = GetSecondBestEdge(baselineBestMoveInfo.BestMoveEdge);
          string secondBestInfo = secondBestEdge.HasValue
            ? $", 2nd best: {secondBestEdge.Value.MoveMG} Q={-secondBestEdge.Value.Q:F3}"
            : "";

          // Color based on Q: red if Q >= 0.25, yellow if Q >= 0.1
          ConsoleColor warningColor = Node.Q >= 0.25 ? ConsoleColor.Red
                                    : Node.Q >= 0.1 ? ConsoleColor.Yellow
                                    : ConsoleColor.White;

          // In yellow zone (Q between 0.1 and 0.3), substitute with second-best move if available
          // and the second-best move's Q (from our perspective) is also above the yellow cutoff
          bool secondBestQSufficient = secondBestEdge.HasValue
                                    && (-secondBestEdge.Value.Q) >= THRESHOLD_SWITCH_AVOID_DRP;
          if (secondBestEdge.HasValue && secondBestQSufficient)
          {
            if (DUMP_TO_CONSOLE)
            {
              ConsoleUtils.WriteLineColored(warningColor,
                $"SWITCH: Best move {baselineBestMoveInfo.BestMove} leads to opponent having draw by repetition available " +
                $"(root Q={Node.Q:F3}). Substituting with 2nd best: {secondBestEdge.Value.MoveMG} Q={-secondBestEdge.Value.Q:F3}.");
            }

            // Create new BestMoveInfoMCGS with fields consistent with the substituted move
            // The original best move becomes the "second best" since we rejected it
            return new BestMoveInfoMCGS(BestMoveInfoMCGS.BestMoveReason.SearchResult,
                                        position,
                                        secondBestEdge.Value,
                                        (float)-secondBestEdge.Value.Q,           // QMaximal: Q of substituted move from our perspective
                                        secondBestEdge.Value.N,                   // BestN: N of substituted move
                                        baselineBestMoveInfo.BestMoveEdge.N,      // BestNSecond: N of rejected original best (now "second")
                                        secondBestEdge.Value,                     // BestNEdge: the substituted move
                                        secondBestEdge.Value);                    // BestQEdge: the substituted move
          }
          else if (DUMP_TO_CONSOLE)
          {
            ConsoleUtils.WriteLineColored(warningColor,
              $"WARNING: Best move {baselineBestMoveInfo.BestMove} leads to opponent having draw by repetition available " +
              $"(root Q={Node.Q:F3}{secondBestInfo}).");
          }
        }
      }


      return baselineBestMoveInfo;
    }
  }


  /// <summary>
  /// N of the move having second best N.
  /// </summary>
  private float BestNSecond
  {
    get
    {
      GEdge[] childrenSortedN = Node.EdgesSorted(node => -node.N);

      return childrenSortedN.Length switch
      {
        0 => 0,
        < 2 => childrenSortedN[0].N,
        _ => childrenSortedN[1].N
      };
    }
  }


  /// <summary>
  /// Gets the second-best edge (the move that would be chosen if the specified best move was disqualified).
  /// Uses the same sorting logic as DoCalcBestMove to find the next best candidate.
  /// </summary>
  /// <param name="bestMoveEdge">The current best move edge to exclude.</param>
  /// <returns>The second-best edge, or null if none exists.</returns>
  private GEdge? GetSecondBestEdge(GEdge bestMoveEdge)
  {
    bool MoveAtIndexAllowed(int childIndex) => Node.ChildEdgeAtIndex(childIndex).N > 0
                                            && Manager.TerminationManager.MoveAtIndexAllowed(childIndex);

    // Sort by N (with Q tiebreaker), same logic as DoCalcBestMove
    GEdge[] childrenSortedN = Node.EdgesSorted(edge => MoveAtIndexAllowed(edge.ParentNode.IndexOfChildInChildEdges(edge.ChildNodeIndex))
                                ? (-edge.N + (float)edge.Q * 0.1f)
                                : 0);

    // Find the first edge that isn't the best move
    foreach (GEdge edge in childrenSortedN)
    {
      if (edge != bestMoveEdge && edge.N > 0)
      {
        return edge;
      }
    }

    return null;
  }


  /// <summary>
  /// Worker method that implements the rules to 
  /// to determine the best move to make.
  /// </summary>
  /// <returns></returns>
  private BestMoveInfoMCGS DoCalcBestMove()
  {
    MGPosition position = Node.IsSearchRoot ? Manager.Engine.Manager.Engine.SearchRootPosMG :  Node.CalcPosition();

    bool MoveAtIndexAllowed(int childIndex) => Node.ChildEdgeAtIndex(childIndex).N > 0
                                            && Manager.TerminationManager.MoveAtIndexAllowed(childIndex);

    // Get nodes sorted by N and Q (with most attractive move into beginning of array)
    // Note that the sort on N is augmented with an additional term based on Q so that tied N leads to lower Q preferred.
    // Also note that if a child is not allowed (filtered out by SearchMoves) then the move goes at the end).
    GEdge[] childrenSortedN = Node.EdgesSorted(edge => MoveAtIndexAllowed(edge.ParentNode.IndexOfChildInChildEdges(edge.ChildNodeIndex))
                                ? (-edge.N + (float)edge.Q * 0.1f)
                                : 0);
    GEdge[] edgesSortedQ = Node.EdgesSorted(edge => MoveAtIndexAllowed(edge.ParentNode.IndexOfChildInChildEdges(edge.ChildNodeIndex))
                             ? (float)edge.Q : float.MaxValue);

    GEdge priorBest = edgesSortedQ[0];

    // First see if any were forced losses for the child (i.e. wins for us)
    if (edgesSortedQ.Length == 1 || ParamsSelect.VIsForcedLoss((float)edgesSortedQ[0].Q))
    {
      return new BestMoveInfoMCGS(BestMoveInfoMCGS.BestMoveReason.SearchResult, position, edgesSortedQ[0],
                                  (float)-edgesSortedQ[0].Q, childrenSortedN[0].N, BestNSecond,
                                  childrenSortedN[0], edgesSortedQ[0]); // TODO: look for quickest win?
    }

    int thisMoveNum = Node.Graph.Store.NodesStore.PositionHistory.Moves.Count / 2; // convert ply to moves

    // Never use Top Q for very small trees
    const int MIN_N_USE_TOP_Q = 100;
    if (Manager.ParamsSearch.BestMoveMode == ParamsSearch.BestMoveModeEnum.RegularizedPolicyOptimizationLow)
    {
      const float LAMBDA = 0.125f;
      const float LAMBDA_POWER = 0.5f;
      GEdge bestChildEdge = RPOUtils.BestMove(Node, float.NaN, Node.NumEdgesExpanded, LAMBDA, LAMBDA_POWER);
      return new BestMoveInfoMCGS(BestMoveInfoMCGS.BestMoveReason.SearchResult, position, bestChildEdge, (float)-edgesSortedQ[0].Q, childrenSortedN[0].N,
                                  BestNSecond, childrenSortedN[0], edgesSortedQ[0]);

    }
    else if (Manager.ParamsSearch.BestMoveMode == ParamsSearch.BestMoveModeEnum.RegularizedPolicyOptimizationHigh)
    {
      const float LAMBDA = 0.175f;
      const float LAMBDA_POWER = 0.5f;
      GEdge bestChildEdge = RPOUtils.BestMove(Node, float.NaN, Node.NumEdgesExpanded, LAMBDA, LAMBDA_POWER);
      return new BestMoveInfoMCGS(BestMoveInfoMCGS.BestMoveReason.SearchResult, position, bestChildEdge, (float)-edgesSortedQ[0].Q, childrenSortedN[0].N,
                                   BestNSecond, childrenSortedN[0], edgesSortedQ[0]);
    }
    else if (Manager.ParamsSearch.BestMoveMode == ParamsSearch.BestMoveModeEnum.TopN
          || Node.N < MIN_N_USE_TOP_Q)
    {
      // Just return best N (note that tiebreaks are already decided with sort logic above)
      BestMoveInfoMCGS result = new BestMoveInfoMCGS(BestMoveInfoMCGS.BestMoveReason.SearchResult, position, childrenSortedN[0], (float)-edgesSortedQ[0].Q, childrenSortedN[0].N,
                                  BestNSecond, childrenSortedN[0], edgesSortedQ[0]);
      return TryOverrideWithMoreIrreversibleMove(position, result, edgesSortedQ);
    }
    else if (Manager.ParamsSearch.BestMoveMode == ParamsSearch.BestMoveModeEnum.TopQIfSufficientNPermissive
          || Manager.ParamsSearch.BestMoveMode == ParamsSearch.BestMoveModeEnum.TopQIfSufficientN)
    {
      float qOfBestNMove = (float)childrenSortedN[0].Q;

      // Only consider moves having number of visits which is some minimum fraction of visits to most visisted move
      int nOfChildWithHighestN = childrenSortedN[0].N;

      for (int i = 0; i < edgesSortedQ.Length; i++)
      {
        GEdge candidate = edgesSortedQ[i];

        // Return if this has a worse Q (for the opponent) and meets minimum move threshold
        if ((float)candidate.Q > qOfBestNMove)
        {
          break;
        }

        float differenceFromQOfBestN = MathF.Abs((float)candidate.Q - (float)childrenSortedN[0].Q);

        float minFrac = MinFractionNToUseQ(Node, differenceFromQOfBestN, Manager.ParamsSearch.BestMoveMode == ParamsSearch.BestMoveModeEnum.TopQIfSufficientNPermissive);

        int minNToBeConsideredForBestQ = (int)(nOfChildWithHighestN * minFrac);
        if (candidate.N > minNToBeConsideredForBestQ)
        {
          BestMoveInfoMCGS candidateResult = new BestMoveInfoMCGS(BestMoveInfoMCGS.BestMoveReason.SearchResult, position, candidate, (float)-edgesSortedQ[0].Q, childrenSortedN[0].N,
                                      BestNSecond, childrenSortedN[0], edgesSortedQ[0]);
          return TryOverrideWithMoreIrreversibleMove(position, candidateResult, edgesSortedQ);
        }
      }

      // We didn't find any moves qualified by Q, fallback to move with highest N
      BestMoveInfoMCGS result = new BestMoveInfoMCGS(BestMoveInfoMCGS.BestMoveReason.SearchResult, position, childrenSortedN[0], (float)-edgesSortedQ[0].Q, childrenSortedN[0].N,
                                  BestNSecond, childrenSortedN[0], edgesSortedQ[0]);
      return TryOverrideWithMoreIrreversibleMove(position, result, edgesSortedQ);

    }
    else
    {
      throw new Exception("Internal error, unknown BestMoveMode");
    }
  }


  /// <summary>
  /// Checks if the chosen move should be overridden with a move that reaches an irreversible position sooner.
  /// 
  /// The override occurs when:
  ///   - Root Q is greater than a threshold (indicating a winning position)
  ///   - There exists an alternative move B with Q not more than 0.02 worse than move A
  ///   - Either A has no irreversible move in its PV and B does, or B's ply until irreversible is less than half of A's
  /// </summary>
  /// <param name="position">The current position.</param>
  /// <param name="currentBest">The currently chosen best move info.</param>
  /// <param name="edgesSortedQ">Edges sorted by Q value.</param>
  /// <returns>The potentially overridden best move info.</returns>
  private BestMoveInfoMCGS TryOverrideWithMoreIrreversibleMove(MGPosition position,
                                                               BestMoveInfoMCGS currentBest,
                                                               GEdge[] edgesSortedQ)
  {
    // Return if not enabled.
    if (!IsFinalBestMoveCalc || Manager.ParamsSearch.AntiShufflingQThreshold == 0)
    {
      return currentBest;
    }

    const bool VERBOSE = false;
    const float MIN_ROOT_Q_FOR_OVERRIDE = 0.25f;

    const int MIN_PLY_ORIGINAL_NOT_IRREVERSIBLE = 12;
    const float MIN_SHORTER_IRREVERSIBLE_DIVISOR = 1.5f;
      
    // Only apply this logic in winning positions
    if (Node.Q <= MIN_ROOT_Q_FOR_OVERRIDE)
    {
      return currentBest;
    }

    GEdge moveA = currentBest.BestMoveEdge;
    if (moveA.IsNull)
    {
      return currentBest;
    }

    // Check if moveA itself is irreversible, or look at PV from child
    int? plyA;
    if (moveA.MoveMG.ResetsMove50Count)
    {
      plyA = 0; // moveA itself is irreversible
    }
    else if (moveA.ChildNode.IsNull)
    {
      return currentBest; // Can't analyze further without child node
    }
    else
    {
      plyA = moveA.ChildNode.PlyUntilPVIsIrreversibleMove();
    }

    // If the preferred move already has low ply to irreversible, no need to override
    if (plyA != null && plyA <= MIN_PLY_ORIGINAL_NOT_IRREVERSIBLE)
    {
      return currentBest;
    }

    // Look for an alternative move B that is better for reaching irreversibility
    foreach (GEdge candidateB in edgesSortedQ)
    {
      // Skip the current best move
      if (candidateB == moveA)
      {
        continue;
      }

      // Skip if child node doesn't exist (but allow irreversible moves which don't need child traversal)
      if (candidateB.IsNull || (!candidateB.MoveMG.ResetsMove50Count && candidateB.ChildNode.IsNull))
      {
        continue;
      }

      // Check Q difference (from opponent's perspective, so lower Q is better for us)
      // We only want to override if candidateB is slightly worse (higher Q) than moveA,
      // but not more than MAX_Q_DIFFERENCE worse.
      // If candidateB has better Q (negative difference), we shouldn't override since
      // the original move selection logic should have already preferred it.
      float qDifference = (float)(candidateB.Q - moveA.Q);
      if (qDifference > Manager.ParamsSearch.AntiShufflingQThreshold)  // Only reject if significantly worse
      {
        continue;
      }

      // Apply the same "sufficient N" logic used elsewhere to ensure the candidate
      // has enough visits to be considered reliable. The candidate must have at least
      // a minimum fraction of the best move's N, where the fraction depends on the Q difference.
      float minFrac = MinFractionNToUseQ(Node, qDifference, 
                                         Manager.ParamsSearch.BestMoveMode == ParamsSearch.BestMoveModeEnum.TopQIfSufficientNPermissive);
      int minNRequired = (int)(moveA.N * minFrac);
      if (candidateB.N < minNRequired)
      {
        continue;
      }

      // treat candidateB.MoveMG.ResetsMove50Count as plyB = 0
      int? plyB;
      GEdge? irreversibleEdgeB;
      if (candidateB.MoveMG.ResetsMove50Count)
      {
        plyB = 0; // Immediate irreversibility
        irreversibleEdgeB = candidateB; // The candidate move itself is the irreversible move
      }
      else
      {
        (plyB, irreversibleEdgeB) = candidateB.ChildNode.PlyUntilPVIsIrreversibleMoveWithMove();
      }

      // Check override conditions:
      // 1. A has no irreversible move in PV and B does, OR
      // 2. B's ply until irreversible is less than half of A's
      bool shouldOverride = false;

      if (!plyA.HasValue && plyB.HasValue)
      {
        // A has no irreversible move, B does
        shouldOverride = true;
      }
      else if (plyA.HasValue && plyB.HasValue && plyB.Value < plyA.Value / MIN_SHORTER_IRREVERSIBLE_DIVISOR)
      {
        // B reaches irreversible move in less than some fraction of the plies of A
        shouldOverride = true;
      }

      if (shouldOverride)
      {
        if (VERBOSE)
        {
          string plyAStr = plyA.HasValue ? plyA.Value.ToString() : "none";
          string plyBStr = plyB.HasValue ? plyB.Value.ToString() : "none";
          string reason = !plyA.HasValue ? "original has no irreversible in PV"
                        : $"plyB({plyBStr}) < plyA({plyAStr})/{MIN_SHORTER_IRREVERSIBLE_DIVISOR}";
          string irreversibleMoveStr = irreversibleEdgeB.HasValue ? irreversibleEdgeB.Value.MoveMG.ToString() : "?";
          string fen = position.ToPosition.FEN;

          Console.WriteLine($"IRREVERSIBLE_OVERRIDE: {moveA.MoveMG} -> {candidateB.MoveMG}  " +
                            $"Q: {-moveA.Q:F3} -> {-candidateB.Q:F3} (diff={qDifference:F3})  " +
                            $"Reason: {reason}  " +
                            $"IrreversibleMove: {irreversibleMoveStr} at ply {plyBStr}  " +
                            $"FEN: {fen}");
        }

        // Create a new BestMoveInfoMCGS with the overridden move
        return new BestMoveInfoMCGS(BestMoveInfoMCGS.BestMoveReason.SearchResult,
                                    position,
                                    candidateB,
                                    currentBest.QMaximal,
                                    currentBest.BestN,
                                    currentBest.BestNSecond,
                                    currentBest.BestNEdge,
                                    currentBest.BestQEdge);
      }
    }

    return currentBest;
  }


  /// <summary>
  /// Given a specified superiority of Q relative to another move,
  /// returns the minimum fraction of N (relative to the other move)
  /// required before the move will be preferred.
  /// 
  /// Relatively unexplored nodes are only chose when the Q difference is very large,
  /// because:
  ///   - the less explored (lower N) a move is, the more uncertain we are of its true Q (could be worse).
  ///   - N partly reflects the influence of policy, which should not be lightly ignored.
  /// </summary>
  /// <param name="qDifferenceFromBestQ"></param>
  /// <returns></returns>
  static internal float MinFractionNToUseQ(GNode node, float qDifferenceFromBestQ, bool permissive)
  {
    // Compute fraction required which decreases slowly below 1.0
    // as the Q difference increases (using a power function).
    // Value of 25 yields these minimum fractions for various sample levels of Q difference:
    //    0.005 --> 88%
    //    0.010 --> 78%
    //    0.020 --> 60%
    // Tests (using matches) suggest play quality is not highly sensitive to this POWER,
    // with a value of 30 possibly very slightly better than 20.
    const float POWER = 25;
    const float POWER_PERMISSIVE = 50;

    const float MIN_FRAC_N_REQUIRED_MIN_PERMISSIVE = 0.15f;

    float minFrac = MathF.Pow(1.0f - qDifferenceFromBestQ, (permissive ? POWER_PERMISSIVE : POWER));

    // Impose absolute minimum fraction.
    if (minFrac < (permissive ? MIN_FRAC_N_REQUIRED_MIN_PERMISSIVE : MIN_FRAC_N_REQUIRED_MIN))
    {
      minFrac = MIN_FRAC_N_REQUIRED_MIN;
    }

    return minFrac;
  }

}
