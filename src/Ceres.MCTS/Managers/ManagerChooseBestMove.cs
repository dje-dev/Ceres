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
using System.Runtime.CompilerServices;
using Ceres.Base.Math;
using Ceres.Base.Math.Random;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.Params;

#endregion

[assembly: InternalsVisibleTo("Ceres.EngineMCTS.Test")] // TODO: move or remove me.

namespace Ceres.MCTS.Managers
{
  /// <summary>
  /// Manager that selects which move at the root of the search is best to play.
  /// </summary>
  public class ManagerChooseBestMove
  {
    public static float MLHMoveModifiedFraction => (float)countBestMovesWithMLHChosenWithModification / (float)countBestMovesWithMLHChosen;

    public static long countBestMovesWithMLHChosen = 0;
    public static long countBestMovesWithMLHChosenWithModification = 0;

    public readonly MCTSNode Node;
    public readonly bool UpdateStatistics;
    public readonly float MBonusMultiplier;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="updateStatistics"></param>
    /// <param name="mBonusMultiplier"></param>
    public ManagerChooseBestMove(MCTSNode node, bool updateStatistics, float mBonusMultiplier)
    {
      Node = node;
      UpdateStatistics = updateStatistics;
      MBonusMultiplier = mBonusMultiplier;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="moveNode"></param>
    /// <param name="mAvgOfBestQ"></param>
    /// <returns></returns>
    float MLHBoostForMove(MCTSNode moveNode, float mAvgOfBestQ)
    {
      const float MLH_DELTA_BOUND = 50; // truncate outliers
      float mHigherBy = StatUtils.Bounded(moveNode.MAvg - mAvgOfBestQ, -MLH_DELTA_BOUND, MLH_DELTA_BOUND);
      if (float.IsNaN(mHigherBy))
      {
        return 0; // no MLH support in network
      }

      const float Q_BOUND = 0.5f;
      float boundedRootQ = StatUtils.Bounded(MathF.Abs((float)Node.Context.Root.Q), -Q_BOUND, Q_BOUND);

      // Produce bonus which is more negative if the game will be longer and we are winning 
      float bonusMagnitude = mHigherBy * -boundedRootQ;

      const float MAX_BONUS = 0.03f;
      return StatUtils.Bounded(MBonusMultiplier * bonusMagnitude, -MAX_BONUS, MAX_BONUS);
#if NOT

      // Return function increasing in MBonusMultiplier, 
      return MBonusMultiplier * mHigherBy * boundedRootQ;

      float scaledDelta = mHigherBy * boundedRootQ * MBonusMultiplier;
      if (Node.Context.Root.Q > 0.10f) // we are winning
        return -scaledDelta; // We are winning therefore don't prefer moves leading to longer games
      else if (Node.Context.Root.Q < -0.10f) // we are losing
        return scaledDelta; // We are losing therefore we 
      else
        return 0;
#endif
    }

    /// <summary>
    /// Calculates the best move to play from root 
    /// given the current state of the search.
    /// </summary>
    public BestMoveInfo BestMoveCalc
    {
      get
      {
        if (Node.N <= 1 && Node.Context.CheckTablebaseBestNextMove != null)
        {
          // TODO: Improve this: in some situations, the best move coming back might not be actually best
          //       (for example, if falls into draw by repetition or if winningMoveListOrderedByDTM is false indicating search required).
          Node.Annotate();
          MGMove tablebaseMove = Node.Context.CheckTablebaseBestNextMove(in Node.Annotation.Pos, 
                                                                         out GameResult result, 
                                                                         out List<MGMove> otherWinningMoves, 
                                                                         out bool winningMoveListOrderedByDTM);
          if (tablebaseMove != default && winningMoveListOrderedByDTM)
          {
            return new BestMoveInfo(BestMoveInfo.BestMoveReason.TablebaseImmediateMove, tablebaseMove, Node.V);
          }
        }

        if (Node.N == 0)
        {
          Node.Annotate();
          if (Node.Annotation.Moves.NumMovesUsed == 1)
          {
            return new BestMoveInfo(BestMoveInfo.BestMoveReason.OneLegalMove, Node.Annotation.Moves.MovesArray[0], 0);
          }
        }


        if (Node.NumPolicyMoves == 0)
        {
          return new BestMoveInfo(BestMoveInfo.BestMoveReason.NoLegalMoves, default, Node.V);
        }
        else if (Node.NumPolicyMoves == 1)
        {
          MCTSNode onlyChild = Node.NumChildrenExpanded == 0 ? Node.CreateChild(0) : Node.ChildAtIndex(0);
          return new BestMoveInfo(onlyChild, (float)-onlyChild.Q, onlyChild.N, 1, 0);
        }
        else if (Node.NumChildrenExpanded == 0)
        {
          // No visits, create a node for the first child (which will be move with highest prior)
          return new BestMoveInfo(Node, BestMoveInfo.BestMoveReason.ImmediateNoSearchPolicyMove, Node.V);
        }
        else if (Node.NumChildrenExpanded == 1)
        {
          MCTSNode onlyChild = Node.ChildAtIndex(0);
          return new BestMoveInfo(onlyChild, (float)-onlyChild.Q, onlyChild.N, BestNSecond, 0);
        }

        return DoCalcBestMove();
      }
    }

    /// <summary>
    /// N of the move having second best N.
    /// </summary>
    private float BestNSecond
    {
      get
      {
        MCTSNode[] childrenSortedN = Node.ChildrenSorted(node => -node.N);

        return childrenSortedN.Length switch
        {
          0 => 0,
          < 2 => childrenSortedN[0].N,
          _ => childrenSortedN[1].N
        };
      }
    }

    /// <summary>
    /// Worker method that implements the rules to 
    /// to determine the best move to make.
    /// </summary>
    /// <returns></returns>
    public BestMoveInfo DoCalcBestMove()
    {
      bool useMLH = MBonusMultiplier > 0 && !float.IsNaN(Node.MAvg);
      if (useMLH && UpdateStatistics)
      {
        countBestMovesWithMLHChosen++;
      }

      bool MoveAtIndexAllowed(int childIndex) => Node.ChildAtIndex(childIndex).N > 0 
                                              && Node.Context.Manager.TerminationManager.MoveAtIndexAllowed(childIndex);

      // Get nodes sorted by N and Q (with most attractive move into beginning of array)
      // Note that the sort on N is augmented with an additional term based on Q so that tied N leads to lower Q preferred.
      // Also note that if a child is not allowed (filtered out by SearchMoves) then the move goes at the end).
      MCTSNode[] childrenSortedN = Node.ChildrenSorted(node => MoveAtIndexAllowed(node.IndexInParentsChildren) ? (-node.N + (float)node.Q * 0.1f) 
                                                                                                               : 0);
      MCTSNode[] childrenSortedQ = Node.ChildrenSorted(n => MoveAtIndexAllowed(n.IndexInParentsChildren) ? (float)n.Q 
                                                                                                         : float.MaxValue);


      float mAvgOfBestQ = childrenSortedQ[0].MAvg;
      MCTSNode priorBest = childrenSortedQ[0];

      if (useMLH)
      {
        // Note that more attractive moves have more negative Q, hence we invert MLHBoostForMove value
        childrenSortedQ = Node.ChildrenSorted(n => (float)n.Q + -MLHBoostForMove(n, mAvgOfBestQ));
      }

      const bool VERBOSE = false;
      if (VERBOSE
        && useMLH
        && Math.Abs(Node.Context.Root.Q) > 0.05f
        && childrenSortedQ[0] != priorBest)
      {
        Console.WriteLine("\r\n" + Node.Context.Root.Q + " " + Node.Context.Root.MPosition + " " + Node.Context.Root.MAvg);
        Console.WriteLine(priorBest + "  ==> " + childrenSortedQ[0]);
        for (int i = 0; i < Node.Context.Root.NumChildrenExpanded; i++)
        {
          MCTSNode nodeInner = Node.Context.Root.ChildAtIndex(i);
          Console.WriteLine($" {nodeInner.Q,6:F3} [MAvg= {nodeInner.MAvg,6:F3}] ==> {MLHBoostForMove(nodeInner, mAvgOfBestQ),6:F3} {nodeInner}");
        }
        Console.ReadKey();
      }


      // First see if any were forced losses for the child (i.e. wins for us)
      if (childrenSortedQ.Length == 1 || ParamsSelect.VIsForcedLoss((float)childrenSortedQ[0].Q))
      {
        return new BestMoveInfo(childrenSortedQ[0], (float)-childrenSortedQ[0].Q, childrenSortedN[0].N, BestNSecond, 0,
                                childrenSortedN[0], childrenSortedQ[0]); // TODO: look for quickest win?
      }

      int thisMoveNum = Node.Context.StartPosAndPriorMoves.Moves.Count / 2; // convert ply to moves

      if (Node.Context.ParamsSearch.SearchNoiseBestMoveSampling != null
       && thisMoveNum < Node.Context.ParamsSearch.SearchNoiseBestMoveSampling.MoveSamplingNumMovesApply
       && Node.Context.NumMovesNoiseOverridden < Node.Context.ParamsSearch.SearchNoiseBestMoveSampling.MoveSamplingMaxMoveModificationsPerGame
       )
      {
        throw new NotImplementedException();
        // TODO: currently only supported for sorting by N
        //MCTSNode bestMoveWithNoise = BestMoveByNWithNoise(childrenSortedN);
        //return new BestMoveInfo(bestMoveWithNoise, (float)-childrenSortedQ[0].Q, childrenSortedN[0].N,
        //                        BestNSecond, MLHBoostForMove(bestMoveWithNoise, mAvgOfBestQ)); // TODO: look for quickest win?
      }
      else
      {
        // Always use Top N for very small trees
        const int MIN_N_USE_TOP_Q = 100;
        if (Node.Context.ParamsSearch.BestMoveMode == ParamsSearch.BestMoveModeEnum.TopN
          || Node.N < MIN_N_USE_TOP_Q)
        {
          // Just return best N (note that tiebreaks are already decided with sort logic above)
          return new BestMoveInfo(childrenSortedN[0], (float)-childrenSortedQ[0].Q, childrenSortedN[0].N,
                                  BestNSecond, 0, childrenSortedN[0], childrenSortedQ[0]); // TODO: look for quickest win?
        }
        else if (Node.Context.ParamsSearch.BestMoveMode == ParamsSearch.BestMoveModeEnum.TopQIfSufficientN)
        {
          float qOfBestNMove = (float)childrenSortedN[0].Q;

          // Only consider moves having number of visits which is some minimum fraction of visits to most visisted move
          int nOfChildWithHighestN = childrenSortedN[0].N;


          for (int i = 0; i < childrenSortedQ.Length; i++)
          {
            MCTSNode candidate = childrenSortedQ[i];

            // Return if this has a worse Q (for the opponent) and meets minimum move threshold
            if ((float)candidate.Q > qOfBestNMove)
            {
              break;
            }

            float differenceFromQOfBestN = MathF.Abs((float)candidate.Q - (float)childrenSortedN[0].Q);

            float minFrac = MinFractionNToUseQ(Node, differenceFromQOfBestN);

            int minNToBeConsideredForBestQ = (int)(nOfChildWithHighestN * minFrac);
            if (candidate.N >= minNToBeConsideredForBestQ)
            {
              if (useMLH && UpdateStatistics)
              {
                ManagerChooseBestMove bestMoveChooserWithoutMLH = new ManagerChooseBestMove(this.Node, false, 0);
                if (bestMoveChooserWithoutMLH.BestMoveCalc.BestMoveNode != candidate)
                  countBestMovesWithMLHChosenWithModification++;
              }

              return new BestMoveInfo(candidate, (float)-childrenSortedQ[0].Q, childrenSortedN[0].N,
                                      BestNSecond, MLHBoostForMove(candidate, mAvgOfBestQ),
                                      childrenSortedN[0], childrenSortedQ[0]); // TODO: look for quickest win?
            }
          }

          // We didn't find any moves qualified by Q, fallback to move with highest N
          return new BestMoveInfo(childrenSortedN[0], (float)-childrenSortedQ[0].Q, childrenSortedN[0].N,
                                  BestNSecond, 0, childrenSortedN[0], childrenSortedQ[0]);
        }
        else
          throw new Exception("Internal error, unknown BestMoveMode");
      }
    }


    /// <summary>
    /// Given a specified superiority of Q relative to another move,
    /// returns the minimum fraction of N (relative to the other move)
    /// required before the move will be preferred.
    /// 
    /// Returned values are fairly close to 1.0 
    /// to avoid choising moves which are relatively much less explored.
    ///
    /// The greater the Q superiority of the cadidate, the lower the fraction required.
    /// </summary>
    /// <param name="qDifferenceFromBestQ"></param>
    /// <returns></returns>
    static internal float MinFractionNToUseQOLD(MCTSNode node, float qDifferenceFromBestQ)
    {
      bool isSmallTree = node.Context.Root.N < 50_000;

      float minFrac;

      if (isSmallTree)
      {
        // For small trees we are even more reluctant to rely upon Q if few visits
        minFrac = qDifferenceFromBestQ switch
        {
          >= 0.06f => MIN_FRAC_N_REQUIRED_MIN + 0.10f,
          >= 0.04f => 0.55f,
          >= 0.02f => 0.75f,
          _ => 0.90f
        };
      }
      else
      {
        minFrac = qDifferenceFromBestQ switch
        {
          >= 0.05f => MIN_FRAC_N_REQUIRED_MIN,
          >= 0.02f => 0.55f,
          >= 0.01f => 0.75f,
          _ => 0.90f
        };
      }

#if EXPERIMENTAL
      bool test = this.Node.Context.ParamsSearch.TestFlag;
      // Not completely successful attempt to simplify 
      // the above overparameterized logic.
      if (test)
      {
        if (qDifferenceFromBestQ < 0.002f)
          return 0.95f;

        // A LTC test (3+ min/game) with J94-100 yielded:
        //   -28 +/- 21 with 25
        //    -7 +/- 13 with 20
        const float POWER = 20f;
        minFrac = MathF.Pow(1.0f - qDifferenceFromBestQ, POWER) - 0.05f;

        if (minFrac < 0.35f) minFrac = 0.35f;
      }
#endif

      return minFrac;
    }


    internal const float MIN_FRAC_N_REQUIRED_MIN = 0.325f;


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
    static internal float MinFractionNToUseQ(MCTSNode node, float qDifferenceFromBestQ)
    {
//      if (!node.Context.ParamsSearch.TestFlag)
//      return MinFractionNToUseQOLD(node, qDifferenceFromBestQ);

      // Compute fraction required which decreases slowly below 1.0
      // as the Q difference increases (using a power function).
      const float POWER = 20f;
      float minFrac = MathF.Pow(1.0f - qDifferenceFromBestQ, POWER);

      // Impose absolute minimum fraction.
      if (minFrac < MIN_FRAC_N_REQUIRED_MIN)
      {
        minFrac = MIN_FRAC_N_REQUIRED_MIN;
      }

      return minFrac;
    }


    private MCTSNode BestMoveByNWithNoise(MCTSNode[] childrenSortedByAttractiveness)
    {
      if (Node.Context.ParamsSearch.BestMoveMode != ParamsSearch.BestMoveModeEnum.TopN)
        throw new NotImplementedException("SearchNoiseBestMoveSampling requires ParamsSearch.BestMoveModeEnum.TopN");

      float threshold = childrenSortedByAttractiveness[0].N * Node.Context.ParamsSearch.SearchNoiseBestMoveSampling.MoveSamplingConsiderMovesWithinFraction;
      float minN = childrenSortedByAttractiveness[0].N - threshold;
      List<MCTSNode> childrenWithinThreshold = new List<MCTSNode>();
      List<float> densities = new List<float>(childrenSortedByAttractiveness.Length);
      for (int i = 0; i < childrenSortedByAttractiveness.Length; i++)
      {
        if (childrenSortedByAttractiveness[i].N >= minN)
        {
          childrenWithinThreshold.Add(childrenSortedByAttractiveness[i]);
          densities.Add(childrenSortedByAttractiveness[i].N);
        }
      }

      if (childrenWithinThreshold.Count == 1)
      {
        return childrenSortedByAttractiveness[0];
      }
      else
      {
        MCTSNode bestMove = childrenWithinThreshold[ThompsonSampling.Draw(densities.ToArray(), Node.Context.ParamsSearch.SearchNoiseBestMoveSampling.MoveSamplingConsideredMovesTemperature)];
        if (bestMove != childrenSortedByAttractiveness[0])
        {
          Node.Context.ParamsSearch.SearchNoiseBestMoveSampling.MoveSamplingMaxMoveModificationsPerGame++;
          MCTSIterator.TotalNumMovesNoiseOverridden++;
        }
        return bestMove;
      }
    }

  }
}

