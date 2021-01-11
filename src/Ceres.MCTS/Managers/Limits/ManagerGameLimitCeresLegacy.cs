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
using System.Text;
using Ceres.Base.Math;
using Ceres.Chess;

#endregion

namespace Ceres.MCTS.Managers.Limits
{
  /// <summary>
  /// The default time manager for Ceres which employes
  /// a variety of techinques to try to optimally allocate time
  /// to moves within a game, possibly using the MLH if available.
  /// </summary>
  [Serializable]
  public class ManagerGameLimitCeresLegacy : IManagerGameLimit
  {
    static readonly SearchLimit zeroSeconds = new SearchLimit(SearchLimitType.SecondsPerMove, 0);
    static readonly SearchLimit zeroNodes = new SearchLimit(SearchLimitType.NodesPerMove, 0);


    /// Determines how much time or nodes resource to
    /// allocate to the the current move in a game subject to
    /// a limit on total numbrer of time or nodes over 
    /// some number of moves (or possibly all moves).
    public ManagerGameLimitOutputs ComputeMoveAllocation(ManagerGameLimitInputs inputs)
    {
      if (inputs.MaxMovesToGo.HasValue && inputs.MaxMovesToGo < 2)
      {
        return new ManagerGameLimitOutputs(new SearchLimit(inputs.TargetLimitType,
                                                           inputs.RemainingFixedSelf * 0.99f));
      }

      ManagerGameLimitOutputs outputs = DoComputeMoveTime(inputs);

      return outputs;
    }


    float EstimatedNumDifficultMovesToGo(ManagerGameLimitInputs inputs)
    {
      int numPieces = inputs.StartPos.PieceCount;

      // On estimate of the number of moves left comes from the piece count.
      // This method is likely imprecise, but is always available.
      // We subtract 6 from number of pieces since game is mostly or completely
      // over when 6 pieces are recahed (either via tablebases or very simple play).
      float estNumMovesLeftPieces = Math.Max(5, (numPieces - 6) * 2);

#if NOT
      // Attempts at using MLH were not immediately successful.
      float newM = inputs.Manager.Root.MAvg;

      // As a second estimate, use the output of the moves left head estimate
      float estNumMovesLeftMLH = (newM / 2);
      if (estNumMovesLeftMLH > 50)
      {
        // Shrink outlier very large estimates (more than 50 moves to go)
        estNumMovesLeftMLH = 50 + MathF.Sqrt(estNumMovesLeftMLH - 50);
      }
// Compute the aggregate estimate (equal weight combo from both if available)
      //      if (inputs.RootN == 0 || float.IsNaN(estNumMovesLeftMLH))
      //      else
      //        estNumMovesLeft = Stats.Average(estNumMovesLeftPieces, estNumMovesLeftMLH);
#endif

      float estNumMovesLeft = estNumMovesLeftPieces;

      // We assume that not all of these moves will be "hard",
      // e.g. at end of game the moves may become easier 
      // due to tablebases or more transpositions available, 
      // or narrower search trees due to simpler positions.
      const float MULTIPLIER = 0.65f;

      if (inputs.MaxMovesToGo.HasValue)
      {
        // Never assume more than 90% of remaining moves are hard
        float maxFromToGo = inputs.MaxMovesToGo.Value * 0.9f;
        float maxDefault = MULTIPLIER * estNumMovesLeft;
        return Math.Min(maxDefault, maxFromToGo);
      }
      else
      {
        return MULTIPLIER * estNumMovesLeft;
      }
    }


    /// <summary>
    /// Returns limits to be used for the special case of first move of a game
    /// (more time is allocated because there is no tree use available).
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="estNumMovesLeftHard"></param>
    /// <returns></returns>
    ManagerGameLimitOutputs MoveTimeForFirstMove(ManagerGameLimitInputs inputs, float estNumMovesLeftHard)
    {
      float fractionFirstMove = 0.05f;
      float targetUnits = inputs.RemainingFixedSelf * fractionFirstMove;
      return new ManagerGameLimitOutputs(new SearchLimit(inputs.TargetLimitType, targetUnits));
    }


    /// <summary>
    /// Main algorithm for determining amount of resources to allocate to this node.
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    ManagerGameLimitOutputs DoComputeMoveTime(ManagerGameLimitInputs inputs)
    {
      // Use more frontloading of early stopping enabled so we don't understoot in time spent
      float frontloadingAggressivemessMult = inputs.SearchParams.FutilityPruningStopSearchEnabled
                                             ? 1.5f
                                             : 1.3f;

      frontloadingAggressivemessMult *= inputs.SearchParams.GameLimitUsageAggressiveness;

      float estNumMovesLeftHard = EstimatedNumDifficultMovesToGo(inputs);

      // Special handling of first move
      if (inputs.IsFirstMoveOfGame)
      {
        return MoveTimeForFirstMove(inputs, estNumMovesLeftHard);
      }

      // When we are behind then it's worth taking a gamble and using more time
      // but when we are ahead, take a little less time to be sure we don't err in time pressure.
      // Testing suggests this feature is helpful (circa 10 elo?)
      float winningnessMultiplier = inputs.RootQ switch
      {
        < -0.25f => 1.20f,
        < -0.15f => 1.10f,
        > 0.25f => 0.90f,
        > 0.15f => 0.95f,
        _ => 1.0f
      };

      // Compute the base amount of time by dividing remaining time
      // by number of estimated "hard" moves left in game
      // where hard moves are those without trivial or tablebase best moves.
      //
      float totalAvailableSearchUnitsOverHardMoves = inputs.RemainingFixedSelf +
                                                     inputs.IncrementSelf * estNumMovesLeftHard;
      float targetUnitsBase = frontloadingAggressivemessMult
                              * winningnessMultiplier
                              * (totalAvailableSearchUnitsOverHardMoves / estNumMovesLeftHard);

      float targetNodes;

      float trailingNPS = inputs.TrailingAvgNPS(3, inputs.StartPos.MiscInfo.SideToMove);

      if (SearchLimit.TypeIsNodesLimit(inputs.TargetLimitType))
      {
        targetNodes = targetUnitsBase;
      }
      else
      {
        // Compute some trailing statistics on how many final nodes were in the search tree
        float trailingFinalNodes = inputs.TrailingAvgFinalNodes(5, inputs.StartPos.MiscInfo.SideToMove);

        // If we haven't accumulated sufficient history of prior moves
        // just return this naive target directly
        if (SearchLimit.TypeIsTimeLimit(inputs.TargetLimitType) && float.IsNaN(trailingNPS))
          return new ManagerGameLimitOutputs(new SearchLimit(inputs.TargetLimitType, targetUnitsBase));

        // Translate the time budget into an estimated nodes budget
        // using 
        float targetNodesRawViaNPS = targetUnitsBase * trailingNPS;

        float targetNodesViaPriorFinalN;
        if (!float.IsNaN(trailingFinalNodes))
        {
          targetNodesViaPriorFinalN = trailingFinalNodes;
        }
        else
          targetNodesViaPriorFinalN = targetNodesRawViaNPS; // not data to compute via final N, so stick with time based estimate

        // Compute how many nodes we
        float targetNodesViaNPS = targetNodesRawViaNPS - inputs.RootN;
        targetNodesViaNPS = MathF.Max(0, targetNodesViaNPS);

        targetNodes = StatUtils.Average(targetNodesViaPriorFinalN, targetNodesViaNPS);
      }

      float MULTIPLIER_BYPASS_SEARCH = 1.75f; // Testing suggests 1.75 as be approximate peak

      if (inputs.RootN > targetNodes * MULTIPLIER_BYPASS_SEARCH)
      {
        // Current tree (thanks to tree reuse) is already 
        // much larger than our target, so decide to do no search at all.
        // TODO: in this situation we could not bother doing MakeNewRoot operation to save time
        return inputs.TargetLimitType == SearchLimitType.SecondsPerMove
                                 ? new ManagerGameLimitOutputs(zeroSeconds)
                                 : new ManagerGameLimitOutputs(zeroNodes);
      }
      else
      {
        float targetAllocUnits;

        if (inputs.TargetLimitType == SearchLimitType.NodesPerMove)
        {
          if (inputs.RemainingFixedSelf < 100)
          {
            // We need to get very conservative with low nodes
            // since they are whole numbers not infinitely divisible.
            targetAllocUnits = Math.Max(1, inputs.RemainingFixedSelf / 50);
          }
          else
          {
            targetAllocUnits = targetNodes;
            float maxAllocUnits = 0.90f * inputs.RemainingFixedSelf;
            if (targetAllocUnits > maxAllocUnits) targetAllocUnits = maxAllocUnits;
          }
        }
        else
        {
          // Convert target nodes back into estimated time
          targetAllocUnits = targetNodes / trailingNPS;

          // As a safety check, never spend too much of remaining time on a single move
          float maxAllocUnits = 0.07f * inputs.RemainingFixedSelf;
          if (targetAllocUnits > maxAllocUnits) targetAllocUnits = maxAllocUnits;
        }

        return new ManagerGameLimitOutputs(new SearchLimit(inputs.TargetLimitType, targetAllocUnits));
      }
    }


  }
}
