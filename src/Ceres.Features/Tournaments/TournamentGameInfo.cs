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
using Ceres.Chess.GameEngines;

#endregion

namespace Ceres.Features.Tournaments
{
  public enum TournamentGameResult 
  { 
    /// <summary>
    /// Game not completed.
    /// </summary>
    None, 

    /// <summary>
    /// Win.
    /// </summary>
    Win, 

    /// <summary>
    /// Loss.
    /// </summary>
    Loss, 

    /// <summary>
    /// Draw.
    /// </summary>
    Draw 
  };


  public enum TournamentGameResultReason
  {
    /// <summary>
    /// Checkmate.
    /// </summary>
    Checkmate,

    /// <summary>
    /// Stalemate.
    /// </summary>
    Stalemate,

    /// <summary>
    /// Adjudicate tablebase known result.
    /// </summary>
    AdjudicateTB,

    /// <summary>
    /// Adjudicate insufficient mating material.
    /// </summary>
    AdjudicateMaterial,

    /// <summary>
    /// Aborted due to too excessive moves
    /// </summary>
    ExcessiveMoves,

    /// <summary>
    /// Draw by repetition.
    /// </summary>
    Repetition,

    /// <summary>
    /// Both players agree on advantage of one side of sufficient magnitude.
    /// </summary>
    AdjudicatedEvaluation,

    /// <summary>
    /// Player forfeited due to exceeding time/node limit.
    /// </summary>
    ForfeitLimits,
  }

  /// <summary>
  /// Record summarizing result of a tournament game.
  /// </summary>
  [Serializable]
  public record TournamentGameInfo
  {
    /// <summary>
    /// Sequence number within tournament.
    /// </summary>
    public int GameSequenceNum;

    /// <summary>
    /// Index of opening from opening book.
    /// </summary>
    public int OpeningIndex;

    /// <summary>
    /// If Engine2 is playing the white pieces.
    /// </summary>
    public bool Engine2IsWhite;

    /// <summary>
    /// Starting FEN of game.
    /// </summary>
    public string FEN;

    /// <summary>
    /// Result of game.
    /// </summary>
    public TournamentGameResult Result;

    /// <summary>
    /// Reason for game termination.
    /// </summary>
    public TournamentGameResultReason ResultReason;

    /// <summary>
    /// Number of ply in game.
    /// </summary>
    public int PlyCount;

    /// <summary>
    /// Total time used by engine 1 in seconds.
    /// </summary>
    public float TotalTimeEngine1;

    /// <summary>
    /// Total time used by engine 2 in seconds.
    /// </summary>
    public float TotalTimeEngine2;

    /// <summary>
    /// If engine 1 should have forfeited due to exceeding specified search limit at least once.
    /// </summary>
    public bool ShouldHaveForfeitedOnLimitsEngine1;

    /// <summary>
    /// If engine 2 should have forfeited due to exceeding specified search limit at least once.
    /// </summary>
    public bool ShouldHaveForfeitedOnLimitsEngine2;

    /// <summary>
    /// Total number of nodes evaluated by engine 1.
    /// </summary>
    public long TotalNodesEngine1;

    /// <summary>
    /// Total number of nodes evaluated by engine 2.
    /// </summary>
    public long TotalNodesEngine2;

    /// <summary>
    /// List of descriptive information relating to all moves played.
    /// </summary>
    public List<GameMoveStat> GameMoveHistory;

    /// <summary>
    /// Returns the reverse of a specified game result.
    /// </summary>
    /// <param name="result"></param>
    /// <returns></returns>
    public static TournamentGameResult InvertedResult(TournamentGameResult result)
    {
      switch (result)
      {
        case TournamentGameResult.Draw:
          return TournamentGameResult.Draw;
        case TournamentGameResult.Win:
          return TournamentGameResult.Loss;
        case TournamentGameResult.Loss:
          return TournamentGameResult.Win;
      }
      throw new Exception("Internal error: unexpected result");
    }
  }
}
