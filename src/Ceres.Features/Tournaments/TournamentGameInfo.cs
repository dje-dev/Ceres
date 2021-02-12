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

using System.Collections.Generic;
using Ceres.Chess.GameEngines;

#endregion

namespace Ceres.Features.Tournaments
{
  public enum TournamentGameResult { None, Win, Loss, Draw };

  /// <summary>
  /// Record summarizing result of a tournament game.
  /// </summary>
  public record TournamentGameInfo
  {
    /// <summary>
    /// Starting FEN of game.
    /// </summary>
    public string FEN;

    /// <summary>
    /// Result of game.
    /// </summary>
    public TournamentGameResult Result;

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
  }
}
