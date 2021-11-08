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
using Ceres.Chess;

#endregion

namespace Ceres.Chess.GameEngines
{
  /// <summary>
  /// Summary statistics describing the search charactersitics for a move in a game.
  /// </summary>
  [Serializable]
  public record GameMoveStat
  {
    /// <summary>
    /// Identifier for player/engine.
    /// </summary>
    public string Id;
    /// <summary>
    /// Ply number within game.
    /// </summary>
    public readonly int PlyNum;

    /// <summary>
    /// Side to move.
    /// </summary>
    public readonly SideType Side;

    /// <summary>
    /// Amount of time (seconds) already consumed on this player's clock at start of move.
    /// </summary>
    public readonly float ClockSecondsAlreadyConsumed;

    /// <summary>
    /// Position evaluation at end of search (Q = probability of winning).
    /// </summary>
    public readonly float ScoreQ;

    /// <summary>
    /// Position evaluation at end of search (in centipawns).
    /// </summary>
    public readonly float ScoreCentipawns;

    /// <summary>
    /// Number of pieces on the board.
    /// </summary>
    public readonly int NumPieces;

    /// <summary>
    /// Average Moves Left estimate (if available, else NaN).
    /// </summary>
    public readonly float MAvg;

    /// <summary>
    /// Final N at root.
    /// </summary>
    public readonly int FinalN;

    /// <summary>
    /// Total N added during search (same as FinalN if no tree reuse).
    /// </summary>
    public readonly int NumNodesComputed;

    /// <summary>
    /// Search resources allocated to search by time manager.
    /// </summary>
    public readonly SearchLimit SearchLimit;

    /// <summary>
    /// Elapsed time actually spent in search.
    /// </summary>
    public readonly float TimeElapsed;

    /// <summary>
    /// Number of nodes at start of search
    /// </summary>
    public int StartN => FinalN - NumNodesComputed;

    /// <summary>
    /// Realized nodes per second during search for move.
    /// </summary>
    public float NodesPerSecond => nps ?? (NumNodesComputed / TimeElapsed);


    /// <summary>
    /// Optional nps (nodes per second) reported directly by an engine.
    /// </summary>
    private float? nps;

    public GameMoveStat(int plyNum, SideType side, float scoreQ, float scoreCP, float clockSecondsAlreadyConsumed,
                        int numPieces, float mAvg, int finalN, int numNodesComputed, 
                        SearchLimit searchLimit, float timeElapsed, float? nps = null)
    {
      PlyNum = plyNum;
      Side = side;
      ScoreQ = scoreQ;
      ScoreCentipawns = scoreCP;
      ClockSecondsAlreadyConsumed = clockSecondsAlreadyConsumed;
      NumPieces = numPieces;
      MAvg = mAvg;
      FinalN = finalN;
      NumNodesComputed = numNodesComputed;
      SearchLimit = searchLimit;
      TimeElapsed = timeElapsed;
      this.nps = nps;
    }


    public override string ToString()
    {
      return $"<GameMoveStat {PlyNum}. {Side} N {StartN} --> {FinalN} in {TimeElapsed,6:F2}sec Q={ScoreQ,6:F2}";
    }

  }
}
