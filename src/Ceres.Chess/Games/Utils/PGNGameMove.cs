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


using Ceres.Chess.Textual.PgnFileTools;

#endregion

namespace Ceres.Chess.Games.Utils
{
  /// <summary>
  /// Represents a single move within a game in a PGN file.
  /// TODO: finish this.
  /// </summary>
  public record PGNGameMove
  {
    public readonly Move Move;

    public readonly float MoveTimeSeconds;
    public readonly float EvalCentipawns;
    public readonly short Depth;
    
    public PGNGameMove(Move move)
    {
#if NOT
      if (move.Comment != null)
      {
        // +1.35/8 0.17s
        string[] parts = move.Comment.Split(new char[] { '/', ' ' });
        if (parts.Length > 0) float.TryParse(parts[0], out EvalCentipawns);
        if (parts.Length > 1) short.TryParse(parts[1], out Depth);
        if (parts.Length > 2) float.TryParse(parts[2].Replace("s",""), out MoveTimeSeconds);
      }
#endif
    }
  }
}
