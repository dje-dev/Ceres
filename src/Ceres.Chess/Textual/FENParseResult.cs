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

#endregion

namespace Ceres.Chess.Textual
{
  /// <summary>
  /// Stores the raw result of parsing a FEN as a set of 
  /// pieces and associated miscellaneous position information.
  /// </summary>
  public readonly struct FENParseResult
  {
    /// <summary>
    /// Set of pieces.
    /// </summary>
    public readonly List<PieceOnSquare> Pieces;

    /// <summary>
    /// Miscellaneous position information.
    /// </summary>
    public readonly PositionMiscInfo MiscInfo;


    /// <summary>
    /// Constructor (from set of pieces and miscellaneous information).
    /// </summary>
    /// <param name="pieces"></param>
    /// <param name="miscInfo"></param>
    public FENParseResult(List<PieceOnSquare> pieces, PositionMiscInfo miscInfo)
    {
      Pieces = pieces;
      MiscInfo = miscInfo;
    }


    /// <summary>
    /// Returns parse result as a Position object.
    /// </summary>
    public Position AsPosition => new Position(Pieces.ToArray(), MiscInfo);
  }
   
  
}
