#region License notice

/*
Adapted from the PgnFileTools project by Clinton Sheppard 
at https://github.com/handcraftsman/PgnFileTools
licensed under Apache License, Version 2.
*/

#endregion

#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

using System;

using Ceres.Chess.Textual.PgnFileTools.MvbaCore;

namespace Ceres.Chess.Textual.PgnFileTools
{
  public class PieceColor : NamedConstant<PieceColor>
  {
    public static readonly PieceColor Black = new PieceColor("b", "Black");
    public static readonly PieceColor White = new PieceColor("w", "White");

    private PieceColor(string symbol, string description)
    {
      Symbol = symbol;
      Description = description;
      Add(symbol, this);
    }

    public string Description { get; private set; }
    public string Symbol { get; private set; }

    public static PieceColor GetFor(char key)
    {
      return GetFor(key + "");
    }

    public static PieceColor GetForFen(char ch)
    {
      return Char.IsUpper(ch) ? White : Black;
    }
  }
}
