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

using Ceres.Chess.Textual.PgnFileTools.MvbaCore;

namespace Ceres.Chess.Textual.PgnFileTools
{
  public class SymbolicMoveAnnotation : NamedConstant<SymbolicMoveAnnotation>
  {
    public static readonly SymbolicMoveAnnotation GoodMove = new SymbolicMoveAnnotation("!", 1);
    public static readonly SymbolicMoveAnnotation PoorMove = new SymbolicMoveAnnotation("?", 2);
    public static readonly SymbolicMoveAnnotation QuestionableMove = new SymbolicMoveAnnotation("?!", 6);
    public static readonly SymbolicMoveAnnotation SpeculativeMove = new SymbolicMoveAnnotation("!?", 5);
    public static readonly SymbolicMoveAnnotation VeryGoodMove = new SymbolicMoveAnnotation("!!", 3);
    public static readonly SymbolicMoveAnnotation VeryPoorMove = new SymbolicMoveAnnotation("??", 4);

    private SymbolicMoveAnnotation(string symbol, int id)
    {
      Id = id;
      Add(symbol, this);
    }

    public int Id { get; private set; }
  }
}
