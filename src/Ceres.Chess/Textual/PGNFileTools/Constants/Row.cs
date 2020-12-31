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
  public class Row : NamedConstant<Row>
  {
    public static readonly Row Row1 = new Row(1, true);
    public static readonly Row Row2 = new Row(2, false);
    public static readonly Row Row3 = new Row(3, false);
    public static readonly Row Row4 = new Row(4, false);
    public static readonly Row Row5 = new Row(5, false);
    public static readonly Row Row6 = new Row(6, false);
    public static readonly Row Row7 = new Row(7, false);
    public static readonly Row Row8 = new Row(8, true);

    private Row(int index, bool isPromotionRow)
    {
      Index = index;
      IsPromotionRow = isPromotionRow;
      Symbol = index + "";
      Add(Symbol, this);
    }

    public int Index { get; private set; }

    public bool IsPromotionRow { get; private set; }
    public string Symbol { get; private set; }

    public static Row GetFor(char ch)
    {
      return GetFor(ch + "");
    }

    public static Row GetFor(int index)
    {
      return GetFor(index + "");
    }
  }
}
