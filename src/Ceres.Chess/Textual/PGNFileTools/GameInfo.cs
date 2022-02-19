#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

using System.Collections.Generic;

namespace Ceres.Chess.Textual.PgnFileTools
{
  public class GameInfo
  {
    public readonly int GameIndex;

    public GameInfo(int gameIndex)
    {
      GameIndex = gameIndex;
      Headers = new Dictionary<string, string>(16);
      Moves = new List<Move>(128);
    }

    public string Comment { get; set; }

    public string ErrorMessage { get; set; }
    public bool HasError { get; set; }
    public IDictionary<string, string> Headers { get; private set; }
    public IList<Move> Moves { get; private set; }
  }
}
