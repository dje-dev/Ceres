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

using Ceres.Chess.Games.Utils;
using System;
using System.Collections.Generic;
using System.IO;

#endregion

namespace Ceres.Chess.Textual.PgnFileTools
{
  public class PgnStreamReader
  {
    public static List<PGNGame> ReadGames(string pgnFileName)
    {
      int index = 0;
      List<PGNGame> games = new();
      foreach (GameInfo game in new PgnStreamReader().Read(pgnFileName))
      {
        index++;
        if (game.HasError)
        {
          Console.WriteLine($"ERROR parsing game {index} in PGN {pgnFileName}: " + game.ErrorMessage);
          continue;
        }
        PGNGame gamePGN = new PGNGame(game, index);

        games.Add(gamePGN);
      }

      return games;
    }


    public IEnumerable<GameInfo> Read(string pgnFileName)
    {
      using (StreamReader reader = System.IO.File.OpenText(pgnFileName))
      {
        GameInfoParser parser = new ();

        while (reader.Peek() != -1)
        {
          yield return parser.Parse(reader);
        }
      }
    }

    public IEnumerable<GameInfo> Read(TextReader reader)
    {
      GameInfoParser parser = new();
      while (reader.Peek() != -1)
      {
        yield return parser.Parse(reader);
      }
    }
  }
}
