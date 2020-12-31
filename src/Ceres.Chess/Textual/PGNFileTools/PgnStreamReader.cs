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

#region Using directives

using System.Collections.Generic;
using System.IO;

#endregion

namespace Ceres.Chess.Textual.PgnFileTools
{
  public class PgnStreamReader
  {
    public IEnumerable<GameInfo> Read(string pgnFileName)
    {
      using (StreamReader reader = System.IO.File.OpenText(pgnFileName))
      {
        var parser = new GameInfoParser();
        while (reader.Peek() != -1)
        {
          var game = parser.Parse(reader);
          yield return game;
        }
      }
    }

    public IEnumerable<GameInfo> Read(TextReader reader)
    {
      var parser = new GameInfoParser();
      while (reader.Peek() != -1)
      {
        var game = parser.Parse(reader);
        yield return game;
      }
    }
  }
}
