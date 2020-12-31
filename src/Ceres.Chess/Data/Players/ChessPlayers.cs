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

#endregion

namespace Ceres.Chess.Data.Players
{
  /// <summary>
  ///  Static collections of various chess players
  /// </summary>
  public static class ChessPlayers
  {
    /// <summary>
    /// Looks up a chess player by last name (or returns null if not found).
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    public static ChessPlayer ByLastName(string lastName)
    {
      lastName = lastName.ToUpper();
      foreach (ChessPlayer player in FamousPlayers)
        if (player.LastName.ToUpper() == lastName 
         || player.AlternateLastName.ToUpper() == lastName)
          return player;

      return null;
    }

    /// <summary>
    /// Static table of some of the strongest players in history.
    /// </summary>
    public readonly static ChessPlayer[] FamousPlayers = new ChessPlayer[]
    {
      new("Pachman", 1924, 1976),
      new("Portisch", 1937,1980),
      new("Huebner", 1948,1978),
      new("Lasker", 1868, 1894),
      new("Capablanca", 1888, 1916),
      new("Alekhine", 1892, 1930),
      new("Euwe", 1901, 1931),
      new("Botvinnik", 1911, 1948), // 1971 reported as peak rating, but that seems silly
      new("Reshevsky", 1911, 1972),
      new("Smyslov", 1921, 1971),
      new("Petrosian", 1929, 1972),
      new("Korchnoi", 1931, 1979, "Kortschnoj"),
      new("Tal", 1936, 1980),
      new("Fischer", 1943, 1972),
      new("Karpov", 1951, 1994),
      new("Kasparov", 1963, 1999),
      new("Anand", 1969, 2011),
      new("Topalov", 1975, 2015),
      new("Kramnik", 1975, 2016),
      new("Nakamura", 1987, 2015),
      new("Carlsen", 1990, 2018),
      new("Caruana", 1992, 2014),
    };

  }
}