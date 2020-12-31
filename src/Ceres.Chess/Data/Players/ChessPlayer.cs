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


#endregion

namespace Ceres.Chess.Data.Players
{
  /// <summary>
  /// Represents a single chess player in history.
  /// </summary>
  public record ChessPlayer
  {
    /// <summary>
    /// Last name of player (but beware muliple spellings!).
    /// </summary>
    public readonly string LastName;

    /// <summary>
    /// Year of birth.
    /// </summary>
    public readonly int BirthYear;

    /// <summary>
    /// Estimated year of peak strength.
    /// Typically as determined by rating, if available.
    /// </summary>
    public readonly int PeakYear;

    /// <summary>
    /// Optional second last name (e.g.different spelling).
    /// </summary>
    public readonly string AlternateLastName;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="lastName"></param>
    /// <param name="birthYear"></param>
    /// <param name="peakYear"></param>
    /// <param name="alternateLastName"></param>
    public ChessPlayer(string lastName, int birthYear, int peakYear, string alternateLastName = null)
    {
      LastName = lastName;
      BirthYear = birthYear;
      PeakYear = peakYear;
      AlternateLastName = alternateLastName;
    }

  }
}