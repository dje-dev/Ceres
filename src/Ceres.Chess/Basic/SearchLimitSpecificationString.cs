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
using System.Globalization;

#endregion

namespace Ceres.Chess
{
  /// <summary>
  /// Manages parsing of SearchLimit specification strings,
  /// textual representations of SearchLimits with syntax
  ///   <number> [+<number>] ["nm"|"ng"|"nt"|"sm"|"sg"]
  /// where the suffix indicates nodes/seconds per game/move.
  /// </summary>
  public static class SearchLimitSpecificationString
  {
    /// <summary>
    /// Resturns a SeachLimit by parsing a string limit specification.
    /// </summary>
    /// <param name="specificationString"></param>
    /// <returns></returns>
    public static SearchLimit Parse(string specificationString)
    {
      SearchLimit limit = TryParse(specificationString, out string errorString);
      if (limit == null)
      {
        throw new Exception($"Error parsing SearchLimit specification string {specificationString}. "
                           + "Expecting a number followed by NM, NG, SM or SG (nodes/seconds per move/game).");
      }
      else
        return limit;     
    }

    /// <summary>
    /// Attempts to parse specification string, returning error string if failure.
    /// </summary>
    public static SearchLimit TryParse(string specificationString, out string errorString)
    {
      if (specificationString == null) throw new ArgumentNullException(nameof(specificationString));
      specificationString = specificationString.Replace(" ", "").Replace("_", "").ToLower();

      const string ERROR_STRING = "Missing SearchLimit specification string, expect <number>[+<number>] followed by one of nm, ng, sm, or sg indicating nodes/seconds per game/move";
      if (specificationString == "")
      {
        errorString = ERROR_STRING;
        return null;
      }

      SearchLimitType limitType;
      if (specificationString.EndsWith("nm"))
        limitType = SearchLimitType.NodesPerMove;
      else if (specificationString.EndsWith("nt"))
        limitType = SearchLimitType.NodesPerTree;
      else if (specificationString.EndsWith("ng"))
        limitType = SearchLimitType.NodesForAllMoves;
      else if (specificationString.EndsWith("sm"))
        limitType = SearchLimitType.SecondsPerMove;
      else if (specificationString.EndsWith("sg"))
        limitType = SearchLimitType.SecondsForAllMoves;
      else
      {
        errorString = "Invalid SearchLimit specification, expected to end with one of nm, nim, ng, sm, or sg indicating nodes/seconds per game/move";
        return null;
      }

      string[] plusParts = specificationString[..^2].Split("+");
      float partIncrement = 0;
      float partBase;
      if (!float.TryParse(plusParts[0], NumberStyles.Any, CultureInfo.InvariantCulture, out partBase))
      {
        errorString = "Invalid SearchLimit specification, exected number of nodes/seconds at beginning of specification";
        return null;
      }

      if (plusParts.Length > 1)
      {
        if (!SearchLimit.TypeIsPerGameLimit(limitType))
        {
          errorString = "Invalid SearchLimit specification, increments are only valid with per game (ng or sg) limit types";
          return null;
        }
        if (!float.TryParse(plusParts[1], NumberStyles.Any, CultureInfo.InvariantCulture, out partIncrement))
        {
          errorString = "Invalid SearchLimit specification, exected + to be followed by a number indicating incremental nodes/seconds per move";
          return null;
        }
      }

      errorString = null;
      return new SearchLimit(limitType, partBase, true, partIncrement);
    }
  }
}
