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
using System.Collections.Generic;
using System.Globalization;

#endregion

namespace Ceres.Chess.NNEvaluators.Specifications.Iternal
{
  /// <summary>
  /// Static helper methods to facilitate parsing of options strings.
  /// </summary>
  internal static class OptionsParserHelpers
  {
    /// <summary>
    /// Delimiter character used to indicate beginning of a weights specification.
    /// </summary>
    const char WEIGHTS_CHAR = '@';

    internal static List<(string, float)> ParseCommaSeparatedWithOptionalWeights(string str)
    {
      List<(string, float)> ret = new List<(string, float)>();

      string[] nets = str.Split(",");

      float sumWeights = 0.0f;
      foreach (string netStr in nets)
      {
        string[] netParts = netStr.Split(WEIGHTS_CHAR);
        float weight;
        string netID = netParts[0];

        if (netParts.Length == 2)
        {
          
          if (float.TryParse(netParts[1], NumberStyles.Any, CultureInfo.InvariantCulture, out float netWeight))
          {
            weight = netWeight;
          }
          else
            throw new Exception($"Expected weight not valid number: {netParts[1]}");
        }
        else
        {
          // Default is equally weighted
          weight = 1.0f / nets.Length;
        }
        sumWeights += weight;


        ret.Add((netID, weight));
      }

      if (MathF.Abs(1.0f - sumWeights) > 0.001) throw new Exception($"Weights must not sum to 1.0, currently {sumWeights}");

      return ret;
    }

  }
}
