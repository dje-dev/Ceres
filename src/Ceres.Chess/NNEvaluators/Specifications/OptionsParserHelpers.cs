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

    /// <summary>
    /// Delimiter character used to indicate separation between value/policy/mlh weight,
    /// for example "1.0;0.5;0.5"
    /// </summary>
    const char SUB_WEIGHTS_CHAR = ';';


    internal static List<(string netID, float wtValue, float wtPolicy, float wtMLH)> 
      ParseCommaSeparatedWithOptionalWeights(string str, bool allowSubWeights)
    {
      List<(string, float, float, float)> ret = new();

      string[] nets = str.Split(",");

      float sumWeightsValue = 0.0f;
      float sumWeightsPolicy = 0.0f;
      float sumWeightsMLH = 0.0f;
      foreach (string netStr in nets)
      {
        string[] netParts = netStr.Split(WEIGHTS_CHAR);
        float weightValue, weightPolicy, weightMLH;
        string netID = netParts[0];
        string netWts = netParts.Length == 1 ? "1" : netParts[1];
        string[] wtParts = allowSubWeights ? netWts.Split(SUB_WEIGHTS_CHAR) : new string[] { netWts };

        if (netParts.Length == 2)
        {
          if (wtParts.Length == 1)
          { 
            weightValue = weightPolicy = weightMLH = ParseWt(wtParts[0]);
          }
          else if (wtParts.Length == 3)
          {
            weightValue = ParseWt(wtParts[0]);
            weightPolicy = ParseWt(wtParts[1]);
            weightMLH = ParseWt(wtParts[2]);
          }
          else
          {
            throw new Exception("Weight string must be of form value_wt;policy_wt;mlh_wt");
          }
        }
        else
        {
          // Default is equally weighted
          weightValue = weightPolicy = weightMLH = 1.0f / nets.Length;
        }
        sumWeightsValue += weightValue;
        sumWeightsPolicy += weightPolicy;
        sumWeightsMLH += weightMLH;


        ret.Add((netID, weightValue, weightPolicy, weightMLH));
      }

      if (MathF.Abs(1.0f - sumWeightsValue) > 0.001)
      {
        if (sumWeightsValue == sumWeightsPolicy && sumWeightsValue == sumWeightsMLH)
        {
          throw new Exception($"Weights must not sum to 1.0, currently {sumWeightsValue}");
        }
        else
        {
          throw new Exception($"Weights value must not sum to 1.0, currently {sumWeightsValue}");
        }
      }

      if (MathF.Abs(1.0f - sumWeightsPolicy) > 0.001)
      {
        throw new Exception($"Weights policy must not sum to 1.0, currently {sumWeightsPolicy}");
      }
      if (MathF.Abs(1.0f - sumWeightsMLH) > 0.001)
      {
        throw new Exception($"Weights MLH must not sum to 1.0, currently {sumWeightsMLH}");
      }

      return ret;
    }

    private static float ParseWt(string wtString)
    {
      if (float.TryParse(wtString, NumberStyles.Any, CultureInfo.InvariantCulture, out float netWeight))
      {
        return netWeight;
      }
      else
        throw new Exception($"Expected weight not valid number: {wtString}");
    }
  }
}
