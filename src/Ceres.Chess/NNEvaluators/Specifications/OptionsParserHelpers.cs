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
using Ceres.Chess.Data.Nets;
using Ceres.Chess.NNEvaluators.Defs;
using Chess.Ceres.NNEvaluators;

#endregion

namespace Ceres.Chess.NNEvaluators.Specifications.Iternal
{
  /// <summary>
  /// Static helper methods to facilitate parsing of options strings.
  /// </summary>
  public static class OptionsParserHelpers
  {
    internal static List<(string netID, NNEvaluatorType type, NNEvaluatorPrecision precision, float wtValue, float wtPolicy, float wtMLH)>
      ParseNetworkOptions(string netSpecStr)
    {
      /// <summary>
      /// Delimiter character used to indicate separation between value/policy/mlh weight,
      /// for example "1.0;0.5;0.5"
      /// </summary>
      const char SUB_WEIGHTS_CHAR = ';';

      List<(string, NNEvaluatorType type, NNEvaluatorPrecision precision, float, float, float)> ret = new();

      string[] nets = netSpecStr.Split(",");

      float sumWeightsValue = 0.0f;
      float sumWeightsPolicy = 0.0f;
      float sumWeightsMLH = 0.0f;
      foreach (string netStrWithPrecision in nets)
      {
        const NNEvaluatorPrecision DEFAULT_PRECISION = NNEvaluatorPrecision.FP16;
        NNEvaluatorPrecision precision = DEFAULT_PRECISION;

        (NNEvaluatorType NN_EVAL_TYPE, string thisNetID) = ExtractEvaluatorTypeAndNetID(netStrWithPrecision);

        string netStr; // without any possible precision indicator

        // Parse precision string, if any (either #8 or #16 at end of network ID)
        if (thisNetID != null && thisNetID.Contains("#"))
        {
          string[] netIDs = thisNetID.Split("#");
          if (!int.TryParse(netIDs[1], out int precisionBits)
            || (precisionBits != 8 && precisionBits != 16 && precisionBits != 32))
          {
            throw new Exception("Network specification has invalid or unsupported precision " + netIDs[1]);
          }

          netStr = netIDs[0];
          precision = precisionBits == 8 ? NNEvaluatorPrecision.Int8 : (precisionBits == 16 ? NNEvaluatorPrecision.FP16 : NNEvaluatorPrecision.FP32);
        }
        else
        {
          netStr = thisNetID;
        }

        // Default to equal weight
        float weightValue = 1.0f / nets.Length;
        float weightPolicy = 1.0f / nets.Length;
        float weightMLH = 1.0f / nets.Length;

        string netID = null;
        if (netStr != null)
        {
          if (netStr.Contains(SUB_WEIGHTS_CHAR))
          {
            string[] netParts = netStr.Split(SUB_WEIGHTS_CHAR);
            thisNetID = netParts[0];

            if (netParts.Length == 4)
            {
              weightValue = ParseWt(netParts[1]);
              weightPolicy = ParseWt(netParts[2]);
              weightMLH = ParseWt(netParts[3]);
            }
            else
            {
              throw new Exception("Weight string must be of form value_wt;policy_wt;mlh_wt");
            }
          }
          else if (netStr.Contains("="))
          {
            string[] netParts = netStr.Split("=");
            thisNetID = netParts[0];

            if (netParts.Length == 2)
            {
              float wt = float.Parse(netParts[1]);
              weightValue = weightPolicy = weightMLH = wt;
            }
            else
            {
              throw new Exception("Weight string must be of form net=weight such as 703810=0.5");
            }
          }
          else
          {
            thisNetID = netStr;
          }
        }

        sumWeightsValue += weightValue;
        sumWeightsPolicy += weightPolicy;
        sumWeightsMLH += weightMLH;

        ret.Add((thisNetID, NN_EVAL_TYPE, precision, weightValue, weightPolicy, weightMLH));
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


    internal static (NNEvaluatorType NN_EVAL_TYPE, string thisNetID) ExtractEvaluatorTypeAndNetID(string netStrWithPrecision)
    {
      NNEvaluatorType NN_EVAL_TYPE;
      string thisNetID = null;

      // Check for a reference net marked by "~" followed by an alias.
      // This character chosen because it is one of the few that does not have a special meaning in Linux.
      if (netStrWithPrecision.StartsWith("~"))
      {
        string netID = netStrWithPrecision.Substring(1);
        if (RegisteredNets.Aliased.TryGetValue(netID, out RegisteredNetInfo baseline))
        {
          // Resolve to underlying network specification, call recursively.
          return ExtractEvaluatorTypeAndNetID(baseline.NetSpecificationString);
        }
        else
        {
          throw new Exception($"Network specification begins with ! but the no such reference net {netID}" 
                            + $" is registered in ReferenceNets.Common");
        }
      }

      if (netStrWithPrecision.ToUpper().StartsWith("LC0:"))
      {
        // Net specification "LC0:703810=0.5,66193=0.5";
        thisNetID = netStrWithPrecision.Substring(4);
        NN_EVAL_TYPE = NNEvaluatorType.LC0Library;
      }
      else if (netStrWithPrecision.ToUpper().StartsWith("LC0_ONNX_ORT:"))
      {
        thisNetID = netStrWithPrecision.Substring(13);
        NN_EVAL_TYPE = NNEvaluatorType.LC0ViaONNXViaORT;
      }
      else if (netStrWithPrecision.ToUpper().StartsWith("LC0_ONNX_TRT:"))
      {
        thisNetID = netStrWithPrecision.Substring(13);
        NN_EVAL_TYPE = NNEvaluatorType.LC0ViaONNXViaTRT;
      }
      else if (netStrWithPrecision.ToUpper().StartsWith("ONNX_TRT:"))
      {
        thisNetID = netStrWithPrecision.Substring(9);
        NN_EVAL_TYPE = NNEvaluatorType.ONNXViaTRT;
      }
      else if (netStrWithPrecision.ToUpper().StartsWith("ONNX_ORT:"))
      {
        thisNetID = netStrWithPrecision.Substring(9);
        NN_EVAL_TYPE = NNEvaluatorType.ONNXViaORT;
      }
      else if (netStrWithPrecision.ToUpper().StartsWith("TRT:"))
      {
        thisNetID = netStrWithPrecision.Substring(4);
        NN_EVAL_TYPE = NNEvaluatorType.TRT;
      }
      else if (netStrWithPrecision.ToUpper().StartsWith("RANDOM_WIDE:"))
      {
        thisNetID = netStrWithPrecision.Substring(12);
        NN_EVAL_TYPE = NNEvaluatorType.RandomWide;
      }
      else if (netStrWithPrecision.ToUpper().StartsWith("RANDOM_NARROW:"))
      {
        thisNetID = netStrWithPrecision.Substring(14);
        NN_EVAL_TYPE = NNEvaluatorType.RandomNarrow;
      }
      else if (netStrWithPrecision.ToUpper().StartsWith("COMBO_PHASED:"))
      {
        thisNetID = netStrWithPrecision.Substring(13);
        NN_EVAL_TYPE = NNEvaluatorType.ComboPhased;
      }
      else if (netStrWithPrecision.ToUpper().StartsWith("CUSTOM1"))
      {
        if (netStrWithPrecision.Contains(":"))
        {
          thisNetID = netStrWithPrecision.Split(":")[1];
        }
        NN_EVAL_TYPE = NNEvaluatorType.Custom1;
      }
      else if (netStrWithPrecision.ToUpper().StartsWith("CUSTOM2"))
      {
        if (netStrWithPrecision.Contains(":"))
        {
          thisNetID = netStrWithPrecision.Split(":")[1];
        }
        NN_EVAL_TYPE = NNEvaluatorType.Custom2;
      }
      else
      {
        // Prefix optionally omitted
        thisNetID = netStrWithPrecision;
        NN_EVAL_TYPE = NNEvaluatorType.LC0Library;
      }

      return (NN_EVAL_TYPE, thisNetID);
    }

    internal static List<(string netID, int? maxBatchSize, int? optimalBatchSize, string batchSizesFileName, float weight)>  ParseDeviceOptions(string str)
    {
      /// <summary>
      /// Delimiter character used to indicate beginning of a weights specification.
      /// </summary>
      const char WEIGHTS_CHAR = '@';

      List<(string, int?, int?, string, float)> ret = new();

      string[] nets = str.Split(",");

      int? maxBatchSize = null;
      int? optimalBatchSize = null;
      string batchSizesFileName = null;
      
      foreach (string netStr in nets)
      {
        string[] netParts = netStr.Split(WEIGHTS_CHAR);
        float weightValue, weightPolicy, weightMLH;
        string netID = netParts[0];

        if (netID.Contains("["))
        {
          ParseBatchSizeSpecification(netID, out maxBatchSize, out optimalBatchSize, out batchSizesFileName);
          netID = netID.Substring(0, netID.IndexOf("["));
        }

        float weight = 1.0f / nets.Length; // default equal weight
        if (netParts.Length > 1)
        {
          weight = float.Parse(netParts[1]);
        }

        ret.Add((netID, maxBatchSize, optimalBatchSize, batchSizesFileName, weight));        
      }


      return ret;
    }


    public static void ParseBatchSizeSpecification(string specStr, out int? maxBatchSize, out int? optimalBatchSize, out string batchSizesFileName)
    {
      maxBatchSize = null;
      optimalBatchSize = null;
      batchSizesFileName = null;

      string maxBatchStr = specStr.Substring(specStr.IndexOf("[") + 1).Replace("]", "");
      if (maxBatchStr.Contains(".."))
      {
        string[] bsSplit = maxBatchStr.Split("..");
        optimalBatchSize = int.Parse(bsSplit[0]);
        maxBatchSize = int.Parse(bsSplit[1]);
      }
      else if (int.TryParse(maxBatchStr, out int maxBatchSizeValue))
      {
        maxBatchSize = maxBatchSizeValue;
      }
      else
      {
        // Interpret as filename of batch file configuration file.
        batchSizesFileName = maxBatchStr;
      }
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
