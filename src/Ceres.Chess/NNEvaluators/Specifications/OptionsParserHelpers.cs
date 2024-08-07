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
    /// <summary>
    /// Delimiter character used to indicate separation between value/policy/mlh weight,
    /// for example "1.0;0.5;0.5"
    /// </summary>
    const char SUB_WEIGHTS_CHAR = ';';

    // The "~" character indicates the name given is an alias and for a registered net (in class RegisteredNets).
    // This character chosen because it is one of the few that does not have a special meaning in Linux.
    const string CHAR_ALIAS = "~";

    /// <summary>
    /// The "|" character delimites the beginning of a possible options string.
    /// </summary>
    const string CHAR_OPTIONS = "|";


    internal static (string options, List<(string netID, NNEvaluatorType type, NNEvaluatorPrecision precision, 
                     float wtValue, float wtValue2, float wtPolicy, float wtMLH, float wtUncertainty, float wtUncertaintyP)>)
      ParseNetworkOptions(string netSpecStr)
    {
      string optionsString = null;

      // Split off the options string if present.
      string[] parts = netSpecStr.Split(CHAR_OPTIONS);
      if (parts.Length > 2)
      {
        throw new Exception("Network specification must have at most one " + CHAR_OPTIONS + " character to separate netID from options");
      }
      else if (parts.Length == 2)
      {
        netSpecStr = parts[0];
        optionsString = parts[1];
      }
      else
      {
        netSpecStr = parts[0];
      }

      List<(string, NNEvaluatorType type, NNEvaluatorPrecision precision, float, float, float, float, float, float)> ret = new();

      string[] nets = netSpecStr.Split(",");

      float sumWeightsValue = 0.0f;
      float sumWeightsValue2 = 0.0f;
      float sumWeightsPolicy = 0.0f;
      float sumWeightsMLH = 0.0f;
      float sumWeightsUncertainty = 0;
      float sumWeightsUncertaintyP = 0;

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

          // Restore possible weights string which might have followed the alisas/precision indicator.
          if (netStrWithPrecision.Contains(";"))
          {
            string weightsPart = netStrWithPrecision.Substring(netStrWithPrecision.IndexOf(";")); 
            netStr = netStr + weightsPart;
          }
          precision = precisionBits == 8 ? NNEvaluatorPrecision.Int8 : (precisionBits == 16 ? NNEvaluatorPrecision.FP16 : NNEvaluatorPrecision.FP32);
        }
        else
        {
          netStr = thisNetID;
        }

        // Default to equal weight
        float weightValue = 1.0f / nets.Length;
        float weightValue2 = 1.0f / nets.Length;
        float weightPolicy = 1.0f / nets.Length;
        float weightMLH = 1.0f / nets.Length;
        float weightUncertainty = 1.0f / nets.Length;
        float weightUncertaintyP = 1.0f / nets.Length;

        string netID = null;
        if (netStr != null)
        {
          if (netStr.Contains(SUB_WEIGHTS_CHAR))
          {
            string[] netParts = netStr.Split(SUB_WEIGHTS_CHAR);
            thisNetID = netParts[0];

            if (netParts.Length == 5)
            {
              weightValue = ParseWt(netParts[1]);
              weightPolicy = ParseWt(netParts[2]);
              weightMLH = ParseWt(netParts[3]);
              weightUncertainty = ParseWt(netParts[4]);
              weightUncertaintyP = ParseWt(netParts[5]);
              weightValue2 = ParseWt(netParts[6]);
            }
            else
            {
              throw new Exception("Weight string must be of form value_wt;policy_wt;mlh_wt;uncertaintyv_wt;uncertaintyp_wt;value2_wt");
            }
          }
          else if (netStr.Contains("="))
          {
            string[] netParts = netStr.Split("=");
            thisNetID = netParts[0];

            if (netParts.Length == 2)
            {
              float wt = float.Parse(netParts[1]);
              weightValue = weightValue2 = weightPolicy = weightMLH = wt;
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
        sumWeightsValue2 += weightValue2;
        sumWeightsPolicy += weightPolicy;
        sumWeightsMLH += weightMLH;
        sumWeightsUncertainty += weightUncertainty;
        sumWeightsUncertaintyP += weightUncertaintyP;

        ret.Add((thisNetID, NN_EVAL_TYPE, precision, weightValue, weightValue2, weightPolicy, weightMLH, weightUncertainty, weightUncertaintyP));
      }

      static void CheckWeightSum(float sumWeights, string weightType)
      {
        if (MathF.Abs(1.0f - sumWeights) > 0.001)
        {
          throw new Exception($"Weights {weightType} must sum to 1.0, currently {sumWeights.ToString("0.##")}");
        }
      }

      CheckWeightSum(sumWeightsValue, "value");
      CheckWeightSum(sumWeightsPolicy, "policy");
      CheckWeightSum(sumWeightsMLH, "MLH");
      CheckWeightSum(sumWeightsUncertainty, "uncertainty");
      CheckWeightSum(sumWeightsUncertaintyP, "uncertaintyP");

      return (optionsString, ret);
    }


    internal static (NNEvaluatorType NN_EVAL_TYPE, string thisNetID) ExtractEvaluatorTypeAndNetID(string netStrWithPrecision)
    {
      NNEvaluatorType NN_EVAL_TYPE;
      string thisNetID = null;

      if (netStrWithPrecision.StartsWith(OptionsParserHelpers.CHAR_ALIAS))
      {
        string netID = netStrWithPrecision.Substring(1);

        // Strip off possible weighting specifications for the netID to look up.
        if (netID.Contains(";"))
        {
          netID = netID.Split(";")[0];
        }

        if (RegisteredNets.Aliased.TryGetValue(netID, out RegisteredNetInfo baseline))
        {
          // Resolve to underlying network specification, call recursively.
          return ExtractEvaluatorTypeAndNetID(baseline.NetSpecificationString);
        }
        else
        {
          throw new Exception($"Network specification begins with {CHAR_ALIAS} but the no such reference net {netID}" 
                            + $" is registered in ReferenceNets.Common");
        }
      }

      if (netStrWithPrecision.ToUpper().StartsWith("LC0:"))
      {
        // Net specification "LC0:703810=0.5,66193=0.5";
        thisNetID = netStrWithPrecision.Substring(4);
        NN_EVAL_TYPE = NNEvaluatorType.LC0;
      }
      else if (netStrWithPrecision.ToUpper().StartsWith("CERES:"))
      {
        thisNetID = netStrWithPrecision.Substring(6);
        NN_EVAL_TYPE = NNEvaluatorType.Ceres;
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
        NN_EVAL_TYPE = NNEvaluatorType.LC0;
      }

      return (NN_EVAL_TYPE, thisNetID);
    }

    internal static (List<(string deviceID, int? maxBatchSize, int? optimalBatchSize, string batchSizesFileName, float weight)>, string overrideEngine)
      ParseDeviceOptions(string deviceSpecString)
    {
      /// <summary>
      /// Delimiter character used to indicate beginning of a optional weights specification.
      /// </summary>
      const char WEIGHTS_CHAR = '@';

      /// <summary>
      /// Delimiter character used to indicate beginning of optional device execution engine
      /// for cases where multiple different engines are available (e.g. CUDA vs TensorRt).
      /// </summary>
      const char DEVICE_ENGINE_CHAR = '#';

      string overrideEngine = null;
      List<(string, int?, int?, string, float)> deviceList = new();

      // Parse precision string, if any (either #8 or #16 at end of network ID)
      if (deviceSpecString.Contains(DEVICE_ENGINE_CHAR))
      {
        string[] deviceSpecParts = deviceSpecString.Split("#");
        deviceSpecString = deviceSpecParts[0];

      }

      string[] nets = deviceSpecString.Split(",");

      int? maxBatchSize = null;
      int? optimalBatchSize = null;
      string batchSizesFileName = null;
      
      foreach (string netStr in nets)
      {
        string[] netParts = netStr.Split(WEIGHTS_CHAR);
        float weightValue, weightPolicy, weightMLH;
        string deviceID = netParts[0];

        if (deviceID.Contains("["))
        {
          ParseBatchSizeSpecification(deviceID, out maxBatchSize, out optimalBatchSize, out batchSizesFileName);
          deviceID = deviceID.Substring(0, deviceID.IndexOf("["));
        }

        float weight = 1.0f / nets.Length; // default equal weight
        if (netParts.Length > 1)
        {
          weight = float.Parse(netParts[1]);
        }

        deviceList.Add((deviceID, maxBatchSize, optimalBatchSize, batchSizesFileName, weight));
      }

      return (deviceList, overrideEngine);
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
