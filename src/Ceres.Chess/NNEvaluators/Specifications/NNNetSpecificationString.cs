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
using System.Linq;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.Specifications.Iternal;
using Chess.Ceres.NNEvaluators;

#endregion

namespace Ceres.Chess.NNEvaluators.Specifications
{
  /// <summary>
  /// Manages parsing of a neural net specification string which 
  /// specifies the neural network(s) to be used for evaluationg positions.
  /// 
  /// Examples: 
  ///   "LC0:42767"
  ///   "LC0:703810=0.5,66193=0.5"
  /// </summary>
  public record NNNetSpecificationString
  {
    /// <summary>
    /// Method used to combine (possibly) multiple nets.
    /// </summary>
    public readonly NNEvaluatorNetComboType ComboType;

    /// <summary>
    /// List of nets to be used, and their fractional weights for the value, policy, and MLH heads.
    /// </summary>
    public readonly List<(NNEvaluatorNetDef def, float wtValue, float wtPolicy, float wtMLH)> NetDefs;


    /// <summary>
    /// Constructor which parses a net specification string.
    /// </summary>
    /// <param name="netString"></param>
    public NNNetSpecificationString(string netString)
    {
      NNEvaluatorType NN_EVAL_TYPE = NNEvaluatorType.LC0Library;

      string netIDs;
      if (netString.ToUpper().StartsWith("LC0:"))
      {
        // Net specification "LC0:703810=0.5,66193=0.5";
        netIDs = netString.Substring(4);
      }
      else if (netString.ToUpper().StartsWith("ONNX_TRT:"))
      {
        netIDs = netString.Substring(9);
        NN_EVAL_TYPE = NNEvaluatorType.ONNXViaTRT;
      }
      else if (netString.ToUpper().StartsWith("ONNX_ORT:"))
      {
        netIDs = netString.Substring(9);
        NN_EVAL_TYPE = NNEvaluatorType.ONNXViaORT;
      }
      else if (netString.ToUpper().StartsWith("RANDOM_WIDE:"))
      {
        netIDs = netString.Substring(12);
        NN_EVAL_TYPE = NNEvaluatorType.RandomWide;
      }
      else if (netString.ToUpper().StartsWith("RANDOM_NARROW:"))
      {
        netIDs = netString.Substring(14);
        NN_EVAL_TYPE = NNEvaluatorType.RandomNarrow;
      }
      else if (netString.ToUpper().StartsWith("COMBO_PHASED:"))
      {
        netIDs = netString.Substring(13);
        NN_EVAL_TYPE = NNEvaluatorType.ComboPhased;
      }
      else if (netString.ToUpper().StartsWith("CUSTOM1:"))
      {
        netIDs = netString.Substring(8);
        NN_EVAL_TYPE = NNEvaluatorType.Custom1;
      }
      else if (netString.ToUpper().StartsWith("CUSTOM2:"))
      {
        netIDs = netString.Substring(8);
        NN_EVAL_TYPE = NNEvaluatorType.Custom2;
      }
      else
      {
        // Prefix optionally omitted
        netIDs = netString;
      }

      // Build network definitions
      List<(string, NNEvaluatorPrecision, float, float, float)> netParts = OptionsParserHelpers.ParseCommaSeparatedWithOptionalWeights(netIDs, true);

      NetDefs = new List<(NNEvaluatorNetDef, float, float, float)>();
      foreach (var netSegment in netParts)
      {
        NetDefs.Add((new NNEvaluatorNetDef(netSegment.Item1, NN_EVAL_TYPE, netSegment.Item2), netSegment.Item3, netSegment.Item4, netSegment.Item5));
      }

      ComboType = NetDefs.Count == 1 ? NNEvaluatorNetComboType.Single : NNEvaluatorNetComboType.WtdAverage;
    }


    /// <summary>
    /// Returns if specified network string is valid.
    /// </summary>
    /// <param name="netString"></param>
    /// <returns></returns>
    public static bool IsValid(string netString)
    {
      try
      {
        new NNNetSpecificationString(netString);
        return true;
      }
      catch
      {
        return false;
      }
    }


    static string NetInfoStr((NNEvaluatorNetDef def, float wtValue, float wtPolicy, float wtMLH) net)
    {
      if (net.wtValue == 1)
        return $"{net.def} ";
      else
        return $"({net.def} {net.wtValue} {net.wtPolicy} {net.wtMLH}) ";
    }


    static string ToSpecificationStringComplex(NNEvaluatorNetComboType comboType, IEnumerable<(NNEvaluatorNetDef, float, float, float)> nets)
    {
      string ret = $"Nets: {comboType} ";
      foreach ((NNEvaluatorNetDef, float, float, float) net in nets)
      {
        ret += NetInfoStr(net);
      }
      return ret;
    }


    public static string ToSpecificationString(NNEvaluatorNetComboType comboType, IEnumerable<(NNEvaluatorNetDef, float, float, float)> nets)
    {
      // TODO: Currently support conversion back to original specification string only for simple cases
      if (comboType == NNEvaluatorNetComboType.Single
       && nets.Count() == 1
       && nets.First().Item1.Type == NNEvaluatorType.LC0Library)
      {
        return $"Network=LC0:{nets.First().Item1.NetworkID}";
      }
      else
      {
        return ToSpecificationStringComplex(comboType, nets);
      }
    }


  }
}
