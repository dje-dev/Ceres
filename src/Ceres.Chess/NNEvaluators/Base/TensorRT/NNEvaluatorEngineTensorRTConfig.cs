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

namespace Chess.Ceres.NNEvaluators.TensorRT
{
  /// <summary>
  /// Defines configuration for a TensorRT neural network evaluation.
  /// </summary>
  public class NNEvaluatorEngineTensorRTConfig : IEquatable<NNEvaluatorEngineTensorRTConfig>
  {
    /// <summary>
    /// Priority level which mirrors that of the PriorityLevel type in TensorRT
    /// </summary>
    public enum TRTPriorityLevel : int
    {
      Low = 0,
      Medium = 1,
      High = 2
    };


    public enum NetTypeEnum { Ceres, LC0 };

    public readonly string UFFFileName;
    public readonly string EngineType;
    public readonly int MaxBatchSize;
    public readonly NNEvaluatorPrecision Precision;
    public readonly int GPUID;
    public readonly bool IsWDL;
    public readonly bool HasM;
    public readonly TRTPriorityLevel PriorityLevel = 0;
    public readonly bool RetrieveValueFCActivations;


    public NNEvaluatorEngineTensorRTConfig(string uFFFileName, string engineType, int batchSize, NNEvaluatorPrecision precision, int gPUID,
                                           bool isWDL, bool hasM,
                                          TRTPriorityLevel priorityLevel, bool retrieveValueFCActivations)
    {
      UFFFileName = uFFFileName;
      EngineType = engineType;
      MaxBatchSize = batchSize;
      Precision = precision;
      GPUID = gPUID;
      IsWDL = isWDL;
      HasM = hasM;
      PriorityLevel = priorityLevel;
      RetrieveValueFCActivations = retrieveValueFCActivations;
    }

    public bool Equals(NNEvaluatorEngineTensorRTConfig other)
    {
      return UFFFileName == other.UFFFileName &&
             EngineType == other.EngineType &&
             MaxBatchSize == other.MaxBatchSize &&
             Precision == other.Precision &&
             GPUID == other.GPUID &&
             IsWDL == other.IsWDL &&
             PriorityLevel == other.PriorityLevel &&
             RetrieveValueFCActivations == other.RetrieveValueFCActivations;
    }

  }

}
