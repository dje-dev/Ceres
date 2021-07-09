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

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Abstract base class for evaluators which are compound, 
  /// i.e. based on underlying set of (typically multiple) evaluators.
  /// </summary>
  public abstract class NNEvaluatorCompound : NNEvaluator
  {
    public readonly NNEvaluator[] Evaluators;

    #region Private data

    InputTypes inputsRequired = InputTypes.Undefined;

    readonly bool isWDL;
    readonly bool hasM;
    readonly bool policyReturnedSameOrderMoveList;

    #endregion

    #region Overrides

    /// <summary>
    /// Types of input(s) required by the evaluator.
    /// </summary>
    public override InputTypes InputsRequired => inputsRequired;


    /// <summary>
    /// If the evaluator has a WDL (win/draw/loss) head.
    /// </summary>
    public override bool IsWDL => isWDL;

    /// <summary>
    /// If the evaluator has an M (moves left) head.    /// </summary>
    public override bool HasM => hasM;

    /// <summary>
    /// If the network returns policy moves in the same order
    /// as the legal MGMoveList.
    /// </summary>
    public override bool PolicyReturnedSameOrderMoveList => policyReturnedSameOrderMoveList;


    #endregion

    public NNEvaluatorCompound(NNEvaluator[] evaluators)
    {
      Evaluators = evaluators;

      // until possibly prove false below
      isWDL = true; 
      hasM = true;
      policyReturnedSameOrderMoveList = true;

      // We must track additional position information if required by any of the evaluators
      foreach (NNEvaluator evaluator in evaluators)
      {
        inputsRequired |= evaluator.InputsRequired;

        if (!evaluator.IsWDL) isWDL = false;
        if (!evaluator.HasM) hasM = false;
        if (!evaluator.PolicyReturnedSameOrderMoveList) policyReturnedSameOrderMoveList = false;
      }
    }

    protected int MinBatchSizeAmongAllEvaluators
    {
      get
      {
        int min = int.MaxValue;
        for (int i = 0; i < Evaluators.Length; i++)
        {
          min = System.Math.Min(min, Evaluators[i].MaxBatchSize);
        }

        return min;
      }
    }

    protected override void DoShutdown()
    {
      foreach (NNEvaluator evaluator in Evaluators)
      {
        evaluator.Shutdown();
      }
    }

  }
}
