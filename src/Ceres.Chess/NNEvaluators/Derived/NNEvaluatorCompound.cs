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
    readonly bool hasUncertaintyV;
    readonly bool hasUncertaintyP;
    readonly bool hasAction;
    readonly bool hasValueSecondary;
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
    /// If action head is present in the network.
    /// </summary>
    public override bool HasAction => hasAction;

    /// <summary>
    /// Optional contextual information to be potentially used 
    /// as supplemental input for the evaluation of children.
    /// </summary>
    public override bool HasState => false; // not meaningful to combine multiple states


    /// <summary>
    /// If Uncertainty of V head is present in the network.
    /// </summary>
    public override bool HasUncertaintyV => hasUncertaintyV;


    /// <summary>
    /// If Uncertainty of policy head is present in the network.
    /// </summary>
    public override bool HasUncertaintyP => hasUncertaintyP;

    /// <summary>
    /// If the evaluator has an secondary value head.
    /// </summary>
    public override bool HasValueSecondary => hasValueSecondary;


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
      hasUncertaintyV = true;
      hasUncertaintyP = true;
      policyReturnedSameOrderMoveList = true;
      hasValueSecondary = true;
      hasAction = true;

      // We must track additional position information if required by any of the evaluators
      foreach (NNEvaluator evaluator in evaluators)
      {
        inputsRequired |= evaluator.InputsRequired;

        if (!evaluator.IsWDL) isWDL = false;
        if (!evaluator.HasM) hasM = false;
        if (!evaluator.HasUncertaintyV) hasUncertaintyV = false;
        if (!evaluator.HasUncertaintyP) hasUncertaintyP = false;
        if (!evaluator.PolicyReturnedSameOrderMoveList) policyReturnedSameOrderMoveList = false;
        if (!evaluator.HasValueSecondary) hasValueSecondary = false;
        if (!evaluator.HasAction) hasAction = false;
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
