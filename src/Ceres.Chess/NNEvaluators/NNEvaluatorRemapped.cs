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

using Ceres.Base.DataTypes;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Wrapper for a neural network evaluator that somehow remaps the inputs or outputs.
  /// </summary>
  public class NNEvaluatorRemapped : NNEvaluator
  {
    public readonly NNEvaluator BaseEvaluator;

    public readonly float Temperature;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="baseEvaluator"></param>
    /// <param name="remappingString"></param>
    public NNEvaluatorRemapped(NNEvaluator baseEvaluator, string remappingString)
    {
      BaseEvaluator = baseEvaluator;
      Temperature = float.Parse(remappingString);
    }


    protected override void DoShutdown()
    {
      // TODO: fix this up somehow. We can access BaseEvaluator.Shutdown because it's protected.
      //BaseEvaluator.DoShutdown();
    }


    public override bool IsWDL => BaseEvaluator.IsWDL;

    public override bool HasM => BaseEvaluator.HasM;

    public override bool HasAction => BaseEvaluator.HasAction;

    public override bool HasUncertaintyV => BaseEvaluator.HasUncertaintyV;

    public override bool HasValueSecondary => BaseEvaluator.HasValueSecondary;

    public override int MaxBatchSize => BaseEvaluator.MaxBatchSize;

    public override bool HasState => BaseEvaluator.HasState;

    public override InputTypes InputsRequired => BaseEvaluator.InputsRequired;

    public override bool IsEquivalentTo(NNEvaluator evaluator) => BaseEvaluator.IsEquivalentTo(evaluator);

    public override bool PolicyReturnedSameOrderMoveList => BaseEvaluator.PolicyReturnedSameOrderMoveList;

    public override bool SupportsParallelExecution => BaseEvaluator.SupportsParallelExecution;

    public override bool UseBestValueMoveUseRepetitionHeuristic { get => base.UseBestValueMoveUseRepetitionHeuristic; set => base.UseBestValueMoveUseRepetitionHeuristic = value; }

    public override EvaluatorInfo Info => BaseEvaluator.Info;


    protected override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      IPositionEvaluationBatch baseBatch = BaseEvaluator.EvaluateIntoBuffers(positions, retrieveSupplementalResults);
      PositionEvaluationBatch batch = (PositionEvaluationBatch)baseBatch;

      // TODO: vectorize
      Span<FP16> w = batch.W.Span;
      Span<FP16> l = batch.L.Span;

      // TODO: remove this from ParamsSearch
      for (int i = 0; i < batch.NumPos; i++)
      {
        (float winPRaw, float drawPRaw, float lossPRaw) = (w[i], 1 - w[i] - l[i], l[i]);
        float winRaw = Math.Clamp((float)w[i], 1e-7f, 1f - 1e-7f); // Clamp to avoid log(0)
        float lossRaw = Math.Clamp((float)l[i], 1e-7f, 1f - 1e-7f);
        float drawRaw = 1 - winRaw - lossRaw;
        drawRaw = Math.Clamp(drawRaw, 1e-7f, 1f - 1e-7f);

        // Calculate logits
        float winLogit = MathF.Log(winRaw) / Temperature;
        float drawLogit = MathF.Log(drawRaw) / Temperature;
        float lossLogit = MathF.Log(lossRaw) / Temperature;

        // Calculate adjusted probabilities
        float winAdj = MathF.Exp(winLogit);
        float drawAdj = MathF.Exp(drawLogit);
        float lossAdj = MathF.Exp(lossLogit);

        // Normalize
        float sumAdj = winAdj + drawAdj + lossAdj;
        w[i] = (FP16)(winAdj / sumAdj);
        l[i] = (FP16)(lossAdj / sumAdj);      
      }

      return baseBatch;
    }

    public override IPositionEvaluationBatch DoEvaluateNativeIntoBuffers(object positionsNativeInput, bool usesSecondaryInputs,
                                                                        int numPositions, Func<int, int, bool> posMoveIsLegal,
                                                                        bool retrieveSupplementalResults = false)
    {
      throw new NotImplementedException();
    }
  }
}
