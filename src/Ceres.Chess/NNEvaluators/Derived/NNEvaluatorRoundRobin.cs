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

using System.Net.NetworkInformation;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// NNEvaluator subclass which dispatches batches to one
  /// of multiple evaluators in a round-robin fashion
  /// </summary>
  public class NNEvaluatorRoundRobin : NNEvaluatorCompound
  {
    int nextIndex;

    /// <summary>
    /// Constructor from an array of evaluators over which to dispatch to round-robin.
    /// </summary>
    /// <param name="evaluators"></param>
    public NNEvaluatorRoundRobin(NNEvaluator[] evaluators) : base (evaluators)
    {
      nextIndex = 0;
    }


    /// <summary>
    /// The maximum number of positions that can be evaluated in a single batch.
    /// </summary>
    public override int MaxBatchSize => MinBatchSizeAmongAllEvaluators;


    public override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      // Evaluate using next evaluator
      IPositionEvaluationBatch batch = Evaluators[nextIndex++].EvaluateIntoBuffers(positions, retrieveSupplementalResults);
      
      // Advance to next evaluator for next time
      nextIndex = nextIndex % Evaluators.Length;

      return batch;
    }

  }
}
