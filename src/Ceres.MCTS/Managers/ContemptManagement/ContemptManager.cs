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

namespace Ceres.MCTS.Managers.Limits
{
  /// <summary>
  /// Manager of continuous estimating the best contempt that
  /// should be used when playing against an opponent based 
  /// on periodic updates of the moves played by the opponent
  /// and their apparent strength.
  /// 
  /// NOTE: experimental.
  /// </summary>
  public class ContemptManager
  {
    const float EMPIRICAL_CONTEMPT_SCALING = 2.0f;
    const float MAX_EMPIRICAL_CONTEMPT = 0.10f;

    public readonly float BaseContempt;
    public readonly float ContemptAutoScaleWeight;

    public ContemptManager(float baseContempt, float contemptAutoScaleWeight)
    {
      BaseContempt = baseContempt;
      ContemptAutoScaleWeight = contemptAutoScaleWeight;
    }

    int numMovesTracked;
    float sumAbsQInferiorityFullGame;
    float movingWeightedAverageAbsQInferiority;

    const float LAMBDA = 0.80f;


    float AvgAbsQInferiorityFullGame => sumAbsQInferiorityFullGame / numMovesTracked;


    public float TargetContempt
    {
      get
      {
        if (ContemptAutoScaleWeight == 0 || numMovesTracked == 0 ) return BaseContempt;

        const float WEIGHT_FULL_GAME = 0.5f;
        float empiricalInferiority = WEIGHT_FULL_GAME * AvgAbsQInferiorityFullGame * 3.0f
                                   + (1.0f - WEIGHT_FULL_GAME) * movingWeightedAverageAbsQInferiority;

        float empiricalContempt = MathF.Min(MAX_EMPIRICAL_CONTEMPT, EMPIRICAL_CONTEMPT_SCALING * empiricalInferiority);

        float ret = (ContemptAutoScaleWeight * empiricalContempt)
          +    (1.0f - ContemptAutoScaleWeight) * BaseContempt;

        return ret;
      }
    }


    public void RecordOpponentMove(float qOpponentMove, float qBest)
    {
      float absInferiority = MathF.Abs(qOpponentMove - qBest);

      const float MAX_ABS_INFERIORITY = 0.20f;
      if (absInferiority > MAX_ABS_INFERIORITY) absInferiority = MAX_ABS_INFERIORITY;

      sumAbsQInferiorityFullGame += absInferiority;

      movingWeightedAverageAbsQInferiority = absInferiority * (1.0f - LAMBDA) +
                                             movingWeightedAverageAbsQInferiority * LAMBDA;

      numMovesTracked++;
    }
  }
}
