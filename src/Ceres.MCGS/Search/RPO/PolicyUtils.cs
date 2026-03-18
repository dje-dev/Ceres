#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

using System;

namespace Ceres.MCGS.Search.RPO;

public static class PolicyUtils
{
  /// <summary>
  /// Pull each Q? toward <paramref name="qPrior"/>.  
  /// The adjustment decays as 1 / ?(visits+1), so
  /// � 0 visits  ? fully replaced by prior  
  /// � many visits ? only a small nudge toward prior.
  /// </summary>
  /// <remarks>
  /// newQ? ? qPrior + (Q? ? qPrior) � (1 ? 1 / ?(visits? + 1)).
  /// </remarks>
  public static void ShrinkTowardQ(Span<float> Q, float qPrior, Span<int> visitCounts, float shrinkScaling)
  {
    if (Q.Length != visitCounts.Length)
    {
      throw new ArgumentException("Q and visitCounts must have identical lengths.");
    }

    for (int i = 0; i < Q.Length; ++i)
    {
      int visits = visitCounts[i];
      if (visits < 0)
      {
        throw new ArgumentOutOfRangeException(nameof(visitCounts), "Visit counts must be non-negative.");
      }

      const float SD = 1.0f;
      Q[i] = BayesianShrinkage.Shrink(Q[i], qPrior, SD, visitCounts[i], shrinkScaling);
#if NOT
			// ? = 1 / ?(v + 1)  ? (0,1]; ensures monotonic shrink as visits grow.
			float alpha = 1.0f / MathF.Sqrt((float)visits + 1.0f);

			// Move Q[i] toward the prior by factor ?.
			Q[i] = qPrior + (Q[i] - qPrior) * alpha * 0.5f;
#endif
#if NOT
			// shrinkFactor ? [0,1):  0 when visits=0  ? full shrink,
			//                        ?1 as visits?? ? minimal shrink.
			//float shrinkFactor = 1.0f - 1.0f / MathF.Sqrt((float)visits + 0.0f); // was + 1.0
			float oneSD = 1 / MathF.Sqrt(visits);
			float newQ = Q[i] - oneSD;		
//float newQ  = qPrior + (Q[i] - qPrior) * shrinkFactor;
			Q[i] = (shrinkScaling * newQ) + (1.0f - shrinkScaling) * Q[i];
#endif
    }
  }
}
