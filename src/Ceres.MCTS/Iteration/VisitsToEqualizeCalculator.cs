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
using Ceres.Base.Math;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Static class with method to calculate or estimate
  /// the number of nodes that would be needed to equalize score,
  /// assuming consecutive visits yielding backup of specified value.
  /// 
  /// Note: experimental.
  /// </summary>
  public static class VisitsToEqualizeCalculator
  {
    /// <summary>
    /// Returns the total number of MCTS steps that would be necessary 
    /// for node2 to catch up to node1 in terms of N, 
    /// under the assumption that steps are with specified V values for the node
    /// </summary>
    /// <param name="N"></param>
    /// <param name="p1"></param>
    /// <param name="q1"></param>
    /// <param name="n1"></param>
    /// <param name="p2"></param>
    /// <param name="q2"></param>
    /// <param name="n2"></param>
    /// <param name="assumedV1"></param>
    /// <param name="assumedV2"></param>
    /// <returns></returns>
    public static int NumVisitsToEqualize(float ucdNumeratorPower, float uctDenominatorPower,
                                          int N,
                                          float p1, float q1, int n1,
                                          float p2, float q2, int n2,
                                          float assumedV1, float assumedV2)
    {
      // Approximate by moving in steps (to reduce runtime) 
      int STEP_SIZE = Math.Max(200, N / 100);

      if (assumedV1 > assumedV2) return 0;

      float w1 = q1 * n1;
      float w2 = q2 * n2;

      int numNodes1 = 0;
      int numNodes2 = 0;
      int loopCount = 0;
      float cpuct = 0;
      while (true)
      {
        int totalNodes = N + numNodes1 + numNodes2;

        if (loopCount % 10 == 0) cpuct = CalcCPUCT(totalNodes);

        if (n2 + numNodes2 > n1 + numNodes1) break;

        float score1 = ScoreCalc(ucdNumeratorPower, uctDenominatorPower, cpuct, totalNodes, p1, w1 + (numNodes1 * assumedV1), n1 + numNodes1);
        float score2 = ScoreCalc(ucdNumeratorPower, uctDenominatorPower, cpuct, totalNodes, p2, w2 + (numNodes2 * assumedV2), n2 + numNodes2);
        if (score1 > score2)
          numNodes1 += STEP_SIZE;
        else
          numNodes2 += STEP_SIZE;
        loopCount++;
      };
     
      return numNodes1 + numNodes2;
    }


    static float ScoreCalc(float uctNumeratorPower, float uctDenominatorPower, float cpuct, int N, float p, float w, int n)
    {
      float q = w / n;
      float denominator = uctDenominatorPower == 1.0f ? (n + 1) : MathF.Pow(n + 1, uctDenominatorPower);
      float u = cpuct * p * (ParamsSelect.UCTParentMultiplier(N, uctNumeratorPower) / denominator);
      return q + u;
    }

    // TODO: undo the hardcoding here
    const float CPUCT = 2.147f;
    const float CPUCT_BASE = 18368;
    const float CPUCT_FACTOR = 2.815f;


    static float CalcCPUCT(int n)
    {
      float CPUCT_EXTRA = (CPUCT_FACTOR == 0) ? 0 : CPUCT_FACTOR * FastLog.Ln((n + CPUCT_BASE + 1.0f) / CPUCT_BASE); // ?? should parentN be min 1.0f as above
      float thisCPUCT = CPUCT + CPUCT_EXTRA;
      return thisCPUCT;
    }

  }

}


