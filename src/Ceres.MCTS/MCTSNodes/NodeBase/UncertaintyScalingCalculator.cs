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

using Ceres.Base.Math;

#endregion

namespace Ceres.MCTS.Managers.Uncertainty
{
    /// <summary>
    /// Manages calculation of possible scaling factor to be applied to
    /// CPUCT based on uncertain of node relative to parent.
    /// </summary>
    public static class UncertaintyScalingCalculator
    {
      /// <summary>
      /// Multiplier applied to difference in MAD between parent and child
      /// which controsl magnitude of exploration bonus/penalty.
      /// </summary>
      const float UNCERTAINTY_DIFF_MULTIPLIER = 2f;

      /// <summary>
      /// Maximum allowed deviation of bonus multiplier from 1.0
      /// (to enhance robusness in the face of possibly noisy outliers).
      /// </summary>
      const float UNCERTAINTY_MAX_DEVIATION = 0.20f;

      public static float ExplorationMultiplier(float childMAD, float parentMAD)
      {
        // The uncertainty scaling is a number centered at 1 which is
        // higher for children with more emprical volatility than the parent and
        // lower for children with less volatility.
        float explorationScaling = 1 + UNCERTAINTY_DIFF_MULTIPLIER * (childMAD - parentMAD);
        return StatUtils.Bounded(explorationScaling, 1.0f - UNCERTAINTY_MAX_DEVIATION, 1.0f + UNCERTAINTY_MAX_DEVIATION);
      }

    }
  }
