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
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace Ceres.MCGS.Search.RPO;

public static class BayesianShrinkage
{
  /// <summary>
  /// Shrinks an empirical proportion <paramref name="p"/> toward a Bayesian
  /// posterior estimate that uses a Student-t prior with DOF = 4.
  ///
  ///     � <paramref name="priorP"/> � prior mean (your prior belief about p).  
  ///     � <paramref name="sd"/>     � sample standard deviation of the data feeding p.  
  ///     � <paramref name="n"/>      � number of independent samples that produced p.  
  ///     � <paramref name="scalingFactor"/> ? [0,1] � how aggressively to pull p toward
  ///       the posterior (1 = fully use the posterior, 0 = leave p unchanged).
  ///
  /// The function first forms a posterior mean under a Normal�Inverse-Gamma model
  /// whose predictive distribution is Student-t with DOF = 4.  In a
  /// Normal�Inverse-Gamma family the posterior mean is
  ///
  ///          ?? = (?? ?? + n p) / (?? + n) ,
  ///
  /// where ?? is the prior "effective sample size."  We tie ?? to the degrees of
  /// freedom (DOF = 4) so that the prior contributes the weight of four notional
  /// observations.  Finally, we shrink the current p toward ?? according to
  ///
  ///          p? = p + scalingFactor � (?? ? p).
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  public static float Shrink(float p, float priorP, float sd, int n, float scalingFactor)
  {
    Debug.Assert(scalingFactor >= 0f && scalingFactor <= 1f, "scalingFactor must lie in [0,1]");
    if (n <= 0 || scalingFactor == 0f) return p;             // nothing to update

    const float PRIOR_EFFECTIVE_SAMPLE_SIZE = 3f;   // prior degrees of freedom

    // Posterior mean under Normal�Inverse-Gamma (predictive Student-t with DOF = 4+n)
    float posteriorMean = (PRIOR_EFFECTIVE_SAMPLE_SIZE * priorP + n * p) / (PRIOR_EFFECTIVE_SAMPLE_SIZE + n);

    // Linear shrinkage of current p toward the posterior mean
    return p + scalingFactor * (posteriorMean - p);
  }
}
