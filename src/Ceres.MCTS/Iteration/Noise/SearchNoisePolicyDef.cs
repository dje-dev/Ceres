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

using System;

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Dirichlet noise applied to root moves (as for example used by Alpha Zero).
  /// 
  /// </summary>
  [Serializable]
  public class SearchNoisePolicyDef
  {
    public float DirichletAlpha = float.NaN; // 0.3
    public float DirichletFraction = float.NaN; // 0.25

    public static SearchNoisePolicyDef Dirichlet(float alpha, float fraction)
    {
      return new SearchNoisePolicyDef()
      {
        DirichletAlpha = alpha,
        DirichletFraction = fraction
      };

    }
  }
}

