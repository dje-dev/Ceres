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

using Ceres.Chess;
using System;


#endregion

namespace Ceres.MCTS.Search.IteratedMCTS
{
  /// <summary>
  /// Defines a single iteration within an iterated MCTS.
  /// </summary>
  [Serializable]
  public class IteratedMCTSStepDef
  {
    /// <summary>
    /// Fraction of the search to be allocated to this step.
    /// </summary>
    public readonly float SearchFraction;

    /// <summary>
    /// Only nodes have N greater than the root N times this fraction 
    /// will be eligible for policy blending
    /// </summary>
    public readonly float NodeNFractionCutoff;

    /// <summary>
    /// Weight to be placed on the adjusted policy after iteration.
    /// </summary>
    public readonly float WeightFractionNewPolicy;

    /// <summary>
    /// Search limit to be used for this step.
    /// </summary>
    public SearchLimit Limit { get; internal set; }



    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="searchFraction"></param>
    /// <param name="nodeNFractionCutoff"></param>
    /// <param name="weightFractionNewPolicy"></param>
    public IteratedMCTSStepDef(float searchFraction, float nodeNFractionCutoff, float weightFractionNewPolicy)
    {
      SearchFraction = searchFraction;
      NodeNFractionCutoff = nodeNFractionCutoff;
      WeightFractionNewPolicy = weightFractionNewPolicy;
    }


    /// <summary>
    /// Sets Limit based on the specified overall limit and target fraction for this step.
    /// </summary>
    /// <param name="overallLimit"></param>
    public void SetForOverallLimit(SearchLimit overallLimit)
    {
      Limit = overallLimit * SearchFraction;
    }
  }
}
