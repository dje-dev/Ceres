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

using Ceres.Base.DataType;
using Ceres.MCTS.LeafExpansion;

#endregion

namespace Ceres.MCTS.MTCSNodes
{
  /// <summary>
  /// Container for transient set of node statistics
  /// (gathered from children)
  /// </summary>
  internal class GatheredChildStats
  {
    const int ALIGNMENT = 64; // For AVX efficiency

    internal SpanAligned<float> N;
    internal SpanAligned<float> InFlight;
    internal SpanAligned<float> P;
    internal SpanAligned<float> W;
    internal SpanAligned<float> U;
    internal SpanAligned<float> A;

    internal GatheredChildStats()
    {
      N = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
      InFlight = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
      P = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
      W = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
      U = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
      A = new SpanAligned<float>(MCTSScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
    }
  }
}
