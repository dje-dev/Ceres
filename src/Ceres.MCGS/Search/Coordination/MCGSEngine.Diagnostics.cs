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
using Ceres.MCGS.Search.Paths;

#endregion

namespace Ceres.MCGS.Search.Coordination;

/// <summary>
/// Diagnostics support methods for MCGSEngine.
/// </summary>
public partial class MCGSEngine
{
  /// <summary>
  /// Dumps the combined path length distribution from all active iterators.
  /// If dual iterators are active, sums the distributions from both iterators.
  /// </summary>
  public void DumpDistributionPathLengths()
  {
    if (iterator0 == null)
    {
      Console.WriteLine("No iterators available.");
      return;
    }

    long[] combinedDistribution = new long[256];
    long totalPaths = 0;
    int maxDepth = 0;
    long sumPathLengths = 0;

    // Add iterator0 distribution
    Array.Copy(iterator0.PathsSet.PathLengthDistribution, combinedDistribution, 256);
    totalPaths = iterator0.PathsSet.CountNonAbortedPathVisits;
    maxDepth = iterator0.PathsSet.MaxNonAbortedPathDepth;
    sumPathLengths = iterator0.PathsSet.SumNonAbortedPathVisits;

    // Add iterator1 distribution if it exists
    if (iterator1 != null)
    {
      for (int i = 0; i < 256; i++)
      {
        combinedDistribution[i] += iterator1.PathsSet.PathLengthDistribution[i];
      }
      totalPaths += iterator1.PathsSet.CountNonAbortedPathVisits;
      maxDepth = Math.Max(maxDepth, iterator1.PathsSet.MaxNonAbortedPathDepth);
      sumPathLengths += iterator1.PathsSet.SumNonAbortedPathVisits;
    }

    double avgDepth = totalPaths > 0 ? sumPathLengths / (double)totalPaths : 0;
    MCGSPathsSet.DumpDistribution(combinedDistribution, totalPaths, maxDepth, avgDepth);
  }
}
