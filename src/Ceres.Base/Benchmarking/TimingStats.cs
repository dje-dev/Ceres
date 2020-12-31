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

namespace Ceres.Base.Benchmarking
{
  /// <summary>
  /// Set of statistics maintained by a TimingContext track time and memory usage.
  /// </summary>
  [Serializable]
  public class TimingStats
  {
    /// <summary>
    /// Number of elapsed seconds
    /// </summary>
    public double ElapsedTimeSecs;

    /// <summary>
    /// Number of elapsed clock ticks
    /// </summary>
    public long   ElapsedTimeTicks;

    /// <summary>
    /// Number of seconds of CPU time consumed
    /// </summary>
    public double CPUTimeSecs;

    /// <summary>
    /// Difference in .NET memory usage between end and start of
    /// </summary>
    public long   MemUsedBytes;

    public override string ToString()
    {
      return $"<TimingStats {ElapsedTimeSecs,6:F3} secs, CPU={CPUTimeSecs,6:F3}, MBytes={MemUsedBytes / 1_000_000,6:F3}>";
    }
  }
}



