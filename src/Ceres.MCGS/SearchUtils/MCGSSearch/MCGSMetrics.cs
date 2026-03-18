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
using System.Diagnostics;
using System.Diagnostics.Metrics;

#endregion

namespace Ceres.MCGS.Search;

/// <summary>
/// Provides metrics for monitoring and analyzing the behavior of the MCGS system.
/// </summary>
public static class MCGSMetrics
{
  public static readonly Meter Meter = new("Ceres.MCGS", "1.0");

  public static readonly Counter<int> PathTerminationResultHits =
      Meter.CreateCounter<int>("MCGSPath TerminationResult", "terminations", "Number of path terminations by type (visits)");

  public static readonly Counter<int> CacheMisses =
      Meter.CreateCounter<int>("cache_misses", "misses", "Number of failed cache lookups");

  public static readonly Counter<int> TerminationCounter =
     Meter.CreateCounter<int>("termination_reasons_total", "items",  "Count of termination reasons");

}

internal static class MetricTagHelper
{
  /// <summary>
  /// Returns a list of TagList objects for each value of the specified enum type.
  /// </summary>
  /// <typeparam name="TEnum"></typeparam>
  /// <param name="keyName"></param>
  /// <returns></returns>
  internal static TagList[] PrecomputeEnumTagLists<TEnum>(string keyName) where TEnum : struct, Enum
  {
    TEnum[] values = (TEnum[])Enum.GetValues(typeof(TEnum));
    TagList[] tagLists = new TagList[values.Length];

    foreach (TEnum value in values)
    {
      tagLists[Convert.ToInt32(value)] = new TagList { { keyName, value.ToString() } };
    }

    return tagLists;
  }
}
