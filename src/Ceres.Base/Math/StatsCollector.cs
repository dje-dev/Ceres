#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directive

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

#endregion

namespace Ceres.Base.Math
{
  /// <summary>
  /// Accepts online values received for one or more types of values, 
  /// and upon request dumps their statistics to the console. 
  /// </summary>
  public class StatsCollector
  {
    /// <summary>
    /// Represents the stats for a single value.
    /// </summary>
    public class ValueStats
    {
      public int CountAdded { get; set; }
      public float Min { get; set; } = float.MaxValue;
      public float Max { get; set; } = float.MinValue;
      public float Total { get; set; }
      public float SumOfSquares { get; set; }

      public void AddValue(float value)
      {
        CountAdded++;
        Min = System.Math.Min(Min, value);
        Max = System.Math.Max(Max, value);
        Total += value;
        SumOfSquares += value * value;
      }

      public float Average => Total / CountAdded;
      public float StandardDeviation => (float)System.Math.Sqrt((SumOfSquares / CountAdded) - (Average * Average));
    }

    public readonly ConcurrentDictionary<string, ValueStats> Stats = new();


    /// <summary>
    /// Updates the stats for a given value.
    /// </summary>
    /// <param name="valueID"></param>
    /// <param name="value"></param>
    public void AddValue(string valueID, float value)
    {
      if (!Stats.ContainsKey(valueID))
      {
        Stats[valueID] = new ValueStats();
      }
      Stats[valueID].AddValue(value);
    }


    /// <summary>
    /// Dumps all stats to the console.
    /// </summary>
    public void DumpAllValueStats()
    {
      // Get length of longest key in Stats 
      int maxKeyLen = Stats.Keys.Max(k => k.Length);

      Console.WriteLine();
      foreach (var pair in Stats)
      {
        ValueStats stats = pair.Value;
        Console.WriteLine($"{pair.Key.PadRight(maxKeyLen)}  {stats.CountAdded,10:F2}  {stats.Min,10:F2}  {stats.Average,10:F2} {stats.StandardDeviation,10:F2}  {stats.Max,10:F2} ");
      }
    }


    /// <summary>
    /// Resets all stats back uninitialized.
    /// </summary>
    public void ResetStats() => Stats.Clear();
  }

}


