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
using System.Threading;
using System.Threading.Tasks;

#endregion

namespace Ceres.MCGS.Utils;

/// <summary>
/// Tracks live statistics and periodically outputs its state.
/// Thread-safe for count updates.
/// </summary>
public class LiveStats
{
  public const bool ENABLED = true;

  public const int INTERVAL_MULTIPLIER = 100;

  public readonly string ID;
  public readonly float UpdateIntervalSecs;

  public long Count1;
  public long Count2;

  // Store last output values to suppress duplicate output
  private long lastDumpCount1 = long.MinValue;
  private long lastDumpCount2 = long.MinValue;



  /// <summary>
  /// Initializes a new instance and starts periodic output if enabled.
  /// </summary>
  public LiveStats(string id, float updateIntervalSecs)
  {
    ID = id;
    UpdateIntervalSecs = updateIntervalSecs;

    if (ENABLED)
    {
      Task.Run(() =>
      {
        while (true)
        {
          Thread.Sleep((int)(UpdateIntervalSecs * INTERVAL_MULTIPLIER * 1000f));
          DumpState();
        }
      });
    }
  }


  /// <summary>
  /// Atomically adds to the tracked counts.
  /// </summary>
  public void Add(long count1, long count2)
  {
    if (ENABLED)
    {
      Interlocked.Add(ref Count1, count1);
      Interlocked.Add(ref Count2, count2);
    }
  }

  /// <summary>
  /// Resets both counts to zero.
  /// </summary>
  public void Reset()
  {
    Count1 = 0;
    Count2 = 0;
  }

  /// <summary>
  /// Outputs the current state if values have changed since last output.
  /// </summary>
  public void DumpState()
  {
    double percent = (Count1 != 0) ? 100.0 * (double)Count2 / Count1 : 0.0;

    // Only output if values have changed since last dump
    if (Count1 == lastDumpCount1 && Count2 == lastDumpCount2)
    {
      return;
    }

    Console.WriteLine($"{System.Environment.NewLine}(LiveStat) {ID} --> {Count1} {Count2} {percent,5:F2}%");
    lastDumpCount1 = Count1;
    lastDumpCount2 = Count2;
  }
}
