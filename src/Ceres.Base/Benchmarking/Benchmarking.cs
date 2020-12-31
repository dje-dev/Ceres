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
using System.Threading;

#endregion

namespace Ceres.Base.Benchmarking
{
  /// <summary>
  /// Static helper methods related to benchmarking.
  /// </summary>
  public static class Benchmarking
  {
    /// <summary>
    /// Measures and outputs to console various statistics about the execution time (and memory usage)
    /// of a specified operation.
    /// </summary>
    /// <param name="operation"></param>
    /// <param name="description"></param>
    /// <param name="warmupIterations"></param>
    /// <returns></returns>
    public static float DumpOperationTimeAndMemoryStats(Action operation, string description, int warmupIterations = 50)
    {
      Stopwatch stopwatch = new Stopwatch();

      int count = 10;
      int total = 0;
      double secsElapsed = 0.0;

      // Possibly warm up (JIT tiered compilation may not kick in until 30 or more executions)
      for (int i = 0; i < warmupIterations; i++) operation();

      // Give some time for JIT (re)-compilation to possibly happen
      Thread.Sleep(100);

      long bytesAllocated = 0;

      while (secsElapsed < 0.5)
      {
        // Run 
        long startBytesAllocated = GC.GetAllocatedBytesForCurrentThread();
        stopwatch.Restart();
        for (int i = 0; i < count; i++) operation();
        stopwatch.Stop();

        // update stats
        bytesAllocated += GC.GetAllocatedBytesForCurrentThread() - startBytesAllocated;
        secsElapsed += stopwatch.Elapsed.TotalSeconds;
        total += count;

        // Exponentially increase iteration count
        count *= 2;
      }

      long bytesAllocPerOp = bytesAllocated / count;
      double secsPerOperation = secsElapsed / total;

      Console.Write($"{  (1.0f / secsPerOperation),12:N0} ops/second");
      Console.WriteLine($", {bytesAllocPerOp,9:N0} bytes alloc/op : {description}");

      return 1.0f / (float)secsPerOperation;
    }
  }
}
