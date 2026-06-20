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

using System.Diagnostics;
using System.Threading;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Accumulates the total wall-clock time during which at least one of a set of
  /// evaluators is executing inside the (C++) backend interop boundary ("backend time").
  ///
  /// A single instance is shared by the (up to two) evaluators of an NNEvaluatorSet
  /// so that overlapping backend calls collapse into a single busy interval (i.e.
  /// the value computed is the UNION of the evaluators' busy intervals, not their sum).
  ///
  /// This is computed cheaply via a reference count of how many evaluators are
  /// currently in the backend, without storing per-batch intervals: a busy period
  /// opens when the count rises from 0 and closes when it returns to 0.
  ///
  /// The complement (total search time minus busy time) is the time during which
  /// NEITHER evaluator is in the backend (i.e. the GPU is idle / pure C# overhead).
  /// </summary>
  public sealed class BackendTimeTracker
  {
    readonly Lock lockObj = new();

    /// <summary>
    /// Number of evaluators currently executing inside the backend.
    /// </summary>
    int busyCount;

    /// <summary>
    /// Timestamp (Stopwatch ticks) at which the current busy period began
    /// (valid only while busyCount > 0).
    /// </summary>
    long periodStartTimestamp;

    /// <summary>
    /// Cumulative backend-busy time (Stopwatch ticks) of all completed busy periods.
    /// </summary>
    long accumulatedBusyTicks;

    /// <summary>
    /// Whether EnterBackend has been called at least once since the last Reset.
    /// Used to distinguish "supported and measured" from "unsupported backend".
    /// </summary>
    bool everUsed;


    /// <summary>
    /// Resets all accumulated state. Call at the start of each search.
    /// </summary>
    public void Reset()
    {
      lock (lockObj)
      {
        busyCount = 0;
        periodStartTimestamp = 0;
        accumulatedBusyTicks = 0;
        everUsed = false;
      }
    }


    /// <summary>
    /// Records that an evaluator has entered the backend interop boundary.
    /// </summary>
    public void EnterBackend()
    {
      lock (lockObj)
      {
        if (busyCount == 0)
        {
          periodStartTimestamp = Stopwatch.GetTimestamp();
        }
        busyCount++;
        everUsed = true;
      }
    }


    /// <summary>
    /// Records that an evaluator has exited the backend interop boundary.
    /// </summary>
    public void ExitBackend()
    {
      lock (lockObj)
      {
        if (--busyCount == 0)
        {
          accumulatedBusyTicks += Stopwatch.GetTimestamp() - periodStartTimestamp;
        }
      }
    }


    /// <summary>
    /// Cumulative backend-busy time in seconds (time during which at least one
    /// evaluator was inside the backend).
    /// </summary>
    public double BusySeconds
    {
      get
      {
        lock (lockObj)
        {
          return accumulatedBusyTicks / (double)Stopwatch.Frequency;
        }
      }
    }


    /// <summary>
    /// Whether the tracker has recorded any backend activity since the last Reset
    /// (false indicates the backend does not support this instrumentation).
    /// </summary>
    public bool EverUsed
    {
      get
      {
        lock (lockObj)
        {
          return everUsed;
        }
      }
    }
  }
}
