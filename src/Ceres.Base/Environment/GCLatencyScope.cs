#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using Directives

using System;
using System.Runtime;

#endregion

namespace Ceres.Base.Environment;

/// <summary>
/// Scope for temporarily setting the garbage collector latency mode.
/// </summary>
public readonly struct GCLatencyScope : IDisposable
{
  /// <summary>
  /// Latency mode which was active prior to this scope being created.
  /// </summary>
  private readonly GCLatencyMode priorLatencyMode;

  /// <summary>
  /// If the scope is active (not default constructed).
  /// </summary>
  private readonly bool isActive;


  /// <summary>
  /// Constructor which sets the GC latency mode for the current process, 
  /// returning a scope which will restore the prior mode on disposal.
  /// </summary>
  /// <param name="mode"></param>
  public GCLatencyScope(GCLatencyMode mode)
  {
    priorLatencyMode = GCSettings.LatencyMode;     // capture current (process-wide)
    GCSettings.LatencyMode = mode;      // set requested mode
    isActive = true;
  }


  /// <summary>
  /// Releases resources used by the current instance and 
  /// restores the previous garbage collection latency mode.
  /// </summary>
  public void Dispose()
  {
    if (isActive)
    {
      GCSettings.LatencyMode = priorLatencyMode;
    }
  }


  #region Convenience factories

  /// <summary>
  /// Creates a scope that sets the garbage collector to low latency mode for the duration of the scope.
  /// </summary>
  public static GCLatencyScope LowLatency() => new(GCLatencyMode.LowLatency);


  /// <summary>
  /// Creates a scope that sets the garbage collector to sustained low latency mode for the duration of the scope.
  /// </summary>
  /// <returns></returns>
  public static GCLatencyScope SustainedLowLatency() => new(GCLatencyMode.SustainedLowLatency);

  #endregion
}
