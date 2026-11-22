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
using System.Collections.Generic;

using Ceres.Chess.Probing;

#endregion

namespace Ceres.MCGS.Search.ProbeGraft;

/// <summary>
/// Process-wide registry of probe source factories, keyed by string ID
/// (following the ParamsSearchExecutionModifier ID-registry pattern).
///
/// ParamsSearch is a serializable record and therefore must not hold an
/// IProbeSource instance directly; instead ParamsProbeGraft.ProbeSourceID
/// names a factory registered here, which MCGSManager resolves once per search when the
/// ProbeGraftCoordinator is constructed. An unknown ID with Mode != Disabled fails fast
/// (exception at search start), never silently.
///
/// A factory may return a fresh instance per call or one shared instance (a pooled engine
/// source typically exposes a shared factory, so its workers and transposition table survive
/// across sessions); the coordinator never disposes the resolved source, it only brackets its
/// use with BeginSession/EndSession.
/// </summary>
public static class ProbeSourceRegistry
{
  [NonSerialized]
  static readonly Dictionary<string, Func<IProbeSource>> dictFactories = new();

  static readonly object lockObj = new();

  /// <summary>
  /// Globally registers a probe source factory under the specified ID.
  /// </summary>
  /// <param name="id"></param>
  /// <param name="factory"></param>
  public static void Register(string id, Func<IProbeSource> factory)
  {
    if (string.IsNullOrEmpty(id))
    {
      throw new ArgumentException("Probe source ID must be non-empty", nameof(id));
    }
    if (factory == null)
    {
      throw new ArgumentNullException(nameof(factory));
    }

    lock (lockObj)
    {
      if (dictFactories.ContainsKey(id))
      {
        throw new Exception($"Already registered with ProbeSourceRegistry.Register: {id}");
      }

      dictFactories[id] = factory;
    }
  }


  /// <summary>
  /// Attempts to create (resolve) a probe source for the specified ID,
  /// returning false if no factory is registered under that ID.
  /// </summary>
  /// <param name="id"></param>
  /// <param name="source"></param>
  /// <returns></returns>
  public static bool TryCreate(string id, out IProbeSource source)
  {
    Func<IProbeSource> factory;
    lock (lockObj)
    {
      if (id == null || !dictFactories.TryGetValue(id, out factory))
      {
        source = null;
        return false;
      }
    }

    source = factory();
    return source != null;
  }
}
