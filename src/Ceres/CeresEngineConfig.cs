#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

namespace Ceres;

/// <summary>
/// Identifies the search engine backend to use.
/// </summary>
public enum CeresEngineVersion
{
  /// <summary>
  /// Legacy MCTS engine (v1).
  /// </summary>
  V1MCTS,

  /// <summary>
  /// New MCGS engine (v2, default).
  /// </summary>
  V2MCGS
}


/// <summary>
/// Global configuration for which search engine version is active.
/// Defaults to V2MCGS; can be overridden with /v1 or ENGINE=V1 on the command line.
/// </summary>
public static class CeresEngineConfig
{
  /// <summary>
  /// The currently active engine version.
  /// </summary>
  public static CeresEngineVersion ActiveEngine { get; set; } = CeresEngineVersion.V2MCGS;

  /// <summary>
  /// Returns true if the active engine is the MCGS (v2) engine.
  /// </summary>
  public static bool IsMCGS => ActiveEngine == CeresEngineVersion.V2MCGS;

  /// <summary>
  /// Returns true if the active engine is the legacy MCTS (v1) engine.
  /// </summary>
  public static bool IsMCTS => ActiveEngine == CeresEngineVersion.V1MCTS;
}
