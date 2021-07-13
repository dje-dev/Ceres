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
using System.Reflection.Metadata.Ecma335;

#endregion

namespace Ceres.Chess.GameEngines
{
  /// <summary>
  /// Definition of a game engine runing as an external UCI process.
  /// </summary>
  [Serializable]
  public record GameEngineUCISpec
  {
    /// <summary>
    /// Name of engine.
    /// </summary>
    public readonly string Name;

    /// <summary>
    /// Path to executable for engine.
    /// </summary>
    public readonly string EXEPath;

    /// <summary>
    /// Number of threads to allocate to search engine.
    /// It is assumed the engine has a corresponding settable option named "Threads" defined.
    /// </summary>
    public readonly int? NumThreads;

    /// <summary>
    /// Size of transposition hash table in megabytes.
    /// It is assumed the engine has a corresponding settable option named "Hash" defined.
    /// </summary>
    public readonly int? HashSizeMB;

    /// <summary>
    /// Path to Szygygy tablebases (or null if no tablebases).
    /// It is assumed the engine has a corresponding settable option named "SyzygyPath" defined.
    /// </summary>
    public readonly string SyzygyPath;

    /// <summary>
    /// Optional set of commands sent to engine in intialization.
    /// </summary>
    public readonly List<string> UCISetOptionCommands;

    /// <summary>
    /// Optional callback that is called to report on engine progress.
    /// </summary>
    public GameEngine.ProgressCallback Callback;

    /// <summary>
    /// If a full reset (flush cache) should be performed before each move.
    /// </summary>
    public bool ResetGameBetweenMoves;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="name"></param>
    /// <param name="exePath"></param>
    /// <param name="numThreads"></param>
    /// <param name="hashSizeMB"></param>
    /// <param name="syzygyPath"></param>
    /// <param name="callback"></param>
    /// <param name="resetGameBetweenMoves"></param>
    /// <param name="uciSetOptionCommands"></param>
    public GameEngineUCISpec(string name, string exePath = null,
                             int? numThreads = null, int? hashSizeMB = null,
                             string syzygyPath = null,
                             GameEngine.ProgressCallback callback = null,
                             bool resetGameBetweenMoves = false,
                             List<string> uciSetOptionCommands = null)
    {
      if (exePath is null) throw new ArgumentNullException(nameof(exePath));

      Name = name;
      EXEPath = exePath;
      NumThreads = numThreads;
      HashSizeMB = hashSizeMB;
      SyzygyPath = syzygyPath;
      UCISetOptionCommands = uciSetOptionCommands;
      Callback = callback;
      ResetGameBetweenMoves = resetGameBetweenMoves;
    }


    /// <summary>
    /// Returns new engine created to match this specification.
    /// </summary>
    /// <returns></returns>
    public GameEngineUCI CreateEngine()
    {
      return new GameEngineUCI(Name, EXEPath,
                               NumThreads,
                               HashSizeMB,
                               SyzygyPath,
                               UCISetOptionCommands,
                               Callback, ResetGameBetweenMoves);
    }


    /// <summary>
    /// Returns string summary.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<GameEngineUCISpec {Name} using {EXEPath}"
           + (NumThreads.HasValue ? $" NumThreads={NumThreads}" : "")
           + (HashSizeMB.HasValue ? $" HashSizeMB={HashSizeMB}" : "")
           + (SyzygyPath != null ? $" SyzygyPath={SyzygyPath}" : "")
           + (ResetGameBetweenMoves ? " ResetGameBetweenMoves" : "")
           + (UCISetOptionCommands != null ? $" (and {UCISetOptionCommands.Count} additional options)" : "")
           + $" >";
    }

  }
}
