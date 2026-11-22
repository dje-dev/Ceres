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

using Ceres.Chess.GameEngines;

#endregion

namespace Ceres.Features.GameEngines;

/// <summary>
/// Definition of a Stockfish engine, configured the way GameEngineStockfish configures one:
/// the executable resolved automatically, Threads/Hash/SyzygyPath set, and UCI_ShowWDL enabled
/// so that Stockfish reports its own win/draw/loss alongside the centipawn score. Callers get
/// named settings instead of hand-assembled setoption lines.
///
/// Nothing is resolved or launched until CreateEngine is called: the executable search (which
/// throws, listing everything tried, when Stockfish cannot be found) and the Ceres.json lookup
/// for the tablebase directory both happen there. A definition is therefore cheap to construct,
/// copy and serialize, which is what a tournament driver expects of one.
/// </summary>
[Serializable]
public class GameEngineDefStockfish : GameEngineDef
{
  /// <summary>Path of the Stockfish executable, or null to resolve it when the engine is created.</summary>
  public readonly string ExePath;

  /// <summary>Threads per engine (one is right when many engines run in parallel).</summary>
  public readonly int? NumThreads;

  /// <summary>Transposition table size in megabytes.</summary>
  public readonly int? HashSizeMB;

  /// <summary>Syzygy tablebase path, or null to take the Ceres.json default.</summary>
  public readonly string SyzygyPath;

  /// <summary>If UCI_ShowWDL is enabled, making Stockfish report its own win/draw/loss.</summary>
  public readonly bool ShowWDL;

  /// <summary>If ucinewgame is sent before each position.</summary>
  public readonly bool ResetGameBetweenMoves;

  /// <summary>Additional setoption command lines applied at engine startup.</summary>
  public readonly List<string> ExtraUCIOptions;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="id">identifying name of the engine (also used as the UCI spec name)</param>
  /// <param name="exePath">path to executable, or null to resolve automatically</param>
  /// <param name="numThreads">threads (default 1)</param>
  /// <param name="hashSizeMB">hash size in megabytes (default 256)</param>
  /// <param name="syzygyPath">tablebase path, or null to take the Ceres.json default</param>
  /// <param name="showWDL">if UCI_ShowWDL should be enabled (default true)</param>
  /// <param name="resetGameBetweenMoves">if ucinewgame should precede each position</param>
  /// <param name="extraUCIOptions">additional setoption command lines</param>
  /// <param name="processorGroupID"></param>
  public GameEngineDefStockfish(string id = "SF",
                                string exePath = null,
                                int? numThreads = GameEngineStockfish.DEFAULT_NUM_THREADS,
                                int? hashSizeMB = GameEngineStockfish.DEFAULT_HASH_MB,
                                string syzygyPath = null,
                                bool showWDL = true,
                                bool resetGameBetweenMoves = false,
                                List<string> extraUCIOptions = null,
                                int processorGroupID = 0)
    : base(id)
  {
    ExePath = exePath;
    NumThreads = numThreads;
    HashSizeMB = hashSizeMB;
    SyzygyPath = syzygyPath;
    ShowWDL = showWDL;
    ResetGameBetweenMoves = resetGameBetweenMoves;
    ExtraUCIOptions = extraUCIOptions;

    ProcessorGroupID = processorGroupID;
  }


  /// <summary>
  /// Not supported: like any UCI engine, Stockfish is driven by its own time management
  /// and has no notion of a node budget spanning a whole game.
  /// </summary>
  public override bool SupportsNodesPerGameMode => false;


  /// <summary>
  /// Returns the UCI specification these settings describe, resolving the executable path and
  /// (if not given) the tablebase directory. Exposed because GameEngineStockfish is a factory
  /// over GameEngineUCISpec rather than a GameEngine subclass, so the spec is the form every
  /// other part of Ceres understands.
  /// </summary>
  /// <returns></returns>
  public GameEngineUCISpec CreateSpec()
    => GameEngineStockfish.Spec(ID, ExePath, NumThreads, HashSizeMB, SyzygyPath,
                                ShowWDL, ResetGameBetweenMoves, ExtraUCIOptions, ProcessorGroupID);


  /// <summary>
  /// Implementation of virtual method to create underlying engine.
  /// </summary>
  /// <returns></returns>
  public override GameEngine CreateEngine() => CreateSpec().CreateEngine();


  /// <summary>
  /// Returns string description.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    return $"<GameEngineDefStockfish {ID} threads={NumThreads} hash={HashSizeMB}mb "
         + $"wdl={ShowWDL} {ExePath ?? "(exe resolved at creation)"}>";
  }
}
