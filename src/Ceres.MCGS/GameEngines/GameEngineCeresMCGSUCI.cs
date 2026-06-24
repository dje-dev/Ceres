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
using System.Reflection;
using System.IO;
using System.Collections.Generic;

using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;

using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Specifications;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.UCI;


#endregion

namespace Ceres.MCGS.GameEngines;

/// <summary>
/// Subclass of GameEngine for Ceres chess engine accessed via UCI protocol.
/// 
/// Typically it will be more flexible and performant to 
/// instead use GameEngineCeresInProcess, but nevertheless this class
/// can be helpful for testing purposes or possibly eventually for distributed engines.
/// </summary>
public class GameEngineCeresMCGSUCI : GameEngineUCI
{
  /// <summary>
  /// Constructor. 
  /// </summary>
  /// <param name="id">descriptive identifier</param>
  /// <param name="evaluatorDef">specification of the neural network to be used</param>
  /// <param name="forceDisableSmartPruning"></param>
  /// <param name="transferCeresParams">if true, the supplied searchParams/selectParams are serialized
  /// to a temporary file and the external engine is instructed (via load-params) to use them for all
  /// of its searches, so the separate process runs with the same parameters as an in-process engine</param>
  /// <param name="searchParams">search parameters to transfer when transferCeresParams is true</param>
  /// <param name="selectParams">select parameters to transfer when transferCeresParams is true</param>
  /// <param name="uciSetOptionCommands"></param>
  /// <param name="callback"></param>
  /// <param name="overrideEXE">optionally the full name of the executable file(otherwise looks for Ceres executable in working directory)</param>
  public GameEngineCeresMCGSUCI(string id,
                                NNEvaluatorDef evaluatorDef,
                                bool forceDisableSmartPruning = false,
                                bool transferCeresParams = false,
                                ParamsSearch searchParams = null,
                                ParamsSelect selectParams = null,
                                List<string> uciSetOptionCommands = null,
                                ProgressCallback callback = null,
                                string overrideEXE = null)
    : base(id, GetExecutableFN(overrideEXE), null, null, null,
           forceDisableSmartPruning ? AddedDisableSetSmartPruning(uciSetOptionCommands) : uciSetOptionCommands,
           callback, false, ExtraArgsForEvaluator(evaluatorDef))
  {
    if (overrideEXE != null)
    {
      overrideEXE = overrideEXE.Replace(".exe", "");
    }
    if (evaluatorDef == null)
    {
      throw new ArgumentNullException(nameof(evaluatorDef));
    }

    if (evaluatorDef.DeviceCombo == NNEvaluatorDeviceComboType.Pooled)
    {
      throw new Exception("Evaluators for GameEngineDefCeresMCGSUCI should not be created as Pooled.");
    }

    // If requested, transfer the supplied in-process search parameters to the external engine so it
    // runs with identical settings. The base constructor has already started the engine process and
    // waited for it to become ready, and the engine processes UCI commands in order, so this
    // load-params is applied before any subsequent position/go commands are issued for this engine.
    if (transferCeresParams)
    {
      TransferParamsToEngine(searchParams ?? new ParamsSearch(), selectParams ?? new ParamsSelect());
    }
  }


  /// <summary>
  /// Temporary file holding the parameters transferred to the external engine (or null). Retained
  /// for the engine's lifetime and removed on Dispose.
  /// </summary>
  string transferredParamsFile;


  /// <summary>
  /// Serializes the specified parameters to a temporary file and instructs the external engine to
  /// load them (via the "load-params" UCI command), so that all of the engine's future searches use
  /// exactly these parameters.
  /// </summary>
  void TransferParamsToEngine(ParamsSearch searchParams, ParamsSelect selectParams)
  {
    string paramsFile = Path.Combine(Path.GetTempPath(), $"ceres_transfer_params_{Guid.NewGuid():N}.ceres.params");
    ParamsFileSerializer.Save(paramsFile, searchParams, selectParams);
    transferredParamsFile = paramsFile;

    UCIRunner.SendCommand("load-params " + paramsFile);
  }


  /// <summary>
  /// If the NodesPerGame time control mode is supported.
  /// </summary>
  public override bool SupportsNodesPerGameMode => true;


  /// <summary>
  /// Optional name of program used as launcher for executable.
  /// </summary>
  public override string OverrideLaunchExecutable => SoftwareManager.IsLinux ? "dotnet" : null;

  public override bool SupportsBestValueMoveMode => true;

  public override bool SupportsBestActionMoveMode => true;


  #region Internal helper methods

  private static List<string> AddedDisableSetSmartPruning(List<string> options)
  {
    if (options == null)
    {
      options = new List<string>();
      options.Add("setoption name smartpruningfactor value 0");
    }
    return options;
  }

  /// <summary>
  /// Returns the network and device arguments string matching specified NNEvaluatorDef.
  /// </summary>
  /// <param name="evaluatorDef"></param>
  /// <returns></returns>
  static string ExtraArgsForEvaluator(NNEvaluatorDef evaluatorDef)
  {
    string netSpecString = NNNetSpecificationString.ToSpecificationString(evaluatorDef.NetCombo, evaluatorDef.Nets);
    if (evaluatorDef.Description.StartsWith("~", StringComparison.OrdinalIgnoreCase))
    {
      netSpecString = "Network=" + evaluatorDef.Description.Split(" ")[0]; // split off only network parg
    }
    else
    {
      if (!netSpecString.Contains("Network="))
      {
        throw new Exception("Unsupported network specification");
      }
    }

    string deviceSpecString = NNDevicesSpecificationString.ToSpecificationString(evaluatorDef.DeviceCombo, evaluatorDef.Devices);
    if (!deviceSpecString.Contains("Device="))
    {
      throw new Exception("Unsupported device specification");
    }

    return "\"" + "UCI " + netSpecString + " " + deviceSpecString + "\"";
  }


  /// <summary>
  /// Determines (and validates existence) the name of the executable to be used.
  /// </summary>
  /// <param name="overrideEXE"></param>
  /// <returns></returns>
  static string GetExecutableFN(string overrideEXE)
  {
    string executableFN = overrideEXE ?? Assembly.GetExecutingAssembly().FullName;
    if (!File.Exists(executableFN))
    {
      throw new ArgumentException(nameof(overrideEXE), $"specified executable not found: {executableFN}");
    }
    return executableFN;
  }

  #endregion


  /// <summary>
  /// Requests a diagnostics dump from the external Ceres engine over UCI (via the "dump-info-block"
  /// command), captures the marker-delimited output, strips the markers, and writes the result to the
  /// specified writer (prefixed with a yellow header identifying this engine). This mirrors, for an
  /// external UCI Ceres engine, the in-process Ctrl-D dump. Safe to call from another thread; blocks
  /// up to timeoutMs waiting for the engine's response, so callers driving this from an interactive
  /// key handler should invoke it on a background task.
  /// </summary>
  /// <param name="description">label identifying the requester (shown in the header, e.g. "Ctrl-D DUMP-INFO")</param>
  /// <param name="writer">destination for the captured dump</param>
  /// <param name="timeoutMs">maximum time to wait for the engine to emit the block</param>
  public void DumpInfoBlockToConsole(string description, TextWriter writer, int timeoutMs = 8000)
  {
    IReadOnlyList<string> lines = UCIRunner.CaptureMarkedBlock("dump-info-block",
                                                               DiagnosticsBlock.BEGIN_MARKER,
                                                               DiagnosticsBlock.END_MARKER,
                                                               timeoutMs);
    lock (DiagnosticsBlock.ConsoleLock)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"===== {description}  engine={ID} (UCI) =====");
      if (lines == null)
      {
        writer.WriteLine($"(dump-info-block timed out after {timeoutMs} ms or engine produced no block)");
      }
      else
      {
        foreach (string line in lines)
        {
          writer.WriteLine(line);
        }
      }
    }
  }


  public override void Dispose()
  {
    UCIRunner.Shutdown();

    // Remove the transferred-params temporary file (if any); the engine has long since loaded it.
    if (transferredParamsFile != null)
    {
      try
      {
        if (File.Exists(transferredParamsFile))
        {
          File.Delete(transferredParamsFile);
        }
      }
      catch (Exception)
      {
        // Best-effort cleanup; ignore failures.
      }
      transferredParamsFile = null;
    }
  }


  /// <summary>
  /// Returns string summary of object.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    return $"<GameEngineCeresMCGSUCI>";
  }

}
