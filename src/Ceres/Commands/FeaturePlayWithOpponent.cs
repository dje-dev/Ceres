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
using System.IO;
using System.Threading.Tasks;
using Ceres.Base.DataTypes;
using Ceres.Base.OperatingSystem;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.Specifications;
using Ceres.Chess.UserSettings;
using Ceres.Features.GameEngines;
using Ceres.Features.Players;

#endregion

namespace Ceres.Commands
{
  public abstract record FeaturePlayWithOpponent
  {
    public static readonly string[] FEATURE_PLAY_COMMON_ARGS
       = new[] { "NETWORK", "DEVICE", "LIMIT", "OPPONENT", 
                 "NETWORK-OPPONENT", "DEVICE-OPPONENT", "LIMIT-OPPONENT",
                 "PRUNING"};

    public string Opponent { protected set; get; } = null;

    public NNNetSpecificationString NetworkSpec { protected set; get; }
    public NNDevicesSpecificationString DeviceSpec { protected set; get; }

    public NNNetSpecificationString NetworkOpponentSpec { protected set; get; }
    public NNDevicesSpecificationString DeviceOpponentSpec { protected set; get; }

    public SearchLimit SearchLimit { protected set; get; } = SearchLimit.SecondsPerMove(30);
    public SearchLimit SearchLimitOpponent { protected set; get; } = SearchLimit.SecondsPerMove(30);

    public bool Pruning { protected set; get; } = true;


    /// <summary>
    /// Constructs player definitions based on parameters.
    /// </summary>
    /// <param name="opponentRequired"></param>
    /// <returns></returns>
    protected (EnginePlayerDef def1, EnginePlayerDef def2) GetEngineDefs(bool opponentRequired)
    {
      NNEvaluatorDef evaluatorDef = new NNEvaluatorDef(NetworkSpec.ComboType, NetworkSpec.NetDefs,
                                                       DeviceSpec.ComboType, DeviceSpec.Devices, null);

      // Check if different network and device speciifed for opponent
      NNEvaluatorDef evaluatorDefOpponent;
      if (NetworkOpponentSpec != null || DeviceOpponentSpec != null)
      {
        if (NetworkOpponentSpec == null || DeviceOpponentSpec == null)
          throw new Exception("Both network-opponent and device-opponent must be provided");

        evaluatorDefOpponent = new NNEvaluatorDef(NetworkOpponentSpec.ComboType, NetworkOpponentSpec.NetDefs,
                                                  DeviceOpponentSpec.ComboType, DeviceOpponentSpec.Devices, null);
      }
      else
      {
        // Share evaluators
        evaluatorDefOpponent = evaluatorDef;
      }

      GameEngineDefCeres defCeres1 = new GameEngineDefCeres("Ceres", evaluatorDef, null);
      defCeres1.SearchParams.FutilityPruningStopSearchEnabled = Pruning;
      EnginePlayerDef player1 = new EnginePlayerDef(defCeres1, SearchLimit);
      
      EnginePlayerDef player2 = null;

      if (Opponent != null && Opponent.ToUpper() == "LC0")
      {
        player2 = new EnginePlayerDef(new GameEngineDefLC0("LC0", evaluatorDefOpponent, !Pruning),
                                                           SearchLimitOpponent ?? SearchLimit);
      }
      else if (Opponent != null && Opponent.ToUpper() == "SF14DJE")
      {
        // TODO: Internal testing option, remove or clean up/document.
        const int SF_NUM_THREADS = 8;
        const int SF_HASH_SIZE_MB = 2048;
        string TB_PATH = CeresUserSettingsManager.Settings.TablebaseDirectory;
        string SF14_EXE = SoftwareManager.IsLinux ? @"/raid/dev/SF14/stockfish_14_linux_x64_avx2"
                                                  : @"\\synology\dev\chess\engines\stockfish_14_x64_avx2.exe";

        GameEngineDef engineDefStockfish14 = new GameEngineDefUCI("SF14", new GameEngineUCISpec("SF14", SF14_EXE, SF_NUM_THREADS,
                                                                  SF_HASH_SIZE_MB, TB_PATH, uciSetOptionCommands: null));
        player2 = new EnginePlayerDef(engineDefStockfish14, SearchLimitOpponent ?? SearchLimit);
      }
      else if (Opponent != null && Opponent.ToUpper() == "CERES"
           || (Opponent == null && opponentRequired))
      {
        GameEngineDefCeres engineCeres = new GameEngineDefCeres("Ceres2", evaluatorDefOpponent, null);
        engineCeres.SearchParams.FutilityPruningStopSearchEnabled = Pruning;
        player2 = new EnginePlayerDef(engineCeres, SearchLimitOpponent ?? SearchLimit);
      }
      else if (Opponent != null)
      {
        // Note that not possible to disable smart pruning
        player2 = BuildUCIOpponent();
      }

      return (player1, player2);
    }


    /// <summary>
    /// The speciifed opponent is an UCI engine; creates a properly configured engine definition.
    /// </summary>
    /// <returns></returns>
    private EnginePlayerDef BuildUCIOpponent()
    {
      EnginePlayerDef player1;
      string externalEnginePath;

      if (CeresUserSettingsManager.Settings.DirExternalEngines != null)
      {
        externalEnginePath = Path.Combine(CeresUserSettingsManager.Settings.DirExternalEngines, Opponent);
      }
      else
        externalEnginePath = Opponent;

      if (OperatingSystem.IsWindows() && !externalEnginePath.ToUpper().EndsWith(".EXE"))
        externalEnginePath += ".exe";

      FileInfo infoEngine = new FileInfo(externalEnginePath);
      if (!infoEngine.Exists) throw new Exception($"Requested opponent engine executable not found {externalEnginePath}");
      string name = infoEngine.Name.Split(".")[0];
      if (name.Length > 12) name = name.Substring(0, 12);
      SearchLimit opponentSearchLimit = SearchLimitOpponent ?? SearchLimit;
      GameEngineUCISpec uciSpec = new GameEngineUCISpec(name, Opponent);
      player1 = new EnginePlayerDef(new GameEngineDefUCI("UCI", uciSpec), opponentSearchLimit, name);
      return player1;
    }


    /// <summary>
    /// Parses base fields common to all sublcasses.
    /// </summary>
    /// <param name="args"></param>
    protected void ParseBaseFields(string args, bool convertGameLimitToMoveLimit)
    {
      KeyValueSetParsed keys = new KeyValueSetParsed(args, null);

      Opponent = keys.GetValue("Opponent");

      NetworkSpec = keys.GetValueOrDefaultMapped<NNNetSpecificationString>("Network", CeresUserSettingsManager.Settings.DefaultNetworkSpecString, true, spec => new NNNetSpecificationString(spec));
      DeviceSpec = keys.GetValueOrDefaultMapped("Device", CeresUserSettingsManager.Settings.DefaultDeviceSpecString, true, spec => new NNDevicesSpecificationString(spec));

      NetworkOpponentSpec = keys.GetValueOrDefaultMapped<NNNetSpecificationString>("Network-opponent", null, false, spec => new NNNetSpecificationString(spec));
      DeviceOpponentSpec = keys.GetValueOrDefaultMapped("Device-opponent", null, false, spec => new NNDevicesSpecificationString(spec));

      SearchLimit = keys.GetValueOrDefaultMapped<SearchLimit>("Limit", "10sm", true, str => SearchLimitSpecificationString.Parse(str));
      SearchLimitOpponent = keys.GetValueOrDefaultMapped<SearchLimit>("Limit-Opponent", null, false, str => SearchLimitSpecificationString.Parse(str));

      Pruning = keys.GetValueOrDefaultMapped<bool>("Pruning", "true", false, str=> bool.Parse(str));

      // If opponent was specified, default to same settings as
      // primary player if overrides were not specified
      if (Opponent != null)
      {
        if (SearchLimitOpponent == null)
          SearchLimitOpponent = SearchLimit;

        if (NetworkOpponentSpec == null)
          NetworkOpponentSpec = NetworkSpec;

        if (DeviceOpponentSpec == null)
          DeviceOpponentSpec = DeviceSpec;
      }

      if (convertGameLimitToMoveLimit)
      {
        if (SearchLimit != null)
        {
          SearchLimit = SearchLimit.ConvertedGameToMoveLimit;
        }

        if (SearchLimitOpponent != null)
        {
          SearchLimitOpponent = SearchLimitOpponent.ConvertedGameToMoveLimit;
        }
      }

    }

  }
}
