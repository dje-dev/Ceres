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
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;
using Ceres.Features.GameEngines;
using Ceres.Features.Players;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.Commands
{
  public record FeatureAnalyzeParams : FeaturePlayWithOpponent
  {
    public string FenAndMovesStr { init; get; }

    /// <summary>
    /// Constructor which parses arguments.
    /// </summary>
    /// <param name="fen"></param>
    /// <param name="args"></param>
    /// <returns></returns>
    public static FeatureAnalyzeParams ParseAnalyzeCommand(string fenAndMovesStr, string args)
    {
      if (fenAndMovesStr == null) throw new Exception("ANALYZE command expected to end with FEN to be analyzed");

      // Make sure string is valid
      try
      {
        PositionWithHistory position = PositionWithHistory.FromFENAndMovesUCI(fenAndMovesStr);
      }
      catch (Exception exc)
      {
        throw new Exception($"Error parsing expected FEN (and possible moves string) {fenAndMovesStr}");
      }

      FeatureAnalyzeParams parms = new FeatureAnalyzeParams()
      {
        FenAndMovesStr = fenAndMovesStr
      };

      // Add in all the fields from the base class
      parms.ParseBaseFields(args, true);

      if (parms.Opponent != null && parms.SearchLimit != parms.SearchLimitOpponent)
      {
        throw new Exception("Unequal search limits not currently supported for ANALYZE command");
      }

      return parms;
    }


    /// <summary>
    /// Executes the command against a specified FEN.
    /// </summary>
    /// <param name="fen"></param>
    public void Execute(string fen)
    {
      NNEvaluatorDef evaluatorDef = new NNEvaluatorDef(NetworkSpec.ComboType, NetworkSpec.NetDefs,
                                                        DeviceSpec.ComboType, DeviceSpec.Devices, null);

      // Check if different network and device specified for opponent
      NNEvaluatorDef evaluatorDefOpponent;
      if (Opponent != null &&
        (NetworkOpponentSpec != null || DeviceOpponentSpec != null))
      {
        if (NetworkOpponentSpec == null || DeviceOpponentSpec == null)
        {
          throw new Exception("Both network-opponent and device-opponent must be provided");
        }

        evaluatorDefOpponent = new NNEvaluatorDef(NetworkOpponentSpec.ComboType, NetworkOpponentSpec.NetDefs,
                                                  DeviceOpponentSpec.ComboType, DeviceOpponentSpec.Devices, null);
      }
      else
      {
        // Share evaluators
        evaluatorDefOpponent = evaluatorDef;
      }

      GameEngineLC0 lc0 = null;
      GameEngine comparisonEngine = null;
      if (Opponent != null && Opponent.ToUpper() == "LC0")
      {
        lc0 = new GameEngineLC0("LC0", evaluatorDefOpponent.Nets[0].Net.NetworkID, !Pruning,
                                false, null, null, evaluatorDefOpponent, verbose: true);
        comparisonEngine = lc0;
      }
      else if (Opponent != null && Opponent.ToUpper() == "CERES")
      {
        ParamsSearch paramsSearch = new();
        ParamsSelect paramsSelect = new();
        paramsSearch.FutilityPruningStopSearchEnabled = Pruning;

        GameEngineCeresInProcess opponentCeres = new GameEngineCeresInProcess("Ceres2", evaluatorDefOpponent, null, 
                                                                              paramsSearch, paramsSelect, moveImmediateIfOnlyOneMove:false);
        comparisonEngine = opponentCeres;
      }
      else if (Opponent != null)
      {
        (EnginePlayerDef def1, EnginePlayerDef def2) = GetEngineDefs(false);

        comparisonEngine = def2.EngineDef.CreateEngine();
      }

      lc0?.Warmup();
      comparisonEngine?.Warmup();

      const bool VERBOSE = true;
      AnalyzePosition.Analyze(fen, SearchLimit, evaluatorDef, !Pruning, lc0?.LC0Engine, comparisonEngine, VERBOSE);

      if (lc0 != null)
      {
        Console.WriteLine();
        Console.WriteLine("LC0 Verbose move stats");
        foreach (string info in lc0.LC0Engine.LastAnalyzedPositionStats.UCIInfo.Infos)
        {
          if (info.Contains("WL:"))
            Console.WriteLine(info);
        }
      }

    }
  }
}
