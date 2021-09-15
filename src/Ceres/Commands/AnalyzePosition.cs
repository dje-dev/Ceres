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
using System.Threading;
using System.Threading.Tasks;

using Ceres.Base.Benchmarking;
using Ceres.Chess;
using Ceres.Chess.ExternalPrograms.UCI;
using Ceres.Chess.GameEngines;
using Ceres.Chess.LC0.Engine;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;
using Ceres.Features.GameEngines;
using Ceres.Features.UCI;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes.Analysis;
using Ceres.MCTS.Params;
using Ceres.MCTS.Utils;

#endregion

namespace Ceres.Commands
{
  public static class AnalyzePosition
  {
    public static void Analyze(string fenAndMoves, SearchLimit searchLimit,
                               NNEvaluatorDef evaluatorDef,
                               bool forceDisablePruning,
                               LC0Engine lc0Engine = null,
                               GameEngine comparisonEngine = null,
                               bool verbose = false)
    {
      Console.WriteLine("=============================================================================");
      Console.WriteLine("Analyzing FEN   : " + fenAndMoves);
      Console.WriteLine("Search limit    : " + searchLimit.ToString());
      Console.WriteLine("Ceres evaluator : " + evaluatorDef.ToString());
      if (comparisonEngine != null)
        Console.WriteLine("Opponent      : " + comparisonEngine.ToString());
      Console.WriteLine();
      Console.WriteLine();

      ParamsSearch searchParams = new ParamsSearch();
      searchParams.FutilityPruningStopSearchEnabled = !forceDisablePruning;

      GameEngineCeresInProcess ceresEngine = new GameEngineCeresInProcess("Ceres", evaluatorDef, null, searchParams,
                                                                          moveImmediateIfOnlyOneMove: false);

      // Warmup (in parallel)    
      lc0Engine?.DoSearchPrepare();
      Parallel.Invoke(
        () => ceresEngine.Warmup(),
        () => comparisonEngine?.Warmup());

      ceresEngine.VerboseMoveStats = verbose;
      bool ceresDone = false;

      UCISearchInfo lastCeresInfo = null;

      GameEngine.ProgressCallback callback;
      callback = manager => lastCeresInfo = new UCISearchInfo(UCIInfo.UCIInfoString((MCTSManager)manager));


      // Launch Ceres
      GameEngineSearchResultCeres ceresResults = null;
      Task searchCeres = Task.Run(() =>
      {
        PositionWithHistory positionWithHistory = PositionWithHistory.FromFENAndMovesUCI(fenAndMoves);
        ceresResults = ceresEngine.Search(positionWithHistory, searchLimit, null, callback) as GameEngineSearchResultCeres;
      });

      // Possibly launch search for other engine
      Task searchComparison = null;
      if (lc0Engine != null || comparisonEngine != null)
      {
        searchComparison = Task.Run(() =>
        {
          if (lc0Engine != null)
          {
            lc0Engine.DoSearchPrepare();
            lc0Engine.AnalyzePositionFromFENAndMoves(fenAndMoves, searchLimit);
          }
          else
          {
            comparisonEngine.Search(PositionWithHistory.FromFENAndMovesUCI(fenAndMoves), searchLimit, verbose:true);
          }
        });
      };

      while (!searchCeres.IsCompleted || (searchComparison != null && !searchComparison.IsCompleted))
      {
        Thread.Sleep(1000);
//Console.WriteLine(DateTime.Now + " --> " + lastCeresInfo?.PVString + " OTHER " + comparisonEngine?.UCIInfo?.RawString);

        int numCharactersSame = int.MaxValue;
        if (lastCeresInfo?.PVString != null || comparisonEngine?.UCIInfo?.RawString != null)
        {
          if (lastCeresInfo != null && comparisonEngine?.UCIInfo != null)
          {
            numCharactersSame = 0;
            string pv1 = lastCeresInfo.PVString;
            UCISearchInfo lastComparisonInfo = comparisonEngine.UCIInfo;
            string pv2 = lastComparisonInfo.PVString;
            while (pv1.Length > numCharactersSame
                && pv2.Length > numCharactersSame
                && pv1[numCharactersSame] == pv2[numCharactersSame])
              numCharactersSame++;
          }
        }

        if (lastCeresInfo != null)
        {
          WriteUCI("Ceres", lastCeresInfo, numCharactersSame);
        }

        if (comparisonEngine != null)
        {
          WriteUCI(comparisonEngine.ID, comparisonEngine.UCIInfo, numCharactersSame);
        }
        Console.WriteLine();
      }

      searchCeres.Wait();
      searchComparison?.Wait();

      string infoUpdate = UCIInfo.UCIInfoString(ceresResults.Search.Manager);

      double q2 = ceresResults.Search.SearchRootNode.Q;
      //SearchPrincipalVariation pv2 = new SearchPrincipalVariation(worker2.Root);
      MCTSPosTreeNodeDumper.DumpPV(ceresResults.Search.SearchRootNode, true);

    }


    static void WriteUCI(string id, UCISearchInfo info, int numGreenPV)
    {
      if (info?.RawString != null)
      {
        string pv = info.PVString;
        string truncatedInfoString = pv.Substring(0, Math.Min(100, pv.Length));
        Console.Write($"{id,10}   {info.EngineReportedSearchTime/1000.0f,6:F2}s {info.ScoreCentipawns,4:F0}cp  {info.Nodes,14:N0} {info.NPS,12:N0}/s   ");

        ConsoleColor priorColor = Console.ForegroundColor;
        for (int i = 0; i < truncatedInfoString.Length; i++)
        {
          Console.ForegroundColor = (i < numGreenPV) ? ConsoleColor.Green : ConsoleColor.Red;
          Console.Write(truncatedInfoString[i]);
        }
        Console.ForegroundColor = priorColor;
        Console.WriteLine();
      }
    }
  }
}
