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

using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;
using Ceres.MCTS.Params;
using Ceres.Features.GameEngines;

#endregion

namespace Ceres.Commands
{
  public record FeatureBenchmarkSearch : FeaturePlayWithOpponent
  {

    /// <summary>
    /// Constructor which parses arguments.
    /// </summary>
    /// <param name="fen"></param>
    /// <param name="args"></param>
    /// <returns></returns>
    public static FeatureBenchmarkSearch ParseBenchmarkCommand(string args)
    {
      // Add in all the fields from the base class
      FeatureBenchmarkSearch parms = new();
      parms.ParseBaseFields(args, true);

      if (parms.Opponent != null && parms.SearchLimit != parms.SearchLimitOpponent)
      {
        throw new Exception("Unequal search limits not supported for BENCHMARK command");
      }

      return parms;
    }

    internal void Execute()
    {
      bool withLC0 = false;

      if (Opponent != null)
      {
        if (Opponent.ToUpper() == "LC0")
        {
          withLC0 = true;
        }
        else
        {
          // TODO: soften this restriction, allow any UCI
          throw new Exception("Only opponent supported is current LC0");
        }
      }


      NNEvaluatorDef evaluatorDef = new NNEvaluatorDef(NetworkSpec.ComboType, NetworkSpec.NetDefs,
                                                       DeviceSpec.ComboType, DeviceSpec.Devices, null);
      Benchmark(evaluatorDef, SearchLimit, withLC0);
    }


    /// <summary>
    /// Runs searches on a set of standard benchmarking positions 
    /// (originally used by Stockfish).
    /// 
    /// </summary>
    /// <param name="netSpec"></param>
    /// <param name="secondsPerMove"></param>
    /// <param name="withLC0"></param>
    /// <param name="maxPositions"></param>
    public static void Benchmark(NNEvaluatorDef nnDefCeres, SearchLimit searchLimit, bool withLC0, int maxPositions = int.MaxValue)
    {
      Console.WriteLine();
      Console.WriteLine($"Benchmark Position Performance Test using {nnDefCeres}");

      ParamsSearch paramsSearch = new ParamsSearch();
      paramsSearch.FutilityPruningStopSearchEnabled = false;
      GameEngineCeresInProcess engineCeres = new GameEngineCeresInProcess("Ceres", nnDefCeres, null, paramsSearch);
      engineCeres.Warmup();
      engineCeres.ResetGame();

      GameEngineLC0 engineLC0 = null;
      if (withLC0)
      {
        //NNEvaluatorDef nnDefLC0 = nnDefCeres.ToEvaluator();
        engineLC0 = new GameEngineLC0("LC0", nnDefCeres.Nets[0].Net.NetworkID, 
                                      forceDisableSmartPruning: true, paramsNN: nnDefCeres,
                                      alwaysFillHistory: true, verbose: false);
        engineLC0.Warmup();
        engineLC0.ResetGame();
      }

      Console.WriteLine("  " + engineCeres);
      if (withLC0)
      {
        Console.WriteLine("  " + engineLC0);
      }
      Console.WriteLine();

      List<float> npsCeres = new();
      List<float> npsLC0 = new();

      searchLimit = searchLimit with { SearchCanBeExpanded = false };
      if (searchLimit.IsTimeLimit)
      {
        // No need for move overhead
        searchLimit = searchLimit with { Value = searchLimit.Value + engineCeres.SearchParams.MoveOverheadSeconds };
      }

      long numNodesCeres = 0;
      long timeMSCeres = 0;
      long numNodesLC0 = 0;
      long timeMSLC0 = 0;
      int countCeresFaster = 0;

      for (int i = 0; i < Math.Min(BENCHMARK_POS.Length, maxPositions); i++)
      {
        string fen = BENCHMARK_POS[i];
        PositionWithHistory benchmarkPos = PositionWithHistory.FromFENAndMovesUCI(fen);

        GameEngineSearchResultCeres resultCeres = engineCeres.SearchCeres(benchmarkPos, searchLimit);
        float ceresSearchSecs = (float)resultCeres.TimingStats.ElapsedTimeSecs;
        float thisNpsCeres = resultCeres.FinalN / ceresSearchSecs;
        npsCeres.Add(thisNpsCeres);
          
        Console.WriteLine($"{ i+1,5:N0}. {resultCeres.FinalN,10:N0} nodes " + $" {ceresSearchSecs,7:F2} secs  {thisNpsCeres,8:N0} / sec"
          + $"  {resultCeres.ScoreCentipawns,6:N0} cp {resultCeres.MoveString,7}      {fen}");

        numNodesCeres += resultCeres.Search.SearchRootNode.N;
        timeMSCeres += (long)(Math.Round(resultCeres.Search.TimingInfo.ElapsedTimeSecs * 1000, 0));

        if (withLC0)
        {
          GameEngineSearchResult resultLC0 = engineLC0.Search(benchmarkPos, searchLimit);
          float lc0SearchSecs = (float)resultLC0.TimingStats.ElapsedTimeSecs;
          float thisNPSLC0 = resultLC0.FinalN / lc0SearchSecs;
          npsLC0.Add(thisNPSLC0);
          Console.WriteLine($"       {resultLC0.FinalN,10:N0} nodes " + $" {lc0SearchSecs,7:F2} secs  {thisNPSLC0,8:N0} / sec"
            + $"  {resultLC0.ScoreCentipawns,6:N0} cp {resultLC0.MoveString,7}    ");

          numNodesLC0 += resultLC0.FinalN;
          timeMSLC0 += (long)(Math.Round(resultLC0.TimingStats.ElapsedTimeSecs * 1000, 0)); // Make TimingStats/TimingInfo consisten between Ceres and LC0, maybe maybe make interface?

          if (thisNpsCeres > thisNPSLC0)
          {
            countCeresFaster++;
          }

        }
        Console.WriteLine();
        engineCeres.ResetGame();
      }

      npsCeres.Sort();
      Console.WriteLine();
      Console.WriteLine("Ceres Benchmark Results =======");
      Console.WriteLine($"Total time(ms)   : {timeMSCeres,12:N0}");
      Console.WriteLine($"Nodes searched   : {numNodesCeres,12:N0}");
      Console.WriteLine($"Avg nodes/sec    : {1000 * numNodesCeres / timeMSCeres,12:N0}");
      Console.WriteLine($"Median nodes/sec : {npsCeres[npsCeres.Count / 2],12:N0}");

      if (withLC0)
      {
        Console.WriteLine($"Positions faster : {countCeresFaster,12:N0}");

        npsLC0.Sort();
        Console.WriteLine();
        Console.WriteLine("LC0 Benchmark Results =========");
        Console.WriteLine($"Total time(ms)   : {timeMSLC0,12:N0}");
        Console.WriteLine($"Nodes searched   : {numNodesLC0,12:N0}");
        Console.WriteLine($"Avg nodes/sec    : {1000 * numNodesLC0 / timeMSLC0,12:N0}");
        Console.WriteLine($"Median nodes/sec : {npsLC0[npsLC0.Count / 2],12:N0}");
        Console.WriteLine($"Positions faster : {npsCeres.Count - countCeresFaster,12:N0}");
      }

      engineCeres.Dispose();
      engineLC0?.Dispose();
    }



    // Standard positions copied from Stockfish and LC0.
    static readonly string[] BENCHMARK_POS = new string[] {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
      "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
      "rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14 moves d4e6",
      "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14 moves g2g4",
      "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
      "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
      "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
      "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
      "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
      "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
      "3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
      "r1q2rk1/2p1bppp/2Pp4/p6b/Q1PNp3/4B3/PP1R1PPP/2K4R w - - 2 18",
      "4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22",
      "3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26",
      "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
      "3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1",
      "2K5/p7/7P/5pR1/8/5k2/r7/8 w - - 0 1 moves g5g6 f3e3 g6g5 e3f3",
      "8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1",
      "7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1",
      "8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1",
      "8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1",
      "8/pp2r1k1/2p1p3/3pP2p/1P1P1P1P/P5KR/8/8 w - - 0 1",
      "8/3p4/p1bk3p/Pp6/1Kp1PpPp/2P2P1P/2P5/5B2 b - - 0 1",
      "5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1",
      "6k1/6p1/P6p/r1N5/5p2/7P/1b3PP1/4R1K1 w - - 0 1",
      "1r3k2/4q3/2Pp3b/3Bp3/2Q2p2/1p1P2P1/1P2KP2/3N4 w - - 0 1",
      "6k1/4pp1p/3p2p1/P1pPb3/R7/1r2P1PP/3B1P2/6K1 w - - 0 1",
      "8/3p3B/5p2/5P2/p7/PP5b/k7/6K1 w - - 0 1",
      "5rk1/q6p/2p3bR/1pPp1rP1/1P1Pp3/P3B1Q1/1K3P2/R7 w - - 93 90",
      "4rrk1/1p1nq3/p7/2p1P1pp/3P2bp/3Q1Bn1/PPPB4/1K2R1NR w - - 40 21",
      "r3k2r/3nnpbp/q2pp1p1/p7/Pp1PPPP1/4BNN1/1P5P/R2Q1RK1 w kq - 0 16",
      "3Qb1k1/1r2ppb1/pN1n2q1/Pp1Pp1Pr/4P2p/4BP2/4B1R1/1R5K b - - 11 40"
  };


  }
}
