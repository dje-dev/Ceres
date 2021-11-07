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
using System.Collections.Concurrent;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

using Ceres.Chess;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;
using Ceres.Features.GameEngines;
using Ceres.Chess.GameEngines;
using System.Text.Json.Serialization;
using System.IO;
using Ceres.Base.Benchmarking;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.Features.BatchAnalysis
{
  public record BatchAnalyzeResult
  {
    public BatchAnalyzeRequest Request { init; get; }
    public int MoveIndex { init; get; }

    public GameEngineSearchResult Result { init; get; }
  }

  public record BatchAnalyzeRequest
  {
    public string ID { init; get; }


//    [JsonIgnore]
//    public PositionWithHistory PosWithHistory;

    
    public string StartFEN { init; get; }
    public string MovesUCI { init; get; }

    public int IndexFirstAnalyzePos { init; get; } = 0;
    public int IndexLastAnalyzePos { init; get; } = int.MaxValue;
    public List<int> IndicesAnalyzePos { init; get; }

    public int NumNodes;
  }


  /// <summary>
  /// Accepts a set of requests to analyze positions
  /// (either standalone or as part or all of a game)
  /// efficiently batches analysis requests and outputs 
  /// summary of search result for each position.
  /// 
  /// The design is inspired by the KataGo implementation of a similar feature.
  /// </summary>
  public class BatchAnalyzer
  {
    public Action<BatchAnalyzeResult> Callback;

    CancellationTokenSource cancellationSource = new ();
    BlockingCollection<BatchAnalyzeRequest> pendingAnalysisQueue = new();
    readonly object outputLockObj = new Object();


    public static void Test()
    {
      BatchAnalyzer analyzer = new();
      string FN_SMALL = @"c:\temp\ceres\match_TOURN_Ceres1_Ceres2_637470679068697070.pgn";
      string FN_BIG = @"c:\temp\ceres\match_TOURN_SF12_SF11_637466862262603189.pgn";

      analyzer.ProcessPGN(FN_SMALL, SearchLimit.NodesPerMove(1000));
    }

    const int NUM_PROCESSORS = 20;

    NNEvaluatorDef evaluatorDef;

    void CreateNNEvaluators()
    {
      const bool POOLED = true;
      string GPUS = POOLED ? "GPU:0,1,2,3:POOLED=BATCHANALYZER"
                           : "GPU:0";
      evaluatorDef = NNEvaluatorDefFactory.FromSpecification("LC0:J64-210", GPUS); // j64-210
    }


    public void ProcessPGN(Action<BatchAnalyzeResult> callback)
    {
      Process(() => DoProcessFromConsole(callback));
    }

    void DoProcessFromConsole(Action<BatchAnalyzeResult> callback)
    {
      CreateNNEvaluators();

      Callback = callback;
      OutputSample();

      //CeresUserSettingsManager.LoadFromDefaultFile();

      StartProcessorThreads(NUM_PROCESSORS);


      while (true)
      {
        string line = Console.ReadLine();
        if (line.ToUpper() == "QUIT")
        {
          break;
        }

        try
        {
          BatchAnalyzeRequest request = JsonSerializer.Deserialize<BatchAnalyzeRequest>(line);
          Console.WriteLine(request);
          pendingAnalysisQueue.Add(request);
        }
        catch (Exception exc)
        {
          lock (outputLockObj)
          {
            Console.WriteLine($"Illegal request {line}");
          }
        }

        //object ox = System.Text.Json.JsonSerializer.Deserialize<CeresUserSettings>(File.ReadAllText(@"c:\dev\ceres\artifacts\release\net5.0\ceres.json"));
        //Console.WriteLine(ox);

      }

      DrainQueueAndCancel();
      WriteStats();
    }

    public void ProcessPGN(string pgnFN, SearchLimit limit)
    {
      Process(() => DoProcessPGN(pgnFN, limit));
    }

    void Process(Action action)
    {
      CreateNNEvaluators();
      TimingStats stats = new TimingStats();
      using (new TimingBlock(stats))
      {
        action();
      }

      totalSearchSeconds += (float)stats.ElapsedTimeSecs;

      DrainQueueAndCancel();
      WriteStats();
    }

    void DoProcessPGN(string pgnFN, SearchLimit limit)
    {
      
      FileInfo info = new FileInfo(pgnFN);
      if (!info.Exists) throw new Exception($"PGN specified not found {pgnFN}");
      string baseName = info.Name;

      StartProcessorThreads(NUM_PROCESSORS);

      int gameIndex = 0;
      foreach (Game g in Game.FromPGN(pgnFN))
      {

        BatchAnalyzeRequest request = new BatchAnalyzeRequest()
        {
          ID = baseName + "_" + gameIndex,
          NumNodes = (int)limit.Value,
          StartFEN = g.InitialPosition.FEN,
          MovesUCI = g.MoveStr
        };

        pendingAnalysisQueue.Add(request);
#if NOT
        int moveIndex = 0;
//        ges.ResetGame();
        foreach (PositionWithHistory pwm in g.PositionsWithHistory)
        {
//          int id = Interlocked.Increment(ref pgnID);
          BatchAnalyzeRequest request =  new BatchAnalyzeRequest() 
          { 
            ID = baseName + "_" + gameIndex + "_" + moveIndex,
            NumNodes = (int)limit.Value, 
            StartFEN = pwm.InitialPosition.FEN,
            MovesUCI = pwm.MovesStr
          };

          pendingAnalysisQueue.Enqueue(request);
          moveIndex++;
        }
#endif
        gameIndex++;
      }
    }


    void StartProcessorThreads(int numProcessors)
    {
      for (int i = 0; i < numProcessors; i++)
      {
        CancellationToken cancellationToken = cancellationSource.Token;
        Task processor = new Task(ProcessAnalysisItemQueue, cancellationToken);
        processor.Start();
      }
    }


    readonly object statsLockObj = new();
    int totalSearchesExecuted = 0;
    int totalNodes = 0;
    int totalDistinctNodes = 0;
    float totalSearchSeconds = 0;


    void ProcessAnalysisItemQueue(object cancellationToken)
    {
      GameEngineCeresInProcess ges = new("BatchAnalyzer", evaluatorDef, null, 
                                         new ParamsSearch() { FutilityPruningStopSearchEnabled = false}, moveImmediateIfOnlyOneMove:false);
      ges.VerboseMoveStats = false;
      Console.WriteLine("analyzer started " + ges);



      foreach (BatchAnalyzeRequest request in pendingAnalysisQueue.GetConsumingEnumerable())
      {
        ges.ResetGame();

        PositionWithHistory pwh = PositionWithHistory.FromFENAndMovesUCI(request.StartFEN, request.MovesUCI);
        int moveIndex = 0;
        foreach (PositionWithHistory pwsh in pwh.PositionWithHistories)
        {

          // Skip if not in requested set of positions to evaluate
          if (moveIndex < request.IndexFirstAnalyzePos) continue;
          if (moveIndex > request.IndexLastAnalyzePos) continue;
          if (request.IndicesAnalyzePos != null && Array.IndexOf(request.IndicesAnalyzePos.ToArray(), moveIndex) == -1) continue;

          GameEngineSearchResult searchResult = ges.Search(pwsh, SearchLimit.NodesPerMove(request.NumNodes));

          lock (statsLockObj)
          {
            totalSearchesExecuted++;
            totalNodes += searchResult.FinalN;
            totalDistinctNodes += searchResult.FinalN - searchResult.StartingN;
          }

          BatchAnalyzeResult result = new() { Request = request, MoveIndex = moveIndex,  Result = searchResult};
          Callback?.Invoke(result);

          lock (outputLockObj)
          {
            Console.WriteLine(request.ID + "," + moveIndex + "," + searchResult.ScoreQ);
          }

          moveIndex++;
        }
      }
    }

    void DrainQueueAndCancel()
    {
      pendingAnalysisQueue.CompleteAdding();
      while (pendingAnalysisQueue.Count > 0)
        Thread.Sleep(20);
    }



    void WriteStats()
    {
      Console.WriteLine();
      Console.WriteLine($"Total seconds          : {totalSearchSeconds:11,F2}");
      Console.WriteLine($"Total searches         : {totalSearchesExecuted:11,N0}");
      Console.WriteLine($"Total nodes            : {totalNodes:11,N0}");
      Console.WriteLine($"Total distinct nodes   : {totalDistinctNodes:11,N0}");
      Console.WriteLine($"Nodes / second         : {totalNodes / totalSearchSeconds:11,N0}");
    }


    void OutputSample()
    {
      var requestx = new BatchAnalyzeRequest() { ID = "test1", StartFEN = Position.StartPosition.FEN, MovesUCI = "e2e4" };

      var reqSerialized = JsonSerializer.Serialize(requestx);
      BatchAnalyzeRequest reqDeserialized = JsonSerializer.Deserialize<BatchAnalyzeRequest>(reqSerialized);

      Console.WriteLine(reqSerialized);
      Console.WriteLine(reqDeserialized);

    }

  }


}
