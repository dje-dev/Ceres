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
using System.Diagnostics;
using System.IO;
using Ceres.Base.Benchmarking;
using Ceres.Chess.External.CEngine;

#endregion

namespace Ceres.Chess.ExternalPrograms.UCI
{
  public enum UCIMoveLimitType
  {
    NodeCount,
    TimeMove,
    TimeGameForWhiteAndBlack
  }

  /// <summary>
  /// Manages execution of an external UCI engine as a separate process,
  /// controled by standard input and output.
  /// </summary>
  public class UCIGameRunner
  {
    public static bool UCI_VERBOSE_LOGGING = false;
    public readonly int Index;

    protected UCIEngineProcess engine;

    public readonly string EngineEXE;

    public readonly string EngineExtraCommand;

    public readonly bool ResetStateAndCachesBeforeMoves;

    public bool IsShutdown => engine == null;

    /// <summary>
    /// The engine identification string as reported by the UCI "id name" response.
    /// </summary>
    public string EngineID { get; private set; }

    protected string curPosAndMoves = null;

    protected int doneCount = 0;
    protected double startTime;
    protected double freq;

    int numWarningsShown = 0;


    public int EngineProcessID => engine.EngineProcess.Id;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="exe"></param>
    /// <param name="extraCommand"></param>
    /// <param name="readHandler"></param>
    /// <param name="numThreads">N.B. when doing single static position evals from LC0, must set to 1</param>
    /// <returns></returns>
    UCIEngineProcess StartEngine(string engineName, string exePath, string extraCommand, ReadEvent readHandler,
                                 int numThreads = 1, bool checkExecutableExists = true)
    {
      UCIEngineProcess engine = new UCIEngineProcess(engineName, exePath, extraCommand);
      engine.ReadEvent += readHandler;
      engine.StartEngine(checkExecutableExists);

      engine.ReadAsync();

      engine.SendCommandLine("uci");
     
      return engine;
    }


    /// <summary>
    /// 
    /// </summary>
    /// <param name="engineEXE"></param>
    /// <param name="engineExtraCommand"></param>
    /// <param name="runnerIndex">optional index of this runner within parallel set of runners</param>
    public UCIGameRunner(string engineEXE,
                          bool resetStateAndCachesBeforeMoves,
                          string extraCommandLineArguments = null, 
                          string[] uciSetOptionCommands = null, 
                          int runnerIndex = -1,
                          bool checkExecutableExists = true)
    {
      EngineEXE = engineEXE;
      ResetStateAndCachesBeforeMoves = resetStateAndCachesBeforeMoves;
      EngineExtraCommand = extraCommandLineArguments;
      Index = runnerIndex;

      ReadEvent readHandler = new ReadEvent(DataRead);

      string engine1Name = new FileInfo(engineEXE).Name;
      engine = StartEngine(engine1Name, engineEXE, extraCommandLineArguments, readHandler, checkExecutableExists: checkExecutableExists);

      System.Threading.Thread.Sleep(20);

      if (uciSetOptionCommands != null)
      {
        foreach (string extraCommand in uciSetOptionCommands)
        {
          engine.SendCommandLine(extraCommand);
        }
      }

      // Only now issue isready (after we've set options).
      engine.SendIsReadyAndWaitForOK();

      freq = Stopwatch.Frequency;
      startTime = Stopwatch.GetTimestamp();
    }


    protected volatile string lastInfo;
    protected volatile string lastBestMove = null;
    protected volatile string lastError = null;

    /// <summary>
    /// Signaled by the engine read thread the instant a "bestmove" line arrives, so that
    /// EvalPosition can block (sleep) until the search completes rather than busy-spinning.
    /// </summary>
    readonly System.Threading.ManualResetEventSlim bestMoveSignal = new System.Threading.ManualResetEventSlim(false);

    public string LastInfoString => lastInfo;
    public string LastBestMove => lastBestMove;

    public UCISearchInfo LastSearchInfo => lastSearchInfo;


    protected volatile UCISearchInfo lastSearchInfo;

    protected UCISearchInfo engine1LastSearchInfo;


    public List<string> InfoStringDict0 = new List<string>();

    /// <summary>
    /// For each multipv index, the most recent (deepest) UCI search info line seen during the
    /// current search. Populated whenever the engine emits lines with a principal variation and a
    /// score (always for MultiPV &gt; 1, and also for MultiPV = 1). Cleared at the start of each search.
    /// </summary>
    public readonly System.Collections.Concurrent.ConcurrentDictionary<int, string> LastInfoLineByMultiPV = new();

    /// <summary>
    /// Sequence of multipv-1 (best line) info lines seen during the current search, in arrival order.
    /// Used to determine the depth at which the best move stabilized. Cleared at the start of each search.
    /// </summary>
    public readonly System.Collections.Concurrent.ConcurrentQueue<string> BestLineHistory = new();


    #region Marker-delimited block capture

    /// <summary>
    /// When non-null, a marker-delimited block capture is in progress; received lines belonging to
    /// the block are collected here (see CaptureMarkedBlock). Null when no capture is active.
    /// </summary>
    volatile List<string> blockCaptureLines;
    volatile string blockBeginMarker;
    volatile string blockEndMarker;
    volatile bool blockCaptureActive;     // true once the begin marker has been seen
    volatile bool blockCaptureComplete;   // true once the end marker has been seen

    #endregion

    void DataRead(int id, string data)
    {
      double elapsedTime = (double)(Stopwatch.GetTimestamp() - startTime) / freq;
      if (UCI_VERBOSE_LOGGING) Console.WriteLine(Math.Round(elapsedTime, 3) + " ENGINE::{0}::{1}", id, data);

      // If a marker-delimited block capture is active, intercept lines belonging to the block so
      // they are collected (and not routed as normal engine output). Lines arriving before the
      // begin marker fall through to normal handling below.
      List<string> capture = blockCaptureLines;
      if (capture != null)
      {
        string trimmed = data.Trim();
        if (!blockCaptureActive)
        {
          if (trimmed == blockBeginMarker)
          {
            blockCaptureActive = true;
            return;
          }
        }
        else
        {
          if (trimmed == blockEndMarker)
          {
            blockCaptureActive = false;
            blockCaptureComplete = true;
            return;
          }
          lock (capture)
          {
            capture.Add(data);
          }
          return;
        }
      }

      if (data.StartsWith("id name "))
      {
        EngineID = data.Substring("id name ".Length).Trim();
      }
      else if (data.Contains("error"))
      {
        lastError = data;
      }
      else if (data.Contains("bestmove"))
      {
        lastBestMove = data;
        // Wake EvalPosition's blocking wait immediately (it is waiting for exactly this line).
        bestMoveSignal.Set();
      }
      else if (data.Contains("info string"))
      {
        InfoStringDict0.Add(data);
      }
      else if (data.Contains("info"))
      {
        if (lastSearchInfo == null || data.Contains("score")) // ignore things like "info time" because it might not contain score info and overwrite prior good info
        {
          lastSearchInfo = new UCISearchInfo(data, lastBestMove, InfoStringDict0);// id == 0 ? InfoStringDict0 : null);
          lastInfo = data;
        }

        // Capture per-multipv lines so a MultiPVResult can be reconstructed after the search
        // (used by the suite builder oracle). Only lines that carry a scored principal variation.
        if (data.Contains(" pv ") && data.Contains("score"))
        {
          int multiPVIndex = UCIInfoParse.ExtractMultiPVIndex(data);
          LastInfoLineByMultiPV[multiPVIndex] = data;
          if (multiPVIndex == 1)
          {
            BestLineHistory.Enqueue(data);
          }
        }
      }
    }


    public UCISearchInfo EvalPositionToMovetime(string fenAndMovesString, int moveTimeMilliseconds)
    {
      return EvalPosition(fenAndMovesString, "movetime", moveTimeMilliseconds);
    }

    public UCISearchInfo EvalPositionRemainingTime(string fenAndMovesString,
                                                   bool whiteToMove,
                                                   int? movesToGo,
                                                   int remainingTimeMilliseconds, 
                                                   int incrementTimeMilliseconds)
    {
      string prefixChar = whiteToMove ? "w" : "b";
      string moveStr = $"go {prefixChar}time {Math.Max(1, remainingTimeMilliseconds)}";
      if (incrementTimeMilliseconds > 0)
      {
        moveStr += $" {prefixChar}inc {incrementTimeMilliseconds}";
      }

      if (movesToGo.HasValue)
      {
        moveStr += " movestogo " + movesToGo.Value;
      }

      return EvalPosition(fenAndMovesString, null, 0, moveStr);
    }


    public UCISearchInfo EvalPositionToNodes(string fenAndMovesString, int numNodes)
    {
      return EvalPosition(fenAndMovesString, "nodes", numNodes);
    }

    public UCISearchInfo EvalPositionRemainingNodes(string fenAndMovesString,
                                                    bool whiteToMove,
                                                    int? movesToGo,
                                                    int remainingNodes,
                                                    int incrementNodes)
    {
      string prefixChar = whiteToMove ? "w" : "b";
      string moveStr = $"go {prefixChar}nodes {Math.Max(1, remainingNodes)}";
      if (incrementNodes > 0)
      {
        moveStr += $" {prefixChar}inc {incrementNodes}";
      }

      if (movesToGo.HasValue)
      {
        moveStr += " movestogo " + movesToGo.Value;
      }

      return EvalPosition(fenAndMovesString, null, 0, moveStr);
    }


    public  void SendCommand(string command)
    {
      SendCommandCRLF(engine, command);
    }


    /// <summary>
    /// Sends a command to the engine and captures the lines it emits between the specified begin and
    /// end marker lines (the markers themselves are excluded). Lines arriving before the begin marker
    /// are routed normally. Returns the captured lines, or null if the end marker was not seen within
    /// the timeout (or the engine exited). Intended for capturing custom multi-line diagnostics output
    /// such as the "dump-info-block" command.
    /// </summary>
    /// <param name="command">the UCI command to send (e.g. "dump-info-block")</param>
    /// <param name="beginMarker">exact line (after trimming) marking the start of the block</param>
    /// <param name="endMarker">exact line (after trimming) marking the end of the block</param>
    /// <param name="timeoutMs">maximum time to wait for the end marker</param>
    public IReadOnlyList<string> CaptureMarkedBlock(string command, string beginMarker, string endMarker, int timeoutMs)
    {
      List<string> lines = new List<string>();
      blockBeginMarker = beginMarker;
      blockEndMarker = endMarker;
      blockCaptureActive = false;
      blockCaptureComplete = false;
      blockCaptureLines = lines;   // set last: arms the intercept in DataRead

      try
      {
        SendCommand(command);

        long deadline = Stopwatch.GetTimestamp() + (long)(timeoutMs / 1000.0 * Stopwatch.Frequency);
        while (!blockCaptureComplete && Stopwatch.GetTimestamp() < deadline)
        {
          if (engine.EngineProcess.HasExited)
          {
            break;
          }
          System.Threading.Thread.Sleep(10);
        }
      }
      finally
      {
        blockCaptureLines = null;   // disarm the intercept
        blockCaptureActive = false;
      }

      return blockCaptureComplete ? lines : null;
    }

    protected void SendCommandCRLF(UCIEngineProcess thisEngine, string cmd)
    {
      if (UCI_VERBOSE_LOGGING) Console.WriteLine("--> CMD " + cmd);
      thisEngine.SendCommandLine(cmd);
    }


    bool havePrepared = false;


    /// <summary>
    /// Executes any preparatory UCI commands before sending a position for evaluation.
    /// These preparatory steps are typically not counted in the search time for the engine.
    /// </summary>
    /// <param name="engineNum"></param>
    public void EvalPositionPrepare()
    {
      if (ResetStateAndCachesBeforeMoves)
      {
        // Not all engines support Clear hash, e.g.
        // "option name Clear Hash type button"
        // so we do not issue this command.
        //thisEngine.SendCommandLine("setoption name Clear Hash");

        // Perhaps ucinewgame helps reset state
        engine.SendCommandLine("ucinewgame");
        engine.SendIsReadyAndWaitForOK();
      }

      havePrepared = true;
    }


    public void StartNewGame()
    {
      SendCommandCRLF(engine, "ucinewgame");
    }


    /// <summary>
    /// 
    /// </summary>
    /// <param name="fen"></param>
    /// <param name="movesString"></param>
    /// <param name="moveType"></param>
    /// <param name="moveMetric"></param>
    /// <param name="moveOverrideString"></param>
    /// <returns></returns>
    public UCISearchInfo EvalPosition(string fen, string movesString,
                                      string moveType, int moveMetric,
                                      string moveOverrideString = null)
    {
      string fenAndMovesStr = fen;
      if (movesString != null && movesString != "")
      {
        fenAndMovesStr = fen + " moves " + movesString;
      }

      return EvalPosition(fenAndMovesStr, moveType, moveMetric, moveOverrideString);
    }


    /// <summary>
    /// Sends a position to UCI engine to be evaluated and waits for response.
    /// </summary>
    /// <param name="fenAndMovesString"></param>
    /// <param name="moveType"></param>
    /// <param name="moveMetric"></param>
    /// <param name="moveOverrideString"></param>
    /// <returns></returns>
    public UCISearchInfo EvalPosition(string fenAndMovesString,
                                      string moveType, int moveMetric,
                                      string moveOverrideString = null)
    {
      if (!havePrepared)
      {
        throw new Exception("UCIGameRunner.EvalPositionPrepare should be called each time before EvalPosition.");
      }

      lastBestMove = null;
      lastInfo = null;
      LastInfoLineByMultiPV.Clear();
      BestLineHistory.Clear();
      bestMoveSignal.Reset();   // clear any prior signal before issuing this search

      //      string posString = fenAndMovesString.Contains("startpos") ? fen
      string curPosCmd = "position ";
      if (fenAndMovesString.Contains("startpos"))
      {
        curPosCmd += fenAndMovesString;
      }
      else
      {
        curPosCmd += "fen " + fenAndMovesString;
      }
      SendCommandCRLF(engine, curPosCmd);

      if (moveOverrideString != null)
      {
        SendCommandCRLF(engine, moveOverrideString);
      }
      else
      {
        SendCommandCRLF(engine, "go " + moveType + " " + moveMetric);
      }

      string desc = $"{curPosCmd} on {EngineEXE} {EngineExtraCommand}";

      bool isShortSearch = moveType == "nodes" && moveMetric <= 1000;

      int waitCount = 0;
      DateTime startTime = DateTime.Now;
      while (lastBestMove == null || !lastBestMove.Contains("bestmove"))
      {
        float elapsedSeconds = (float)(DateTime.Now - startTime).TotalSeconds;

        const int MAX_WARNINGS_SHOW = 10;
        if (lastError != null && numWarningsShown < MAX_WARNINGS_SHOW)
        {
          Console.WriteLine($"WARNING: UCI error on external engines {desc} : {lastError}");
          if (++numWarningsShown == MAX_WARNINGS_SHOW)
          {
            Console.WriteLine("WARNING: Suppressing any possible additional warnings.");
          }
        }

        // Block (without consuming CPU) until the read thread signals "bestmove", or until a short
        // timeout elapses so the error / process-exit / long-wait checks below still run periodically.
        bestMoveSignal.Wait(500);
        if (elapsedSeconds > 5)
        {
          if (engine.EngineProcess.HasExited)
          {
            throw new Exception($"The engine process has exited: {desc}");
          }
          else if (isShortSearch)
          {
            Console.WriteLine($"--------------> Warn: waiting >{waitCount}ms for {desc}");
          }
          startTime = DateTime.Now;
        }

        waitCount++;
      }

      havePrepared = false;
      if (lastBestMove != null)
      {
        UCISearchInfo ret = new UCISearchInfo(lastInfo, lastBestMove, InfoStringDict0);
        return ret;
      }
      else
      {
        return null;
      }
    }


    public void Shutdown()
    {
      engine.SendCommandLine("stop");
      engine.SendCommandLine("quit");
      engine.Shutdown();
      engine = null;
    }

  }

}
