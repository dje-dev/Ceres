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

    public string LastInfoString => lastInfo;
    public string LastBestMove => lastBestMove;

    public UCISearchInfo LastSearchInfo => lastSearchInfo;


    protected volatile UCISearchInfo lastSearchInfo;

    protected UCISearchInfo engine1LastSearchInfo;


    public List<string> InfoStringDict0 = new List<string>();

    void DataRead(int id, string data)
    {
      double elapsedTime = (double)(Stopwatch.GetTimestamp() - startTime) / freq;
      if (UCI_VERBOSE_LOGGING) Console.WriteLine(Math.Round(elapsedTime, 3) + " ENGINE::{0}::{1}", id, data);

      if (data.Contains("error"))
      {
        lastError = data;
      }
      else if (data.Contains("bestmove"))
      {
        lastBestMove = data;
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

        System.Threading.Thread.Sleep(isShortSearch ? 0 : 1);
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
