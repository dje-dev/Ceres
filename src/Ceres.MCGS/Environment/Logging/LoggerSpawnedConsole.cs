#region Using directives

using System;
using System.IO;
using Ceres.Base.Misc;
using Microsoft.Extensions.Logging;


#endregion

#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

namespace Ceres.MCGS.Environment;

public sealed class LoggerSpawnedConsole : ILogger
{
  private readonly string _category;
  private readonly TextWriter _writer;

  public readonly bool EchoInline;

  public LoggerSpawnedConsole(string category, TextWriter writer, bool echoInline)
  {
    _category = category;
    _writer = writer;
    EchoInline = echoInline;
  }

  public IDisposable BeginScope<TState>(TState state)
  {
    return null;
  }


  public bool IsEnabled(LogLevel logLevel)
  {

    return true;
  }

  static bool haveLoggedDisconnect = false;

  const bool ECHO_INLINE = true;

  public void Log<TState>(LogLevel logLevel, EventId eventId,
                          TState state, Exception exception, Func<TState, Exception, string> formatter)
  {
    if (formatter == null)
    {
      throw new ArgumentNullException(nameof(formatter));
    }

    string message = formatter(state, exception);
    string output = $"[{DateTime.Now:HH:mm:ss}] {_category} [{logLevel}] {message}";

    if (output.Contains("[GREEN]"))
    {
      output = output.Replace("[GREEN]", "");
      if (EchoInline)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Green, output);
      }
    }
    else if (output.Contains("[BLUE]"))
    {
      output = output.Replace("[BLUE]", "");
      if (EchoInline)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Blue, output);
      }
    }
    if (output.Contains("[YELLOW]"))
    {
      output = output.Replace("[YELLOW]", "");
      if (EchoInline)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, output);
      }
    }
    else if (output.Contains("[RED]"))
    {
      output = output.Replace("[RED]", "");
      if (EchoInline)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, output);
      }
    }
    else
    {
//        Console.WriteLine(output);
    }
    
    try
    {
      _writer.WriteLine(output);
    }
    catch (Exception exc)
    {
      if (!haveLoggedDisconnect)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, "Failed to write to secondary console logger, likely disconnected. " + exc.Message);
        haveLoggedDisconnect = true;
      }
    }
  }
}


#if NOT
  public static void MakeMoveTest()
  {
    MGPosition mg = Position.StartPosition.ToMGPosition;

    Position pos = Position.StartPosition;
    Chess.Move m1 = Chess.Move.FromSAN(mg.ToPosition, "e4");

    while (false)
      Benchmarking.DumpOperationTimeAndMemoryStats(() => { Position p = pos; p.MakeMove(m1); }, "conv");
//      Benchmarking.DumpOperationTimeAndMemoryStats(() => MGChessPositionConverter.PositionFromMGChessPosition(in mg), "conv");

          PGNFileEnumerator pgn = new(@"z:\chess\data\pgn\ccrl.pgn");// xxx
//      PGNFileEnumerator pgn = new(@"z:\chess\data\pgn\TCEC_Season_25_-_Frd_1_Final_League.pgn");

    foreach ((Chess.Textual.PgnFileTools.GameInfo gameInfo, int posIndex, PositionWithHistory pp) in pgn.EnumeratePositionWithDetail())
    {
      Position startPos = pp.InitialPosition;
      MGPosition startPosMG = startPos.ToMGPosition;
      Position priorPos;
      foreach (MGMove moveMG in pp.Moves)
      {
        if (false)
        {
          const string BAD = "b4rkr/5p2/N1p1p1q1/1p6/P1pP3P/4R1P1/1P1Q1P2/R4K2 w - - 3 27";
          startPos = Position.FromFEN(BAD);
          Move moveX = new Move(Move.MoveType.MoveCastleShort);
          startPos.MakeMove(moveX);
          Console.WriteLine(startPos.FEN);
        }

        if (startPos.PieceCount != startPosMG.PieceCount)
          throw new NotImplementedException();
        priorPos = startPos;
        Move move = MGMoveConverter.ToMove(moveMG);

//          Console.WriteLine(startPos.FEN + " " + moveMG + " " + move);

//          if (startPos.FEN.StartsWith("rnbqk2r/pp2ppbp/6p1/2p5/3PP3/2P2N2/P4PPP/1RBQKB1R b Kkq"))
//            Console.WriteLine("here");

        startPos.MakeMove(move);
        startPosMG.MakeMove(moveMG);

//          if (!startPos.PiecesEqual(startPosMG.ToPosition))
          bool fensSame = startPos.FEN == startPosMG.ToPosition.FEN;
          if (!fensSame)
          {
            Console.WriteLine();
            Console.WriteLine(startPos.FEN);
            Console.WriteLine(startPos.FEN + " " + move);
            Console.WriteLine(startPosMG.ToPosition.FEN + " " + moveMG);
            Console.WriteLine(fensSame);
          }
          //throw new Exception("bad");
        else
        {
          Console.Write(".");
        }
      }
    }

#endif
