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
using System.Linq;
using Ceres.Base.Misc;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators.LC0DLL;

#endregion

namespace Ceres.Chess.TBBackends.Fathom
{
  public class FathomEvaluator : ISyzygyEvaluatorEngine
  {
    // *******************************************************
    // TEST ONLY 
    const bool TESTING_MODE_COMPARE_RESULTS = false;
    static LC0DLLSyzygyEvaluator compEvaluator = null;
    // *******************************************************

    public static void Install()
    {
      SyzygyEvaluatorPool.OverrideEvaluatorFactory = () => new FathomEvaluator();
    }

    public FathomEvaluator()
    {
    }


    /// <summary>
    /// Maximum number of pieces of available tablebase positions.
    /// </summary>
    public int MaxCardinality => FathomTB.MaxPieces;


    /// <summary>
    /// If the DTZ files supported by the engine and potentially usable
    /// (if the necessary tablebase files are found for a given piece combination).
    /// </summary>
    public bool DTZAvailable => NumDTZTablebaseFiles > 0;

    /// <summary>
    /// The number of DTM tablebase files available, if known.
    /// </summary>
    public int? NumWDLTablebaseFiles => FathomProbe.numWdl;

    /// <summary>
    /// The number of DTM tablebase files available, if known.
    /// </summary>
    public int? NumDTZTablebaseFiles => FathomProbe.numDtz;


    /// <summary>
    /// Shuts down the evaluators (releasing associated resources).
    /// </summary>
    public void Dispose()
    {
      FathomTB.Release();
    }

    /// <summary>
    /// Initializes tablebases to use a specied set of paths.
    /// </summary>
    /// <param name="paths"></param>
    /// <returns></returns>
    public bool Initialize(string paths)
    {
      if (TESTING_MODE_COMPARE_RESULTS && compEvaluator == null)
      {
        compEvaluator = new LC0DLLSyzygyEvaluator(0);
        compEvaluator.Initialize(paths);
      }

      bool ok = FathomTB.Initialize(paths);

      if (ok)
      {
        Console.WriteLine($"Loading Syzygy tablebases from { paths }, found {FathomProbe.numWdl} WDL and {FathomProbe.numDtz} DTZ files.");
        if (FathomProbe.numDtz != FathomProbe.numWdl && FathomProbe.numWdl > 0)
        {
          ConsoleUtils.WriteLineColored(ConsoleColor.Red, "WARNING: WDL/DTZ files appear incomplete, possibly negatively impacting engine play.");
        }
        Console.WriteLine();
      }

      return ok;
    }


#if NOT
//         n < -100 : loss, but draw under 50-move rule
// -100 <= n < -1   : loss in n ply (assuming 50-move counter == 0)
//         0        : draw
//     1 < n <= 100 : win in n ply (assuming 50-move counter == 0)
//   100 < n        : win, but draw under 50-move rule
#endif

#if NOT
See the file syzygy.h (struct SyzygyTb) of the open source Arasan chess engine
for an example of a simple interface used by a chess engine to the underlying Fathom code.

However it seems like (in search.cpp) the probe_wdl is only called for move 50 counter equal to 0,
so maybe this engine is not doing this optimally?

#endif

    public MGMove CheckTablebaseBestNextMoveViaDTZ(in Position currentPos, out GameResult result, out List<MGMove> fullWinningMoveList)
    {
      fullWinningMoveList = null;

      if (currentPos.PieceCount > MaxCardinality
       || currentPos.MiscInfo.CastlingRightsAny)
      {
        result = GameResult.Unknown;
        return default;
      }

      FathomProbeMove fathomResult = FathomTB.ProbeDTZ(currentPos, out int minDTZ, out List<FathomTB.DTZMove> results);

      if (fathomResult.Result == FathomWDLResult.Failure)
      {
        result = GameResult.Unknown;
      }
      else if (fathomResult.Result == FathomWDLResult.Loss)
      {
        // TODO: Due to limitation of expressivness of GameResult we have to return Unknown
        //       See also code in MCTSManager which is affected.
        //       (This is not a problem of correctness, but just ugly).
        result = GameResult.Unknown;
      }
      else if (fathomResult.Result == FathomWDLResult.Draw
            || fathomResult.Result == FathomWDLResult.CursedWin
            || fathomResult.Result == FathomWDLResult.BlessedLoss)
      {
        result = GameResult.Draw;
      }
      else if (fathomResult.Result == FathomWDLResult.Win)
      {
        fullWinningMoveList = new List<MGMove>();
        results = results.OrderBy(fpm => fpm.DistanceToZero).ToList();// put shortest mates at beginning
        foreach (FathomTB.DTZMove fpm in results)
        {
          if (fpm.WDL == FathomWDLResult.Win)
          {
            fullWinningMoveList.Add(fpm.Move);
          }
        }

        result = GameResult.Checkmate;
      }
      else
      {
        throw new Exception($"Internal error: unexpected Fathom game result {fathomResult.Result} in lookup of {currentPos.FEN}");
      }

      if (TESTING_MODE_COMPARE_RESULTS)
      {
        MGMove compMove = compEvaluator.CheckTablebaseBestNextMoveViaDTZ(in currentPos, out GameResult compResult, out _);
        if (result != compResult)
        {
          Console.WriteLine("DTZ check failure " + currentPos.FEN + " " + result + " " + fathomResult.Move + " vs. compare " + compResult + " " + compMove);
        }
      }

      if (result != GameResult.Unknown)
      {
        LC0DLLSyzygyEvaluator.NumTablebaseHits++;
      }

      return fathomResult.Move;
    }


    public void ProbeWDL(in Position pos, out LC0DLLSyzygyEvaluator.WDLScore score, out LC0DLLSyzygyEvaluator.ProbeState result)

    {
      // Make sure sufficiently few pieces remain on board
      // and not castling rights
      if (pos.PieceCount > MaxCardinality || pos.MiscInfo.CastlingRightsAny)
      {
        result = LC0DLLSyzygyEvaluator.ProbeState.Fail;
        score = LC0DLLSyzygyEvaluator.WDLScore.WDLDraw; // actually, unknown
        return;
      }

      FathomWDLResult probeResult = FathomTB.ProbeWDL(in pos);

      if (probeResult == FathomWDLResult.Failure)
      {
        result = LC0DLLSyzygyEvaluator.ProbeState.Fail;
        score = LC0DLLSyzygyEvaluator.WDLScore.WDLDraw; // actually, unknown
        return;
      }

      LC0DLLSyzygyEvaluator.NumTablebaseHits++;

      result = LC0DLLSyzygyEvaluator.ProbeState.Ok;
      score = (LC0DLLSyzygyEvaluator.WDLScore)((int)probeResult) - 2;

      // TODO: The original Fathom interface has a wrapper which
      //       will reject positions with the 50 move counter nonzero.
      //       This code really should do a DTZ lookup if it gets a hit
      //       and make sure the win/loss is reachable in time.
      //       However the ProbeDTZOnly is possibly not working, a
      //       and calling ProveDTZ is probably too expensive.
      const bool TEST = false;
      if (TEST && (score == LC0DLLSyzygyEvaluator.WDLScore.WDLWin 
                || score == LC0DLLSyzygyEvaluator.WDLScore.WDLLoss))
      {
        FathomProbeMove fathomResult = FathomTB.ProbeDTZ(in pos, out int minDTZ, out _);
        int numMovesAvailable = 100 - pos.MiscInfo.Move50Count;
        if (Math.Abs(minDTZ) > numMovesAvailable)
        {
          Console.WriteLine(pos.FEN + " " + score + " " + fathomResult + " but DTZ " + minDTZ);
          // int dtz = FathomTB.ProbeDTZOnly(currentPos.FEN, out int success); does not work
        }
      }

      if (TESTING_MODE_COMPARE_RESULTS)
      {
        compEvaluator.ProbeWDL(in pos, out LC0DLLSyzygyEvaluator.WDLScore compScore, out LC0DLLSyzygyEvaluator.ProbeState compResult);
        if (compResult == LC0DLLSyzygyEvaluator.ProbeState.ZeroingBestMove) compResult = LC0DLLSyzygyEvaluator.ProbeState.Ok;

        if (compScore != score || compResult != result)
        {
          throw new Exception($"DoProbeWDL disagrees with comparison on {pos.FEN}  yields {result},{ score } versus compare {compResult},{compScore}");
        }

      }

    }
  }

}
