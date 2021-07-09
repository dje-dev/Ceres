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
    public bool DTZAvailable => true;


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

      return FathomTB.Initialize(paths);
    }


    public MGMove CheckTablebaseBestNextMoveViaDTZ(in Position currentPos, out GameResult result)
    {
      if (currentPos.PieceCount > MaxCardinality)
      {
        result = GameResult.Unknown;
        return default;
      }

      FathomProbeMove fathomResult = FathomTB.ProbeDTZ(currentPos.FEN, null);
      MGMove fathomMove = default;
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
        result = GameResult.Checkmate;
      }
      else
      {
        throw new Exception($"Internal error: unexpected Fathom game result {fathomResult.Result} in lookup of {currentPos.FEN}");
      }

      if (TESTING_MODE_COMPARE_RESULTS)
      {
        MGMove compMove = compEvaluator.CheckTablebaseBestNextMoveViaDTZ(in currentPos, out GameResult compResult);
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
      DoProbeWDL(in pos, out score, out result);
    }


    void DoProbeWDL(in Position pos, out LC0DLLSyzygyEvaluator.WDLScore score, out LC0DLLSyzygyEvaluator.ProbeState result)

    {
      // Make sure sufficiently few pieces remain on board
      // and not castling rights
      if (pos.PieceCount > MaxCardinality || pos.MiscInfo.CastlingRightsAny)
      {
        result = LC0DLLSyzygyEvaluator.ProbeState.Fail;
        score = LC0DLLSyzygyEvaluator.WDLScore.WDLDraw; // actually, unknown
        return;
      }

      FathomWDLResult probeResult = FathomTB.ProbeWDL(pos.FEN);

      if (probeResult == FathomWDLResult.Failure)
      {
        result = LC0DLLSyzygyEvaluator.ProbeState.Fail;
        score = LC0DLLSyzygyEvaluator.WDLScore.WDLDraw; // actually, unknown
        return;
      }

      LC0DLLSyzygyEvaluator.NumTablebaseHits++;

      result = LC0DLLSyzygyEvaluator.ProbeState.Ok;
      score = (LC0DLLSyzygyEvaluator.WDLScore)((int)probeResult) - 2;

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
