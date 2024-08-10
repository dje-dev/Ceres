#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using Directives

using System;
using System.Runtime.CompilerServices;
using System.Threading;

using Ceres.Base.DataTypes;
using Ceres.Base.Threading;

using Ceres.Chess;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Evaluators
{
  /// <summary>
  /// Leaf evaluator which probes Syzygy tablebases 
  /// and returns optimal WDL evaluation if found.
  /// </summary>
  public sealed class LeafEvaluatorSyzygy : LeafEvaluatorBase
  {
    /// <summary>
    /// Maximum cardinality (number of pieces) supported by tablebases.
    /// </summary>
    public readonly int MaxCardinality;

    /// <summary>
    /// Number of probe successes (global accumulator for Ceres engine).
    /// </summary>
    public static AccumulatorMultithreaded NumHitsGlobal;

    /// <summary>
    /// Number of probe successes (for this evaluator).
    /// </summary>
    public int NumHits;

    /// <summary>
    /// 
    /// </summary>
    public readonly ISyzygyEvaluatorEngine Evaluator;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="paths">Set of paths in which tablebase are found; if null then Ceres.json settings used.</param>
    /// <param name="forceNoTablebaseTerminals"></param>
    public LeafEvaluatorSyzygy(string paths = null, bool forceNoTablebaseTerminals = false)
    {
      Evaluator = SyzygyEvaluatorPool.GetSessionForPaths(paths);
      MaxCardinality = Evaluator.MaxCardinality;

      if (forceNoTablebaseTerminals)
      {
        Mode = LeafEvaluatorMode.SetAuxilliaryEval;
      }
    }

    protected override LeafEvaluationResult DoTryEvaluate(MCTSNode node)
    {
      // Do not enable at root, since we need need to choose actual best move
      // which is not available without at least one level of search, since
      // DTZ access is not yet implemented.
      //
      // TODO: when implement DTZ access then we would know which move to make immediate
      //       then this would not be needed
      if (node.IsRoot)
      {
        return default;
      }

      ref readonly Position pos = ref node.Annotation.Pos;

      LeafEvaluationResult result = Lookup(in pos);

      return result;
    }

    internal LeafEvaluationResult Lookup(in Position pos)
    {
      Evaluator.ProbeWDL(in pos, out SyzygyWDLScore score, 
                                 out SyzygyProbeState resultCode);
      if (resultCode == SyzygyProbeState.Fail ||
          resultCode == SyzygyProbeState.ChangeSTM)
      {
        return default;
      }   

      LeafEvaluationResult result;
      switch (score)
      {
        // TODO: use information about distance to draw/mate to fill in the M and also argumet to WinPForProvenWin/WinPForProvenLoss
        case SyzygyWDLScore.WDLDraw:
          result = new LeafEvaluationResult(GameResult.Draw, 0, 0, -1, 0);
          break;

        case SyzygyWDLScore.WDLWin:
          result = new LeafEvaluationResult(GameResult.Checkmate, (FP16)ParamsSelect.WinPForProvenWin(1, false), 0, 1, 0);
          break;

        case SyzygyWDLScore.WDLLoss:
          result = new LeafEvaluationResult(GameResult.Checkmate, 0, (FP16)ParamsSelect.LossPForProvenLoss(1, false), 1, 0);
          break;

        case SyzygyWDLScore.WDLCursedWin:
          // Score as almost draw, just slightly positive (since opponent might err in obtaining draw)
          result = new LeafEvaluationResult(GameResult.Draw, (FP16)0.05f, 0, 100, 0);
          break;

        case SyzygyWDLScore.WDLBlessedLoss:
          // Score as almost draw, just slightly negative (since we might err in obtaining draw)
          result = new LeafEvaluationResult(GameResult.Draw, 0, (FP16)0.05f, 100, 0);
          break;

        default:
          throw new Exception("Internal error: unknown Syzygy tablebase result code");
      }

      // Update tablebase hits counter.
      Interlocked.Increment(ref NumHits);
      NumHitsGlobal.Add(1, pos.PiecesShortHash);

      return result;
    }


    [ModuleInitializer]
    internal static void ModuleInitialize()
    {
      NumHitsGlobal.Initialize();
    }
  }
}


