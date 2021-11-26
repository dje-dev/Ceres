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

using Ceres.Chess;
using System;
using Ceres.MCTS.MTCSNodes;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.Base.DataTypes;
using Ceres.MCTS.LeafExpansion;
using System.Diagnostics;
using Ceres.MCTS.Params;
using Ceres.Base.Threading;
using Ceres.Base.Environment;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.MCTS.Evaluators
{
  /// <summary>
  /// Leaf evaluator which consults Syzygy tablebases 
  /// on disk via Leela Chess Zero DLL.
  /// 
  /// NOTE: Currently only simple WDL support (no DTZ).
  /// </summary>
  public sealed class LeafEvaluatorSyzygyLC0 : LeafEvaluatorBase
  {
    /// <summary>
    /// Maximum cardinality (number of pieces) supported by tablebases.
    /// </summary>
    public readonly int MaxCardinality;

    /// <summary>
    /// Number of probe successes.
    /// </summary>
    internal static AccumulatorMultithreaded NumHits;

    /// <summary>
    /// 
    /// </summary>
    public readonly ISyzygyEvaluatorEngine Evaluator;


    public LeafEvaluatorSyzygyLC0(string paths, bool forceNoTablebaseTerminals)
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
      if (node.IsRoot) return default;

      ref readonly Position pos = ref node.Annotation.Pos;

      LeafEvaluationResult result = Lookup(in pos);

      return result;
    }

    internal LeafEvaluationResult Lookup(in Position pos)
    {
      Evaluator.ProbeWDL(in pos, out LC0DLLSyzygyEvaluator.WDLScore score, 
                                 out LC0DLLSyzygyEvaluator.ProbeState resultCode);
      if (resultCode == LC0DLLSyzygyEvaluator.ProbeState.Fail ||
          resultCode == LC0DLLSyzygyEvaluator.ProbeState.ChangeSTM)
      {
        return default;
      }   

      LeafEvaluationResult result;
      switch (score)
      {
        // TODO: use information about distance to draw/mate to fill in the M and also argumet to WinPForProvenWin/WinPForProvenLoss
        case LC0DLLSyzygyEvaluator.WDLScore.WDLDraw:
          result = new LeafEvaluationResult(GameResult.Draw, 0, 0, -1);
          break;

        case LC0DLLSyzygyEvaluator.WDLScore.WDLWin:
          result = new LeafEvaluationResult(GameResult.Checkmate, (FP16)ParamsSelect.WinPForProvenWin(1, false), 0, 1);
          break;

        case LC0DLLSyzygyEvaluator.WDLScore.WDLLoss:
          result = new LeafEvaluationResult(GameResult.Checkmate, 0, (FP16)ParamsSelect.LossPForProvenLoss(1, false), 1);
          break;

        case LC0DLLSyzygyEvaluator.WDLScore.WDLCursedWin:
          // Score as almost draw, just slightly positive (since opponent might err in obtaining draw)
          result = new LeafEvaluationResult(GameResult.Draw, (FP16)0.05f, 0, 100);
          break;

        case LC0DLLSyzygyEvaluator.WDLScore.WDLBlessedLoss:
          // Score as almost draw, just slightly negative (since we might err in obtaining draw)
          result = new LeafEvaluationResult(GameResult.Draw, 0, (FP16)0.05f, 100);
          break;

        default:
          throw new Exception("Internal error: unknown Syzygy tablebase result code");
      }

      if (CeresEnvironment.MONITORING_METRICS)
      {
        NumHits.Add(1, pos.PiecesShortHash);
      }

      return result;
    }

    [ModuleInitializer]
    internal static void ModuleInitialize()
    {
      NumHits.Initialize();
    }
  }
}


