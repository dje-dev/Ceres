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

using Ceres.Base.Environment;
using Ceres.Base.Threading;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.MCTS.MTCSNodes;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.MCTS.Evaluators
{
  /// <summary>
  /// A leaf evaluator which can evaluate positions just 1 ply short
  /// of being covered by a tablebase. Succeeds only in the special situation where:
  ///   - the number of pieces on board is exactly one more than our tablebases cover, and
  ///   - there exists at least one capture move which leads to a tablebase loss for the opponent
  ///   
  /// Empirically the number of successful evaluations
  /// of LeafEvaluatorSyzygyPly1 is typically approximately 10% that 
  /// of LeafEvaluatorSyzygyPly0 successful evaluations.
  /// </summary>
  public sealed class LeafEvaluatorSyzygyPly1 : LeafEvaluatorBase
  {
    /// <summary>
    /// Supporting tablebase evaluator.
    /// </summary>
    public readonly LeafEvaluatorSyzygyLC0 Ply0Evaluator;

    /// <summary>
    /// Number of probe successes.
    /// </summary>
    internal static AccumulatorMultithreaded NumHits;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="ply0Evaluator"></param>
    public LeafEvaluatorSyzygyPly1(LeafEvaluatorSyzygyLC0 ply0Evaluator, bool forceNoTablebaseTerminals)
    {
      Ply0Evaluator = ply0Evaluator;
      if (forceNoTablebaseTerminals)
      {
        Mode = LeafEvaluatorMode.SetAuxilliaryEval;
      }
    }


    /// <summary>
    /// Implementation of evaluation method.
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    protected override LeafEvaluationResult DoTryEvaluate(MCTSNode node)
    {
      if (node.Depth == 0)
      {
        // Don't attempt at the root to avoid short-circuiting search here
        // (we need to build a tree to be able to choose a best move).
        return default;
      }

      Position pos = node.Annotation.Pos;

      // Abort immediately unless this position has exactly
      // one more piece than the max cardinality of our tablebases.
      if (pos.PieceCount != (Ply0Evaluator.MaxCardinality + 1))
      { 
        return default;
      }

      // Iterate over the capture moves.
      foreach ((MGMove move, Position newPos) in PositionsGenerator1Ply.GenPositions(pos, move => move.Capture))
      {
        // Check if this position is in tablebase and it is a definitive win for our side.
        LeafEvaluationResult result = Ply0Evaluator.Lookup(in newPos);
        if (result.TerminalStatus == GameResult.Checkmate)
        {
          // Check if loss for them (win for us)
          bool posLoses = result.V < 0;
          if (!posLoses)
          {
            return default;
          }

          if (CeresEnvironment.MONITORING_METRICS)
          {
            NumHits.Add(1, pos.PiecesShortHash);
          }

          return new LeafEvaluationResult(GameResult.Checkmate, result.LossP, result.WinP, result.M);
        }
      }

      return default;
    }


    [ModuleInitializer]
    internal static void ModuleInitialize()
    {
      NumHits.Initialize();
    }
  }
}
