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
using Ceres.Chess;
using Ceres.Chess.LC0.Positions;
using Ceres.Chess.MoveGen;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;

#endregion


namespace Ceres.MCTS.Utils
{
  /// <summary>
  /// Utility methods relating to UCI output.
  /// </summary>
  public static class UCIInfo
  {
    /// <summary>
    /// Returns an UCI info string appropriate for a given search state.
    /// </summary>
    /// <param name="manager"></param>
    /// <param name="overrideRootMove"></param>
    /// <returns></returns>
    public static string UCIInfoString(MCTSManager manager,
                                       MCTSNode overrideRootMove = default,
                                       MCTSNode overrideBestMoveNodeAtRoot = default,
                                       int? multiPVIndex = null,
                                       bool useParentN = true,
                                       bool showWDL = false,
                                       bool scoreAsQ = false)
    {
      using (new SearchContextExecutionBlock(manager.Context))
      {
        if (manager.TablebaseImmediateBestMove != default)
        {
          if (multiPVIndex.HasValue && multiPVIndex != 1)
          {
            return null;
          }
          else
          {
            return OutputUCIInfoTablebaseImmediate(manager, overrideRootMove.IsNotNull ? overrideRootMove : manager.Root, scoreAsQ);
          }
        }

        bool wasInstamove = manager.StopStatus == MCTSManager.SearchStopStatus.Instamove;

        // If no override bestMoveRoot was specified
        // then it is assumed the move chosen was from the root (not an instamove)
        MCTSNode thisRootNode = overrideRootMove.IsNotNull ? overrideRootMove : manager.Root;

        if (thisRootNode.NumPolicyMoves == 0)
        {
          // Terminal position, nothing to output
          return null;
        }

        float elapsedTimeSeconds = wasInstamove ? 0 : (float)(DateTime.Now - manager.StartTimeThisSearch).TotalSeconds;

        // Get the principal variation (the first move of which will be the best move)
        SearchPrincipalVariation pv;
        BestMoveInfo bestMoveInfo;
        using (new SearchContextExecutionBlock(manager.Context))
        {
          bestMoveInfo = thisRootNode.BestMoveInfo(false);
          pv = new SearchPrincipalVariation(thisRootNode, overrideBestMoveNodeAtRoot);
        }

        MCTSNode bestMoveNode = pv.Nodes.Count > 1 ? pv.Nodes[1] : pv.Nodes[0];

        float thisQ;
        if (overrideBestMoveNodeAtRoot.IsNotNull)
        {
          thisQ = (float)-overrideBestMoveNodeAtRoot.Q;
        }
        else
        {
          thisQ = bestMoveInfo.QOfBest;
        }

        // The score displayed corresponds to
        // the Q (average visit value) of the move to be made.
        float scoreToShow;
        if (scoreAsQ)
        {
          scoreToShow = MathF.Round(thisQ * 1000, 0);
        }
        else
        {
          scoreToShow = MathF.Round(EncodedEvalLogistic.LogisticToCentipawn(thisQ), 0);
        }

        float nps = manager.NumNodesVisitedThisSearch / elapsedTimeSeconds;

        // If somehow the nps looks unreasonable then truncate it to zero
        // to avoid making graphs in tournament managers show outlier points.
        const float MAX_NPS = 3_000_000;
        if (nps > MAX_NPS)
        {
          nps = 0;
        }

        //info depth 12 seldepth 27 time 30440 nodes 51100 score cp 105 hashfull 241 nps 1678 tbhits 0 pv e6c6 c5b4 d5e4 d1e1 
        int selectiveDepth = pv.Nodes.Count;
        int depthOfBestMoveInTree = wasInstamove ? thisRootNode.Depth : 0;
        int depth = 1 + (int)MathF.Round(manager.Context.AvgDepth - depthOfBestMoveInTree, 0);

        string pvString = multiPVIndex.HasValue ? $"multipv {multiPVIndex} pv {pv.ShortStr()}"
                                                : $"pv {pv.ShortStr()}";

        int n = thisRootNode.N;
        if (!useParentN && overrideBestMoveNodeAtRoot.IsNotNull) n = overrideBestMoveNodeAtRoot.N;

        //score cp 27 wdl 384 326 290
        string strWDL = "";
        if (showWDL)
        {
          // Note that win and loss inverted to reverse perspective.
          // (except if only one node evaluated so we only have the root and not actual moves).
          bool isRoot = bestMoveNode.IsRoot;
          strWDL = $" wdl {Math.Round((isRoot ? bestMoveNode.WinP : bestMoveNode.LAvg) * 1000)} "
                 + $"{Math.Round((isRoot ? bestMoveNode.DrawP : bestMoveNode.DAvg) * 1000)} "
                 + $"{Math.Round((isRoot ? bestMoveNode.LossP : bestMoveNode.WAvg) * 1000)}";
        }

        if (wasInstamove)
        {
          // Note that the correct tablebase hits cannot be easily calculated and reported
          return $"info depth {depth} seldepth {selectiveDepth} time 0 "
               + $"nodes {n:F0} score cp {scoreToShow}{strWDL} tbhits {manager.CountTablebaseHits} nps 0 "
               + $"{pvString} string M= {thisRootNode.MAvg:F0} ";
        }
        else
        {
          return $"info depth {depth} seldepth {selectiveDepth} time {elapsedTimeSeconds * 1000.0f:F0} "
               + $"nodes {n:F0} score cp {scoreToShow}{strWDL} tbhits {manager.CountTablebaseHits} nps {nps:F0} "
               + $"{pvString} string M= {thisRootNode.MAvg:F0}";
        }
      }
    }


    /// <summary>
    /// Handles special case that move was selected immediately at root from tablebase.
    /// </summary>
    /// <param name="manager"></param>
    /// <param name="searchRootNode"></param>
    /// <param name="scoreAsQ"></param>
    /// <returns></returns>
    static string OutputUCIInfoTablebaseImmediate(MCTSManager manager, MCTSNode searchRootNode, bool scoreAsQ)
    {
      GameResult result = searchRootNode.StructRef.Terminal;

      string scoreStr;
      if (result == GameResult.Checkmate)
      {
        scoreStr = scoreAsQ ? "1.0" : "9999";
      }
      else if (result == GameResult.Draw)
      {
        scoreStr = "0";
      }
      else
      {
        // TODO: cleanup, see comment in MCTSManager.TrySetImmediateBestMove
        //       explaining special meeting of Unknown status to actually mean loss.
        scoreStr = scoreAsQ ? "-1.0" : "-9999";
      }

      string moveStr = manager.TablebaseImmediateBestMove.MoveStr(MGMoveNotationStyle.LC0Coordinate);
      string str = $"info depth 1 seldepth 1 time 0 nodes 1 score cp {scoreStr} pv {moveStr}";
      return str;
    }

  }
}
