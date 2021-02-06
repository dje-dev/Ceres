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
using Ceres.Chess.LC0.Positions;

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
                                       MCTSNode overrideRootMove = null,
                                       MCTSNode overrideBestMoveNodeAtRoot = null,
                                       int? multiPVIndex = null,
                                       bool useParentN = true,
                                       bool showWDL = false,
                                       bool scoreAsQ = false)
    {
      bool wasInstamove = manager.Root != overrideRootMove;

      // If no override bestMoveRoot was specified
      // then it is assumed the move chosen was from the root (not an instamove)
      MCTSNode thisRootNode = overrideRootMove ?? manager.Root;

      float elapsedTimeSeconds = wasInstamove ? 0 : (float)(DateTime.Now - manager.StartTimeThisSearch).TotalSeconds;

      float scoreToShow;
      if (scoreAsQ)
      {
        scoreToShow = MathF.Round((float)thisRootNode.Q * 1000, 0);
      }
      else
      { 
        scoreToShow = MathF.Round(EncodedEvalLogistic.LogisticToCentipawn((float)thisRootNode.Q), 0);
      }

      float nps = manager.NumStepsTakenThisSearch / elapsedTimeSeconds;

      // Get the principal variation (the first move of which will be the best move)
      SearchPrincipalVariation pv;
      using (new SearchContextExecutionBlock(manager.Context))
      {
        pv = new SearchPrincipalVariation(thisRootNode, overrideBestMoveNodeAtRoot);
      }

      //info depth 12 seldepth 27 time 30440 nodes 51100 score cp 105 hashfull 241 nps 1678 tbhits 0 pv e6c6 c5b4 d5e4 d1e1 
      int selectiveDepth = pv.Nodes.Count;
      int depthOfBestMoveInTree = wasInstamove ? thisRootNode.Depth : 0;
      int depth = 1 + (int)MathF.Round(manager.Context.AvgDepth - depthOfBestMoveInTree, 0);

      string pvString = multiPVIndex.HasValue ? $"multipv {multiPVIndex} pv {pv.ShortStr()}"
                                              : $"pv {pv.ShortStr()}";

      int n = thisRootNode.N;
      if (!useParentN && overrideBestMoveNodeAtRoot != null) n = overrideBestMoveNodeAtRoot.N;

      //score cp 27 wdl 384 326 290
      string strWDL = "";
      if (showWDL)
      {
        strWDL = $" wdl {Math.Round(thisRootNode.WinP * 1000)} " 
               + $"{Math.Round(thisRootNode.DrawP * 1000)} " 
               + $"{Math.Round(thisRootNode.LossP * 1000)}";
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
}
