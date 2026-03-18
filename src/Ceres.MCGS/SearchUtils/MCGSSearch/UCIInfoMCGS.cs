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
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Search;
using Ceres.MCGS.Search.Coordination;

#endregion

namespace Ceres.MCGS.Utils;

/// <summary>
/// Utility methods relating to UCI output.
/// </summary>
public static class UCIInfoMCGS
{
  static float ScoreToShow(bool scoreAsQ, float thisQ)
  {
    // The score displayed corresponds to
    // the Q (average visit value) of the move to be made.
    float scoreToShow = scoreAsQ ? MathF.Round(thisQ * 1000, 0) : MathF.Round(EncodedEvalLogistic.LogisticToCentipawn(thisQ), 0);

    if (float.IsNaN(scoreToShow))
    {
      // Make sure not to emit NaN for NPS because this might crash
      // some GUIs or tournament managers.
      scoreToShow = 0;
    }
    return scoreToShow;
  }


  /// <summary>
  /// Returns an UCI info string appropriate for a given search state.
  /// </summary>
  /// <param name="manager"></param>
  /// <param name="overrideRootMove"></param>
  /// <returns></returns>
  public static string UCIInfoString(MCGSManager manager,
                                     GNode overrideBestMoveNodeAtRoot = default,
                                     int? multiPVIndex = null,
                                     bool useParentN = true,
                                     bool showWDL = false,
                                     bool scoreAsQ = false,
                                     bool isChess960 = false)
  {
    if (manager.TablebaseImmediateBestMove != default)
    {
      if (multiPVIndex.HasValue && multiPVIndex != 1)
      {
        return null;
      }
      else
      {
        return OutputUCIInfoTablebaseImmediate(manager, manager.Engine.SearchRootNode, scoreAsQ);
      }
    }
    else if (manager.TopVForcedMove != default)
    {
      GNode root = manager.Engine.SearchRootNode;
      float topVScoreToShow = ScoreToShow(scoreAsQ, (float)root.Q);
      string moveStr = manager.TopVForcedMove.MoveStr(MGMoveNotationStyle.Coordinates, isChess960: isChess960);
      string str = $"info depth 1 seldepth 1 time 0 nodes 1 score cp {topVScoreToShow} pv {moveStr}";
      return str;
    }

    bool wasInstamove = manager.StopStatus == MCGSManager.SearchStopStatus.Instamove;

    // If no override bestMoveRoot was specified
    // then it is assumed the move chosen was from the root (not an instamove)
    GNode thisRootNode = manager.Engine.SearchRootNode;

    if (thisRootNode.NumPolicyMoves == 0)
    {
      // Terminal position, nothing to output
      return null;
    }

    float elapsedTimeSeconds = wasInstamove ? 0 : (float)(DateTime.Now - manager.StartTimeThisSearch).TotalSeconds;

    // Avoid showing PV which extends to tiny fraction of the graph
    int minN = (int)Math.Max(1, (int)manager.Engine.SearchRootNode.N * 0.0001f);

    // Get the principal variation (the first move of which will be the best move)
    SearchPrincipalVariationMCGS pv = null;
    try
    {
      pv = new SearchPrincipalVariationMCGS(manager, thisRootNode, overrideBestMoveNodeAtRoot, true, minN);
    }
    catch (Exception ex)
    {
      return ex.StackTrace + System.Environment.NewLine + "UCI Error " +  ex.Message;
    }


    GNodeAndOptionalEdge bestMoveEdge = pv.Nodes[0];

    float thisQ;
    if (!overrideBestMoveNodeAtRoot.IsNull)
    {
      thisQ = (float)-overrideBestMoveNodeAtRoot.Q;
    }
    else if (manager.TopVForcedMove != default || manager.Engine.SearchRootNode.N == 1)
    {
      thisQ = (float)manager.Engine.SearchRootNode.Q;
    }
    else
    {
      thisQ = (float)-bestMoveEdge.Edge.Q;
    }

    // The score displayed corresponds to
    // the Q (average visit value) of the move to be made.
    float scoreToShow = ScoreToShow(scoreAsQ, thisQ);

    if (float.IsNaN(scoreToShow))
    {
      // Make sure not to emit NaN for NPS because this might crash
      // some GUIs or tournament managers.
      scoreToShow = 0;
    }

    float nps = manager.NumNodesVisitedThisSearch / elapsedTimeSeconds;
    float eps = manager.NumEvalsThisSearch / elapsedTimeSeconds; // positions evaluated per second

    //info depth 12 seldepth 27 time 30440 nodes 51100 score cp 105 hashfull 241 nps 1678 tbhits 0 pv e6c6 c5b4 d5e4 d1e1 
    int selectiveDepth = manager.MaxDepth;
    int depth = (int)MathF.Round(manager.AvgDepth);

    string pvString = multiPVIndex.HasValue ? $"multipv {multiPVIndex} pv {pv.ShortStr(isChess960)}"
                                            : $"pv {pv.ShortStr(isChess960)}";

    int n = thisRootNode.N;
    if (!useParentN && !overrideBestMoveNodeAtRoot.IsNull)
    {
      n = overrideBestMoveNodeAtRoot.N;
    }

    ManagerChooseBestMoveMCGS moveChooser = new(manager, thisRootNode, false, default, false);
    BestMoveInfoMCGS bestMoveInfo = moveChooser.BestMoveCalc;

    //score cp 27 wdl 384 326 290
    string strWDL = "";
    if (showWDL)
    {
      // Note that win and loss inverted to reverse perspective.
      // (except if only one node evaluated so we only have the root and not actual moves).
      bool isRoot = bestMoveInfo.BestMoveEdge.IsNull;
      // For root: use ComputeDFromChildren() for fresh D from immediate children.
      // For non-root: use backed-up node.D (running average).
      double displayD = isRoot ? bestMoveEdge.ParentNode.DrawP : bestMoveEdge.ParentNode.ComputeDFromChildren();
      double displayW = isRoot ? bestMoveEdge.ParentNode.WinP : (bestMoveEdge.ParentNode.Q + 1 - displayD) / 2.0;
      double displayL = isRoot ? bestMoveEdge.ParentNode.LossP : (1 - displayD - bestMoveEdge.ParentNode.Q) / 2.0;
      strWDL = $" wdl {Math.Round(displayW * 1000)} "
             + $"{Math.Round(displayD * 1000)} "
             + $"{Math.Round(displayL * 1000)}";
    }


    if (wasInstamove)
    {
      // Note that the correct tablebase hits cannot be easily calculated and reported
      return $"info depth {depth} seldepth {selectiveDepth} time 0 "
           + $"nodes {n:F0} score cp {scoreToShow}{strWDL} tbhits {manager.CountTablebaseHits} nps 0 eps 0"
           + $"{pvString} string M= {thisRootNode.NodeRef.MRaw:F0} ";
    }
    else
    {
      return $"info depth {depth} seldepth {selectiveDepth} time {elapsedTimeSeconds * 1000.0f:F0} "
           + $"nodes {n:F0} score cp {scoreToShow}{strWDL} tbhits {manager.CountTablebaseHits} nps {nps:F0} eps {eps:F0} "
           + $"{pvString} string M= {thisRootNode.NodeRef.MRaw:F0}";
    }

  }



  /// <summary>
  /// Handles special case that move was selected immediately at root from tablebase.
  /// </summary>
  /// <param name="manager"></param>
  /// <param name="searchRootNode"></param>
  /// <param name="scoreAsQ"></param>
  /// <returns></returns>
  static string OutputUCIInfoTablebaseImmediate(MCGSManager manager, GNode searchRootNode, bool scoreAsQ)
  {
    GameResult result = searchRootNode.Terminal;

    string scoreStr;
    if (result == GameResult.Checkmate)
    {
      string signChar = searchRootNode.Q > 0 ? "" : "-";
      scoreStr = signChar + (scoreAsQ ? "1.0" : "9999");
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

    string moveStr = manager.TablebaseImmediateBestMove.MoveStr(MGMoveNotationStyle.Coordinates);
    string str = $"info depth 1 seldepth 1 time 0 nodes 1 score cp {scoreStr} pv {moveStr}";
    return str;
  }
}