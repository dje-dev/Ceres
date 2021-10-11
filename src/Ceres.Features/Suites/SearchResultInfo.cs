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

using Ceres.Chess.MoveGen;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;
using System;

#endregion

namespace Ceres.Features.Suites
{
  public class SearchResultInfo
  {
    public readonly double Q;
    public readonly string UCIInfoString;
    public readonly MGMove BestMove;
    public readonly int N;
    public readonly int NumNodesWhenChoseTopNNode;
    public readonly int NumNNBatches;
    public readonly int NumNNNodes;
    public readonly int TopNNodeN;
    public readonly float FractionNumNodesWhenChoseTopNNode;
    public readonly float AvgDepth;
    public readonly float MAvg;
    public readonly float NodeSelectionYieldFrac;
    public readonly string PickedNonTopNMoveStr;

    public SearchResultInfo(MCTSManager manager, BestMoveInfo bestMove)
    {
      using (new SearchContextExecutionBlock(manager.Context))
      {
        Q = manager.Root.Q;
        //UCIInfoString = manager.UCIInfoString();
        // SearchPrincipalVariation pv1 = new SearchPrincipalVariation(worker1.Root);
        BestMove = bestMove.BestMove;
        N = manager.Context.Root.N;
        NumNodesWhenChoseTopNNode = manager.NumNodesWhenChoseTopNNode;
        NumNNBatches = manager.Context.NumNNBatches;
        NumNNNodes = manager.Context.NumNNNodes;

        TopNNodeN = manager.TopNChildIndex is null ? 0 : manager.TopNChildN;
        FractionNumNodesWhenChoseTopNNode = manager.FractionNumNodesWhenChoseTopNNode;
        AvgDepth = manager.Context.AvgDepth;
        MAvg = manager.Context.Root.MAvg;
        NodeSelectionYieldFrac = manager.Context.NodeSelectionYieldFrac;

        PickedNonTopNMoveStr = bestMove.BestMoveWasTopN ? " " : "!";

      }
    }
  }

}
