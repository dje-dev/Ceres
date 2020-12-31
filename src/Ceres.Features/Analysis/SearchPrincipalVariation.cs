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
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Annotation;
using Ceres.MCTS.MTCSNodes.Struct;
using System;
using System.Collections.Generic;
using System.Text;

#endregion

namespace Ceres.Features.Analysis
{
  /// <summary>
  /// The principal variation of a search tree.
  /// </summary>
  public class SearchPrincipalVariation
  {
    public readonly MCTSNode Root;
    public readonly List<MCTSNode> Nodes;

    public SearchPrincipalVariation(MCTSNode root)
    {
      Nodes = new List<MCTSNode>();

      MCTSNode node = root;
      do
      {
        root.Context.Tree.Annotate(node);

        Nodes.Add(node);

        if (node.NumPolicyMoves > 0)
        {
          if (node.IsRoot)
          {
            // Apply special logic at root for best move
            node = node.BestMove(false);
          }
          else
          {
            // Non-root nodes follow visits with maximum number of visits.
            node = node.ChildWithLargestValue(n => n.N);
          }
        }
        else
          node = null;
      } while (node != null);

    }


    public string ShortStr()
    {
      StringBuilder sb = new StringBuilder();

      foreach (MCTSNode node in Nodes)
      {
        if (!node.IsRoot)
        {
          sb.Append(node.Annotation.PriorMoveMG.MoveStr(MGMoveNotationStyle.LC0Coordinate) + " ");
        }
      }
      return sb.ToString();
    }


    public float AverageM()
    {
      float acc = 0;
      int count = 0;
      int depth = 0;
      Console.WriteLine();
      foreach (MCTSNode node in Nodes)
      {
        if (node.Parent != null && node.Parent.N > 10)
        {
          //node.Context.Tree.Annotate(node);
          //Console.WriteLine(node.Annotation.Pos.FEN + " " + node.MPosition);
          acc = (node.MPosition + depth);
          count++;
        }
        depth++;
      }

      return acc / count;
    }


    float MoveDistributionEntropy(MCTSNode node)
    {
      int totalChildren = node.N;
      float acc = 0;
      foreach (MCTSNodeStructChild child in node.Ref.Children)
      {
        if (child.IsExpanded)
          acc += MathF.Log(child.N / node.N, 2.0f);
      }
      return -acc;    
    }


    public float Positionality
    {
      get
      {
        return 0;
        float acc = 0;
        float depth = 1;
        float denominator = 0;
        foreach (MCTSNode node in Nodes)
        {
          float weight = 1.0f / (float)depth;
          denominator += weight;
          acc += weight * MoveDistributionEntropy(node);
          depth++;
        }

        float positionality = acc / denominator;
        return positionality;
      }
    }

  }
}
