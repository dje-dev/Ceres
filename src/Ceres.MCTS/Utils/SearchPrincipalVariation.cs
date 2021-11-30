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


using Ceres.Base.Math;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Analysis;
using Ceres.MCTS.MTCSNodes.Annotation;
using Ceres.MCTS.MTCSNodes.Struct;
using System;
using System.Collections.Generic;
using System.Text;

#endregion

namespace Ceres.MCTS.Utils
{
  /// <summary>
  /// The principal variation of a search tree.
  /// </summary>
  public class SearchPrincipalVariation
  {
    /// <summary>
    /// Search root for the PV.
    /// </summary>
    public readonly MCTSNode SearchRoot;

    /// <summary>
    /// Set of nodes comprising the PV.
    /// </summary>
    public readonly List<MCTSNode> Nodes;

    public static int? IndexOfChildWithBestQSlope(MCTSNode rootNode, int minN, float minFraction)
    {
      float bestSlope = float.MinValue;
      int? bestIndex = null;
        
      using (new SearchContextExecutionBlock(rootNode.Context))
      {
        float rootQ = (float)rootNode.Q;
        for (int i=0; i<rootNode.NumChildrenExpanded;i++)
        {
          MCTSNode node = rootNode.ChildAtIndex(i);
          if (node.N > minN)
          {
            SearchPrincipalVariation spv = new SearchPrincipalVariation(node);
            float pvSlope = spv.PVSlopeQ(rootQ, minFraction);
            if (!float.IsNaN(pvSlope) && pvSlope > bestSlope)
            {
              bestIndex = i;
              bestSlope = pvSlope;
            }
          }
        }
        return bestIndex;
      }
    }

    public static void DumpChildQSlopes(MCTSNode rootNode)
    {
      using (new SearchContextExecutionBlock(rootNode.Context))
      {
        float rootQ = (float)rootNode.Q;
        foreach (MCTSNode node in rootNode.ChildrenSorted(n => -n.N))
        {
          if (node.N > 100)
          {
            SearchPrincipalVariation spv = new SearchPrincipalVariation(node);
            Console.WriteLine("\r\n");
            Console.WriteLine((spv.PVSlopeQ(rootQ, 0.05f) * 100) + " " + node);
            MCTSPosTreeNodeDumper.DumpPV(node, true);
          }
        }
      }
    }

    public float PVSlopeQ(float priorQ, float minNFraction)
    {
      int i = 0;
      List<float> x = new();
      List<float> y = new();

      if (!float.IsNaN(priorQ))
      {
        x.Add(i++);
        y.Add(priorQ);
      }
      foreach (MCTSNode node in Nodes)
      {
        float frac = (float)node.N / Nodes[0].N;

//        Console.WriteLine(frac + " " + node.N + " " + LinearRegression(x.ToArray(), y.ToArray()).slope);
        
        if (frac < minNFraction)
        {
          break;
        }
        x.Add(i++);
        y.Add((float)node.Q * (node.IsOurMove ? -1f : 1f));
      }

      var regression = StatUtils.LinearRegression(x.ToArray(), y.ToArray());
      return -regression.slope;
    }


    public SearchPrincipalVariation(MCTSNode searchRoot, MCTSNode overrideBestMoveNodeAtRoot = default)
    {
      Nodes = new List<MCTSNode>();

      // Omit positions with low N on both absolute and relative basis (too noisy)
      int totalN = searchRoot.N;
      MCTSNode node = searchRoot;
      do
      {
        // 
        float fracN = (float)node.N / totalN;
        const float CUTOFF_FRACTION = 0.0005f;
        if (totalN > 1
         && node.Terminal == Chess.GameResult.Unknown 
         && node.N < 1000 
         && fracN < CUTOFF_FRACTION)
        {
          break;
        }
        searchRoot.Tree.Annotate(node);

        Nodes.Add(node);

        if (node.NumPolicyMoves > 0)
        {
          if (node == searchRoot)
          {
            // Apply special logic at root for best move
            node = overrideBestMoveNodeAtRoot.IsNotNull ? overrideBestMoveNodeAtRoot : node.BestMove(false);
          }
          else
          {
            // Non-root nodes follow visits with maximum number of visits.
            node = node.ChildWithLargestValue(n => n.N);
          }
        }
        else
          node = default;
      } while (node.IsNotNull);
    }


    /// <summary>
    /// A short descriptive string listing the PV moves (suitable for UCI output).
    /// </summary>
    /// <returns></returns>
    public string ShortStr()
    {
      StringBuilder sb = new StringBuilder();

      bool haveSkippedSearchRoot = false;
      foreach (MCTSNode node in Nodes)
      {
        if (haveSkippedSearchRoot)
        {
          sb.Append(node.Annotation.PriorMoveMG.MoveStr(MGMoveNotationStyle.LC0Coordinate) + " ");
        }
        else
        {
          haveSkippedSearchRoot = true;
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
        if (node.Parent.IsNotNull && node.Parent.N > 10)
        {
          //node.Tree.Annotate(node);
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
      foreach (MCTSNodeStructChild child in node.StructRef.Children)
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