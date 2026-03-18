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
using System.Collections.Generic;
using System.Text;

using Ceres.Chess.MoveGen;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Search;
using Ceres.MCGS.Search.Coordination;

#endregion

namespace Ceres.MCGS.Utils;

/// <summary>
/// The principal variation of a search tree.
/// </summary>
public class SearchPrincipalVariationMCGS
{
  /// <summary>
  /// Parent search manager.
  /// </summary>
  public readonly MCGSManager Manager;

  /// <summary>
  /// Search root for the PV.
  /// </summary>
  public readonly GNode SearchRoot;

  /// <summary>
  /// Set of edges comprising the PV.
  /// </summary>
  public readonly List<GNodeAndOptionalEdge> Nodes;


  public SearchPrincipalVariationMCGS(MCGSManager manager,
                                      GNode searchRoot,
                                      GNode overrideBestMoveNodeAtRoot = default,
                                      bool startFromRoot = false,
                                      int minN = 1)
  {
    Manager = manager;

    HashSet<int> useNodes = startFromRoot ? MCGSPosGraphNodeDumper.NodesToRootSet(searchRoot) : null;


    Nodes = [];

    // Omit positions with low N on both absolute and relative basis (too noisy)
    int totalN = searchRoot.N;

    if (!startFromRoot) throw new NotImplementedException();
    //GNode node = startFromRoot ? searchRoot.Tree.Root : searchRoot;

    GNode node = searchRoot;
    GEdge edge = default;

    // Track visited node indices to detect cycles (especially in PositionEquivalence mode).
    HashSet<int> visitedNodeIndices = new();

#if DEBUG
    HashSet<PosHash64> seenStandaloneHashes = new();
#endif

    do
    {
      // In PositionEquivalence mode, paths can lead back to the same node (forming cycles).
      // When a repeated node is encountered, terminate the PV since this indicates a draw by repetition.
      if (!visitedNodeIndices.Add(node.Index.Index))
      {
        // Node already visited - this is a repetition, stop building PV here.
        break;
      }

      if (node.NumPolicyMoves > 0)
      {
#if DEBUG
        if (!seenStandaloneHashes.Add(node.HashStandalone))
        {
          Console.WriteLine("PV had a repitition!");
        }
#endif

        if (node.N < minN || (node.NumEdgesExpanded == 0 && !node.IsSearchRoot))
        {
          break;
        }

        if (node == searchRoot)
        {
          // Apply special logic at root for best move
          if (!overrideBestMoveNodeAtRoot.IsNull)
          {
            edge = node.EdgeForNode(overrideBestMoveNodeAtRoot);
            Nodes.Add(new GNodeAndOptionalEdge(node, edge));
            node = overrideBestMoveNodeAtRoot;
          }
          else
          {
            ManagerChooseBestMoveMCGS bm = new ManagerChooseBestMoveMCGS(Manager, node, false, default, false);
            edge = bm.BestMoveCalc.BestMoveEdge;
            Nodes.Add(new GNodeAndOptionalEdge(node, edge));
            node = bm.BestMoveCalc.BestMoveEdge.ChildNode;
          }
        }
        else
        {
          GEdge mustVisitEdge = default;
          if (useNodes != null)
          {
            foreach (GEdge nodeChild in node.ChildEdgesExpanded)
            {
              if (useNodes.Contains(nodeChild.ChildNode.Index.Index))
              {
                mustVisitEdge = nodeChild;
              }
            }
          }

          // Non-root nodes follow visits with maximum number of visits.
          // N.B. Use simple max N node because BestMove
          // can trigger CreateNode which is problematic in some contexts.
          // TODO: improve this, remove CreateNode logic there?
          GEdge[] childrenSortedN = node.EdgesSorted(node => -node.N);

          edge = mustVisitEdge.IsNull ? childrenSortedN[0] : mustVisitEdge;
          Nodes.Add(new GNodeAndOptionalEdge(node, edge));

          node = edge.ChildNode;
        }
      }
      else
      {
        node = default;
      }

      if (Nodes.Count > 500)
      {
        // Safeguard against infinite loops (should not normally be reached
        // since cycle detection above should catch repetitions first).
        throw new Exception("SearchPrincipalVariationMCGS: exceeded 500 edges, apparent cycle.");
      }

    } while (!node.IsNull);
  }


  /// <summary>
  /// A short descriptive string listing the PV moves (suitable for UCI output).
  /// </summary>
  /// <returns></returns>
  public string ShortStr(bool isChess960 = false)
  {
    if (Nodes.Count == 1)
    {
      ManagerChooseBestMoveMCGS bm = new(Manager, Nodes[0].ParentNode, false, default, false);
      //  public BestMoveInfoMCGS BestMoveInfo(bool updateStatistics, MGMove forcedMove = default)
      BestMoveInfoMCGS bestMoveInfo = bm.BestMoveCalc;

      //      return new ManagerChooseBestMoveMCGS(this, updateStatistics, ParamsSearch.MLHBonusFactor, forcedMove).BestMoveCalc;

      // In the special case of only root evaluated, show the best policy move so the pv is not completely empty.
      // Otherwise, we choose not to show the best policy move at the terminal node.        
      return bestMoveInfo.BestMove.MoveStr(MGMoveNotationStyle.Coordinates, isChess960: isChess960);
    }

    StringBuilder sb = new();

    foreach (GNodeAndOptionalEdge nodeEdge in Nodes)
    {
      if (nodeEdge.HasEdge && !nodeEdge.Edge.IsNull)
      {
        sb.Append(nodeEdge.Edge.MoveMG.MoveStr(MGMoveNotationStyle.Coordinates, isChess960) + " ");
      }
    }
    return sb.ToString();
  }


  public float AverageM()
  {
    throw new NotImplementedException();
  }


}
