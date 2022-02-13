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
using System.Collections.Generic;
using System.IO;
using System.Text;

using Ceres.Base.Benchmarking;
using Ceres.Base.DataType.Trees;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.LC0.Positions;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.Specifications;
using Ceres.Chess.Positions;

using Ceres.MCTS.Iteration;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Utils;

#endregion

namespace Ceres.Features.Visualization.AnalysisGraph
{
  public class AnalysisGraphGenerator
  {
    const bool VERBOSE = false;
    public readonly int MaxDepth = 30;
    const string BASE_FN = "CeresGraph";

    public readonly AnalysisGraphOptions Options;

    /// <summary>
    /// The temporary directory in which all output files are placed.
    /// </summary>
    public readonly string TempDir;

    // bundle svg images into 1
    // https://stackoverflow.com/questions/56484196/how-can-i-bundle-many-svg-images-inside-just-one
    //https://stackoverflow.com/questions/8028746/embedding-svg-in-svg


    /// <summary>
    /// Constructor for SVG generator from a specified positions with specified options.
    /// </summary>
    /// <param name="search"></param>
    /// <param name="options"></param>
    public AnalysisGraphGenerator(MCTSearch search, AnalysisGraphOptions options)
    {
      Options = options;
      analysisPosition = search.Manager.Context.Tree.Store.Nodes.PriorMoves;
      specNet = search.Manager.Context.EvaluatorDef;
      this.search = search;
      spv = new SearchPrincipalVariation(search.SearchRootNode);
      tree = search.Manager.Context.Tree;

      // These parameters tuned to give good user experience,
      // with level 0 very simple and level 9 showing much detail.
      float SCALE = 3 * (10 - Options.DetailLevel);
      minFractionRoot = 0.005f * SCALE;
      minFractionParent = 0.01f * SCALE;

      if (VERBOSE)
      {
        Console.WriteLine($"Min fraction parent {100 * minFractionParent:F0}% root {100 * minFractionRoot:F0}%");
      }

      // Graph generation visits potentially many nodes needing annotation,
      // therefore make sure cache does not happen to already be almost full.
      tree.PossiblyPruneCache();

      TempDir = GraphvizUtils.CreateUniqueTempDirectory("Ceres_Graph_");
    }

    float minFractionRoot;
    float minFractionParent;


    PositionWithHistory analysisPosition;
    NNEvaluatorDef specNet;
    MCTSearch search;
    MCTSTree tree;
    SearchPrincipalVariation spv;

    /// <summary>
    /// StringBuilder that accumulates generated text.
    /// </summary>
    StringBuilder sb = new();

    /// <summary>
    /// Set of nodes added so far to graph.
    /// </summary>
    HashSet<int> addedNodes = new();


    /// <summary>
    /// Gererates SVG with analysis details, returning FN of generated file.
    /// </summary>
    /// <param name="launchWithBrowser"></param>
    /// <returns></returns>
    public string Write(bool launchWithBrowser)
    {
      WriteHeader();

      TimingStats timingStatsGen = new TimingStats();
//      using (new TimingBlock(timingStatsGen))
      {
        // Write root node only
        WriteNodes(ref search.SearchRootNode.StructRef, sb, (in MCTSNodeStruct n, int d) => d == 0, (in MCTSNodeStruct n, int d) => d == 0);

        // Write top-level move nodes 
        foreach (var child in tree.Root.ChildrenExpanded)
        {
          child.Annotate();
          int index = child.IndexInParentsChildren;

          StringBuilder thisSB = new();
          int numWritten = WriteNodes(ref child.StructRef, thisSB,
                                      (in MCTSNodeStruct n, int d) => true,
                                      (in MCTSNodeStruct n, int d) =>
                                      {
                                        if (d == 1)
                                        {
                                          // Don't process if this is a top-level move
                                          // other than one being processed in this iteration.
                                          return n.Index.Index != child.Index;
                                        }
                                        else
                                        {
                                          if (d > MaxDepth) return false;

                                          float nFractionParent = n.IsRoot ? 1 : (float)n.N / n.ParentRef.N;
                                          if (nFractionParent < minFractionParent) return false;

                                          if ((float)n.N / tree.Root.N < minFractionRoot) return false;
                                          return true;
                                        }
                                      });

          if (numWritten > 0)
          {
            string label = MGMoveConverter.ToMove(child.Annotation.PriorMoveMG).ToSAN(in child.Parent.Annotation.Pos);
            float scoreCP = -EncodedEvalLogistic.LogisticToCentipawn((float)child.Q);
            float fracRoot = (float)child.N / child.Parent.N;
            label = label + $" ({scoreCP:F0}cp, {100.0f * fracRoot:F0}% visits)";
            bool isOnPV = spv.Nodes.Contains(child);
            string color = isOnPV ? "#f7fcf0" : "whitesmoke"; // very slight light green background if best move
            sb.AppendLine($"subgraph cluster_move_" + index + " {" + $" label=\"{label}\" bgcolor=\"{color}\"");
            sb.Append(thisSB);
            sb.AppendLine("}");

            if (VERBOSE)
            {
              Console.WriteLine(child.Parent.Index + " ==> " + child.Index);
            }

            WriteEdgeFromParent(sb, child, isOnPV);
          }
        }
      }

      if (Options.ShowTranspositions)
      {
        // Add transposition edges
        foreach (int nodeIndex in addedNodes)
        {
          var node = search.Manager.Context.Tree.GetNode(new MCTSNodeStructIndex(nodeIndex));
          foreach (int transpositionNodeIndex in GetRelated(nodeIndex, node.StructRef.ZobristHash))
          {
            sb.AppendLine(nodeIndex + "->" + transpositionNodeIndex + " [arrowhead=\"empty\"]");
          }
        }
      }

      WriteFooter();

      // Covvert to SVG.
      GraphvizUtils.Convert(sb.ToString(), BASE_FN, TempDir);
      string fn = Path.Combine(TempDir, $"{BASE_FN}.svg");

      if (launchWithBrowser)
      {
        StringUtils.LaunchBrowserWithURL(fn);
      }
    
      return fn;
    }


    delegate bool PredicateNode(in MCTSNodeStruct nodeRef, int depth);

    void WriteHeader()
    {
      string specString = NNNetSpecificationString.ToSpecificationString(specNet.NetCombo, specNet.Nets);
      string posString = analysisPosition.FENAndMovesString.Replace(Position.StartPosition.FEN, "startpos");
      string searchNodesString = $" ({tree.Root.N:N0} visits)";
      string title = $"Ceres ({CeresVersion.VersionString}) Search Visualization\r\n{specString}{searchNodesString}\r\n{posString}\r\n ";

      sb.AppendLine("digraph \"title\" ");
      //      sb.AppendLine("{ nodesep=2 ranksep=1 "); //compound=false;
      sb.AppendLine("{ nodesep=1.5 ranksep=0.5 "); //compound=false;
      sb.AppendLine($"bgcolor=\"whitesmoke\" rankdir=LR; fontsize=\"40\" labelloc=\"t\" label=\"{title}\"");
    }

    void WriteFooter()
    {
      sb.AppendLine("}");
    }

    private int WriteNodes(ref MCTSNodeStruct rootNodeRef, StringBuilder outSB,
                            PredicateNode filterPredicate, PredicateNode hardPredicate)
    {
      int count = 0;
      int rootN = search.Manager.Root.N;

      rootNodeRef.Traverse(
      search.SearchRootNode.Context.Tree.Store,
      (ref MCTSNodeStruct nodeRef, int depth) =>
      {
        if (!hardPredicate(in nodeRef, depth))
        {
          return false;
        }
        else if (!filterPredicate(in nodeRef, depth))
        {
          return true;
        }

        MCTSNode node = tree.GetNode(nodeRef.Index);
        node.Annotate();

        bool isOnPV = spv.Nodes.Contains(node);

        float nFractionParent = node.IsRoot ? 1 : (float)node.N / node.Parent.N;
        float nFractionRoot = (float)node.N / rootN;
        bool isTopN = IsTopN(node);
        //node.IsRoot ? true : node.Parent.ChildrenSorted(n => -n.N)[0] == node;

        if (nFractionRoot > minFractionRoot
        && (isTopN || nFractionParent > minFractionParent))
        {
          AddNode(outSB, node, isOnPV, TempDir);
          count++;
          addedNodes.Add(node.Index);
          if (VERBOSE)
          {
            Console.WriteLine("add " + node.Index + " depth " + node.Depth);
          }
          return true;
        }
        return false;
      }, TreeTraversalType.BreadthFirst);

      return count;
    }


    /// <summary>
    /// Enumerates over all addedNodes having specified hash
    /// (but excluding any nodes with a specified skipIndex).
    /// </summary>
    /// <param name="skipIndex"></param>
    /// <param name="hash"></param>
    /// <returns></returns>
    private IEnumerable<int> GetRelated(int skipIndex, ulong hash)
    {
      foreach (int nodeIndex in addedNodes)
      {
        if (nodeIndex != skipIndex)
        {
          var node = search.Manager.Context.Tree.GetNode(new MCTSNodeStructIndex(nodeIndex));
          if (node.StructRef.ZobristHash == hash)
          {
            yield return node.Index;
          }
        }
      }
      yield break;
    }


    #region Static helpers

    /// <summary>
    /// Adds a node to the graph.
    /// </summary>
    /// <param name="sb"></param>
    /// <param name="node"></param>
    /// <param name="isOnPV"></param>
    /// <param name="tempDir"></param>
    static void AddNode(StringBuilder sb, MCTSNode node, bool isOnPV, string tempDir)
    {
      node.Annotate();

      string tooltip = GetNodeTooltip(node);
      if (!node.IsRoot && node.Depth > 1)
      {
        WriteEdgeFromParent(sb, node, isOnPV);
      }

      Move move = default;
      if (!node.IsRoot)
      {
        move = MGMoveConverter.ToMove(node.Annotation.PriorMoveMG);
      }
      string posFN = GraphvizUtils.WritePosToSVGFile(node.Annotation.Pos, node.StructRef.ZobristHash, node.Index, move, tempDir);


      sb.AppendLine($"{node.Index} [shape=none");
      sb.AppendLine($"href=\"https://lichess.org/editor/{node.Annotation.Pos.FEN}\"");
      sb.AppendLine($"tooltip=\"{tooltip}]\"");
      float scoreCP = EncodedEvalLogistic.LogisticToCentipawn((float)node.Q);
      string wdlStr = $"{100 * node.StructRef.WAvg:F0}/{100 * node.DAvg:F0}/{100 * node.LAvg:F0})";

      //  sb.AppendLine($"xlabel=\"{scoreCP,5:F0}\"");

      sb.Append($" fontsize=\"18\"");

      //<<table border="0" cellspacing="0">
      string colorAttr = node.Annotation.Pos.MiscInfo.SideToMove == SideType.White ? "bgcolor=\"white\"" : "bgcolor=\"silver\"";
      sb.AppendLine("label= <<TABLE cellspacing=\"0\">");
      sb.AppendLine($"<TR><TD border=\"0\" {colorAttr} ><IMG SRC=\"{posFN}\"/></TD></TR>");
      string bgColorScore = GraphvizUtils.ColorStr((float)node.Q);

      sb.AppendLine($"<TR><TD bgcolor=\"linen\" border=\"0\" >{scoreCP,5:F0}cp   ({wdlStr}</TD></TR>");
      sb.AppendLine($"<TR><TD bgcolor=\"linen\" border=\"0\" >N= {node.N:N0}  P={node.P * 100,5:F2}</TD></TR>");

      double qDiffRoot = node.Tree.Root.Q - (node.Depth % 2 == 0 ? 1 : -1) * node.Q;
      sb.AppendLine($"<TR><TD bgcolor=\"{bgColorScore}\" border=\"0\" >Q= {node.Q,6:F3} ({ qDiffRoot,4:F2})  V= {node.V,6:F3}</TD></TR>");
      //sb.AppendLine($"<TR><TD>V= {node.V,6:F3}</TD></TR>");
      sb.AppendLine("</TABLE>>]");
    }


    /// <summary>
    /// Writes the edge from a node to its parent.
    /// </summary>
    /// <param name="sb"></param>
    /// <param name="node"></param>
    /// <param name="isOnPV"></param>
    private static void WriteEdgeFromParent(StringBuilder sb, MCTSNode node, bool isOnPV)
    {
      sb.Append(node.ParentIndex.Index + "->" + node.Index);

      float nFractionParent = (float)node.N / node.Parent.N;
      //float qDelta = (float)node.Q - -node.Parent.N;
      sb.Append("[ ");

      // Output move number and move played
      int moveNum = node.Annotation.Pos.MiscInfo.MoveNum;
      string moveNumStr = (moveNum / 2) + (moveNum % 2 == 0 ? ".. " : ". ");
      string san = node.IsRoot ? "" : MGMoveConverter.ToMove(node.Annotation.PriorMoveMG).ToSAN(in node.Parent.Annotation.Pos);

      sb.AppendLine($"xlabel=\" {moveNumStr}{san}\"");

      // Fraction of visits to this node.
      sb.Append($"label=\"{nFractionParent * 100,4:F0}%  \"");

      if (isOnPV)
      {
        sb.Append($" color=red");
      }

      GraphvizUtils.WriteArrowheadForFractionParent(sb, nFractionParent);
      sb.AppendLine("]");
    }


    /// <summary>
    /// Returns if a specified node is one of the topN nodes within the parent (by N).
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    bool IsTopN(MCTSNode node)
    {
      if (node.IsRoot)
      {
        return true;
      }
      else
      {
        for (int i = 0; i < node.Parent.NumChildrenExpanded; i++)
        {
          if (node.Parent.StructRef.ChildAtIndexRef(i).N > node.N)
          {
            // Found on larger, we cannot possily be largest.
            return false;
          }
        }
        return true;
      }
    }

    static string GetNodeTooltip(MCTSNode node)
    {
      // Show detail on all possible moves
      StringWriter strWriter = new StringWriter();
      node.Dump(node.Depth + 1, int.MaxValue, 0, null, null, int.MaxValue, strWriter, node.Context.Tree.Root, true);
      return GraphvizUtils.ConvertedCRLF(strWriter.ToString());
    }
  }

  #endregion
}
