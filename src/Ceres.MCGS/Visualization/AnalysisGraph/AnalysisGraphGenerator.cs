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

using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.LC0.Positions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.Specifications;
using Ceres.Chess.Positions;
using Ceres.Chess.SearchResultVerboseMoveInfo;
using Ceres.Chess.UserSettings;
using Ceres.Chess.Visualization;
using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Utils;

#endregion

namespace Ceres.MCGS.Visualization.AnalysisGraph;

/// <summary>
/// Generates interactive SVG search-graph visualizations via Graphviz
/// for MCGS (graph-based) search results.
/// </summary>
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

  /// <summary>
  /// Constructor for SVG generator from a specified MCGS search with specified options.
  /// </summary>
  public AnalysisGraphGenerator(MCGSSearch search, AnalysisGraphOptions options)
  {
    Options = options;
    analysisPosition = search.Manager.StartPosAndPriorMoves;
    specNet = search.Manager.EvaluatorDef;
    this.search = search;
    graph = search.Manager.Engine.Graph;
    searchRoot = search.SearchRootNode;

    SearchPrincipalVariationMCGS spv = new SearchPrincipalVariationMCGS(
      search.Manager, searchRoot, startFromRoot: true);
    pvNodeIndices = new HashSet<int>();
    foreach (GNodeAndOptionalEdge nodeEdge in spv.Nodes)
    {
      pvNodeIndices.Add(nodeEdge.ParentNode.Index.Index);
      if (nodeEdge.HasEdge && nodeEdge.Edge.IsExpanded)
      {
        pvNodeIndices.Add(nodeEdge.Edge.ChildNode.Index.Index);
      }
    }

    if (options.RelativeTimeReferenceEngine > 0 && referenceEngine == null)
    {
      TryStartReferenceEngine();
    }

    // These parameters tuned to give good user experience,
    // with level 0 very simple and level 9 showing much detail.
    float SCALE = 3 * (10 - Options.DetailLevel);
    minFractionRoot = 0.005f * SCALE;
    minFractionParent = 0.01f * SCALE;

    if (VERBOSE)
    {
      Console.WriteLine($"Min fraction parent {100 * minFractionParent:F0}% root {100 * minFractionRoot:F0}%");
    }

    TempDir = GraphvizUtils.CreateUniqueTempDirectory("Ceres_Graph_");
  }


  static bool IsMCTSReferenceEngine =>
    CeresUserSettingsManager.Settings.RefEngineEXE?.ToUpper() == "CERES" ||
    CeresUserSettingsManager.Settings.RefEngineEXE?.ToUpper() == "LC0";


  /// <summary>
  /// Optional delegate that callers can set to provide a reference engine factory.
  /// The delegate receives the NNEvaluatorDef from the current search
  /// and should return a ready-to-use GameEngine (or null if unavailable).
  /// </summary>
  public static Func<NNEvaluatorDef, GameEngine> ReferenceEngineFactory { get; set; }


  private void TryStartReferenceEngine()
  {
    if (referenceEngine == null)
    {
      if (ReferenceEngineFactory != null)
      {
        referenceEngine = ReferenceEngineFactory(search.Manager.EvaluatorDef);
      }
      else if (CeresUserSettingsManager.Settings.RefEngineEXE == null)
      {
        Console.WriteLine("Required entry for RefEngineEXE missing in Ceres.json (or set AnalysisGraphGenerator.ReferenceEngineFactory).");
      }
      else if (!File.Exists(CeresUserSettingsManager.Settings.RefEngineEXE))
      {
        Console.WriteLine($"Specified file for RefEngineEXE of {CeresUserSettingsManager.Settings.RefEngineEXE} does not exist.");
      }
      else
      {
        Console.Write($"Starting reference engine {CeresUserSettingsManager.Settings.RefEngineEXE} ");
        Console.WriteLine($"with {CeresUserSettingsManager.Settings.RefEngineThreads} threads "
                        + $"and {CeresUserSettingsManager.Settings.RefEngineHashMB}mb hash...");

#if NOT
        Console.Write($"Starting reference engine {CeresUserSettingsManager.Settings.RefEngineEXE} ");
        if (!isLC0 && !isCeres)
        {
        Console.WriteLine($"with {CeresUserSettingsManager.Settings.RefEngineThreads} threads "
                        + $"and {CeresUserSettingsManager.Settings.RefEngineHashMB}mb hash...");
        }
        else
        {
          Console.WriteLine();
        }

        NNEvaluatorDef GetNNDef()
        {
          NNEvaluatorDef nnDef = search.Manager.EvaluatorDef;
          if (CeresUserSettingsManager.Settings.RefEngineNetworkFile != null)
          {
            string deviceString = NNDevicesSpecificationString.ToSpecificationString(nnDef.DeviceCombo, nnDef.Devices);
            deviceString = deviceString.Replace("Device=", "");
            nnDef = NNEvaluatorDef.FromSpecification(CeresUserSettingsManager.Settings.RefEngineNetworkFile, deviceString);
          }
          return nnDef;
        }

        if (isLC0)
        {
          referenceEngine = new GameEngineDefLC0("LC0", GetNNDef(), true, null, verbose: true).CreateEngine();
        }
        else if (isCeres)
        {
          referenceEngine = new GameEngineDefCeres("Ceres", GetNNDef(), null, null).CreateEngine();
        }
        else
        {
#endif

        GameEngineUCISpec spec = new("Ref",
                                     CeresUserSettingsManager.Settings.RefEngineEXE,
                                     CeresUserSettingsManager.Settings.RefEngineThreads,
                                     CeresUserSettingsManager.Settings.RefEngineHashMB,
                                     CeresUserSettingsManager.Settings.TablebaseDirectory);
        referenceEngine = spec.CreateEngine();
      }
    }
  }


  float minFractionRoot;
  float minFractionParent;

  PositionWithHistory analysisPosition;
  NNEvaluatorDef specNet;
  MCGSSearch search;
  Graph graph;
  GNode searchRoot;
  HashSet<int> pvNodeIndices;

  /// <summary>
  /// StringBuilder that accumulates generated text.
  /// </summary>
  StringBuilder sb = new();

  /// <summary>
  /// Set of node indices added so far to graph.
  /// </summary>
  HashSet<int> addedNodes = new();


  /// <summary>
  /// Generates SVG with analysis details, returning FN of generated file.
  /// </summary>
  public string Write(bool launchWithBrowser)
  {
    WriteHeader();

    int rootN = searchRoot.N;

    // Write root node only.
    WriteNodeIfPassesFilter(searchRoot, 0, rootN);
    addedNodes.Add(searchRoot.Index.Index);

    // Write top-level move nodes.
    int subgraphIndex = 0;
    foreach (GEdge childEdge in searchRoot.ChildEdgesExpanded)
    {
      GNode child = childEdge.ChildNode;
      StringBuilder thisSB = new();
      int numWritten = WriteSubtreeNodes(child, 1, rootN, thisSB);

      if (numWritten > 0)
      {
        MGMove mgMove = childEdge.MoveMGFromPos(searchRoot.CalcPosition());
        Position rootPos = searchRoot.CalcPosition().ToPosition;
        Move move = MGMoveConverter.ToMove(mgMove);
        string label = move.ToSAN(in rootPos);
        float scoreCP = -EncodedEvalLogistic.LogisticToCentipawn((float)childEdge.Q);
        float fracRoot = (float)childEdge.N / rootN;
        label = label + $" ({scoreCP:F0}cp, {100.0f * fracRoot:F0}% visits)";
        bool isOnPV = pvNodeIndices.Contains(child.Index.Index);
        string color = isOnPV ? "#f7fcf0" : "whitesmoke";
        sb.AppendLine($"subgraph cluster_move_" + subgraphIndex + " {" + $" label=\"{label}\" bgcolor=\"{color}\"");
        sb.Append(thisSB);
        sb.AppendLine("}");

        if (VERBOSE)
        {
          Console.WriteLine(searchRoot.Index.Index + " ==> " + child.Index.Index);
        }

        WriteEdgeFromParent(sb, child, searchRoot, childEdge, isOnPV);
      }
      subgraphIndex++;
    }

    // Add transposition edges (nodes reachable via multiple parents).
    foreach (int nodeIndex in addedNodes)
    {
      GNode node = graph[new NodeIndex(nodeIndex)];
      if (node.NumParentsMoreThanOne)
      {
        foreach (GEdge parentEdge in node.ParentEdges)
        {
          int parentIdx = parentEdge.ParentNode.Index.Index;
          if (parentIdx != node.TreeParentNodeIndex.Index && addedNodes.Contains(parentIdx))
          {
            sb.AppendLine(parentIdx + "->" + nodeIndex + " [arrowhead=\"empty\" style=\"dashed\" color=\"gray\"]");
          }
        }
      }
    }

    WriteFooter();

    // Convert to SVG.
    GraphvizUtils.Convert(sb.ToString(), BASE_FN, TempDir);
    string fn = Path.Combine(TempDir, $"{BASE_FN}.svg");

    if (launchWithBrowser)
    {
      StringUtils.LaunchBrowserWithURL(fn);
    }

    return fn;
  }


  /// <summary>
  /// BFS traversal of the subtree rooted at the given node, writing nodes that pass filters.
  /// </summary>
  int WriteSubtreeNodes(GNode subtreeRoot, int startDepth, int rootN, StringBuilder outSB)
  {
    int count = 0;
    Queue<(GNode node, int depth)> bfsQueue = new();
    HashSet<int> visited = new();
    List<(GNode node, int depth, bool isOnPV)> nodesToProcess = new();
    long sumNAllSelectedNodes = 0;

    bfsQueue.Enqueue((subtreeRoot, startDepth));
    visited.Add(subtreeRoot.Index.Index);

    while (bfsQueue.Count > 0)
    {
      (GNode node, int depth) = bfsQueue.Dequeue();

      // Hard predicate: stop traversal beyond max depth or if N is too small.
      if (depth > MaxDepth)
      {
        continue;
      }

      if (depth == 1)
      {
        // At top level, only process the subtree root.
        if (node.Index.Index != subtreeRoot.Index.Index)
        {
          continue;
        }
      }
      else
      {
        float nFractionParent = node.IsSearchRoot ? 1 : (float)node.N / graph[node.TreeParentNodeIndex].N;
        if (nFractionParent < minFractionParent)
        {
          continue;
        }

        if ((float)node.N / rootN < minFractionRoot)
        {
          continue;
        }
      }

      float nodeNFractionRoot = (float)node.N / rootN;
      bool isTopN = IsTopN(node);

      if (nodeNFractionRoot > minFractionRoot && (isTopN || (float)node.N / Math.Max(1, graph[node.TreeParentNodeIndex].N) > minFractionParent))
      {
        count++;
        bool isOnPV = pvNodeIndices.Contains(node.Index.Index);
        nodesToProcess.Add((node, depth, isOnPV));
        addedNodes.Add(node.Index.Index);
        sumNAllSelectedNodes += node.N;

        if (VERBOSE)
        {
          Console.WriteLine("add " + node.Index.Index + " depth " + depth);
        }

        // Enqueue children for BFS.
        foreach (GEdge childEdge in node.ChildEdgesExpanded)
        {
          if (childEdge.IsExpanded && !childEdge.Type.IsTerminal())
          {
            int childIdx = childEdge.ChildNode.Index.Index;
            if (visited.Add(childIdx))
            {
              bfsQueue.Enqueue((childEdge.ChildNode, depth + 1));
            }
          }
        }
      }
    }

    // The root node will itself appear to take up 50% of time, then sum of children the other 50%.
    const float RELATIVE_FRAC_REF_ENGINE = 0.5f;

    float fracThisParent = (float)subtreeRoot.N / rootN;
    foreach ((GNode node, int depth, bool isOnPV) in nodesToProcess)
    {
      SearchLimit limit = null;
      if (Options.RelativeTimeReferenceEngine > 0)
      {
        if (IsMCTSReferenceEngine)
        {
          int numNodes = (int)(Options.RelativeTimeReferenceEngine * node.N);
          limit = SearchLimit.NodesPerMove(numNodes);
        }
        else
        {
          float fracOfTheseNodes = (float)node.N / sumNAllSelectedNodes;
          float thisFrac = RELATIVE_FRAC_REF_ENGINE * fracOfTheseNodes * fracThisParent;
          float timeRootSearch = (float)(DateTime.Now - search.Manager.StartTimeThisSearch).TotalSeconds;
          float refEngineSearchTime = timeRootSearch * thisFrac;
          limit = SearchLimit.SecondsPerMove(refEngineSearchTime * Options.RelativeTimeReferenceEngine);
        }
      }

      AddNode(outSB, node, isOnPV, TempDir, limit);
    }

    return count;
  }


  void WriteHeader()
  {
    string specString = NNNetSpecificationString.ToSpecificationString(specNet.NetCombo, specNet.Nets);
    string posString = analysisPosition.FENAndMovesString.Replace(Position.StartPosition.FEN, "startpos");
    string searchNodesString = $" ({searchRoot.N:N0} visits)";
    string title = $"Ceres MCGS ({GameEngineCeresMCGSInProcess.CERES_MCGS_VERSION_STR}) Search Visualization\r\n";
    title += $"{specString}";
    if (Options.RelativeTimeReferenceEngine > 0)
    {
      string netDesc = "";
      if (IsMCTSReferenceEngine && CeresUserSettingsManager.Settings.RefEngineNetworkFile != null)
      {
        netDesc += " with " + CeresUserSettingsManager.Settings.RefEngineNetworkFile ?? "";
      }
      if (File.Exists(CeresUserSettingsManager.Settings.RefEngineEXE))
      {
        title += " (ref engine " + new FileInfo(CeresUserSettingsManager.Settings.RefEngineEXE).Name + netDesc + ")\r\n";
      }
      else
      {
        title += " (ref engine " + CeresUserSettingsManager.Settings.RefEngineEXE + netDesc + ")\r\n";
      }
    }
    title += $"{searchNodesString}\r\n{posString}";
    title += "\r\n";

    sb.AppendLine("digraph \"title\" ");
    sb.AppendLine("{ nodesep=1.5 ranksep=0.5 ");
    sb.AppendLine($"bgcolor=\"whitesmoke\" rankdir=LR; fontsize=\"40\" labelloc=\"t\" label=\"{title}\"");
  }


  void WriteFooter()
  {
    sb.AppendLine("}");
  }


  void WriteNodeIfPassesFilter(GNode node, int depth, int rootN)
  {
    // Root node is always added.
    bool isOnPV = pvNodeIndices.Contains(node.Index.Index);
    AddNode(sb, node, isOnPV, TempDir, null);
  }


  #region Static helpers

  static GameEngine referenceEngine = null;


  /// <summary>
  /// Adds a node to the graph.
  /// </summary>
  void AddNode(StringBuilder sb, GNode node, bool isOnPV, string tempDir, SearchLimit searchLimit)
  {
    MGPosition mgPos = node.CalcPosition();
    Position pos = mgPos.ToPosition;
    int nodeIdx = node.Index.Index;

    string tooltip = GetNodeTooltip(node, pos);
    if (!node.IsSearchRoot && node.DepthFromSearchRoot() > 1)
    {
      GEdge treeParentEdge = node.TreeParentEdge;
      GNode parent = graph[node.TreeParentNodeIndex];
      WriteEdgeFromParent(sb, node, parent, treeParentEdge, isOnPV);
    }

    Move move = default;
    if (!node.IsSearchRoot)
    {
      GEdge treeParentEdge = node.TreeParentEdge;
      MGMove mgMove = treeParentEdge.MoveMGFromPos(graph[node.TreeParentNodeIndex].CalcPosition());
      move = MGMoveConverter.ToMove(mgMove);
    }
    string posFN = GraphvizUtils.WritePosToSVGFile(pos, (ulong)node.HashStandalone.Hash, nodeIdx, move, tempDir);

    sb.AppendLine($"{nodeIdx} [shape=none");
    sb.AppendLine($"href=\"https://lichess.org/editor/{pos.FEN}\"");
    sb.AppendLine($"tooltip=\"{tooltip}]\"");
    float scoreCP = EncodedEvalLogistic.LogisticToCentipawn((float)node.Q);
    float posEvalCP = EncodedEvalLogistic.LogisticToCentipawn((float)node.V);
    string wdlStr = $"{100 * node.W:F0}/{100 * node.D:F0}/{100 * node.L:F0})";

    sb.Append($" fontsize=\"18\"");

    string colorAttr = pos.MiscInfo.SideToMove == SideType.White ? "bgcolor=\"white\"" : "bgcolor=\"silver\"";
    sb.AppendLine("label= <<TABLE cellspacing=\"0\">");
    sb.AppendLine($"<TR><TD border=\"0\" {colorAttr} ><IMG SRC=\"{posFN}\"/></TD></TR>");

    string bgColorScore = GraphvizUtils.ColorStr((float)node.Q);
    sb.AppendLine($"<TR><TD bgcolor=\"{bgColorScore}\" border=\"0\" >{scoreCP,5:N0}cp   ({wdlStr}  {posEvalCP:N0}cp</TD></TR>");

    float pValue = node.IsSearchRoot ? 1.0f : (float)node.TreeParentEdge.P;
    sb.AppendLine($"<TR><TD bgcolor=\"linen\" border=\"0\" >{node.N:N0}  {pValue * 100,5:F2}%</TD></TR>");

    bool posIsTerminal = pos.CalcTerminalStatus().IsTerminal();
    if (searchLimit != null && !posIsTerminal)
    {
      TryStartReferenceEngine();
      if (referenceEngine != null)
      {
        referenceEngine.ResetGame();
        GameEngineSearchResult refEngineResult = referenceEngine.Search(new PositionWithHistory(in pos), searchLimit);

        float refEngineEvalCP = refEngineResult.ScoreCentipawns;
        float refEngineEvalNodes = refEngineResult.FinalN;
        string refEngineEvalSF = refEngineResult.MoveString;
        Move refEngineMoveMove = Move.FromUCI(in pos, refEngineEvalSF);
        string refEngineMoveSAN = refEngineMoveMove.ToSAN(in pos);

        string refEngineScoreStr = $"{refEngineEvalCP,5:N0}cp";
        string evalDiffStr = "";
        bool movesSame = true;

        if (node.NumEdgesExpanded > 0)
        {
          GEdge bestEdge = node.EdgeWithMaxValue(e => e.N);
          MGMove mgMovePrimary = bestEdge.MoveMGFromPos(mgPos);
          Move movePrimary = MGMoveConverter.ToMove(mgMovePrimary);
          if (refEngineMoveMove != movePrimary)
          {
            movesSame = false;
            if (!IsMCTSReferenceEngine)
            {
              GNode bestChild = bestEdge.ChildNode;
              Position bestChildPos = bestChild.CalcPosition().ToPosition;
              PositionWithHistory pwh = new PositionWithHistory(in bestChildPos);
              GameEngineSearchResult searchResultReferenceOfPrimaryBestMove = referenceEngine.Search(pwh, searchLimit);
              float diff = searchResultReferenceOfPrimaryBestMove.ScoreCentipawns - (-1 * refEngineResult.ScoreCentipawns);
              if (Math.Abs(diff) > 10)
              {
                string signStr = diff > 0 ? "+" : "-";
                evalDiffStr = $"({signStr}{Math.Round(MathF.Abs(diff))}cp)";
                if (diff < 0)
                {
                  evalDiffStr += "?";
                }
              }
            }
          }
        }
        string colorRefEval = movesSame ? "lightyellow3" : "orangered";
        sb.AppendLine($"<TR><TD bgcolor=\"{colorRefEval}\" border=\"0\" >{refEngineScoreStr}  {refEngineEvalNodes:N0}  {refEngineMoveSAN} {evalDiffStr}</TD></TR>");
      }
    }

    sb.AppendLine("</TABLE>>]");
  }


  /// <summary>
  /// Writes the edge from a parent to a child node.
  /// </summary>
  private static void WriteEdgeFromParent(StringBuilder sb, GNode child, GNode parent, GEdge edge, bool isOnPV)
  {
    sb.Append(parent.Index.Index + "->" + child.Index.Index);

    float nFractionParent = (float)edge.N / parent.N;
    sb.Append("[ ");

    // Output move number and move played.
    MGPosition parentPos = parent.CalcPosition();
    Position parentPosChess = parentPos.ToPosition;
    int moveNum = parentPosChess.MiscInfo.MoveNum;
    string moveNumStr = (moveNum / 2) + (moveNum % 2 == 0 ? ".. " : ". ");
    MGMove mgMove = edge.MoveMGFromPos(parentPos);
    string san = MGMoveConverter.ToMove(mgMove).ToSAN(in parentPosChess);

    sb.AppendLine($"xlabel=\" {moveNumStr}{san}\"");

    // Fraction of visits to this edge.
    sb.Append($"label=\"{nFractionParent * 100,4:F0}%  \"");

    if (isOnPV)
    {
      sb.Append($" color=red");
    }

    GraphvizUtils.WriteArrowheadForFractionParent(sb, nFractionParent);
    sb.AppendLine("]");
  }


  /// <summary>
  /// Returns if a specified node is the top-N node within its tree parent (by edge N).
  /// </summary>
  bool IsTopN(GNode node)
  {
    if (node.IsSearchRoot)
    {
      return true;
    }

    GNode parent = graph[node.TreeParentNodeIndex];
    int thisEdgeN = 0;

    // Find the edge N for this node from its tree parent.
    foreach (GEdge edge in parent.ChildEdgesExpanded)
    {
      if (edge.ChildNode.Index.Index == node.Index.Index)
      {
        thisEdgeN = edge.N;
        break;
      }
    }

    // Check if any sibling has higher N.
    foreach (GEdge edge in parent.ChildEdgesExpanded)
    {
      if (edge.N > thisEdgeN)
      {
        return false;
      }
    }
    return true;
  }


  static string GetNodeTooltip(GNode node, Position pos)
  {
    string fen = pos.FEN;
    int depth = node.IsSearchRoot ? 0 : node.DepthFromSearchRoot();
    return GraphvizUtils.ConvertedCRLF(
      $"FEN: {fen}\n" +
      $"N={node.N:N0} Q={node.Q:F4} V={node.V:F4}\n" +
      $"W={100 * node.W:F1}% D={100 * node.D:F1}% L={100 * node.L:F1}%\n" +
      $"Depth={depth} Edges={node.NumEdgesExpanded}");
  }

  #endregion
}
