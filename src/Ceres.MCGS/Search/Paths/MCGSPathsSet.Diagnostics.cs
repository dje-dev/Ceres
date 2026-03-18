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
using System.Linq;
using Ceres.Base.Misc;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;

#endregion

namespace Ceres.MCGS.Search.Paths;

/// <summary>
/// Diagnostics support methods for MCGSPathsSet.
/// </summary>
public partial class MCGSPathsSet
{
  /// <summary>
  /// Dumps the distribution of path lengths to the console.
  /// Outputs one line for each path length seen and the corresponding count of visits.
  /// </summary>
  public void DumpDistribution()
  {
    DumpDistribution(PathLengthDistribution, CountNonAbortedPathVisits, MaxNonAbortedPathDepth, 
                     SumNonAbortedPathVisits / (double)CountNonAbortedPathVisits);
  }

  /// <summary>
  /// Dumps the distribution of path lengths to the console.
  /// Outputs one line for each path length seen and the corresponding count of visits.
  /// </summary>
  /// <param name="distribution">Array of path length counts indexed by path length</param>
  /// <param name="totalPaths">Total number of paths</param>
  /// <param name="maxDepth">Maximum path depth observed</param>
  /// <param name="avgDepth">Average path depth</param>
  public static void DumpDistribution(long[] distribution, long totalPaths, int maxDepth, double avgDepth)
  {
    Console.WriteLine("=== Path Length Distribution ===");
    Console.WriteLine($"{"Path Length",-15} {"Count",-15} {"Percentage",-15}");
    Console.WriteLine(new string('-', 45));

    if (totalPaths == 0)
    {
      Console.WriteLine("No paths recorded.");
      return;
    }

    for (int i = 0; i < distribution.Length; i++)
    {
      long count = distribution[i];
      if (count > 0)
      {
        double percentage = (count * 100.0) / totalPaths;
        Console.WriteLine($"{i,-15} {count,-15} {percentage,-15:F2}%");
      }
    }

    Console.WriteLine(new string('-', 45));
    Console.WriteLine($"{"Total Paths:",-15} {totalPaths,-15}");
    Console.WriteLine($"{"Max Depth:",-15} {maxDepth,-15}");
    Console.WriteLine($"{"Avg Depth:",-15} {avgDepth,-15:F2}");
  }


  /// <summary>
  /// Checks the set of all MCGSPathVisit instances contained in all paths for overlaps:
  /// multiple distinct visit slots (in the path visit pool) that refer to the same child node.
  /// Optionally dumps a summary to the console.
  /// </summary>
  /// <param name="dumpToConsole">If true, writes diagnostic output to the console.</param>
  /// <returns>The number of detail lines (one per involved path-visit) that would be output for crossings.</returns>
  public int CheckCrossingPathVisits(bool dumpToConsole = false)
  {
    if (dumpToConsole)
    {
      Console.WriteLine("=== Analyzing Path Crossings (by visit slots) ===");
    }

    MCGSPath[] pathsArray = Paths.ToArray();

    // Map: child node index -> list of (path, visit, absoluteSlotIndex)
    Dictionary<NodeIndex, List<(MCGSPath Path, MCGSPathVisit Visit, int SlotIndex)>> crossingGroups = new();

    // First pass: collect every visit slot from each path (local slots only)
    foreach (MCGSPath path in pathsArray)
    {
      // iterate local slots; these correspond 1:1 to slots in the pool
      for (int localIndex = 0; localIndex < path.numSlotsUsed; localIndex++)
      {
        ref MCGSPathVisit visit = ref path.slots[localIndex];
        GEdge edge = visit.ParentChildEdge;
        if (edge.IsNull || edge.Type != GEdgeStruct.EdgeType.ChildEdge)
        {
          continue;
        }

        GNode childNode = edge.ChildNode;
        if (childNode.IsNull)
        {
          continue;
        }

        int absoluteSlotIndex = path.slots.StartIndex + localIndex; // identity of this visit slot in the pool
        NodeIndex nodeIndex = childNode.Index;

        if (!crossingGroups.TryGetValue(nodeIndex, out List<(MCGSPath Path, MCGSPathVisit Visit, int SlotIndex)> list))
        {
          list = new List<(MCGSPath, MCGSPathVisit, int)>();
          crossingGroups[nodeIndex] = list;
        }
        list.Add((path, visit, absoluteSlotIndex));
      }
    }

    // Filter to nodes with more than one distinct slot (ignore ourself)
    List<KeyValuePair<NodeIndex, List<(MCGSPath Path, MCGSPathVisit Visit, int SlotIndex)>>> crossingNodes = crossingGroups
      .Select(kvp => new KeyValuePair<NodeIndex, List<(MCGSPath Path, MCGSPathVisit Visit, int SlotIndex)>>(
        kvp.Key,
        kvp.Value
          .GroupBy(x => x.SlotIndex)
          .Select(g => g.First())
          .ToList()))
      .Where(kvp => kvp.Value.Count > 1)
      .ToList();

    if (dumpToConsole)
    {
      if (crossingNodes.Count == 0)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Green, "No path crossing visits detected.");
      }
      else
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Found {crossingNodes.Count} child nodes with overlapping visit slots:");
        Console.WriteLine();
      }
    }

    int detailLineCount = 0;

    foreach (var crossing in crossingNodes)
    {
      NodeIndex nodeIndex = crossing.Key;
      List<(MCGSPath Path, MCGSPathVisit Visit, int SlotIndex)> uniqueVisitSlots = crossing.Value;

      // Count lines for this node
      detailLineCount += uniqueVisitSlots.Count;

      if (dumpToConsole)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Cyan, $"Node {nodeIndex.Index} visited by {uniqueVisitSlots.Count} visit slots:");
        foreach (var (path, visit, slotIndex) in uniqueVisitSlots)
        {
          GEdge edge = visit.ParentChildEdge;
          string parentNode = edge.IsNull ? "NULL" : edge.ParentNode.Index.ToString();
          int numVisitsAttempted = visit.NumVisitsAttempted;
          string numVisitsAccepted = visit.NumVisitsAccepted?.ToString() ?? "NULL";
          int pendingBackup = visit.NumVisitsAttemptedPendingBackup;
          int accumulatorAttempted = visit.Accumulator.NumVisitsAttempted;
          int accumulatorAccepted = visit.Accumulator.NumVisitsAccepted;

          Console.WriteLine($"  Slot#{slotIndex} Path {path.PathID}: Parent={parentNode} Att={numVisitsAttempted} Acc={numVisitsAccepted} " +
                           $"Pending={pendingBackup} AccAtt={accumulatorAttempted} AccAcc={accumulatorAccepted} {path.PathSequenceString}");

          if (pendingBackup != numVisitsAttempted && pendingBackup != 0)
          {
            ConsoleUtils.WriteLineColored(ConsoleColor.Red,
              $"    WARNING: PendingBackup ({pendingBackup}) != NumVisitsAttempted ({numVisitsAttempted})");
          }

          if (accumulatorAttempted > 0)
          {
            ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
              $"    INFO: Accumulator has {accumulatorAttempted} attempted visits");
          }
        }
        Console.WriteLine();
      }
    }

    if (dumpToConsole)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Magenta,
        $"Summary: {crossingNodes.Count} crossing nodes, {detailLineCount} detail lines");
    }

    return detailLineCount;
  }


  /// <summary>
  /// Checks the set of all MCGSPathVisit instances contained in all paths for overlaps by edge:
  /// multiple distinct visit slots (in the path visit pool) that refer to the same edge.
  /// Uses the GEdge equality operator to group edges.
  /// </summary>
  /// <param name="dumpToConsole">If true, writes diagnostic output to the console.</param>
  /// <returns>The number of detail lines (one per involved path-visit) that would be output for crossings.</returns>
  public int CheckCrossingPathVisitsByEdge(bool dumpToConsole = false)
  {
    if (dumpToConsole)
    {
      Console.WriteLine("=== Analyzing Path Crossings (by edge) ===");
    }

    MCGSPath[] pathsArray = Paths.ToArray();

    // Map: edge -> list of (path, visit, absoluteSlotIndex)
    Dictionary<GEdge, List<(MCGSPath Path, MCGSPathVisit Visit, int SlotIndex)>> crossingGroups = new();

    foreach (MCGSPath path in pathsArray)
    {
      for (int localIndex = 0; localIndex < path.numSlotsUsed; localIndex++)
      {
        ref MCGSPathVisit visit = ref path.slots[localIndex];
        GEdge edge = visit.ParentChildEdge;
        if (edge.IsNull || edge.Type != GEdgeStruct.EdgeType.ChildEdge)
        {
          continue;
        }

        int absoluteSlotIndex = path.slots.StartIndex + localIndex;

        if (!crossingGroups.TryGetValue(edge, out List<(MCGSPath Path, MCGSPathVisit Visit, int SlotIndex)> list))
        {
          list = new List<(MCGSPath, MCGSPathVisit, int)>();
          crossingGroups[edge] = list;
        }
        list.Add((path, visit, absoluteSlotIndex));
      }
    }

    // Filter to edges with more than one distinct slot (ignore ourself)
    List<KeyValuePair<GEdge, List<(MCGSPath Path, MCGSPathVisit Visit, int SlotIndex)>>> crossingEdges = crossingGroups
      .Select(kvp => new KeyValuePair<GEdge, List<(MCGSPath Path, MCGSPathVisit Visit, int SlotIndex)>>(
        kvp.Key,
        kvp.Value
          .GroupBy(x => x.SlotIndex)
          .Select(g => g.First())
          .ToList()))
      .Where(kvp => kvp.Value.Count > 1)
      .ToList();

    if (dumpToConsole)
    {
      if (crossingEdges.Count == 0)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Green, "No path crossing visits by edge detected.");
      }
      else
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Found {crossingEdges.Count} edges with overlapping visit slots:");
        Console.WriteLine();
      }
    }

    int detailLineCount = 0;

    foreach (var crossing in crossingEdges)
    {
      GEdge edge = crossing.Key;
      List<(MCGSPath Path, MCGSPathVisit Visit, int SlotIndex)> uniqueVisitSlots = crossing.Value;

      detailLineCount += uniqueVisitSlots.Count;

      if (dumpToConsole)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Cyan, $"Edge {edge} visited by {uniqueVisitSlots.Count} visit slots:");
        foreach (var (path, visit, slotIndex) in uniqueVisitSlots)
        {
          string parentNode = edge.IsNull ? "NULL" : edge.ParentNode.Index.ToString();
          int numVisitsAttempted = visit.NumVisitsAttempted;
          string numVisitsAccepted = visit.NumVisitsAccepted?.ToString() ?? "NULL";
          int pendingBackup = visit.NumVisitsAttemptedPendingBackup;
          int accumulatorAttempted = visit.Accumulator.NumVisitsAttempted;
          int accumulatorAccepted = visit.Accumulator.NumVisitsAccepted;

          Console.WriteLine($"  Slot#{slotIndex} Path {path.PathID}: Parent={parentNode} Att={numVisitsAttempted} Acc={numVisitsAccepted} " +
                           $"Pending={pendingBackup} AccAtt={accumulatorAttempted} AccAcc={accumulatorAccepted} {path.PathSequenceString}");

          if (pendingBackup != numVisitsAttempted && pendingBackup != 0)
          {
            ConsoleUtils.WriteLineColored(ConsoleColor.Red,
              $"    WARNING: PendingBackup ({pendingBackup}) != NumVisitsAttempted ({numVisitsAttempted})");
          }

          if (accumulatorAttempted > 0)
          {
            ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
              $"    INFO: Accumulator has {accumulatorAttempted} attempted visits");
          }
        }
        Console.WriteLine();
      }
    }

    if (dumpToConsole)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Magenta,
        $"Summary: {crossingEdges.Count} crossing edges, {detailLineCount} detail lines");
    }

    return detailLineCount;
  }
}
