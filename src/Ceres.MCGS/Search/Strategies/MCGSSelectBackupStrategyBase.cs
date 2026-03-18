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
using System.Diagnostics;
using Ceres.Base.DataTypes;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Search.Coordination;

#endregion

namespace Ceres.MCGS.Search.Strategies;

/// <summary>
/// Accumulator computed during the selection phase that captures:
///   - the sum of edge visit counts (N) (plus 1 for itself)
///   - sum of (N * Q) over children (plus V of the node itself)
/// over both the node itself and all of its children.
/// </summary>
/// <param name="SumN"></param>
/// <param name="SumW"></param>
/// <param name="SumD"></param>
/// <param name="NumVisitsAccepted"></param>
public readonly record struct NodeSelectAccumulator(double SumN, double SumW, double SumD, int NumVisitsAccepted);


/// <summary>
/// Abstract base class for any selection and backup (backpropagation) strategy.
/// </summary>
public abstract class MCGSSelectBackupStrategyBase
{
  public MCGSEngine Engine { get; internal set; }


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="engine"></param>
  public MCGSSelectBackupStrategyBase(MCGSEngine engine)
  {
    Engine = engine;
  }


  /// <summary>
  /// Runs an algorithm to select the set of children to be next visited.
  /// </summary>
  /// <param name="parentNode"></param>
  /// <param name="selectorID"></param>
  /// <param name="depth"></param>
  /// <param name="numChildrenToConsider"></param>
  /// <param name="numTargetVisits"></param>
  /// <param name="alsoComputeScores"></param>
  /// <param name="explorationMultiplier"></param>
  /// <param name="temperatureMultiplier"></param>
  /// <param name="childVisitCounts">output number visits to be applied to each child</param>
  /// <param name="childScores"></param>
  /// <returns></returns>
  public abstract NodeSelectAccumulator SelectChildren(GNode parentNode,
                                                       int selectorID,
                                                       int depth,
                                                       int numChildrenToConsider,
                                                       int numTargetVisits,
                                                       bool alsoComputeScores,
                                                       float explorationMultiplier,
                                                       float temperatureMultiplier,
                                                       bool refreshStaleEdges,
                                                       MCGSFutilityPruningStatus[] rootMovePruningStatus,
                                                       out Span<short> childVisitCounts, // TODO: remove out, use a single shared array passed from caller
                                                       out Span<double> childScores);


  /// <summary>
  /// Backs up the value of a newly evaluated leaf node to a specified node.
  /// </summary>
  /// <param name="node"></param>
  /// <param name="deltaN"></param>
  /// <param name="deltaW"></param>
  /// <param name="deltaD"></param>
  public abstract void BackupToNode(GNode node, int deltaN, double deltaW, double deltaD);


  /// <summary>
  /// Backs up the value of a newly evaluated leaf node to a specified edge.
  /// </summary>
  /// <param name="edge"></param>
  /// <param name="deltaN"></param>
  /// <param name="newChildQ"></param>
  /// <param name="newD"></param>
  /// <param name="drawKnownToExistAtChild"></param>
  public abstract void BackupToEdge(GEdge edge, int deltaN, double newQChild, double newD, bool drawKnownToExistAtChild);


  /// Returns the number of children (starting from index 0) that are eligible to be considered).
  /// </summary>
  /// <param name="node"></param>
  /// <param name="numTargetVisits"></param>
  /// <returns></returns>
  internal abstract int NumChildrenToConsider(GNode node, int numTargetVisits);
}
