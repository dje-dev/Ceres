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
using System.Threading;

#endregion

namespace Ceres.MCGS.Search.Phases;

/// <summary>
/// Stores coordination information for merging information
/// from multiple paths which overlap on a given node.
/// </summary>
public struct MCGSBackupAccumulator
{
  /// <summary>
  /// Accumulated number of visits attempted to this node (see so far during backup).
  /// </summary>
  public volatile short NumVisitsAttempted;

  /// <summary>
  /// Accumulated number of visits accepted to this node (see so far during backup).
  /// </summary>
  public volatile short NumVisitsAccepted;

  /// <summary>
  /// Accumulated sum of leaf values (in this node's child-edge perspective) backed up through this
  /// node so far during backup. Used together with <see cref="SumV2"/> and the accepted-visit count
  /// to maintain the per-node leaf-value volatility estimate exactly across merge points.
  /// </summary>
  public double SumV;

  /// <summary>
  /// Accumulated sum of squared leaf values backed up through this node so far (perspective-invariant).
  /// </summary>
  public double SumV2;


  /// <summary>
  /// Adds into the accumulator without locking.
  /// </summary>
  /// <param name="numVisitsAttempted"></param>
  /// <param name="numVisitsAccepted"></param>
  internal void DoAdd(int numVisitsAttempted, int numVisitsAccepted, double sumV, double sumV2)
  {
    NumVisitsAttempted += (short)numVisitsAttempted;
    NumVisitsAccepted += (short)numVisitsAccepted;
    SumV += sumV;
    SumV2 += sumV2;
  }


  /// <summary>
  /// Returns a string representation of the object.
  /// </summary>
  /// <returns></returns>
  public override string ToString() => $"<MCGSBackupAccumulator "
                                     + $"Att={NumVisitsAttempted} Acc={NumVisitsAccepted}> ";
}
