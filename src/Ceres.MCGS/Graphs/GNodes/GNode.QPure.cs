#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

using System;
using System.Diagnostics;

using Ceres.Chess;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Phases;
using Ceres.MCGS.Search.RPO;

namespace Ceres.MCGS.Graphs.GNodes;

public partial struct GNode
{
  private static long s_rpoUpdateQErrorCount = 0;
  private static double s_rpoUpdateQSumSquaredError = 0.0;
  private static double s_rpoUpdateQMaxAbsError = 0.0;
  private static long s_rpoUpdateQLastOutputCount = 0;
  private static readonly object s_rpoUpdateQStatsLock = new object();

  const bool TRACK_STATS = false;

  public void ResetQUsingSumWChildrenAndSelf(double sumWChildrenAndSelf, bool refreshSiblingContribution)
  {
    if (CheckmateKnownToExistAmongChildren)
    {
      Debug.Assert(Q > 0.995);
      return; // Q remains fixed as a win
    }

    // Note we divide by full N at parent which may include some NDrawByRepetition (with 0 contribution to W)
    double newQPure = N == 0 ? 0 : sumWChildrenAndSelf / N; 

    ResetNodeQUsingNewQPure(newQPure, refreshSiblingContribution);
  }


  internal void RPOUpdateQ(bool refreshSiblingContribution, double rpoLambdaP)
  {
    Debug.Assert(!Terminal.IsTerminal());
    if (CheckmateKnownToExistAmongChildren)
    {
      return;
    }

    double orgQ = double.NaN;
    if (TRACK_STATS)
    {
      Debug.Assert(!Terminal.IsTerminal());

      double w = V;
      int N = 1;
      for (int i = 0; i < NumEdgesExpanded; i++)
      {
        GEdge edge = ChildEdgeAtIndex(i);
        int n = edge.N;
        if (n > 0)
        {
          N += edge.N;
          w += -edge.QChild * edge.N;
        }
      }
      orgQ = w / N;
    }

    double rpoPureQ = RPOTestsNEW.CalcRPOQ(this, Q, NumEdgesExpanded, rpoLambdaP);

    if (TRACK_STATS)
    {
      // Keep static running static sufficient statistics for standard deviation of difference between orgQ and rpoPureQ and also max absolute difference
      lock (s_rpoUpdateQStatsLock)
      {
        double error = orgQ - rpoPureQ;
        double absError = Math.Abs(error);
        s_rpoUpdateQSumSquaredError += error * error;
        s_rpoUpdateQErrorCount++;

        if (absError > s_rpoUpdateQMaxAbsError)
        {
          s_rpoUpdateQMaxAbsError = absError;
        }

        if (absError > 0.5)
        {
          for (int x=0;x<NumEdgesExpanded;x++)
          {
            Console.WriteLine(x + " " + ChildEdgeAtIndex(x));
          }
          Console.WriteLine("whoaxx " + this);
        }

        // Every 1_000,000 dump these stats to Console
        if (s_rpoUpdateQErrorCount - s_rpoUpdateQLastOutputCount >= 1_000_000)
        {
          double rmse = Math.Sqrt(s_rpoUpdateQSumSquaredError / s_rpoUpdateQErrorCount);
          Console.WriteLine($"RPOUpdateQ Error Statistics - Count: {s_rpoUpdateQErrorCount}, RMSE: {rmse:F6}, Max Abs Error: {s_rpoUpdateQMaxAbsError:F6}");
          s_rpoUpdateQLastOutputCount = s_rpoUpdateQErrorCount;
        }
      }
    }
    
    ResetNodeQUsingNewQPure(rpoPureQ, refreshSiblingContribution);
  }


  internal void ResetNodeQFromChildren(bool refreshSiblingContribution)
  {
    ResetNodeQUsingNewQPure(ComputeQPure(), refreshSiblingContribution);
  }


  /// <summary>
  /// Resets this node's Q value using the given pure Q value, 
  /// optionally also recomputing and incorporating the sibling contribution.
  /// </summary>
  /// <param name="newQPure"></param>
  /// <param name="refreshSiblingContribution"></param>
  internal void ResetNodeQUsingNewQPure(double newQPure, bool refreshSiblingContribution)
  {
    Debug.Assert(!double.IsNaN(newQPure));

    if (Terminal.IsTerminal())
    {
      return;
    }
    
    if (CheckmateKnownToExistAmongChildren)
    {
      Debug.Assert(Q > 0.995);
      return; // Q remains fixed as a win
    }

    Debug.Assert(!double.IsNaN(newQPure));
    Debug.Assert(Math.Abs(newQPure) < 1.2);

    double newSiblingQWt;
    double newSiblingQ;

    if (refreshSiblingContribution && NumEdgesExpanded > 0)
    {
      // TODO: possibly move this CalcPseudotranspositionContribution method here
      MCGSSelect.CalcPseudotranspositionContribution(this, 0,
                                                     out double siblingAvgQ,
                                                     out float extraNFromTranspositionAlias);
      newSiblingQWt = (double)extraNFromTranspositionAlias / (N + extraNFromTranspositionAlias);
      newSiblingQ = extraNFromTranspositionAlias == 0
        ? 0
        : siblingAvgQ;

      NodeRef.SiblingsQ = newSiblingQ;
      NodeRef.SiblingsQFrac = newSiblingQWt;

      // Need to read back the saved versions (which were subject to quantization)
      // to be used in the recomputed Q below (to allow exact invertability subsequently).
      newSiblingQWt = NodeRef.SiblingsQFrac;
      newSiblingQ = NodeRef.SiblingsQ;

    }
    else
    {
      newSiblingQWt = NodeRef.SiblingsQFrac;
      newSiblingQ = NodeRef.SiblingsQ;
    }

    // Stored Q is the weighted average of pure Q and sibling Q.
    double newQ = (newSiblingQWt * newSiblingQ) + ((1.0 - newSiblingQWt) * newQPure);

    if (newQ < 0 && DrawKnownToExistAmongChildren)
    {
      newQ = 0;
      NodeRef.SiblingsQFrac = 0;
    }
    else
    {
      Debug.Assert(!double.IsNaN(newQ));
      Debug.Assert(Math.Abs(newQ) < 1.2);
    }


    if (MCGSParamsFixed.TRACK_NODE_EDGE_UNCERTAINTY)
    {
      double priorMean = N <= 1 ? newQ : Q; // first point is just the new sample
      NodeRef.StdDevEstimate.AddSample(priorMean, newQ);
    }

    NodeRef.Q = newQ;
    if (DrawKnownToExistAmongChildren)
    {
      Debug.Assert(NodeRef.Q >= 0);
    }
    else
    {
      Debug.Assert(Math.Abs(newQPure - ComputeQPure()) < 1E-10);
    }
  }


  /// <summary>
  /// Computes the pure Q value (without sibling contribution) for this node.
  /// </summary>
  /// <returns></returns>
  internal double ComputeQPure()
  {
    if (N == 0)
    {
      return 0;
    }

    double priorSiblingQWtFrac = NodeRef.SiblingsQFrac;
    if (priorSiblingQWtFrac == 0)
    {
      return Q;
    };

    double priorSiblingQ = NodeRef.SiblingsQ;

    double qPure = (Q - (priorSiblingQWtFrac * priorSiblingQ)) / (1.0 - priorSiblingQWtFrac);

    Debug.Assert(!double.IsNaN(qPure));
    Debug.Assert(Math.Abs(qPure) < 1.2);

    return qPure;
  }


  /// <summary>
  /// Removes the sibling contribution from this node's Q value.
  /// </summary>
  internal void RemoveSiblingContribution()
  {
    if (NodeRef.SiblingsQFrac != 0)
    {
      NodeRef.Q = ComputeQPure();
      NodeRef.SiblingsQFrac = 0;
    }
  }


}
