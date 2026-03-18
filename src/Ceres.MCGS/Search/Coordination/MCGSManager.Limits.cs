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

using Ceres.Base.Math;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;

#endregion

namespace Ceres.MCGS.Search.Coordination;

/// <summary>
/// Partial class containing methods and properties 
/// related to search limits and stopping criteria.
/// </summary>
public partial class MCGSManager : IDisposable
{
  /// <summary>
  /// Returns the maximum batch size allowed when the 
  /// search may be nearing its time limit exhaustion.
  /// </summary>
  public int MaxBatchSizeDueToPossibleNearTimeExhaustion
  {
    get
    {
      if (SearchLimit.Type != SearchLimitType.SecondsPerMove)
      {
        return int.MaxValue;
      }

      float elapsedTime = (float)(DateTime.Now - StartTimeThisSearch).TotalSeconds;
      float remainingTime = SearchLimit.Value - elapsedTime;

      // TODO: tune these based on hardware and network (EstimatedNPS)
      return  remainingTime switch
      {
        < 0.05f => 96,
        < 0.10f => 192,
        _ => int.MaxValue
      };
    }
  }



  //  public float FractionSearchRemaining => 1.0f - (LastSearchLimit.Value / Engine.Graph.RootNode.N); // ** TEMPRARY

  public float FractionSearchCompleted => 1.0f - FractionSearchRemaining;

  public float FractionSearchRemaining
  {
    get
    {
      return SearchLimit.Type switch
      {
        SearchLimitType.SecondsPerMove => MathHelpers.Bounded(RemainingTime / SearchLimit.Value, 0, 1),
        SearchLimitType.NodesPerMove => MathHelpers.Bounded((SearchLimit.Value - NumNodesVisitedThisSearch) / SearchLimit.Value, 0, 1),
        SearchLimitType.NodesPerTree => MathHelpers.Bounded(1.0f - (NumNodesVisitedThisSearch / (SearchLimit.Value - RootNWhenSearchStarted)), 0, 1),
        SearchLimitType.BestValueMove => MathHelpers.Bounded(1.0f - (NumNodesVisitedThisSearch / (SearchLimit.Value - RootNWhenSearchStarted)), 0, 1),
        SearchLimitType.BestActionMove => MathHelpers.Bounded(1.0f - (NumNodesVisitedThisSearch / (SearchLimit.Value - RootNWhenSearchStarted)), 0, 1),
        _ => throw new NotImplementedException()
      };
    }
  }


  /// <summary>
  /// Returns remaining time in seconds if the search limit is time based.
  /// </summary>
  public float RemainingTime
  {
    get
    {
      if (SearchLimit.Type != SearchLimitType.SecondsPerMove)
      {
        return float.MaxValue;
      }

      float elapsedTime = (float)(DateTime.Now - StartTimeThisSearch).TotalSeconds;
      float remainingTime = SearchLimit.Value - elapsedTime;
      return remainingTime;
    }
  }


  /// <summary>
  /// Sets the current search stop status by evaluating all stopping criteria.
  /// </summary>
  public void UpdateSearchStopStatus()
  {
    try
    {
      StopStatus = CalcSearchStopStatus();
    }
    catch (Exception exc)
    {
      Console.WriteLine(exc);
    }
  }


  /// <summary>
  /// The move associated with root edge currently having the largest N value.
  /// </summary>
  public MGMove? TopNMove;
  public int TopNChildN;

  /// <summary>
  /// The N value when the current best node came to be best.
  /// </summary>
  public int NumNodesWhenChoseTopNNode;

  public float FractionNumNodesWhenChoseTopNNode => (float)NumNodesWhenChoseTopNNode / Engine.SearchRootNode.N;


  /// <summary>
  /// Calculates the current search stop status by evaluating all stopping criteria.
  /// </summary>
  /// <returns></returns>
  SearchStopStatus CalcSearchStopStatus()
  {
    if (Engine.SearchRootNode.N < 2)
    {
      return SearchStopStatus.Continue; // Have to search at least two nodes to successfully get a move
    }

    UpdateTopNodeInfo();

    if (LastGameLimitInputs != null && LimitManager.CheckStopSearch(LastGameLimitInputs))
    {
      return SearchStopStatus.LimitsManagerRequestedStop;
    }

    if (ExternalStopRequested)
    {
      return SearchStopStatus.ExternalStopRequested;
    }

    if (RemainingTime <= 0.01)
    {
      return SearchStopStatus.TimeLimitReached;
    }

    if (SearchLimit.Type == SearchLimitType.NodesPerMove
    && (Engine.SearchRootNode.N - RootNWhenSearchStarted) >= SearchLimit.Value)
    {
      return SearchStopStatus.NodeLimitReached;
    }

    if (SearchLimit.MaxTreeVisits != null
     && Engine.SearchRootNode.N >= SearchLimit.MaxTreeVisits
     && NumNodesVisitedThisSearch > 0) // always allow a little search to insure state fully initialized
    {
      return SearchStopStatus.MaxGraphVisitsExceeded;
    }

    if (SearchLimit.MaxTreeNodes != null
     && Engine.SearchRootNode.Graph.Store.NodesStore.NumTotalNodes >= (SearchLimit.MaxTreeNodes - 2048)
     && NumNodesVisitedThisSearch > 0) // always allow a little search to insure state fully initialized
    {
      return SearchStopStatus.MaxGraphAllocatedNodesExceeded;
    }

    int numNotShutdowChildren = TerminationManager.NumberOfNotShutdownChildren();

    // Exit if only one possible move, and smart pruning is turned on
    if (ParamsSearch.FutilityPruningStopSearchEnabled)
    {
      if (Engine.SearchRootNode.N > 0 && Engine.SearchRootNode.NumPolicyMoves == 1)
      {
        return SearchStopStatus.FutilityPrunedAllMoves;
      }
      else if (numNotShutdowChildren == 1)
      {
        return SearchStopStatus.FutilityPrunedAllMoves;
      }
    }

    return SearchStopStatus.Continue;
  }



  float estimatedNPS;
  internal float EstimatedNPS => estimatedNPS;

  /// <summary>
  /// Updates the estimated nodes per second rate for the current search.
  /// </summary>
  internal void UpdateEstimatedNPS()
  {
    const float MIN_TIME = 0.02f;
    const float MIN_VISITS = 10;

    float elapsedSecs = (float)(DateTime.Now - StartTimeFirstVisit).TotalSeconds;
    bool insufficientData = elapsedSecs < MIN_TIME || NumNodesVisitedThisSearch < MIN_VISITS;
    estimatedNPS = insufficientData ? float.NaN : NumNodesVisitedThisSearch / elapsedSecs;
  }



  /// <summary>
  /// Returns the estimated number of visits remaining in the search (if available).
  /// </summary>
  /// <returns></returns>
  public int? EstimatedNumVisitsRemaining()
  {
    if (SearchLimit.Type == SearchLimitType.NodesPerMove)
    {
      int nodesProcessedAlready = Engine.SearchRootNode.N - RootNWhenSearchStarted;
      return (int)MathF.Max(0, SearchLimit.Value - nodesProcessedAlready);
    }
    else if (SearchLimit.Type == SearchLimitType.NodesPerTree)
    {
      return (int)MathF.Max(0, SearchLimit.Value - Engine.SearchRootNode.N);
    }
    else if (SearchLimit.Type == SearchLimitType.BestValueMove)
    {
      return (int)MathF.Max(0, SearchLimit.Value - Engine.SearchRootNode.N);
    }
    else if (SearchLimit.Type == SearchLimitType.BestActionMove)
    {
      return (int)MathF.Max(0, SearchLimit.Value - Engine.SearchRootNode.N);
    }
    else if (SearchLimit.Type == SearchLimitType.SecondsPerMove)
    {
      float estNPS = EstimatedNPS;
      if (float.IsNaN(estNPS))
      {
        return null; // unkown
      }

      float elapsedTime = (float)(DateTime.Now - StartTimeThisSearch).TotalSeconds;
      float remainingTime = SearchLimit.Value - elapsedTime;

      return (int)MathF.Max(0, remainingTime * estNPS);
    }
    else
      throw new NotImplementedException();
  }


  /// <summary>
  /// Updates the TopNMove and TopNChildN properties.
  /// </summary>
  public void UpdateTopNodeInfo()
  {
    Debug.Assert(Engine.SearchRootNode.NodeRef.LockRef.IsLocked); 

    if (Engine.SearchRootNode.N > 1 && Engine.SearchRootNode.NumEdgesExpanded > 0)
    {
      GEdge edgeBestEdge = Engine.SearchRootNode.EdgesSorted(n => -n.N)[0];
      if (edgeBestEdge.Type == GEdgeStruct.EdgeType.ChildEdge)
      {
        using (new NodeLockBlock(edgeBestEdge.ChildNode))
        {
          if (TopNMove != default(MGMove) || (edgeBestEdge.MoveMG != TopNMove))
          {
            TopNMove = edgeBestEdge.MoveMGFromPos(in Engine.SearchRootPosMG);
            TopNChildN = edgeBestEdge.N;

            NumNodesWhenChoseTopNNode = Engine.SearchRootNode.N;
          }
        }
      }
      else
      {
        // TODO: how to get MGMove corresponding to a terminal edge
      }
    }
  }

}
