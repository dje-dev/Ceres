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

#endregion

namespace Ceres.Chess
{
  /// <summary>
  /// Defines the resource constraints applied to a search,
  /// such as maximum number of nodes or seconds per move or game.
  /// </summary>
  [Serializable]
  public record SearchLimit
  {
    /// <summary>
    /// Type of search limit (time or nodes)
    /// </summary>
    public SearchLimitType Type { init; get; }

    /// <summary>
    /// Search limit maximum value (time or nodes)
    /// </summary>
    public float Value { init; get; }

    /// <summary>
    /// Possible incremental search allowance for each move played.
    /// </summary>
    public float ValueIncrement { init; get; }

    /// <summary>
    /// Optional. If type is SecondsForAllMoves, the maxiumum number of 
    /// moves that must be made within the allotted time.
    /// </summary>
    public int? MaxMovesToGo { init; get; } = null;

    /// <summary>
    /// Optionally the maximum number of nodes which are allowed in the search tree
    /// at one time. Search is stopped if this limit is reached.
    /// </summary>
    public int? MaxTreeNodes { init; get; } = null;

    /// <summary>
    /// Optionally the maximum number of visits which are allowed in the search tree
    /// at one time. Search is stopped if this limit is reached.
    /// </summary>
    public int? MaxTreeVisits { init; get; } = null;

    /// <summary>
    /// If true then sufficent node storage is reserved so that
    /// the search can subsequently be expanded (without any practical bound).
    /// 
    /// Typically this is false only for specialized situations, such
    /// as when very large numbers of small fixed-sized searches are to be run concurrently.
    /// </summary>
    public bool SearchCanBeExpanded { init; get; } = true;

    /// <summary>
    /// Hint to the limits manager indicating the fraction by which 
    /// the search limit can be expanded dynamically
    /// if it is determined this would be particularly useful.
    /// </summary>
    public float FractionExtensibleIfNeeded { init; get; } = 0f;


    /// <summary>
    /// Optionally a list of moves to which the search is restricted.
    /// </summary>
    public List<Move> SearchMoves;

    #region Static Factory methods

    public static SearchLimit NodesPerMove(int nodes, bool searchContinuationSupported = true)
    {
      return new SearchLimit(SearchLimitType.NodesPerMove, nodes, searchContinuationSupported);
    }

    public static SearchLimit NodesPerTree(int nodes, bool searchContinuationSupported = true)
    {
      return new SearchLimit(SearchLimitType.NodesPerTree, nodes, searchContinuationSupported);
    }

    public static SearchLimit SecondsPerMove(float seconds, bool searchContinuationSupported = true)
    {
      return new SearchLimit(SearchLimitType.SecondsPerMove, seconds, searchContinuationSupported);
    }

    public static SearchLimit SecondsForAllMoves(float seconds, float secondsIncrement = 0, int? maxMovesToGo = null, bool searchContinuationSupported = true)
    {
      return new SearchLimit(SearchLimitType.SecondsForAllMoves, seconds, searchContinuationSupported, secondsIncrement, maxMovesToGo);
    }

    public static SearchLimit NodesForAllMoves(int nodes, int nodesIncrement = 0, int? maxMovesToGo = null, bool searchContinuationSupported = true)
    {
      return new SearchLimit(SearchLimitType.NodesForAllMoves, nodes, searchContinuationSupported, nodesIncrement, maxMovesToGo);
    }

    #endregion



    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="type"></param>
    /// <param name="value"></param>
    /// <param name="searchCanBeExpanded"></param>
    /// <param name="valueIncrement"></param>
    /// <param name="maxMovesToGo"></param>
    /// <param name="searchMoves"></param>
    /// <param name="fractionExtensibleIfNeeded"></param>
    /// <param name="maxTreeNodes"></param>
    /// <param name="maxTreeVisits"></param>
    public SearchLimit(SearchLimitType type, float value, bool searchCanBeExpanded = true,
                       float valueIncrement = 0, int? maxMovesToGo = null, List<Move> searchMoves = null,
                       float fractionExtensibleIfNeeded = 0f,
                       int? maxTreeNodes = null,
                       int? maxTreeVisits = null)
    {
      if (value < 0)
      {
        throw new ArgumentOutOfRangeException(nameof(value), "cannot be negative");
      }

      if (valueIncrement > 0 && !TypeIsPerGameLimit(type))
      {
        throw new Exception("SearchLimit having increment not supported in NodesPerMove or SecondsPerMove");
      }

      Type = type;
      Value = value;
      SearchCanBeExpanded = searchCanBeExpanded;
      ValueIncrement = valueIncrement;
      MaxMovesToGo = maxMovesToGo;
      MaxTreeNodes = maxTreeNodes;
      MaxTreeVisits = maxTreeVisits;
      if (searchMoves != null && searchMoves.Count > 0)
      {
        SearchMoves = searchMoves;

      }
      FractionExtensibleIfNeeded = fractionExtensibleIfNeeded;
    }

    #region Predicates

    public static bool TypeIsPerGameLimit(SearchLimitType type) => type == SearchLimitType.NodesForAllMoves 
                                                                || type == SearchLimitType.SecondsForAllMoves;
    public static bool TypeIsNodesLimit(SearchLimitType type) => type == SearchLimitType.NodesPerMove 
                                                              || type == SearchLimitType.NodesForAllMoves 
                                                              || type == SearchLimitType.NodesPerTree;
    public static bool TypeIsTimeLimit(SearchLimitType type) => type == SearchLimitType.SecondsPerMove 
                                                             || type == SearchLimitType.SecondsForAllMoves;

    public bool IsPerGameLimit => TypeIsPerGameLimit(Type);
    public bool IsNodesLimit => TypeIsNodesLimit(Type);
    public bool IsTimeLimit => TypeIsTimeLimit(Type);

    #endregion

    public SearchLimit WithIncrementApplied()
    {
      // Add search limit increment, if any
      if (ValueIncrement != 0 && IsPerGameLimit)
      {
        return this with { Value = Value + ValueIncrement };
      }
      else
      {
        return this;
      }
    }

    #region Multiplication and Addition operators

    /// <summary>
    /// Returns a new SearchLimit with the duration scaled by a specified factor.
    /// </summary>
    /// <param name="left"></param>
    /// <param name="scalingFactor"></param>
    /// <returns></returns>
    public static SearchLimit operator *(SearchLimit left, float scalingFactor)
    {
      if (scalingFactor <= 0)
      {
        throw new ArgumentOutOfRangeException(nameof(scalingFactor) + " must be a positive value");
      }

      return left with
      {
        Value = left.Value * scalingFactor,
        ValueIncrement = left.ValueIncrement * scalingFactor
      };
    }


    /// <summary>
    /// Returns the maximum number of nodes possible for this search
    /// if this can be determined, otherwise null.
    /// </summary>
    public int? KnownMaxNumNodes
    {
      get
      {
        if (Type == SearchLimitType.NodesPerMove 
         || Type == SearchLimitType.NodesPerTree) // upper bound
        {
          return (int)Value;
        }
        else
        {
          return null;
        }
      }
    }


    public int EstNumNodes(int curNumNodes, int estNumNodesPerSecond, bool estIsObserved)
    {
      // TODO: make the estimations below smarter
      return Type switch
      {
        SearchLimitType.NodesPerMove => (int)Value, 
        SearchLimitType.NodesPerTree  => (int)MathF.Max(Value - curNumNodes, 1),
        SearchLimitType.SecondsPerMove => (int)SecsToNodes(Value, estNumNodesPerSecond, estIsObserved),
        SearchLimitType.SecondsForAllMoves => (int)((Value / 20.0f) * estNumNodesPerSecond),
        SearchLimitType.NodesForAllMoves => (int)(Value / 20.0f),
        _ => throw new NotImplementedException()
      };
    }

    static float SecsToNodes(float secs, int estNumNodesPerSecond, bool estNodesIsObserved)
    {
      if (!estNodesIsObserved && secs < 0.1)
      {
        // first nodes are much slower due to lagency
        return secs * estNumNodesPerSecond * 0.3f;
      }
      else
      {
        return secs * estNumNodesPerSecond;
      }
    }

    #endregion

    /// <summary>
    /// Converts SearchLimit to from a per-game limit to a per-move limit, if applicable.
    /// </summary>
    public SearchLimit ConvertedGameToMoveLimit
    {
      get
      {
        if (Type == SearchLimitType.NodesForAllMoves)
        {
          return this with { Type = SearchLimitType.NodesPerMove };
        }
        else
        {
          return Type == SearchLimitType.SecondsForAllMoves ? (this with { Type = SearchLimitType.SecondsPerMove }) : this;
        }
      }
    }

    /// <summary>
    /// The maximum value (nodes or seconds) that would be allowed
    /// cumulatively after a given number of moves.
    /// </summary>
    /// <param name="numMoves"></param>
    /// <returns></returns>
    public float MaxValueAfterMoves(int numMoves)
    {
      if (IsPerGameLimit)
      {
        return Value + ValueIncrement * numMoves;
      }
      else
      {
        // Not applicable.
        return 0;
      }
    }


    /// <summary>
    /// Returns a short code string indicating the type of limit.
    /// </summary>
    public string TypeShortStr => Type switch
    {
      SearchLimitType.NodesForAllMoves => "NG",
      SearchLimitType.NodesPerMove => "NM",
      SearchLimitType.NodesPerTree => "NT",
      SearchLimitType.SecondsForAllMoves => "SG",
      SearchLimitType.SecondsPerMove => "SM",
      _ => throw new Exception($"Internal error: unsupported SearchLimitType type {Type}")
    };


    /// <summary>
    /// Returns string summary of SearchLimit.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      string movesToGoPart = MaxMovesToGo.HasValue ? $" Moves { MaxMovesToGo }" : "";
      string valuePart = IsTimeLimit ? $"{Value,8:F2}s" : $"{Value,12:N0} nodes";
      string incrPart = IsTimeLimit ? $"{ValueIncrement,8:F2}s" : $"{ValueIncrement,12:N0} nodes";

      string maxNodes = MaxTreeNodes != null ? $" Nodes max {MaxTreeNodes,12:N0}" : "";
      string maxVisits = MaxTreeVisits != null ? $" Visits max {MaxTreeVisits,12:N0}" : "";
      string searchMovesPart = "";
      if (SearchMoves != null)
      {
        searchMovesPart = " searchmoves ";
        for (int i = 0; i < SearchMoves.Count; i++)
        {
          searchMovesPart += SearchMoves[i] + " ";
        }
      }

      return $"<{TypeShortStr,-4}{valuePart}{(ValueIncrement > 0 ? $" +{incrPart}" : "")}{movesToGoPart}{maxNodes}{maxVisits}{searchMovesPart}>";
    }
  }
}
