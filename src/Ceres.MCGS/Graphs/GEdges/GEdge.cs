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
using System.Runtime.CompilerServices;
using System.Threading;
using Ceres.Base.DataTypes;
using Ceres.Base.Misc;

using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Utils;

#endregion

namespace Ceres.MCGS.Graphs.GEdges;

/// <summary>
/// Represents an edge from a parent to a child node, providing convenient access to
/// the parent, child, and edge-specific data.
/// 
/// </summary>
public unsafe readonly struct GEdge : IEquatable<GEdge>
{
  const int MAX_CHILD_INDEX = 255;

  /// <summary>
  /// Pointer to the edge structure
  /// </summary>
  internal readonly GEdgeStruct* edgeStructPtr;

  /// <summary>
  /// Graph reference shared by both parent and child nodes.
  /// </summary>
  private readonly Graph graph;

  /// <summary>
  /// Pointer to the parent node structure.
  /// </summary>
  private readonly GNodeStruct* parentNodePtr;

  /// <summary>
  /// Pointer to the child node structure.
  /// </summary>
  private readonly GNodeStruct* childNodePtr;

  /// <summary>
  /// Parent node (constructed on the fly).
  /// </summary>
  public readonly GNode ParentNode => new GNode(graph, parentNodePtr);

  /// <summary>
  /// Child node (constructed on the fly).
  /// </summary>
  public readonly GNode ChildNode => new GNode(graph, childNodePtr);


  /// <summary>
  /// Constructor for an expanded edge from a pointer to a GEdgeStruct.
  /// </summary>
  /// <param name="edgePtr"></param>
  /// <param name="parent"></param>
  /// <param name="child"></param>
  internal GEdge(GEdgeStruct* edgePtr, GNode parent, GNode child)
  {
    edgeStructPtr = edgePtr;
    graph = parent.Graph;
    parentNodePtr = parent.nodePtr;
    childNodePtr = child.nodePtr;
  }


  /// <summary>
  /// Constructor from a reference to a GEdgeStruct.
  /// </summary>
  /// <param name="edgeRef"></param>
  /// <param name="parent"></param>
  /// <param name="child"></param>
  internal GEdge(ref GEdgeStruct edgeRef, GNode parent, GNode child)
    : this((GEdgeStruct*)Unsafe.AsPointer(ref edgeRef), parent, child)
  {

  }


  /// <summary>
  /// Gets the graph associated with the current node.
  /// </summary>
  public Graph Graph => graph;

  /// <summary>
  /// If the GEdge is uninitialized.
  /// </summary>
  public readonly bool IsNull => Type == GEdgeStruct.EdgeType.Uninitialized;


  /// <summary>
  /// Issues processor prefetch memory hint for this edge structure.
  /// </summary>
  public readonly void Prefetch()
  {
    if (MCGSParamsFixed.PrefetchCacheLevel != Prefetcher.CacheLevel.None && edgeStructPtr != null)
    {
      Prefetcher.PrefetchLevel1(edgeStructPtr);
    }
  }


  #region Accessors to underlying struct

  #region Fields from parent's policy head

  /// <summary>
  /// Policy probability from policy head as a fraction.
  /// </summary>
  public readonly FP16 P => edgeStructPtr->P;


  /// <summary>
  /// Move that transitions board state between parent and child of this edge.
  /// </summary>
  public readonly EncodedMove Move => edgeStructPtr->Move;



#if ACTION_ENABLED
  /// <summary>
  /// Action value from action head (if available).
  /// </summary>
  public readonly FP16 ActionV => edgeStructPtr->ActionV;


  /// <summary>
  /// Action U placeholder (currently non-functional).
  /// </summary>
  public readonly FP16 ActionU => edgeStructPtr->ActionU;
#endif


  #endregion

  #region Fields propagated from child

  /// <summary>
  /// Value uncertainty.
  /// </summary>
  public readonly float UncertaintyV
  {
    get => edgeStructPtr == null ? float.NaN : edgeStructPtr->UncertaintyV;
    set => edgeStructPtr->UncertaintyV = value;
  }

  /// <summary>
  /// Policy uncertainty.
  /// </summary>
  public float UncertaintyP
  {
    readonly get => edgeStructPtr == null ? float.NaN : edgeStructPtr->UncertaintyP;
    set => edgeStructPtr->UncertaintyP = value;
  }

  /// <summary>
  /// Sets both uncertainty values (value and policy) efficiently in one operation.
  /// </summary>
  /// <param name="uncertaintyValue">Value uncertainty</param>
  /// <param name="uncertaintyPolicy">Policy uncertainty</param>
  public void SetUncertaintyValues(float uncertaintyValue, float uncertaintyPolicy)
   => edgeStructPtr->SetUncertaintyValues(uncertaintyValue, uncertaintyPolicy);
  

  #endregion

  #region Search accumulation values

  /// <summary>
  /// Number of visits to this child.
  /// </summary>
  public int N
  {
    readonly get => edgeStructPtr->N;
    set => edgeStructPtr->N = value;
  }


  /// <summary>
  /// Count of the N which were actually draw by repetition visits
  /// (not actually recorded in the child N).
  /// </summary>
  public int NDrawByRepetition
  {
    readonly get => edgeStructPtr->NDrawByRepetition;
    set => edgeStructPtr->NDrawByRepetition = value;
  }


  /// <summary>
  /// Number of in-flight virtual visits (iterator 0).
  /// </summary>
  public readonly int NumInFlight0 => edgeStructPtr->NumInFlight0;


  /// <summary>
  /// Number of in-flight virtual visits (iterator 1).
  /// </summary>
  public readonly int NumInFlight1 => edgeStructPtr->NumInFlight1;


  /// <summary>
  /// If the child node has DrawKnownToExist set to true.
  /// </summary>
  public bool ChildNodeHasDrawKnownToExist
  {
    get => edgeStructPtr != null && edgeStructPtr->ChildNodeHasDrawKnownToExist;
    set => edgeStructPtr->ChildNodeHasDrawKnownToExist = value;
  }


  /// <summary>
  /// If the child Q is known to have changed since 
  /// this edge Q was last updated and propagated to the parent Q.
  /// </summary>
  public bool IsStale
  {
    readonly get => edgeStructPtr != null && edgeStructPtr->IsStale;
    set => edgeStructPtr->IsStale = value;
  }

  /// <summary>
  /// Updates the standard deviation estimate with a new sample.
  /// </summary>
  /// <param name="mean"></param>
  /// <param name="sample"></param>
  public void AddUpdateSample(double mean, double sample)  => edgeStructPtr->StdDevEstimate.AddSample(mean, sample);

  #endregion

  #region Updates

  /// <summary>
  /// Adds specified delta to NDrawByRepetition in a thread-safe manner.
  /// </summary>
  /// <param name="delta"></param>
  public void IncrementNDrawRepetition(int delta) => Interlocked.Add(ref edgeStructPtr->nDrawByRepetition, delta);
 
  
  #endregion

  #region Pointers and metadata

  /// <summary>
  /// Index of child node to which this edge leads.
  /// </summary>
  public NodeIndex ChildNodeIndex
  {
    readonly get => edgeStructPtr->ChildNodeIndex;
    set => edgeStructPtr->ChildNodeIndex = value;
  }


  /// <summary>
  /// True if this edge has been expanded (i.e. child node exists).
  /// </summary>
  public readonly bool IsExpanded => Type != GEdgeStruct.EdgeType.Uninitialized;


  /// <summary>
  /// Aggregate Q which combines visits to child and any possible draw by repetition visits).
  /// (the NDrawByRepetition count is used to dilute child Q 
  /// proportionally to the draw visits treating draws as 0).
  /// </summary>
  public readonly double Q => edgeStructPtr->Q;
  

  /// <summary>
  /// Q value of the child node as of most recent update.
  /// </summary>
  public double QChild
  {
    readonly get => edgeStructPtr->QChild;
    set => edgeStructPtr->QChild = value;
  }


  /// <summary>
  /// Type of edge (uninitialized, child, or terminal).
  /// </summary>
  public readonly GEdgeStruct.EdgeType Type => edgeStructPtr == null ? GEdgeStruct.EdgeType.Uninitialized : edgeStructPtr->Type;

  #endregion

  /// <summary>
  /// Returns MGMove corresponding to this edge.
  /// N.B. For greater efficiency use MoveMGFromPos() instead of this property.
  /// </summary>
  public readonly MGMove MoveMG => ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(Move, ParentNode.CalcPosition());


  /// <summary>
  /// Returns MGMove corresponding to this edge (starting from a given position).
  /// </summary>
  /// <param name="parentPos"></param>
  /// <returns></returns>
  public readonly MGMove MoveMGFromPos(in MGPosition parentPos) => ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(Move, in parentPos);


  /// <summary>
  /// Returns the NInFlight for a specified iterator.
  /// </summary>
  /// <param name="iteratorID"></param>
  /// <returns></returns>
  /// <exception cref="Exception"></exception>
  public readonly int NInFlightForIterator(int iteratorID)
      => iteratorID switch
      {
        0 => NumInFlight0,
        1 => NumInFlight1,
        _ => throw new Exception("Invalid iterator ID for GEdge")
      };


  /// <summary>
  /// Returns child position.
  /// </summary>
  /// <returns></returns>
  public MGPosition CalcChildPosition()
  {
    if (Type == GEdgeStruct.EdgeType.ChildEdge)
    {
      return ChildNode.CalcPosition();
    }
    else
    {
      MGPosition parentPos = ParentNode.CalcPosition();
      parentPos.MakeMove(MoveMG);
      return parentPos;
    }
  }

  #endregion

  #region Equality and Comparison operators

  /// <summary>
  /// Returns if two GEdge represent the same edge.
  /// </summary>
  /// <param name="edge"></param>
  /// <returns></returns>
  public bool Equals(GEdge edge) => edge.edgeStructPtr == edgeStructPtr;

  public static bool operator ==(GEdge lhs, GEdge rhs) => lhs.Equals(rhs);

  public static bool operator !=(GEdge lhs, GEdge rhs) => !lhs.Equals(rhs);

  public override bool Equals(object obj)
  {
    if (obj is GEdge edge)
    {
      return Equals(edge);
    }
    return false;
  }

  public override int GetHashCode() => ParentNode.Index.Index.GetHashCode() ^ ChildNode.Index.Index.GetHashCode();

  public override string ToString() => DoToString(ParentNode.IsWhite, false); // not knowable if is white or black


  public string ToString(bool shortFor) => DoToString(ParentNode.IsWhite, true); // not knowable if is white or black


  private string DoToString(bool isWhite, bool shortForm)
  {
    string transposeStr = "";
    if (this.IsExpanded && !Type.IsTerminal() && ChildNode.NumParentsMoreThanOne)
    {
      transposeStr = " T";
    }

    string ret = $"<GEdge {Type.ToString().Replace("Edge", "")} {MoveMG} {transposeStr} (#{ParentNode.Index.Index}";
    ret += Type == GEdgeStruct.EdgeType.ChildEdge ? $"->#{ChildNode.Index.Index}"
           : (Type.IsTerminal() ? "->TERMINAL" : "");
    ret += $")  {(isWhite ? Move : Move.Flipped)} ";
    string qPrefix = IsStale ? "*" : "";
    if (IsExpanded)
    {
      if (shortForm)
      {
        ret += $" N={N:N0} Q={Q,5:F3} SD={edgeStructPtr->StdDevEstimate.RunningStdDev,5:F2} ";

      }
      else
      {
#if ACTION_ENABLED
    ret+= $" P={P * 100.0f,6:F2}% A={ActionV,6:F3} N={N,8:N0} Q={Q,6:F3} W={W,6:F3} UV={UncertaintyV,6:F3} UP={UncertaintyP,6:F3} "; 
#else
        ret += $" P={P * 100.0f,6:F2}% N={N:N0} ND={NDrawByRepetition:N0} Q={Q,5:F3}{qPrefix} QC={QChild,5:F3} SD={edgeStructPtr->StdDevEstimate.RunningStdDev,5:F2} UV={UncertaintyV,3:F2} UP={UncertaintyP,3:F2} ";
#endif
      }

      ret += $" InFl={NumInFlight0}/{NumInFlight1} ";

      if (Type.IsTerminal() || shortForm)
      {
        ret += $"[{Type}]";
      }
      else if (Type == GEdgeStruct.EdgeType.ChildEdge)
      {
        ret += $"->[{ChildNode.Index.Index} N={ChildNode.NodeRef.N} Q={ChildNode.NodeRef.Q,6:F3}{qPrefix} "
             + $"{ChildNode.NodeRef.Terminal} ";
      }
    }
    else
    {
      ret += $" P={P * 100.0f,5:F2}% UV={UncertaintyV,3:F2} UP={UncertaintyP,3:F2} ";
    }

    return ret;
  }

  #endregion
}
