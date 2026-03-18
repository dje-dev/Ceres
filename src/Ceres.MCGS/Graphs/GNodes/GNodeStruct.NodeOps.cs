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

using Ceres.Base.Math;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GraphStores;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Various support methods for GNodeStruct.
/// </summary>
public partial struct GNodeStruct
{
  public delegate float MCTSNodeStructMetricFunc(in GNodeStruct node);

  /// <summary>
  /// If this node is possibly reachable (appears as a descendent in the full game tree)
  /// from a specified prior node (using approximate heuristics).
  /// </summary>
  /// <param name="priorNode"></param>
  /// <returns></returns>
  public readonly bool IsPossiblyReachableFrom(in GNodeStruct priorNode)
    => NumPieces <= priorNode.NumPieces
    && NumRank2Pawns <= priorNode.NumRank2Pawns;


  /// <summary>
  /// Returns the MGPosition corresponding to this node.
  /// 
  /// NOTE: this is inefficient, requiring descent from root including move generation at each level.
  /// </summary>
  /// <param name="store"></param>
  /// <param name="nodeRef"></param>
  /// <returns></returns>
  public MGPosition CalcPosition(GraphStore store)
  {
    if (IsOldGeneration)
    {
      throw new Exception("Internal error: CalcPosition net yet supported for old generation nodes.");
    }

    ref readonly GNodeStruct visitorNodeRef = ref this;

    // Ascend up to root, keeping track of all moves along the way.
    Span<GNodeStruct> nodes = store.NodesStore.Span;
    Span<ushort> moves = stackalloc ushort[255];
    int index = 0;
    while (!visitorNodeRef.IsGraphRoot)
    {
      throw new NotImplementedException("next line needs remediation");
//        moves[index++] = visitorNodeRef.PriorMove.RawValue;
      visitorNodeRef = ref nodes[visitorNodeRef.ParentIndex.Index];
    }

    // Now reverse the ascent, tracking the position along the way.
    // TODO: would it be possible to sometimes or always avoid the move generation
    //       and instead use the EncodedMove directly?
    MGPosition pos = store.NodesStore.PositionHistory.FinalPosMG;
    if (index > 0)
    {
      for (int i = index - 1; i >= 0; i--)
      {
        EncodedMove move = new(moves[i]);
        MGMove moveMG = MGMoveConverter.ToMGMove(in pos, move);
        pos.MakeMove(moveMG);
      }
    }
    return pos;
  }


  #region Updates

  /// <summary>
  /// Helper method to atomically add a signed short to a ushort using a 16-bit CAS retry loop.
  /// 
  /// A 16-big CAS retry-loop is used because Interlocked primitives do not work on 16 bit types.
  /// </summary>
  /// <param name="target"></param>
  /// <param name="delta"></param>
  /// <returns></returns>
  private static ushort InterlockedAddUShort(ref ushort target, short delta)
  {
    Debug.Assert(delta != 0);

    while (true)
    {
      // ake the current snapshot (atomic because ushort is naturally aligned).
      ushort oldVal = Volatile.Read(ref target);

      // Calculate the proposed new value in a wider type.
      int newInt = oldVal + delta;
      Debug.Assert((uint)newInt <= ushort.MaxValue);

      ushort newVal = (ushort)newInt;

      // Try to swap.  If another thread beat us, retry.
      ushort observed = Interlocked.CompareExchange(ref target, newVal, oldVal);
      if (observed == oldVal)
      { 
        return newVal;               
      }

      Thread.SpinWait(0);   // tiny back-off under heavy contention
    }
  }


  /// <summary>
  /// Atomically adds delta to either NumInFlight0 or NumInFlight1.
  /// </summary>
  /// </remarks>
  public unsafe static void UpdateEdgeNInFlightForIterator(GEdge edge, int iteratorID, int adjust)
  {
    Debug.Assert(iteratorID is 0 or 1);
    Debug.Assert(adjust >= short.MinValue && adjust <= short.MaxValue);

    if (iteratorID == 0)
    {
      Debug.Assert(edge.edgeStructPtr->NumInFlight0 + adjust >= 0);
//    Interlocked.Add(ref edge.edgeStructPtr->NumInFlight0, adjust);
      InterlockedAddUShort(ref edge.edgeStructPtr->NumInFlight0, (short)adjust);
    }
    else
    {
      Debug.Assert(edge.edgeStructPtr->NumInFlight1 + adjust >= 0);
//    Interlocked.Add(ref edge.edgeStructPtr->NumInFlight1, adjust);
      InterlockedAddUShort(ref edge.edgeStructPtr->NumInFlight1, (short)adjust);
    }
  }




#endregion


  /// <summary>
  /// Returns if node corresponds to a position with white to move.
  /// </summary>
  public bool IsWhite
  {
    readonly get => miscFields.IsWhite;
    internal set => miscFields.IsWhite = value;
    
  }


  /// <summary>
  /// Returns the new value for a variance accumulator with exponential weighting
  /// that reflects an update with a specified new update (repeated thisN times).
  /// </summary>
  static double NewEMWVarianceAcc(double priorAcc, double priorN, double squaredDeviation, int thisN, double lambda)
  {
    double newAcc = priorAcc;
    for (int i = 0; i < thisN; i++)
    {
      double priorVariance = newAcc / (priorN + i);

      double newVariance = priorVariance * (1.0f - lambda)
                        + squaredDeviation * lambda;
      newAcc = newVariance * (priorN + i + 1);
    }

    // Return the variance accumulator value which would now return 
    // our new variance target after the current sample is recorded
    return newAcc;
  }
}

