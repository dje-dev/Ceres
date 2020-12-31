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
using System.Runtime.InteropServices;

using Ceres.Chess;
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions.Basic;
using System.Diagnostics;

#endregion

namespace Ceres.MCTS.MTCSNodes.Struct
{
  /// <summary>
  /// Raw data fields appearing in MCTS node structure. 
  /// 
  /// Note that the structure size is exactly 64 bytes 
  /// to optimize memory access efficiency (cache line alignment).
  /// </summary>
  [Serializable]
  [StructLayout(LayoutKind.Sequential, Pack = 1, Size = 64)]
  public partial struct MCTSNodeStruct
  {
    /// <summary>
    /// Number of visits to children
    /// N.B. This must appear first in the struture (see static constructor where refererence in fixed statement)
    /// </summary>
    public int N;

    /// <summary>
    /// Sum of all V from children
    /// Represented in high precision (double) to avoid rounding errors causing:
    ///   - slight non-determinism of results due to floating point underflow behavior, and
    ///   - potentially substantial errors in large (millions of nodes) search trees because
    ///     the W for nodes higher in the tree can become very large and small updates 
    ///     could potentially be completely lost (rounded to zero change)
    /// </summary>
    public double W;

    /// <summary>
    /// Policy probability
    /// </summary>
    public FP16 P;

    /// <summary>
    /// Number of times node has been visited during current batch (from selector 0).
    /// </summary>
    public short NInFlight;

    /// <summary>
    /// Number of times node has been visited during current batch (from selector 1)
    /// </summary>
    public short NInFlight2;

    /// <summary>
    /// Win probability
    /// </summary>
    public FP16 WinP;

    /// <summary>
    /// Loss probabilty
    /// </summary>
    public FP16 LossP;

    /// <summary>
    ///  Moves left estimate for this position
    /// </summary>
    public FP16 MPosition;

    /// <summary>
    /// Accumulator for M (moves left) values across all visits.
    /// </summary>
    private float mSum;

    /// <summary>
    /// Accumulator for D (draw) values across all visits.
    /// </summary>
    private float dSum;

    /// <summary>
    /// Number of policy moves (children)
    /// Possibly this set of moves is incomplete due to either:
    ///   - implementation decision to "throw away" lowest probability moves to save storage, or
    ///   - error in policy evaluation which resulted in certain legal moves not being recognized
    /// </summary>
    public byte NumPolicyMoves;

    /// <summary>
    /// Game terminal status
    /// </summary>
    public GameResult Terminal;

    /// <summary>
    /// Index of the parent, or null if root node
    /// </summary>
    public MCTSNodeStructIndex ParentIndex;


    public int childStartBlockIndex;

    /// <summary>
    /// The move was just played to reach this node (or default if root node)
    /// </summary>
    public EncodedMove PriorMove;

    /// <summary>
    /// Zobrist hash value of the position.
    /// </summary>
    public ulong ZobristHash;

    /// <summary>
    /// The number of children that have been visited in the current search
    /// Note that children are always visited in descending order by the policy prior probability.
    /// Also note that in rare circumstances nodes could have been visited (as counted by this nubmer)
    /// but might actually have an N of 0 (e.g if they were abandoned nodes due to NN buffer hitting its optimal batch size).
    /// NOTE: Values above 64 are special values used by transposition logic (to store NumNodesTranspositionExtracted)
    /// </summary>
    byte numChildrenVisited;

    /// <summary>
    /// Number of children which have been expanded 
    /// (a corresponding node created in the tree).
    /// </summary>
    public byte NumChildrenExpanded;

    /// <summary>
    /// If at least one of the children has been found to 
    /// be a draw (terminal node).
    /// </summary>
    public byte DrawKnownToExistAmongChildren;

    /// <summary>
    /// Nodes active in current tree have generation 0.
    /// Nodes with higher generations can exist when tree reuse is enabled,
    /// and indicate nodes left behind by prior search (but no longer valid).
    /// The generation number indicates how many moves prior the node was last active.
    /// TODO: currently we only maintain these values if enabled TREE_REUSE_RETAINED_POSITION_CACHE_ENABLED
    /// </summary>
    public byte ReuseGenerationNum;

    /// <summary>
    /// If nonzero represents the index of the node an associated IMCTSNodeCache.
    /// 
    /// TODO: this could be compressed into 3 bytes
    /// </summary>
    public int CacheIndex;

    /// <summary>
    /// V value as returned by an optional secondary network.
    /// </summary>
    public FP16 VSecondary;

    public short Unused1;

    public void ResetSearchInProgressState()
    {
      Debug.Assert(!IsTranspositionLinked);

      NInFlight = 0;
      NInFlight2 = 0;
      CacheIndex = 0;
    }

    public void ResetExpandedState()
    {
      Debug.Assert(!IsTranspositionLinked);

      N = 0;
      NInFlight = 0;
      NInFlight2 = 0;

      W = 0;
      mSum = 0;
      dSum = 0;

      numChildrenVisited = 0;
      DrawKnownToExistAmongChildren = 0;
      CacheIndex = 0;

      VSumSquares = 0;

      QUpdatesWtdAvg = 0;
    }

    public void Dump()
    {
      Console.WriteLine($"[Inferred index {Index}]  ParentIndex {ParentIndex} ReuseGenerationNum {ReuseGenerationNum}");
      Console.WriteLine($"PriorMove {PriorMove}");
      Console.WriteLine($"ZobristHash {ZobristHash}");

      Console.WriteLine($"N {N,-11}  NInFlight {NInFlight,-4}  NInFlight2 {NInFlight2,-4} ");
      Console.WriteLine($"P {P,10:F3}  MPosition {MPosition,10:F3}");
      Console.WriteLine($"NumPolicyMoves         {NumPolicyMoves,-4}  NumChildrenExpanded {NumChildrenExpanded,-4} NumChildrenExpanded {NumChildrenExpanded,-4}");
      Console.WriteLine($"ChildStartBlockINdex   { childStartBlockIndex}");

      Console.WriteLine($"WinP {WinP,10:F3} LossP {LossP,10:F3}");
      Console.WriteLine($"W/N    (V)       {W/N,10:F4}");
      Console.WriteLine($"DSum/N (DAvg)    {dSum / N,10:F4}");
      Console.WriteLine($"MSum/N (MAvg)    {mSum / N,10:F4}");
      Console.WriteLine($"VSecondary       {VSecondary ,10:F4}");
      Console.WriteLine($"Terminal         {Terminal}");
      Console.WriteLine($"DrawCanBeClaimed {DrawKnownToExistAmongChildren}");

    }
  }

}