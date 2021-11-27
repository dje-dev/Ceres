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
using System.Runtime.CompilerServices;

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
  public unsafe partial struct MCTSNodeStruct
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
    /// Set of packed miscellaneous fields.
    /// </summary>
    MCTSNodeStructMiscFields miscFields;
    
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


    const byte MAX_M = 255;

    private byte mPosition;

    /// <summary>
    ///  Moves left estimate for this position
    /// </summary>
    public byte MPosition
    {
      readonly get => mPosition;
      set => mPosition = Math.Min(MAX_M, value);
    }


    /// NOTE: TEMPORARILY REMOVED/DISABLED to leave space for variance term
    const bool MSUM_ENABLED = false;

    /// <summary>
    /// Accumulator for M (moves left) values across all visits.
    /// 
    /// </summary>
    internal FP16 mSum
    {
      // NOTE: this field probably suffers from an accumulation precision problem
      readonly get => FP16.NaN;
      set
      {
        Debug.Assert(!MSUM_ENABLED);
        //if (!FP16.IsNaN(value)) throw new NotImplementedException();
      }

    }


    /// <summary>
    /// Accumulator for D (draw) values across all visits.
    /// </summary>
    internal float dSum;

    /// <summary>
    /// Game terminal status
    /// </summary>
    public GameResult Terminal
    {
      readonly get => miscFields.Terminal;
      set => miscFields.Terminal = value;
    }

    /// <summary>
    /// If the position has one more repetitions in the move history.
    /// </summary>
    public bool HasRepetitions
    {
      readonly get => miscFields.HasRepetitions;
      set => miscFields.HasRepetitions = value;
    }

    /// <summary>
    /// If the node was evaluated by the secondary (alternate) neural network.
    /// </summary>
    public bool SecondaryNN
    {
      readonly get => miscFields.SecondaryNN;
      set => miscFields.SecondaryNN = value;
    }

    /// <summary>
    /// Value of test flag (miscellaneous ad hoc tests).
    /// </summary>
    public bool TestFlag
    {
      readonly get => miscFields.TestFlag;
      set => miscFields.TestFlag = value;
    }


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
    /// Number of policy moves (children)
    /// Possibly this set of moves is incomplete due to either:
    ///   - implementation decision to "throw away" lowest probability moves to save storage, or
    ///   - error in policy evaluation which resulted in certain legal moves not being recognized
    /// </summary>
    public byte NumPolicyMoves;

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
    /// Only start accumulating variance statistics after a minimum number of visits
    /// to avoid building statistics when the Q is still highly noisy.
    /// </summary>
    public const int VARIANCE_START_ACCUMULATE_N = 10;

    /// <summary>
    /// Accumulate the squared deviations from mean (Q) at the time of each backup
    /// (starting after VARIANCE_START_ACCUMULATE_N visits).
    /// </summary>
    public float VarianceAccumulator;

#if NOT
/// <summary>
    /// The 64 bit Zobrist hash is used to find nodes in the transposition table
    /// within the same hash equivlance class. However hash collisions will
    /// occur (perhaps ever 300 million to 3 billion positions) and establishing
    /// incorrect linkages could lead to incorrect valuations or invalid move lists
    /// being propagated to the linked nodes.
    /// 
    /// The HashCrossheck is an independent 8 bit hash value used 
    /// as an additional crosscheck for equality before establishing linkages
    /// to transposition nodes to greatly reduce their likelihood.
    /// </summary>
    public byte HashCrosscheck;
#endif

    /// <summary>
    /// If at least one of the children has been found to 
    /// be a checkmate (terminal node).
    /// </summary>
    public bool CheckmateKnownToExistAmongChildren
    {
      readonly get => miscFields.CheckmateKnownToExistAmongChildren;
      set => miscFields.CheckmateKnownToExistAmongChildren = value;
    }

    /// <summary>
    /// If at least one of the children has been found to 
    /// be a draw (terminal node).
    /// </summary>
    public bool DrawKnownToExistAmongChildren
    {
      readonly get => miscFields.DrawKnownToExistAmongChildren;
      set => miscFields.DrawKnownToExistAmongChildren = value;
    }

    /// <summary>
    /// Number of pieces on the board.
    /// </summary>
    public byte NumPieces
    {
      readonly get => miscFields.NumPieces;
      set => miscFields.NumPieces = value;
    }

    /// <summary>
    /// Number of pawns still on their starting square.
    /// </summary>
    public byte NumRank2Pawns
    {
      readonly get => miscFields.NumRank2Pawns;
      set => miscFields.NumRank2Pawns = value;
    }


    /// <summary>
    /// Nodes active in current tree have generation 0.
    /// Nodes with higher generations can exist when tree reuse is enabled,
    /// and indicate nodes left behind by prior search (but no longer valid).
    /// The generation number indicates how many moves prior the node was last active.
    /// TODO: currently we only maintain these values if enabled TREE_REUSE_RETAINED_POSITION_CACHE_ENABLED
    /// </summary>
    public bool IsOldGeneration   
    {
      readonly get => miscFields.IsOldGeneration;
      set => miscFields.IsOldGeneration = value;
    }

    /// <summary>
    /// If the node was successfully added to the transposition root dictionary
    /// as a transposition root.
    /// </summary>
    public bool IsTranspositionRoot 
    {
      readonly get => miscFields.IsTranspositionRoot;
      set => miscFields.IsTranspositionRoot = value;
    }

    /// <summary>
    /// If the node is in the (very brief) process of being transposition unlinked.
    /// </summary>
    internal bool TranspositionUnlinkIsInProgress
    {
      readonly get => miscFields.TranspositionUnlinkIsInProgress;
      set => miscFields.TranspositionUnlinkIsInProgress = value;
    }

    /// <summary>
    /// If nonzero represents the index of the node an associated IMCTSNodeCache.
    /// 
    /// TODO: this could be compressed into 3 bytes
    /// </summary>
    public void* CachedInfoPtr;

    /// <summary>
    /// Returns reference to associated MCTSNodeInfo.
    /// </summary>
    public ref MCTSNodeInfo InfoRef => ref Unsafe.AsRef<MCTSNodeInfo>(CachedInfoPtr);

    /// <summary>
    /// If the associated MCTSNodeInfo is present in the cache.
    /// </summary>
    public readonly bool IsCached => CachedInfoPtr != null;


    /// <summary>
    /// V value as returned by an optional secondary network.
    /// </summary>
    public FP16 VSecondary
    {
      readonly get => FP16.NaN;
      set 
      { 
        if (!FP16.IsNaN(value)) throw new NotImplementedException(); 
      }
    }

    public FP16 Uncertainty
    {
      readonly get => FP16.NaN;
      set
      {
        if (!FP16.IsNaN(value)) throw new NotImplementedException();
      }
    }


  // TODO: make compile time constant
  internal FP16 UNCERTAINTY_PRIOR => (FP16)0.10f;

    public void ResetSearchInProgressState()
    {
      Debug.Assert(!IsTranspositionLinked);

      NInFlight = 0;
      NInFlight2 = 0;
      CachedInfoPtr = null;
    }


    /// <summary>
    /// 
    /// NOTE: try to keep changes in sync with MCTSNodeStruct.Initialize.
    /// </summary>
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
      DrawKnownToExistAmongChildren = false;
      CachedInfoPtr = null;

#if FEATURE_UNCERTAINTY
      Uncertainty = UNCERTAINTY_PRIOR;
#endif
      VSumSquares = 0;
      VarianceAccumulator = 0;

      QUpdatesWtdAvg = 0;

    }

    public readonly void Dump()
    {
      Console.WriteLine($"[Inferred index {Index}]  ParentIndex {ParentIndex} ReuseGenerationNum {IsOldGeneration}");
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