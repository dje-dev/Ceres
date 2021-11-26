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
using System.Diagnostics;
using System.Runtime.CompilerServices;

using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.MCTS.Evaluators;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes.Annotation;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.Params;
using Ceres.Chess.MoveGen;
using System.Runtime.InteropServices;

#endregion

[assembly: InternalsVisibleTo("Ceres.EngineMCTS.Test")]

namespace Ceres.MCTS.MTCSNodes
{
  /// <summary>
  /// Wrapper around a raw MCTSNodeStruct existing in the node store.
  /// 
  /// Also contains additional properties used transiently in tree operations such as search.
  /// </summary>
  public unsafe partial struct MCTSNodeInfo :
      IComparable<MCTSNodeInfo>,
      IEquatable<MCTSNodeInfo>,
      IEqualityComparer<MCTSNodeInfo>
  {
    /// <summary>
    /// Pointer directly to this structure
    /// </summary>
    internal MCTSNodeStruct* ptr { get; private set; }


    #region Object references

    // N.B. These object references need special treatment.
    //      Care needs to be taken to keep them up to date
    //      with the active search, and clear them when done.
    //      Additions/changes to these fields also need 
    //      coordinated code changes in IMCTSNodeCache (SetContext).

    /// <summary>
    /// Search context within which this node exists
    /// </summary>
    public MCTSIterator Context { get; internal set; }

    /// <summary>
    /// Shortcut directly to associated MCTSTree.
    /// </summary>
    public MCTSTree Tree { get; internal set; }

    /// <summary>
    /// Shortcut directly to associated MCTSNodeStore.
    /// </summary>
    public MCTSNodeStore Store { get; internal set; }

    public void ClearObjectReferences()
    {
      Context = null;
      Tree = null;
      Store = null;
    }

    public void SetUninitialized()
    {
      ptr = null;
      index = default;
      parent = default;
      ClearObjectReferences();
    }

    #endregion

    MCTSNode parent;

    /// <summary>
    /// Index of this structure within the array
    /// </summary>
    internal MCTSNodeStructIndex index;


    public enum NodeActionType : short { NotInitialized, None, MCTSApply, CacheOnly };

    public NodeActionType ActionType;


    /// <summary>
    /// Constructor which creates an MCTSNode wrapper for the raw node at specified index.
    /// </summary>
    /// <param name="context"></param>
    /// <param name="index"></param>
    /// <param name="parent">optionally the parent node</param>
    internal MCTSNodeInfo(MCTSIterator context, MCTSNodeStructIndex index, MGMoveList moveList)
    {
      Debug.Assert(context.Tree.Store.Nodes != null);
      Debug.Assert(index.Index <= context.Tree.Store.Nodes.MaxNodes);

      Context = context;
      Tree = context.Tree;
      Store = context.Tree.Store;

      this.parent = default;
      Span<MCTSNodeStruct> parentArray = context.Tree.Store.Nodes.Span;

      ptr = (MCTSNodeStruct*)Unsafe.AsPointer(ref parentArray[index.Index]);
      this.index = index;

      cachedDepth = -1;
      ActionType = NodeActionType.NotInitialized;
      PendingTranspositionV = FP16.NaN;
      PendingTranspositionM = FP16.NaN;
      PendingTranspositionD = FP16.NaN;
      Annotation = new MCTSNodeAnnotation();
      EvalResult = default;
      EvalResultAuxilliary = FP16.NaN;
      InFlightLinkedNode = default;
      LastAccessedSequenceCounter = 0;
      SiblingEval = null;

      TranspositionRootNodeIndex = default;
  }

    public MCTSNodeStructIndex TranspositionRootNodeIndex;

    /// <summary>
    /// The pending evaluation result 
    /// (cached here after evaluation but before backup)
    /// </summary>
    public LeafEvaluationResult EvalResult;

    /// <summary>
    /// Value possibly set by a LeafEvaluator to pass back auxilliary information.
    /// </summary>
    public FP16 EvalResultAuxilliary;

    /// <summary>
    /// If present, information relating to the evaluation a node which 
    /// is a (lower probability) sibling of a node currently in flight.
    /// </summary>
    public MCTSNodeSiblingEval? SiblingEval;


    #region Data

    /// <summary>
    /// Number of visits to children
    /// N.B. This must appear first in the struture (see static constructor where refererence in fixed statement)
    /// </summary>
    public int N => (*ptr).N;

    /// <summary>
    /// Sum of all V from children
    /// </summary>
    public double W => (*ptr).W;

    /// <summary>
    /// Moves left estimate for this position
    /// </summary>
    public FP16 MPosition => (*ptr).MPosition;

    /// <summary>
    /// Moves left estimate for this subtree
    /// </summary>
    public float MAvg => (*ptr).MAvg;

    /// <summary>
    /// Average win probability of subtree
    /// </summary>
    public float WAvg => (*ptr).WAvg;

    /// <summary>
    /// Average draw probability of subtree
    /// </summary>
    public float DAvg => (*ptr).DAvg;

    /// <summary>
    /// Average loss probability of subtree
    /// </summary>
    public float LAvg => (*ptr).LAvg;

    /// <summary>
    /// Index of the parent, or null if root node
    /// </summary>
    public MCTSNodeStructIndex ParentIndex => (*ptr).ParentIndex;

    /// <summary>
    /// The starting index of entries for this node within the child info array
    /// Value is zero before initialization and thereafter set to either:
    ///   - -1 if it was determined there were no children, otherwise
    ///   - positive value representing start index in child store if initialized
    /// </summary>
    internal long ChildStartIndex => (*ptr).ChildStartIndex;

    internal int TranspositionRootIndex => (*ptr).TranspositionRootIndex;

    /// <summary>
    /// The move was just played to reach this node (or default if root node)
    /// </summary>
    public EncodedMove PriorMove => (*ptr).PriorMove;

    /// <summary>
    /// Policy probability
    /// </summary>
    public FP16 P => (*ptr).P;

    /// <summary>
    /// Node estimated value 
    /// </summary>
    public FP16 V => (*ptr).V;


    public FP16 VSecondary => (*ptr).VSecondary;

    public FP16 WinP => (*ptr).WinP;

    public FP16 DrawP => (*ptr).DrawP;

    public FP16 LossP => (*ptr).LossP;

    /// <summary>
    /// Number of times node has been visited during current batch
    /// </summary>
    public short NInFlight => (*ptr).NInFlight;

    /// <summary>
    /// Number of times node has been visited during current batch
    /// </summary>
    public short NInFlight2 => (*ptr).NInFlight2;

    /// <summary>
    /// If the node is in flight (from one or both selectors)
    /// </summary>
    public bool IsInFlight => NInFlight > 0 || NInFlight2 > 0;

    /// <summary>
    /// Number of policy moves (children)
    /// Possibly this set of moves is incomplete due to either:
    ///   - implementation decision to "throw away" lowest probability moves to save storage, or
    ///   - error in policy evaluation which resulted in certain legal moves not being recognized
    /// </summary>
    public byte NumPolicyMoves => (*ptr).NumPolicyMoves;

    /// <summary>
    /// Game terminal status
    /// </summary>
    public GameResult Terminal => (*ptr).Terminal;

    /// <summary>
    /// Returns if the children (if any) with policy values have been initialized
    /// </summary>
    public bool PolicyHasAlreadyBeenInitialized => ChildStartIndex != 0 || Terminal.IsTerminal();

    /// <summary>
    /// Variance of all V values backed up from subtree
    /// </summary>
    public float VVariance => (*ptr).VVariance;

#if FEATURE_UNCERTAINTY
    public FP16 Uncertainty => (*ptr).Uncertainty;
#endif

    #endregion


    #region Fields used by search

    /// <summary>
    /// If a transposition match for this node is already 
    /// "in flight" for evaluation in another batch by another selector,
    /// then we record the node so we can copy its evaluation 
    /// when evaluation finishes (which is guaranteed to be before
    /// we need it because it was launched in a prior batch).
    /// </summary>
    public MCTSNode InFlightLinkedNode;

    #endregion

    /// <summary>
    /// If the tree is truncated at this node and generating position
    /// values via the subtree linked to its tranposition root
    /// </summary>
    public bool IsTranspositionLinked => (*ptr).IsTranspositionLinked && !Ref.TranspositionUnlinkIsInProgress;


    /// <summary>
    /// The number of visits yet to be processed which will have their values taken from the 
    /// the transposition root (or zero if not transposition linked).
    /// This is encoded in the numChildrenVisited.
    /// </summary>
    public int NumVisitsPendingTranspositionRootExtraction => (*ptr).NumVisitsPendingTranspositionRootExtraction;

    // Values to use for pending transposition extractions 
    // (if NumVisitsPendingTranspositionRootExtraction > 0).
    public FP16 PendingTranspositionV;
    public FP16 PendingTranspositionM;
    public FP16 PendingTranspositionD;

    /// <summary>
    /// Returns the side to move as of this node.
    /// </summary>
    public SideType SideToMove
    {
      get
      {
        // TODO: this only works if we are part of the fixed global array
        // WARNING : probably slow

        SideType rawSide = Context.Tree.Store.Nodes.PriorMoves.FinalPosition.MiscInfo.SideToMove;
        return (Depth % 2 == 0) ? rawSide : rawSide.Reversed();
      }
    }


    /// <summary>
    /// The number of children that have been visited in the current search
    /// Note that children are always visited in descending order by the policy prior probability.
    /// </summary>
    public byte NumChildrenVisited => (*ptr).NumChildrenVisited;

    public byte NumChildrenExpanded => (*ptr).NumChildrenExpanded;


    /// <summary>
    /// An integral index unique to each node, which is ascending in order of node creation.
    /// </summary>
    public int Index
    {
      get
      {
        return index.Index;
      }
    }

    internal MCTSNodeAnnotation Annotation;

    /// <summary>
    /// Returns if the associated annotation has been initialized.
    /// </summary>
    public bool IsAnnotated => Annotation.IsInitialized;


    /// <summary>
    /// Counter used for LRU caching (keeps track of last time accessed)
    /// </summary>
    public int LastAccessedSequenceCounter;


    /// <summary>
    /// Returns reference to underlying MCTSNodeStruct.
    /// </summary>
    public ref MCTSNodeStruct Ref => ref Unsafe.AsRef<MCTSNodeStruct>(ptr);


    /// <summary>
    /// Returns node which is the parent of this node (or null if none).
    /// </summary>
    public MCTSNode Parent
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        if (!parent.IsNull)
        {
          return parent;
        }

        return Ref.ParentIndex.IsNull ? default : (parent = Context.Tree.GetNode(ParentIndex));
      }
    }


    /// <summary>
    /// Records that a specified number of visits are being made to a specified child
    /// (the child is not updated).
    /// </summary>
    /// <param name="childIndex"></param>
    /// <param name="numVisits"></param>
    public void UpdateRecordVisitsToChild(int selectorID, int childIndex, int numVisits)
    {
      Debug.Assert(numVisits > 0);
      Debug.Assert(!IsTranspositionLinked);

      ref MCTSNodeStruct nodeRef = ref Ref;

      if (selectorID == 0)
      {
        Ref.UpdateNInFlight(numVisits, 0);
      }
      else
      {
        Ref.UpdateNInFlight(0, numVisits);
      }

      // Update statistics if we are visting this child for the first time
      if (childIndex >= nodeRef.NumChildrenVisited)
      {
        // The children are expected to be visited strictly in order
        // This is because when we visited a new unvisited child we are
        // always choosing the child with highest P, and the univisted children
        // cluster at the end and are maintained in order (by P) at all times
        // almost always true, not when doing tree purification(?)
        // Debug.Assert(childIndex == nodeRef.NumChildrenVisited);
        nodeRef.NumChildrenVisited = (byte)(childIndex + 1);
      }
    }


    public double Q => N == 0 ? 0 : (W / N);

    #region Children

    public bool IsRoot => ParentIndex.IsNull;

    short cachedDepth;

    public bool IsOurMove => Depth % 2 == 0;

    /// <summary>
    /// Depth of node within tree.
    /// </summary>
    public short Depth
    {
      get
      {
        if (cachedDepth == -1)
        {
          if (IsRoot)
          {
            cachedDepth = Ref.DepthInTree;
          }
          else
          {
            cachedDepth = (short)(Parent.Depth + 1);
          }
        }
        return cachedDepth;
      }
    }

    #endregion


    #region Miscellaneous

    internal static float CPUCT(bool isRoot, int n, ParamsSelect parms)
    {
      return CalcCPUCT(n,
                       isRoot ? parms.CPUCTAtRoot : parms.CPUCT,
                       isRoot ? parms.CPUCTBaseAtRoot : parms.CPUCTBase,
                       isRoot ? parms.CPUCTFactorAtRoot : parms.CPUCTFactor);
    }

    static float CalcCPUCT(int n, float cpuct, float cpuctBase, float cpuctFactor)
    {
      float CPUCT_EXTRA = (cpuctFactor == 0) ? 0 : cpuctFactor * FastLog.Ln((n + cpuctBase + 1.0f) / cpuctBase);
      float thisCPUCT = cpuct + CPUCT_EXTRA;
      return thisCPUCT;
    }

    #endregion


    #region Overrides (object)

    public int GetHashCode() => Index;

    public bool Equals(MCTSNodeInfo other) => index.Index == other.index.Index;

    public bool Equals(MCTSNodeInfo x, MCTSNodeInfo y) => x.index.Index == y.index.Index;
    public int CompareTo(MCTSNodeInfo other) => Index.CompareTo(other.Index);

    public int GetHashCode(MCTSNodeInfo obj) => obj.index.Index;

    #endregion
  }
}

