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

using Ceres.Base.DataType.Trees;
using Ceres.Base.DataTypes;
using Ceres.Base.OperatingSystem;

using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.EncodedPositions;

using Ceres.MCTS.Evaluators;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.Managers;
using Ceres.MCTS.MTCSNodes.Annotation;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.Params;

#endregion

[assembly: InternalsVisibleTo("Ceres.EngineMCTS.Test")]

namespace Ceres.MCTS.MTCSNodes
{
  /// <summary>
  /// Wrapper around a raw MCTSNodeStruct existing in the node store
  /// (just a pointer to struct at a fixed address).
  /// </summary>
  /// 
  public unsafe partial struct MCTSNode
    : ITreeNode,
      IComparable<MCTSNode>,
      IEquatable<MCTSNode>
  {
    /// <summary>
    /// Pointer directly to this structure
    /// </summary>
    private readonly MCTSNodeStruct* structPtr;

    /// <summary>
    /// Pointer directly to associated MCTSNodeInfo.
    /// </summary>
    private void* infoPtr => structPtr->CachedInfoPtr;

    /// <summary>
    /// Returns if the node is null.
    /// </summary>
    public bool IsNull => structPtr == default;

    /// <summary>
    /// Returns if the node is not null.
    /// </summary>
    public bool IsNotNull => structPtr != default;

    /// <summary>
    /// Search context within which this node exists
    /// </summary>
    public MCTSIterator Context => InfoRef.Context;

    /// <summary>
    /// Index of this structure within the array
    /// </summary>
    private MCTSNodeStructIndex index => InfoRef.index;


    public ref MCTSNodeInfo.NodeActionType ActionType => ref InfoRef.ActionType;

    /// <summary>
    /// Shortcut directly to associated MCTSTree.
    /// </summary>
    public MCTSTree Tree => InfoRef.Tree;

    /// <summary>
    /// Shortcut directly to associated MCTSNodeStore.
    /// </summary>
    public MCTSNodeStore Store => InfoRef.Store;


    /// <summary>
    /// Constructor which creates an MCTSNode wrapper for the raw node at specified index.
    /// </summary>
    /// <param name="context"></param>
    /// <param name="index"></param>
    /// <param name="parent">optionally the parent node</param>
    internal MCTSNode(MCTSIterator context, MemoryBufferOS<MCTSNodeStruct> nodes,
                      MCTSNodeStructIndex index, bool reinitializeInfo)
    {
      if (index == default)
      {
        structPtr = null;
        return;
      }

      Debug.Assert(context.Tree.Store.Nodes != null);
      Debug.Assert(index.Index <= context.Tree.Store.Nodes.MaxNodes);

      structPtr = (MCTSNodeStruct*)Unsafe.AsPointer(ref nodes[index.Index]);
      if (reinitializeInfo)
      {
        // Reuse the prior move list if there was one present (but mark as not initialized).
        MGMoveList priorMoveList = null;
        if (infoPtr != null)
        {
          ref MCTSNodeInfo refI = ref Unsafe.AsRef<MCTSNodeInfo>(infoPtr);
          priorMoveList = refI.Annotation.moves;
          if (priorMoveList != null)
          {
            priorMoveList.NumMovesUsed = -1;
          }
        }

        Unsafe.AsRef<MCTSNodeInfo>(infoPtr) = new MCTSNodeInfo(context, index, priorMoveList);
      }
    }


    /// <summary>
    /// The pending evaluation result 
    /// (cached here after evaluation but before backup)
    /// </summary>
    public ref LeafEvaluationResult EvalResult => ref InfoRef.EvalResult;


    /// <summary>
    /// If present, information relating to the evaluation a node which 
    /// is a (lower probability) sibling of a node currently in flight.
    /// </summary>
    public ref MCTSNodeSiblingEval? SiblingEval => ref InfoRef.SiblingEval;


    #region Data

    /// <summary>
    /// Number of visits to children
    /// N.B. This must appear first in the struture (see static constructor where refererence in fixed statement)
    /// </summary>
    public int N => (*structPtr).N;

    /// <summary>
    /// Sum of all V from children
    /// </summary>
    public double W => (*structPtr).W;

    /// <summary>
    /// Moves left estimate for this position
    /// </summary>
    public FP16 MPosition => (*structPtr).MPosition;

    /// <summary>
    /// Moves left estimate for this subtree
    /// </summary>
    public float MAvg => (*structPtr).MAvg;

    /// <summary>
    /// Average win probability of subtree
    /// </summary>
    public float WAvg => (*structPtr).WAvg;

    /// <summary>
    /// Average draw probability of subtree
    /// </summary>
    public float DAvg => (*structPtr).DAvg;

    /// <summary>
    /// Average loss probability of subtree
    /// </summary>
    public float LAvg => (*structPtr).LAvg;

    /// <summary>
    /// Index of the parent, or null if root node
    /// </summary>
    public MCTSNodeStructIndex ParentIndex => (*structPtr).ParentIndex;

    /// <summary>
    /// The starting index of entries for this node within the child info array
    /// Value is zero before initialization and thereafter set to either:
    ///   - -1 if it was determined there were no children, otherwise
    ///   - positive value representing start index in child store if initialized
    /// </summary>
    internal long ChildStartIndex => (*structPtr).ChildStartIndex;

    internal int TranspositionRootIndex => (*structPtr).TranspositionRootIndex;

    /// <summary>
    /// The move was just played to reach this node (or default if root node)
    /// </summary>
    public EncodedMove PriorMove => (*structPtr).PriorMove;

    /// <summary>
    /// Policy probability
    /// </summary>
    public FP16 P => (*structPtr).P;

    /// <summary>
    /// Node estimated value 
    /// </summary>
    public FP16 V => (*structPtr).V;


    public FP16 VSecondary => (*structPtr).VSecondary;

    public FP16 WinP => (*structPtr).WinP;

    public FP16 DrawP => (*structPtr).DrawP;

    public FP16 LossP => (*structPtr).LossP;

    /// <summary>
    /// Number of times node has been visited during current batch
    /// </summary>
    public short NInFlight => (*structPtr).NInFlight;

    /// <summary>
    /// Number of times node has been visited during current batch
    /// </summary>
    public short NInFlight2 => (*structPtr).NInFlight2;

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
    public byte NumPolicyMoves => (*structPtr).NumPolicyMoves;

    /// <summary>
    /// Game terminal status
    /// </summary>
    public GameResult Terminal => (*structPtr).Terminal;

    /// <summary>
    /// Returns if the children (if any) with policy values have been initialized
    /// </summary>
    public bool PolicyHasAlreadyBeenInitialized => ChildStartIndex != 0 || Terminal.IsTerminal();

    /// <summary>
    /// Variance of all V values backed up from subtree
    /// </summary>
    public float VVariance => (*structPtr).VVariance;

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
    public MCTSNode InFlightLinkedNode
    {
      get => InfoRef.InFlightLinkedNode;
      set => InfoRef.InFlightLinkedNode = value;
    }

    #endregion

    /// <summary>
    /// If the tree is truncated at this node and generating position
    /// values via the subtree linked to its tranposition root
    /// </summary>
    public bool IsTranspositionLinked => (*structPtr).IsTranspositionLinked && !StructRef.TranspositionUnlinkIsInProgress;


    /// <summary>
    /// The number of visits yet to be processed which will have their values taken from the 
    /// the transposition root (or zero if not transposition linked).
    /// This is encoded in the numChildrenVisited.
    /// </summary>
    public int NumVisitsPendingTranspositionRootExtraction => (*structPtr).NumVisitsPendingTranspositionRootExtraction;

    // Values to use for pending transposition extractions 
    // (if NumVisitsPendingTranspositionRootExtraction > 0).
    public ref FP16 PendingTranspositionV => ref InfoRef.PendingTranspositionV;
    public ref FP16 PendingTranspositionM => ref InfoRef.PendingTranspositionM;
    public ref FP16 PendingTranspositionD => ref InfoRef.PendingTranspositionD;

    /// <summary>
    /// Returns the side to move as of this node.
    /// </summary>
    public SideType SideToMove => InfoRef.SideToMove;


    /// <summary>
    /// The number of children that have been visited in the current search
    /// Note that children are always visited in descending order by the policy prior probability.
    /// </summary>
    public byte NumChildrenVisited => (*structPtr).NumChildrenVisited;

    public byte NumChildrenExpanded => (*structPtr).NumChildrenExpanded;


    /// <summary>
    /// An integral index unique to each node, which is ascending in order of node creation.
    /// </summary>
    public int Index => InfoRef.index.Index;


    /// <summary>
    /// Returns reference to annotation associated with node.
    /// N.B. Annotate must have been already called.
    /// </summary>
    public ref MCTSNodeAnnotation Annotation
    {
      get 
      {
        Debug.Assert(IsAnnotated); // too expensive to always check
        return ref InfoRef.Annotation;
      }
    }

    /// <summary>
    /// Returns if the associated annotation has been initialized.
    /// </summary>
    public bool IsAnnotated => InfoRef.Annotation.IsInitialized;


    /// <summary>
    /// Makes sure this node is annotated.
    /// </summary>
    public void Annotate()
    {
      if (!IsAnnotated)
      {
        Tree.Annotate(this);
      }
    }


    /// <summary>
    /// Returns reference to underlying MCTSNodeStruct.
    /// </summary>
    public ref MCTSNodeStruct StructRef => ref Unsafe.AsRef<MCTSNodeStruct>(structPtr);


    /// <summary>
    /// Returns reference to associated MCTSNodeInfo.
    /// </summary>
#if DEBUG
    public ref MCTSNodeInfo InfoRef
    {
      get
      {
        ref MCTSNodeInfo refI = ref Unsafe.AsRef<MCTSNodeInfo>(infoPtr);
        Debug.Assert(refI.Index != 0);
        Debug.Assert(refI.index == StructRef.Index);
        return ref refI;
      }
    }
#else
    public ref MCTSNodeInfo InfoRef => ref Unsafe.AsRef<MCTSNodeInfo>(infoPtr);
#endif


    /// <summary>
    /// Returns node which is the parent of this node (or null if none).
    /// </summary>
    public MCTSNode Parent => Tree.GetNode(StructRef.ParentIndex);


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public MCTSNode Child(MCTSNodeStructChild childInfo)
    {
      Debug.Assert(childInfo.IsExpanded);

      // First look to see if already created in annotation cache
      return Tree.GetNode(childInfo.ChildIndex);
    }


    /// <summary>
    /// Enumerator over all expanded children (as MCTSNode).
    /// </summary>
    public IEnumerable<MCTSNode> ChildrenExpanded
    {
      get
      {
        for (int i = 0; i < NumChildrenExpanded; i++)
        {
          yield return ChildAtIndex(i);
        }
      }
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public MCTSNode ChildAtIndex(int childIndex)
    {
      Debug.Assert(childIndex < NumPolicyMoves);

      MCTSNodeStructChild childInfo = Store.Children.childIndices[ChildStartIndex + childIndex];
      Debug.Assert(childInfo.IsExpanded);

      return Tree.GetNode(childInfo.ChildIndex);
    }


    /// <summary>
    /// For a child at a given index, returns either:
    ///   - a node representing this child (if it has been expanded), otherwise
    ///   - the value of the move and policy prior probability corresponding to this unexpanded child
    /// </summary>
    /// <param name="childIndex"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public (MCTSNode node, EncodedMove move, FP16 p) ChildAtIndexInfo(int childIndex)
    {
      Debug.Assert(childIndex < NumPolicyMoves);

      if (IsTranspositionLinked)
      {
        // Recursively ask our transposition root for this value.
        MCTSNode transpositionRootNode = Tree.GetNode(new MCTSNodeStructIndex(TranspositionRootIndex));
        return transpositionRootNode.ChildAtIndexInfo(childIndex);
      }
      else
      {
        ref readonly MCTSNodeStructChild childRef = ref Store.Children.childIndices[ChildStartIndex + childIndex];
        if (childRef.IsExpanded)
        {
          MCTSNode childObj = Tree.GetNode(childRef.ChildIndex);
          return (childObj, childObj.PriorMove, childObj.P);
        }
        else
        {
          return (default, childRef.Move, childRef.P);
        }
      }
    }



    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ref MCTSNodeStruct ChildNodeAtIndexRef(int childIndex)
    {
      Debug.Assert(childIndex < NumChildrenExpanded);
      Debug.Assert(ChildStartIndex > 0); // child at slot 0 is reserved for null

      ref MCTSNodeStructChild childRef = ref ChildAtIndexRef(childIndex);
      return ref childRef.ChildRef;
    }

    // TODO: someday add another method that returns MCTSNodeStructChild (not ref as below), 
    // use this in places to avoid the expensive MCTSNode creation above


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ref MCTSNodeStructChild ChildAtIndexRef(int childIndex)
    {
      Debug.Assert(childIndex >= NumChildrenExpanded);
      Debug.Assert(ChildStartIndex > 0); // child at slot 0 is reserved for null

      return ref Store.Children.childIndices[ChildStartIndex + childIndex];
    }


    /// <summary>
    /// Records that a specified number of visits are being made to a specified child
    /// (the child is not updated).
    /// </summary>
    /// <param name="childIndex"></param>
    /// <param name="numVisits"></param>
    public void UpdateRecordVisitsToChild(int selectorID, int childIndex, int numVisits) => InfoRef.UpdateRecordVisitsToChild(selectorID, childIndex, numVisits);


    /// <summary>
    /// Creates a new child node at specified index.
    /// </summary>
    /// <param name="childIndex"></param>
    /// <param name="possiblyOutOfOrder">normally children added only strictly sequentially unless true</param>
    /// <returns></returns>
    public MCTSNode CreateChild(int childIndex, bool possiblyOutOfOrder = false)
    {
      MCTSNodeStructIndex childNodeIndex = StructRef.CreateChild(Store, childIndex, possiblyOutOfOrder);

      return Tree.GetNode(childNodeIndex);
    }


    public void MaterializeAllTranspositionLinks()
    {
      MCTSTree tree = Tree;

      // Sequentially traverse tree nodes and materialize any that are currently just linked.
      StructRef.Traverse(Store,
                   (ref MCTSNodeStruct nodeRef) =>
                   {
                     if (!nodeRef.IsOldGeneration && nodeRef.IsTranspositionLinked)
                     {
                       nodeRef.MaterializeSubtreeFromTranspositionRoot(tree);
                     }
                     return true;
                   }, TreeTraversalType.Sequential);
    }


    public double Q => (*structPtr).Q;

#region Children

    public bool IsRoot => (*structPtr).ParentIndex.IsNull;


    public bool IsOurMove => InfoRef.Depth % 2 == 0;

    /// <summary>
    /// Returns the MTSNode corresponding to a specified top-level child. 
    /// </summary>
    /// <param name="move"></param>
    /// <returns></returns>
    public MCTSNode NodeForMove(MGMove move)
    {
      for (int i = 0; i < NumPolicyMoves; i++)
      {
        MCTSNode child = this.ChildAtIndex(i);
        if (child.Annotation.PriorMoveMG == move)
        {
          return child;
        }
      }

      // It is (rarely) possible that a legal move is not in the tree
      // (since we truncate the maximum number of moves considered around 64)
      return default;
    }


    /// <summary>
    /// Depth of node within tree.
    /// </summary>
    public short Depth => InfoRef.Depth;

#endregion

#region Helpers


    /// <summary>
    /// Returns list of all the children MCTSNode which are currently expanded.
    /// </summary>
    private List<MCTSNode> ExpandedChildrenList
    {
      get
      {
        List<MCTSNode> ret = new List<MCTSNode>(NumChildrenExpanded);
        for (int i = 0; i < NumChildrenExpanded; i++)
        {
          ret.Add(ChildAtIndex(i));
        }

        return ret;
      }
    }

    public MCTSNode BestMove(bool updateStatistics) => BestMoveInfo(updateStatistics).BestMoveNode;

    public BestMoveInfo BestMoveInfo(bool updateStatistics)
    {
      return new ManagerChooseBestMove(this, updateStatistics, Context.ParamsSearch.MLHBonusFactor).BestMoveCalc;
    }


    /// <summary>
    /// Returns the MCTSNode among all children having largest value returned by specified function.
    /// </summary>
    /// <param name="sortFunc"></param>
    /// <returns></returns>
    public MCTSNode ChildWithLargestValue(Func<MCTSNode, float> sortFunc)
    {
      if (NumChildrenExpanded == 0)
      {
        return default;
      }
      else
      {
        MCTSNode maxNode = default;
        float maxN = float.MinValue;
        for (int i = 0; i < NumChildrenExpanded; i++)
        {
          MCTSNode thisNode = ChildAtIndex(i);
          float thisN = sortFunc(thisNode);
          if (thisN > maxN)
          {
            maxNode = thisNode;
            maxN = thisN;
          }
        }

        return maxNode;
      }
    }



    /// <summary>
    /// Returns array of all children MCTSNodes, sorted by specified function.
    /// </summary>
    /// <param name="sortValueFunc"></param>
    /// <returns></returns>
    public MCTSNode[] ChildrenSorted(Func<MCTSNode, float> sortValueFunc)
    {
      MCTSNode[] children = ExpandedChildrenList.ToArray();

      Array.Sort(children, (v1, v2) => sortValueFunc(v1).CompareTo(sortValueFunc(v2)));
      return children;
    }

    /// <summary>
    /// Returns the expanded node having the largest N.
    /// </summary>
    public MCTSNode ChildWithLargestN => ChildWithLargestValue(n => n.N);

    /// <summary>
    /// Returns the expanded node having the largest Q.
    /// </summary>
    public MCTSNode ChildWithLargestQ => ChildWithLargestValue(n => (float)n.Q);



    /// <summary>
    /// Returns the index of this child within the parent's child array.
    /// </summary>
    public int IndexInParentsChildren => StructRef.IndexInParent;

#endregion


    public void SetPolicy(float policySoftmax, float minPolicyProbability,
                      in MGPosition mgPos, MGMoveList moves,
                      in CompressedPolicyVector policyVector,
                      bool returnedMovesAreInSameOrderAsMGMoveList)
    {
      InfoRef.SetPolicy(policySoftmax, minPolicyProbability, in mgPos, moves, in policyVector, returnedMovesAreInSameOrderAsMGMoveList);
    }

    
    #region Miscellaneous

    /// <summary>
    /// Attempts to find a subnode by following specified moves from root.
    /// </summary>
    /// <param name="priorRoot"></param>
    /// <param name="movesMade"></param>
    /// <returns></returns>
    public MCTSNode FollowMovesToNode(IEnumerable<MGMove> movesMade)
    {
      using (new SearchContextExecutionBlock(Context))
      {
        PositionWithHistory startingPriorMove = Context.StartPosAndPriorMoves;
        MGPosition position = startingPriorMove.FinalPosMG;
        MCTSIterator context = Context;

        // Advance root node and update prior moves
        MCTSNode newRoot = this;
        foreach (MGMove moveMade in movesMade)
        {
          bool foundChild = false;

          if (!newRoot.IsTranspositionLinked)
          {
            // Find this new root node (after these moves)
            foreach (MCTSNodeStructChild child in newRoot.StructRef.Children)
            {
              if (child.IsExpanded)
              {
                MGMove thisChildMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(child.Move, in position);
                if (thisChildMove == moveMade)
                {
                  // Advance new root to reflect this move
                  newRoot = context.Tree.GetNode(child.ChildIndex);

                  // Advance position
                  position.MakeMove(thisChildMove);

                  // Done looking for match
                  foundChild = true;
                  break;
                }
              }
            }
          }

          if (!foundChild)
          {
            return default;
          }
        }

        // Found it
        return newRoot;
      }
    }


    /// <summary>
    /// Calculates exploratory U value (in PUCT) for a child at a given index.
    /// NOTE: currently only supported at root.
    /// </summary>
    /// <param name="childIndex"></param>
    /// <returns></returns>
    /// <summary>
    /// Calculates exploratory U value (in PUCT) for a child at a given index.
    /// NOTE: currently only supported at root.
    /// </summary>
    /// <param name="childIndex"></param>
    /// <returns></returns>
    public float ChildU(int childIndex)
    {
      ParamsSelect parms = Context.ParamsSelect;
      if (parms.PolicyDecayFactor != 0) throw new NotImplementedException();

      float cpuct = MCTSNodeInfo.CPUCT(IsRoot, N, parms);

      (MCTSNode node, EncodedMove move, FP16 p) child = ChildAtIndexInfo(childIndex);
      float n = child.node.IsNull  ? 0 : child.node.N;
      float p = child.p;

      float denominator = parms.UCTRootDenominatorExponent == 1.0f ? (n + 1) : MathF.Pow(n + 1, parms.UCTRootDenominatorExponent);
      float u = cpuct * p * (ParamsSelect.UCTParentMultiplier(N, parms.UCTRootNumeratorExponent) / denominator);

      return u;
    }




#endregion

#region ITreeNode

    ITreeNode ITreeNode.IParent => Parent;

    IEnumerable<ITreeNode> ITreeNode.IChildren
    {
      get
      {
        for (int i = 0; i < NumPolicyMoves; i++)
        {
          (MCTSNode childNode, EncodedMove move, FP16 p) info = ChildAtIndexInfo(i);
          if (!info.Item1.IsNull)
          {
            yield return info.childNode;
          }
        }
      }
    }

    ITreeNode ITreeNode.IChildAtIndex(int index) => ChildAtIndex(index);

#endregion


#region Overrides (object)


    public bool Equals(MCTSNode node) => structPtr == node.structPtr;
    public override int GetHashCode() => ((IntPtr)structPtr).GetHashCode();

    public static bool operator ==(MCTSNode lhs, MCTSNode rhs) => lhs.structPtr == rhs.structPtr;
    public static bool operator !=(MCTSNode lhs, MCTSNode rhs) => lhs.structPtr != rhs.structPtr;

    public int CompareTo(MCTSNode other) => ((IntPtr)structPtr).CompareTo((IntPtr)other.structPtr);

#endregion
  }
}


