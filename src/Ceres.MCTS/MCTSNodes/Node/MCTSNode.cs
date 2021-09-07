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
using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.MCTS.Evaluators;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.Managers;
using Ceres.MCTS.MTCSNodes.Annotation;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.Iteration;
using Ceres.Chess.Positions;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCTS.Params;
using Ceres.Base.Math;

#endregion

[assembly: InternalsVisibleTo("Ceres.EngineMCTS.Test")]

namespace Ceres.MCTS.MTCSNodes
{
  /// <summary>
  /// Wrapper around a raw MCTSNodeStruct existing in the node store.
  /// 
  /// Also contains additional properties used transiently in tree operations such as search.
  /// </summary>
  public unsafe sealed partial class MCTSNode
    : ITreeNode,
      IComparable<MCTSNode>,
      IEquatable<MCTSNode>,
      IEqualityComparer<MCTSNode>
  {
    /// <summary>
    /// Pointer directly to this structure
    /// </summary>
    private readonly MCTSNodeStruct* ptr;

    /// <summary>
    /// Search context within which this node exists
    /// </summary>
    public readonly MCTSIterator Context;

    /// <summary>
    /// Index of this structure within the array
    /// </summary>
    private readonly MCTSNodeStructIndex index;

    public enum NodeActionType : short { NotInitialized, None, MCTSApply, CacheOnly };

    public NodeActionType ActionType;

    internal bool startedAsCacheOnlyNode = false;

    public readonly MCTSTree Tree;
    /// <summary>
    /// Constructor which creates an MCTSNode wrapper for the raw node at specified index.
    /// </summary>
    /// <param name="context"></param>
    /// <param name="index"></param>
    /// <param name="parent">optionally the parent node</param>
    internal MCTSNode(MCTSIterator context, MCTSNodeStructIndex index, MCTSNode parent = null)
    {
      Debug.Assert(context.Tree.Store.Nodes != null);
      Debug.Assert(index.Index <= context.Tree.Store.Nodes.MaxNodes);

      Context = context;
      Tree = context.Tree;

      this.parent = parent;
      Span<MCTSNodeStruct> parentArray = context.Tree.Store.Nodes.Span;

      ptr = (MCTSNodeStruct*)Unsafe.AsPointer(ref parentArray[index.Index]);
      this.index = index;
    }


    LeafEvaluationResult evalResult;

    /// <summary>
    /// The pending evaluation result 
    /// (cached here after evaluation but before backup)
    /// </summary>
    public LeafEvaluationResult EvalResult
    {
      get => evalResult;
      set => evalResult = value;
    }

    LeafEvaluationResult evalResultSecondary;
    public LeafEvaluationResult EvalResultSecondary
    {
      get => evalResultSecondary;
      set => evalResultSecondary = value;
    }


    private ref MCTSNodeStruct Value => ref Unsafe.AsRef<MCTSNodeStruct>(ptr);


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
#else
    public short Unused;
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
    public float PendingTranspositionV = float.NaN;
    public float PendingTranspositionM;
    public float PendingTranspositionD;

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

    internal MCTSNodeAnnotation annotation;
    public ref MCTSNodeAnnotation Annotation
    {
      get
      {
        Debug.Assert(annotation.IsInitialized);
        return ref annotation;
      }
    }

    /// <summary>
    /// Returns if the associated annotation has been initialized.
    /// </summary>
    public bool IsAnnotated => annotation.IsInitialized;


    /// <summary>
    /// Makes sure this node is annotated.
    /// </summary>
    public void Annotate()
    {
      if (!IsAnnotated)
      {
        Context.Tree.Annotate(this);
      }
    }


    /// <summary>
    /// Counter used for LRU caching (keeps track of last time accessed)
    /// </summary>
    public long LastAccessedSequenceCounter;


    /// <summary>
    /// Returns reference to underlying MCTSNodeStruct.
    /// </summary>
    public ref MCTSNodeStruct Ref => ref Unsafe.AsRef<MCTSNodeStruct>(ptr);

    MCTSNode parent;


    /// <summary>
    /// Returns node which is the parent of this node (or null if none).
    /// </summary>
    public MCTSNode Parent
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        if (parent != null)
        {
          return parent;
        }

        return Value.ParentIndex.IsNull ? null : (parent = Context.Tree.GetNode(ParentIndex));
      }
    }



    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public MCTSNode Child(MCTSNodeStructChild childInfo)
    {
      Debug.Assert(childInfo.IsExpanded);

      // First look to see if already created in annotation cache
      return Context.Tree.GetNode(childInfo.ChildIndex, this);
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public MCTSNode ChildAtIndex(int childIndex)
    {
      Debug.Assert(childIndex < NumPolicyMoves);

      MCTSNodeStructChild childInfo = Context.Tree.Store.Children.childIndices[ChildStartIndex + childIndex];
      Debug.Assert(childInfo.IsExpanded);

      return Context.Tree.GetNode(childInfo.ChildIndex, this);
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
        ref readonly MCTSNodeStructChild childRef = ref Context.Tree.Store.Children.childIndices[ChildStartIndex + childIndex];
        if (childRef.IsExpanded)
        {
          MCTSNode childObj = Context.Tree.GetNode(childRef.ChildIndex, this);
          return (childObj, childObj.PriorMove, childObj.P);
        }
        else
        {
          return (null, childRef.Move, childRef.P);
        }
      }
    }



    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ref MCTSNodeStruct ChildNodeAtIndexRef(int childIndex)
    {
      Debug.Assert(childIndex < NumPolicyMoves);
      Debug.Assert(ChildStartIndex > 0); // child at slot 0 is reserved for null

      ref MCTSNodeStructChild childRef = ref ChildAtIndexRef(childIndex);
      return ref childRef.ChildRef;
    }

    // TODO: someday add another method that returns MCTSNodeStructChild (not ref as below), 
    // use this in places to avoid the expensive MCTSNode creation above


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ref MCTSNodeStructChild ChildAtIndexRef(int childIndex)
    {
      Debug.Assert(childIndex < NumPolicyMoves);
      Debug.Assert(ChildStartIndex > 0); // child at slot 0 is reserved for null

      return ref Context.Tree.Store.Children.childIndices[ChildStartIndex + childIndex];
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


    /// <summary>
    /// Creates a new child node at specified index.
    /// </summary>
    /// <param name="childIndex"></param>
    /// <param name="possiblyOutOfOrder">normally children added only strictly sequentially unless true</param>
    /// <returns></returns>
    public MCTSNode CreateChild(int childIndex, bool possiblyOutOfOrder = false)
    {
      MCTSNodeStructIndex childNodeIndex = Ref.CreateChild(Context.Tree.Store,  childIndex, possiblyOutOfOrder);
      return Context.Tree.GetNode(childNodeIndex, this, false);
    }


    public void MaterializeAllTranspositionLinks()
    {
      // Sequentially traverse tree nodes and materialize any that are currently just linked.
      Ref.Traverse(Context.Tree.Store,
                   (ref MCTSNodeStruct nodeRef) =>
                   {
                     if (nodeRef.IsTranspositionLinked)
                     {
                       throw new NotImplementedException();
                       // need to call: MaterializeSubtreeFromTranspositionRoot() with argument for  "exclusive access guaranteed"  set to true
                     }
                     return true;
                   }, TreeTraversalType.Sequential);
    }


    public double Q => N == 0 ? 0 : (W / N);

    #region Children

    public bool IsRoot => ParentIndex.IsNull;

    short cachedDepth = -1;

    public bool IsOurMove => Depth % 2 == 0;

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
      return null;
    }


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
    /// Returns the MCTSNode among all children having largest value returned by speciifed function.
    /// </summary>
    /// <param name="sortFunc"></param>
    /// <returns></returns>
    public MCTSNode ChildWithLargestValue(Func<MCTSNode, float> sortFunc)
    {
      if (NumChildrenExpanded == 0)
      {
        return null;
      }
      else
      {
        MCTSNode maxNode = null;
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
    public int IndexInParentsChildren
    {
      get
      {
        // TODO: This is slow. We could store this index, but it is subject to shuffling and usually unneeded.
        //       Therefore probably we should just document that this property is slow 
        if (IsRoot) throw new Exception("Can't call IndexInParentsChildren at the root");

        int thisIndex = this.Index;
        EncodedMove thisMove = this.PriorMove;


        MCTSNode parent = Parent;

        for (int i = 0; i < parent.NumPolicyMoves; i++)
        {
          (MCTSNode childNode, EncodedMove move, FP16 _) = parent.ChildAtIndexInfo(i);
          if ((childNode != null && childNode.Index == thisIndex)
            || (childNode == null && move == thisMove))
          {
            return i;
          }
        }
        throw new Exception("Not found");
      }
    }

    #endregion

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
            foreach (MCTSNodeStructChild child in newRoot.Ref.Children)
            {
              if (child.IsExpanded)
              {
                MGMove thisChildMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(child.Move, in position);
                if (thisChildMove == moveMade)
                {
                  // Advance new root to reflect this move
                  newRoot = context.Tree.GetNode(child.ChildIndex, newRoot);

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
            return null;
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
    public float ChildU(int childIndex)
    {
      ParamsSelect parms = Context.ParamsSelect;
      if (parms.PolicyDecayFactor != 0) throw new NotImplementedException();

      float cpuct = CPUCT(IsRoot, N, parms);

      (MCTSNode node, EncodedMove move, FP16 p) child = ChildAtIndexInfo(childIndex);
      float n = child.node == null ? 0 : child.node.N;
      float p = child.p;

      float denominator = parms.UCTRootDenominatorExponent == 1.0f ? (n + 1) : MathF.Pow(n + 1, parms.UCTRootDenominatorExponent);
      float u = cpuct * p * (ParamsSelect.UCTParentMultiplier(N, parms.UCTRootNumeratorExponent) / denominator);

      return u;
    }


    static float CPUCT(bool isRoot, int n, ParamsSelect parms)
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

    #region ITreeNode

    ITreeNode ITreeNode.IParent => Parent;

    IEnumerable<ITreeNode> ITreeNode.IChildren
    {
      get
      {
        for (int i = 0; i < NumPolicyMoves; i++)
        {
          (MCTSNode childNode, EncodedMove move, FP16 p) info = ChildAtIndexInfo(i);
          if (info.Item1 != null)
          {
            yield return info.childNode;
          }
        }
      }
    }

    ITreeNode ITreeNode.IChildAtIndex(int index) => ChildAtIndex(index);

    #endregion


    #region Overrides (object)

    public int GetHashCode() => Index;

    public bool Equals(MCTSNode other) => index.Index == other.index.Index;

    public bool Equals(MCTSNode x, MCTSNode y) => x.index.Index == y.index.Index;
    public int CompareTo(MCTSNode other) => Index.CompareTo(other.Index);

    public int GetHashCode(MCTSNode obj) => obj.index.Index;

    #endregion
  }
}
