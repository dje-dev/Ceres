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

using Ceres.Base.DataType;
using Ceres.Base.DataType.Trees;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Base.OperatingSystem;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCTS.Environment;
using Ceres.MCTS.Evaluators;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.Params;
using Ceres.MCTS.Search;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading;

#endregion

namespace Ceres.MCTS.MTCSNodes.Struct
{

  /// <summary>
  /// Various support methods for MCTSNodeStruct.
  /// </summary>
  public partial struct MCTSNodeStruct
  {
    public delegate float MCTSNodeStructMetricFunc(in MCTSNodeStruct node);

    public void SwapFirst()
    {
      throw new Exception("Experimental code.");
      Debug.Assert(NumChildrenExpanded == 0);

      ref MCTSNodeStructChild child0 =  ref MCTSNodeStoreContext.Children[ChildStartIndex + 0];
      ref MCTSNodeStructChild child1 = ref MCTSNodeStoreContext.Children[ChildStartIndex + 1];

      MCTSNodeStructChild temp = child0;
      child0 = child1;
      child1 = temp;

    }

    /// <summary>
    /// If this node is possibly reachable (appears as a descendent in the full game tree)
    /// from a specified prior node (using approximate heuristics).
    /// </summary>
    /// <param name="priorNode"></param>
    /// <returns></returns>
    public readonly bool IsPossiblyReachableFrom(in MCTSNodeStruct priorNode)
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
    public MGPosition CalcPosition(MCTSNodeStore store)
    {
      if (IsOldGeneration)
      {
        throw new Exception("Internal error: CalcPosition net yet supported for old generation nodes.");
      }

      ref readonly MCTSNodeStruct visitorNodeRef = ref this;

      // Ascend up to root, keeping track of all moves along the way.
      Span<MCTSNodeStruct> nodes = store.Nodes.Span;
      Span<ushort> moves = stackalloc ushort[255];
      int index = 0;
      while (!visitorNodeRef.IsRoot)
      {
        moves[index++] = visitorNodeRef.PriorMove.RawValue;
        visitorNodeRef = ref nodes[visitorNodeRef.ParentIndex.Index];
      }

      // Now reverse the ascent, tracking the position along the way.
      // TODO: would it be possible to sometimes or always avoid the move generation
      //       and instead use the EncodedMove directly?
      MGPosition pos = store.Nodes.PriorMoves.FinalPosMG;
      if (index > 0)
      {
        for (int i = index - 1; i >= 0; i--)
        {
          EncodedMove move = new EncodedMove(moves[i]);
          MGMove moveMG = MGMoveConverter.ToMGMove(in pos, move);
          pos.MakeMove(moveMG);
        }
      }
      return pos;
    }


    /// <summary>
    /// Returns the index within the child array of the expanded child haivng specified node index.
    /// </summary>
    /// <param name="store"></param>
    /// <param name="childNodeIndex"></param>
    /// <returns></returns>
    public readonly int IndexOfExpandedChildForIndex(MCTSNodeStore store, MCTSNodeStructIndex childNodeIndex)
    {
      Span<MCTSNodeStructChild> children = ChildrenFromStore(store).Slice(0, NumChildrenExpanded);
      Span<int> castChildren = MemoryMarshal.Cast<MCTSNodeStructChild, int>(children);
      return MemoryExtensions.IndexOf(castChildren, -childNodeIndex.Index);
    }


    /// <summary>
    /// Creates a new child node at specified index.
    /// </summary>
    /// <param name="childIndex"></param>
    /// <param name="possiblyOutOfOrder">normally children added only strictly sequentially unless true</param>
    /// <returns></returns>
    public MCTSNodeStructIndex CreateChild(MCTSNodeStore store, int childIndex, bool possiblyOutOfOrder = false)
    {
      // Make sure expanded children appear strictly clustered at lowest indices
      Debug.Assert(possiblyOutOfOrder || childIndex == 0 || ChildAtIndex(childIndex - 1).IsExpanded);
      Debug.Assert(LeafSelectorMulti.LAX_CHILD_ORDER || ( NumChildrenExpanded == childIndex));

      // Get child info and make sure not already expanded
      ref MCTSNodeStructChild thisChildRef = ref store.Children.childIndices[ChildStartIndex + childIndex];
      Debug.Assert(!thisChildRef.IsExpanded); // should not already be expanded!
      Debug.Assert(!thisChildRef.IsNull);

      // Allocate new node
      MCTSNodeStructIndex childNodeIndex = store.Nodes.AllocateNext();
      ref MCTSNodeStruct childNodeRef = ref store.Nodes.nodes[childNodeIndex.Index];

      // Create new wrapper object and use it to initialize fields
      childNodeRef.Initialize(new MCTSNodeStructIndex(Index.Index), childIndex, thisChildRef.p, thisChildRef.Move);

      // Modify child entry to refer to this new child
      // N.B: It is essential to only swap out the child index here
      //      after the new child node has been fully initialized (above)
      //      because another thread might see and then follow the child reference.
      thisChildRef.SetExpandedChildIndex(childNodeIndex);

      NumChildrenExpanded++;

      return childNodeIndex;
    }


    /// <summary>
    /// Modifies the node index within the child array having a specified prior value
    /// to have a specified new value.
    /// </summary>
    /// <param name="store"></param>
    /// <param name="fromChildIndex"></param>
    /// <param name="toChildIndex"></param>
    public void ModifyExpandedChildIndex(MCTSNodeStore store, MCTSNodeStructIndex fromChildIndex, MCTSNodeStructIndex toChildIndex)
    {
      if (NumChildrenExpanded > 0)
      {
        int index = store.Nodes.nodes[fromChildIndex.Index].IndexInParent;
        Span<MCTSNodeStructChild> children = ChildrenFromStore(store);
        children[index].SetExpandedChildIndex(toChildIndex);
      }
    }


    /// <summary>
    /// Copy over all the child information from another node
    /// which is in the same transposition equivalence class.
    /// </summary>
    /// <param name="tree"></param>
    /// <param name="otherNodeIndex"></param>
    public void CopyUnexpandedChildrenFromOtherNode(MCTSTree tree, MCTSNodeStructIndex otherNodeIndex)
    {
      DoCopyUnexpandedChildrenFromOtherNode(tree, otherNodeIndex);
    }


    /// <summary>
    /// Worker method which copies unexpanded children.
    /// </summary>
    /// <param name="tree"></param>
    /// <param name="otherNodeIndex"></param>
    public void DoCopyUnexpandedChildrenFromOtherNode(MCTSTree tree, MCTSNodeStructIndex otherNodeIndex)
    {
      ref readonly MCTSNodeStruct otherNode = ref tree.Store.Nodes.nodes[otherNodeIndex.Index];
      Debug.Assert(!otherNode.IsTranspositionLinked);

      TranspositionUnlinkIsInProgress = true;

      NextTranspositionLinked = 0;
      NumVisitsPendingTranspositionRootExtraction = 0;
      TranspositionRootIndex = 0;

      SetNumPolicyMovesAndAllocateChildInfo(tree, otherNode.NumPolicyMoves);

      int otherNodeNumPolicyMoves = otherNode.NumPolicyMoves;
      if (otherNodeNumPolicyMoves > 0)
      {
        // First, copy any expanded nodes
        // We have to descend to the expanded node to retrieve 
        // the move and policy needed for the new child (which will be not yet expanded)
        MCTSNodeStructChildStorage childrenStore = tree.Store.Children;
        Span<MCTSNodeStructChild> children = childrenStore.SpanForNode(in this);
        Span<MCTSNodeStructChild> otherChildren = childrenStore.SpanForNode(in otherNode);

        for (int i = 0; i < otherNodeNumPolicyMoves; i++)
        {
          // This code is robust to concurrency where
          // other threads are possibly in the process of converting 
          // unexpanded children to expanded children. That operation on
          // the other thread (in CreateChild) is atomic where the childInfo
          // object below is swapped within the array atomically
          // from the pre-expansion unexpanded move/probability structure
          // to the index of the newly created child node.
          // The swapping happens only after the child node is fully initialized.
          // In the next line we observe one or the other of these forms,
          // and both are equally valid and containing the same data for move/probability.
          MCTSNodeStructChild childInfo = otherChildren[i];
          if (childInfo.IsExpanded)
          {
            ref MCTSNodeStruct childNodeRef = ref childInfo.ChildRefFromStore(tree.Store);
            children[i].SetUnexpandedPolicyValues(childNodeRef.PriorMove, childNodeRef.P);
          }
          else
          {
            children[i] = childInfo;
          }
        }
      }

      TranspositionUnlinkIsInProgress = false;
    }


    /// <summary>
    /// Destructively repacks child array such that the linkages to any 
    /// expanded child nodes are undone and rewritten with just original
    /// the policy move and prior probability.
    /// </summary>
    /// <param name="store"></param>
    public void RepackChildren(MCTSNodeStore store)
    {
      if (NumChildrenExpanded > 0)
      {
        Span<MCTSNodeStructChild> children = ChildrenFromStore(store);

        for (int i = 0; i < NumChildrenExpanded; i++)
        {
          MCTSNodeStructChild childInfo = children[i];

          ref readonly MCTSNodeStruct otherNode = ref store.Nodes.nodes[childInfo.ChildIndex.Index];

          children[i].SetUnexpandedPolicyValues(otherNode.PriorMove, otherNode.P);
        }

        NumChildrenExpanded = 0;
        NumChildrenVisited = 0;
      }
    }


    /// <summary>
    /// Allocates space for specified number of children
    /// </summary>
    /// <param name="numPolicyMoves">number of children to allocate space for (degenerate case of zero is acceptable)</param>
    public void SetNumPolicyMovesAndAllocateChildInfo(MCTSTree tree, int numPolicyMoves)
    {
      Debug.Assert(numPolicyMoves >= 0 && numPolicyMoves < 255);

      NumPolicyMoves = (byte)numPolicyMoves;

      if (numPolicyMoves > 0)
      {
        childStartBlockIndex = (int)tree.Store.Children.AllocateEntriesStartBlock(numPolicyMoves);
      }
      else
      {
        childStartBlockIndex = 0;
      }
    }


    /// <summary>
    /// Returns the sum of policy probabilities of all expanded nodes.
    /// </summary>
    /// <returns></returns>
    public float SumPVisited
    {
      get
      {
        if (NumChildrenVisited == 0)
        {
          if (N <= 1)
          {
            // First visit (N) is to the node itself, not children.
            return 0;
          }
          else if (Terminal.IsTerminal())
          {
            return 1.0f;

          }
          else if (IsTranspositionLinked)
          {
            Debug.Assert(N <= 3);

            MCTSNodeStore store = MCTSNodeStoreContext.Store;
            ref readonly MCTSNodeStruct transpositionRootRef = ref store[TranspositionRootIndex];
            ref readonly MCTSNodeStruct transpositionRootChild = ref store[in transpositionRootRef, 0];

            float sumP = transpositionRootChild.P;
            if (N == 3)
            {
              if (transpositionRootRef.NumChildrenVisited > 1)
              {
                // Sibling exists and possibly the 3rd visit must have been to the sibling.
                ref readonly MCTSNodeStruct transpositionRootSibling = ref store[in transpositionRootRef, 1];
                sumP += transpositionRootSibling.P;
              }
            }
            return sumP;
          }
          else
          {
            throw new Exception();
          }
        }


        Span<MCTSNodeStruct> nodes = MCTSNodeStoreContext.Store.Nodes.nodes.Span;

        // Get a slice of children in a way that avoids range checking in loop.
        int numChildrenExpanded = numChildrenVisited;
        Span<MCTSNodeStructChild> theseChildren = Children.Slice(0, numChildrenExpanded);

        float sumPVisited = 0;
        for (int i = 0; i < numChildrenExpanded; i++)
        {
          sumPVisited += nodes[theseChildren[i].ChildIndex.Index].P.ToFloatApprox;
        }

        return sumPVisited;
      }
    }




    public enum CacheLevel
    {
      None,
      Level0,
      Level1,
      Level2
    };


    public readonly void PossiblyPrefetchChild(MCTSNodeStore store, MCTSNodeStructIndex nodeIndex, int childIndex)
    {
      PossiblyPrefetchNodeAndChildrenInRange(store, nodeIndex, childIndex, 1);
    }


    public static void PossiblyPrefetchNodeAndChildrenInRange(MCTSNodeStore store, MCTSNodeStructIndex nodeIndex,
                                                               int firstChildIndex, int numChildren)
    {
      unsafe
      {
        if (MCTSParamsFixed.PrefetchCacheLevel != CacheLevel.None)
        {
          Span<MCTSNodeStruct> nodes = store.Nodes.Span;

          // Prefetch node data
          void* nodePtr = Unsafe.AsPointer(ref nodes[nodeIndex.Index]);
          PrefetchDataAt(nodePtr);

          if (numChildren == 0) return;

          // Prefetch each of the expanded child nodes
          Span<MCTSNodeStructChild> childSpan = store.Children.SpanForNode(in nodes[nodeIndex.Index]);
          for (int i = firstChildIndex; i < (firstChildIndex + numChildren); i++)
          {
            MCTSNodeStructChild child = childSpan[i];

            if (child.IsExpanded)
            {
              void* childNodePtr = Unsafe.AsPointer(ref nodes[child.ChildIndex.Index]);
              PrefetchDataAt(childNodePtr);
            }
            else
            {
              break;
            }
          }
        }
      }
    }


    internal static unsafe void PrefetchNode(ref MCTSNodeStruct nodeRef)
    {
      PrefetchDataAt(Unsafe.AsPointer<MCTSNodeStruct>(ref nodeRef));
    }


    private static unsafe void PrefetchDataAt(void* nodePtr)
    {
      if (MCTSParamsFixed.PrefetchCacheLevel == CacheLevel.Level0)
      {
        Sse.Prefetch0(nodePtr);
      }
      else if (MCTSParamsFixed.PrefetchCacheLevel == CacheLevel.Level1)
      {
        Sse.Prefetch1(nodePtr);
      }
      else if (MCTSParamsFixed.PrefetchCacheLevel == CacheLevel.Level2)
      {
        Sse.Prefetch2(nodePtr);
      }
      else if (MCTSParamsFixed.PrefetchCacheLevel == CacheLevel.None)
      {
      }
      else
        throw new Exception("Internal error: unsupported cache level");
    }


    public readonly void PossiblyPrefetchChildArray(MCTSNodeStore store, MCTSNodeStructIndex index)
    {
      PossiblyPrefetchNodeAndChildrenInRange(store, index, 0, NumChildrenExpanded);
    }


    #region Gather fields

    const float LAMBDA = 0.96f;

#if NOT
// probably ill conceived, using draw probabilities a more direct way

    /// <summary>
    /// 
    /// Example: if contempt 0.05 and entropy is -0.25 (little entropy)
    /// </summary>
    /// <param name="contempt"></param>
    /// <param name="ourMove"></param>
    /// <returns></returns>
    public float EntropyBonus(float contempt, bool ourMove)
    {
      if (!ourMove) return 0;// SHOULD BE SYMMETRIC? -0.1f;
      
      // Don't apply if few entropy estimate from few samples
      const int MIN = 10;
      if (N < MIN) return 0;

      // Scale based on contempt, attaining minimum/maximum value at contempt of +/- 0.10
      float contemptScaling = MathHelpers.Bounded(contempt / 0.1f, -0.1f, 0.1f);

      // Entropy between 0 and -1.5, approximately center
      float centeredEntropy = 1.0f + ESubtree;

      const float MULT = 0.1f;
      float bonus = centeredEntropy * contemptScaling * MULT;

      // Finally, bound in range [-0.03 to 0.03]
      const float MAX_ENTROPY_BONUS_MAGNITUDE = 0.03f;
      return MathHelpers.Bounded(bonus, -MAX_ENTROPY_BONUS_MAGNITUDE, MAX_ENTROPY_BONUS_MAGNITUDE);
    }

#endif


    public readonly float TrendBonusToP
    {
      get
      {
        const int MIN_N = 20;

        //if (ParentIndex.Index != 1) return 0;

        if (N > MIN_N)
        {
          float val = QUpdatesWtdAvg;
          float std = MathF.Sqrt(QUpdatesWtdVariance);
          float stdErr = val / std;
          if (MathF.Abs(stdErr) < 0.2f) return 0;

          const float MULT = 0.04f;
          float bonus = MathHelpers.Bounded(stdErr * MULT, -0.02f, 0.02f);
          //          if (COUNT++ % 9999 == 9998) Console.WriteLine(Q + " " + bonus);
          return bonus;
        }
        else
          return 0;
      }
    }


    /// <summary>
    /// Extracts values of 4 fields (N, NInFlight, P, and W) into arrays
    /// from a contiguous range of children at specified indices.
    /// </summary>
    /// <param name="selectorID"></param>
    /// <param name="minIndex"></param>
    /// <param name="maxIndex"></param>
    /// <param name="n"></param>
    /// <param name="nInFlight"></param>
    /// <param name="p"></param>
    /// <param name="w"></param>
    /// <param name="u"></param>
    public void GatherChildInfo(MCTSIterator context, MCTSNodeStructIndex index,
                                int selectorID, int depth, int maxIndex,
                                Span<float> n, Span<float> nInFlight, Span<float> p, Span<float> w, Span<float> u)
    {
      MCTSNodeStore store = context.Tree.Store;
      Span<MCTSNodeStruct> nodes = store.Nodes.nodes.Span;

      Debug.Assert(maxIndex >= 0 && (maxIndex + 1) <= n.Length);

      // N.B. This prefetch is critical for performance, circa +15% for CPU bound searches
      PossiblyPrefetchNodeAndChildrenInRange(store, index, 0, (maxIndex + 1));

      bool applyTrendBonus = context.ParamsSearch.ApplyTrendBonus;
      bool applyMBonus = false;// WE NO LONGER APPLY M BONUS ANYWHERE EXCEPT ROOT. context.ParamsSearch.MLHBonusFactor;
      float powerMeanNExponent = context.ParamsSelect.PowerMeanNExponent;

      // if the power mean coefficient ends up being very lose to 1 (less than this value) then don't bother to use power mean
      const float POWER_MEAN_SUBTRACT = 2.0f;
      const float MIN_POWER_MEAN_COEFF = POWER_MEAN_SUBTRACT + 1;
      float powerMeanMinN = powerMeanNExponent == 0 ? float.MaxValue :
                                                      MathF.Pow(MIN_POWER_MEAN_COEFF, 1.0f / powerMeanNExponent);

      float dualCollisionFraction = context.ParamsSearch.Execution.DualSelectorAlternateCollisionFraction;
      Span<MCTSNodeStructChild> children = store.Children.SpanForNode(in this);
      for (int i = 0; i <= maxIndex; i++)
      {
        MCTSNodeStructChild child = children[i];

        if (child.IsExpanded)
        {
          ref MCTSNodeStruct childNode = ref nodes[child.ChildIndex.Index];
          n[i] = childNode.N;
          if (selectorID == 0)
          {
            nInFlight[i] = childNode.NInFlight + dualCollisionFraction * childNode.NInFlight2;
          }
          else
          {
            nInFlight[i] = childNode.NInFlight2 + dualCollisionFraction * childNode.NInFlight;
          }

          p[i] = childNode.P.ToFloatApprox;// * childNode.Weight;
          u[i] = childNode.VarianceAccumulator;

          bool isOurMove = depth % 2 == 0;

          if (applyMBonus)
          {
            float mBonus = context.MBonusForNode(ref childNode, isOurMove);
            w[i] = (float)childNode.W + (mBonus * childNode.N);
          }
          else
          {
            if (powerMeanNExponent != 0 && childNode.N > powerMeanMinN)
            {
              float power = MathF.Pow(childNode.N, powerMeanNExponent) - POWER_MEAN_SUBTRACT;
              //float power = (MathF.Log(childNode.N) - 7) * powerMeanNExponent; (try 1 or 2 for powerMeanNExponent, but works poorly)
              float q = childNode.QPowerMean(power, !isOurMove);
              w[i] = q * childNode.N;
            }
            else
            {
              w[i] = (float)childNode.W;
            }
          }

          if (applyTrendBonus)
          {
            Debug.Assert(MCTSParamsFixed.TRACK_NODE_TREND);

            float pBonus = childNode.TrendBonusToP;
            //if (pBonus > 0 || p[i] > -pBonus)
            w[i] += pBonus * childNode.N;
          }
        }
        else
        {
          n[i] = 0;
          nInFlight[i] = 0;
          p[i] = child.P.ToFloatApprox;
          w[i] = 0;
          u[i] = 0;
        }
      }
    }




#if WIP

    /// <summary>
    /// Extracts values of 4 fields (N, NInFlight, P, and W) into arrays
    /// from a contiguous range of children at specified indices.
    /// </summary>
    /// <param name="selectorID"></param>
    /// <param name="minIndex"></param>
    /// <param name="maxIndex"></param>
    /// <param name="n"></param>
    /// <param name="nInFlight"></param>
    /// <param name="p"></param>
    /// <param name="w"></param>
    public void GatherChildInfo2(SearchContext context, MCTSNodeStructIndex index, int selectorID, int depth, int maxIndex,
                                 Span<float> n, Span<float> nInFlight, Span<float> p, Span<float> w)
    {
      MCTSNodeStore store = context.Store;
      Debug.Assert(maxIndex >= 0 && (maxIndex + 1) <= n.Length);

      bool applyTrendBonus = context.ParamsSearch.ApplyTrendBonus;
      bool applyMBonus = false;// WE NO LONGER APPLY M BONUS ANYWHERE EXCEPT ROOT. context.ParamsSearch.MLHBonusFactor;
      float powerMeanNExponent = context.ParamsSelect.POWER_MEAN_N_EXPONENT;

      // if the power mean coefficient ends up being very lose to 1 (less than this value) then don't bother to use power mean
      const float POWER_MEAN_SUBTRACT = 2.0f;
      const float MIN_POWER_MEAN_COEFF = POWER_MEAN_SUBTRACT + 1;
      float powerMeanMinN = powerMeanNExponent == 0 ? float.MaxValue :
                                                      MathF.Pow(MIN_POWER_MEAN_COEFF, 1.0f / powerMeanNExponent);

      float dualCollisionFraction = context.ParamsSearch.Execution.DUAL_SELECTOR_ALTERNATE_COLLISION_FRACTION;

      // Process the expanded nodes
      int i = 0;
      foreach (ref var childNode in context.Store.ChildEnumerator(index, maxIndex + 1))
      {
        n[i] = childNode.N;
        if (selectorID == 0)
          nInFlight[i] = childNode.NInFlight + dualCollisionFraction * childNode.NInFlight2;
        else
          nInFlight[i] = childNode.NInFlight2 + dualCollisionFraction * childNode.NInFlight;

        p[i] = childNode.P;// * childNode.Weight;

        bool isOurMove = depth % 2 == 0;

        if (applyMBonus)
        {
          float mBonus = context.MBonusForNode(ref childNode, isOurMove);
          w[i] = (float)childNode.W + (mBonus * childNode.N);
        }
        else
        {
          if (powerMeanNExponent != 0 && childNode.N > powerMeanMinN)
          {
            float power = MathF.Pow(childNode.N, powerMeanNExponent) - POWER_MEAN_SUBTRACT;
            //float power = (MathF.Log(childNode.N) - 7) * powerMeanNExponent; (try 1 or 2 for powerMeanNExponent, but works poorly)
            float q = childNode.QPowerMean(power, !isOurMove);
            w[i] = q * childNode.N;
          }
          else
          {
            w[i] = (float)childNode.W;
          }
        }

        if (applyTrendBonus)
        {
          Debug.Assert(MCTSParamsFixed.TRACK_NODE_TREND);

          float pBonus = childNode.TrendBonusToP;
          //if (pBonus > 0 || p[i] > -pBonus)
          w[i] += pBonus * childNode.N;
        }
        i++;
      }

      // Finally, fill in the non-expanded nodes
      Span<MCTSNodeStructChild> children = store.Children.SpanForNode(in this);
      for (int ix = maxIndex + 1; ix < NumPolicyMoves; ix++)
      {
        n[ix] = 0;
        nInFlight[ix] = 0;
        p[i] = children[i].P;
        w[ix] = 0;
      }
      
    }
#endif
    #endregion

    #region Updates

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void UpdateNInFlight0(short adjustNInFlight)
    {
      Debug.Assert(adjustNInFlight >= 0 || NInFlight >= -adjustNInFlight);
      Debug.Assert(System.Math.Abs(NInFlight + adjustNInFlight) < short.MaxValue);

      Debug.Assert(NInFlight + adjustNInFlight >= 0);
      NInFlight += adjustNInFlight;

      //Interlocked.Add(ref NInFlight, adjustNInFlight);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void UpdateNInFlight1(short adjustNInFlight)
    {
      Debug.Assert(adjustNInFlight >= 0 || NInFlight2 >= -adjustNInFlight);
      Debug.Assert(System.Math.Abs(NInFlight2 + adjustNInFlight) < short.MaxValue);
      Debug.Assert(NInFlight2 + adjustNInFlight >= 0);
      NInFlight2 += adjustNInFlight;

      // Interlocked.Add(ref NInFlight2, adjustNInFlight);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void UpdateNInFlight(int adjustNInFlight1, int adjustNInFlight2)
    {
      if (adjustNInFlight1 != 0) UpdateNInFlight0((short)adjustNInFlight1);
      if (adjustNInFlight2 != 0) UpdateNInFlight1((short)adjustNInFlight2);
    }


    /// <summary>
    /// Decrements N in flight by specified value for this node and all predecessors.
    /// </summary>
    public void BackupDecrementInFlight(int numInFlight1, int numInFlight2) => BackupUpdateInFlight(-numInFlight1, -numInFlight2);


    /// <summary>
    /// Increments N in flight by specified value for this node and all predecessors.
    /// </summary>
    public void BackupIncrementInFlight(int numInFlight1, int numInFlight2) => BackupUpdateInFlight(numInFlight1, numInFlight2);


    /// <summary>
    /// Updates N in flight by a specified value for this node and all predecessors.
    /// </summary>
    void BackupUpdateInFlight(int numInFlight1, int numInFlight2)
    {
      Span<MCTSNodeStruct> nodes = MCTSNodeStoreContext.Store.Nodes.Span;

      ref MCTSNodeStruct node = ref this;
      while (true)
      {
        node.UpdateNInFlight(numInFlight1, numInFlight2);

        if (node.IsRoot)
          return;
        else
          node = ref nodes[node.ParentRef.Index.Index];
      }
    }


    /// <summary>
    /// Undoes updates in flight (decrements NInFlight by specified value) for this node and all predecessors.
    /// </summary>
    public void BackupAbort0(short numInFlight)
    {
      Span<MCTSNodeStruct> nodes = MCTSNodeStoreContext.Store.Nodes.Span;

      short updateAmount = (short)-numInFlight;
      ref MCTSNodeStruct node = ref this;
      while (true)
      {
        node.UpdateNInFlight0(updateAmount);

        if (node.IsRoot)
          return;
        else
          node = ref nodes[node.ParentRef.Index.Index];
      }

    }


    /// <summary>
    /// Undoes updates in flight (decrements NInFlight by specified value) for this node and all predecessors.
    /// </summary>
    public void BackupAbort1(short numInFlight)
    {
      Span<MCTSNodeStruct> nodes = MCTSNodeStoreContext.Store.Nodes.Span;

      short updateAmount = (short)-numInFlight;
      ref MCTSNodeStruct node = ref this;
      while (true)
      {
        node.UpdateNInFlight1(updateAmount);

        if (node.IsRoot)
          return;
        else
          node = ref nodes[node.ParentRef.Index.Index];
      }
    }


    /// <summary>
    /// Applies update to this node and all predecessors (increment N, add to W, and decrement NInFlight).
    /// </summary>
    /// <param name="vToApply"></param>
    /// <param name="mToApply"></param>
    /// <param name="numInFlight1"></param>
    /// <param name="numInFlight2"></param>
    public unsafe void BackupApply(Span<MCTSNodeStruct> nodes,
                                   int numToApply,
                                   float vToApplyFirst, float vToApplyNonFirst, float mToApply, 
                                   float dToApplyFirst, float dToApplyNonFirst,
                                   int numInFlight1, int numInFlight2,
                                   out MCTSNodeStructIndex indexOfChildDescendentFromRoot)
    {
      Debug.Assert(!float.IsNaN(vToApplyFirst));

      MCTSIterator context = MCTSManager.ThreadSearchContext;
      bool enableDeepTranspositionBackup = context.ParamsSearch.EnableDeepTranspositionBackup;
      Span<MCTSNodeStruct> nodesSpan = context.Tree.Store.Nodes.nodes.Span;

      indexOfChildDescendentFromRoot = default;
      bool first = true;
      float vToApply = vToApplyFirst;
      float dToApply = dToApplyFirst;

      ref MCTSNodeStruct node = ref this;
      while (true)
      {
        ref MCTSNodeStruct parentRef = ref nodes[node.ParentIndex.Index];

        node.UpdateNInFlight(-numInFlight1, -numInFlight2);

        // If a draw could have been claimed here, 
        // assume it would have been if the alternative was worse
        // (use value of 0 here and further up in tree)
        if (vToApply < 0 && node.DrawKnownToExistAmongChildren)
        {
          vToApply = vToApplyNonFirst = 0;
          mToApply = 0;
          dToApply = dToApplyFirst = 1; // TODO: is this ok even if not WDL network?
        }

        // Compute statistics used for tracking uncertainty (variance)
        float qDiff = vToApply - (float)node.Q;

        const float VARIANCE_LAMBDA = 0.0f; // Possible exponentially weighted moving average variance if nonzero

        // NOTE: It is not possible to make the updates to both N and W atomic as a group.
        //       Therefore there is a very small possibility that another thread will observe one updated but not the other
        //       (e.g. thread gathering nodes which reaches over to use this as a transposition root and references Q)
        //       To mitigate the possible distortion, N is updated before W so any distortions will shrink toward 0.
        if (node.N >= VARIANCE_START_ACCUMULATE_N)
        {
          if (VARIANCE_LAMBDA != 0)
          {
            throw new NotImplementedException();
            //node.VarianceAccumulator = NewEMWVarianceAcc(node.VarianceAccumulator, node.N, qDiffSquared, numToApply, VARIANCE_LAMBDA);
          }
          else
          {
            node.VarianceAccumulator += MathF.Abs(qDiff) * numToApply;
          }
        }

        node.N += numToApply;

        bool haveApplied = false;

        // Try to apply deep transposition backup (if requested).
        int transpositionNodeIndex = node.InfoRef.TranspositionRootNodeIndex.Index;
        if (enableDeepTranspositionBackup && node.InfoRef.TranspositionRootNodeIndex != default)
        {
          ref readonly MCTSNodeStruct transpositionRootNodeRef = ref nodes[transpositionNodeIndex];

          if (!transpositionRootNodeRef.IsTranspositionRoot)
          {
            // The root has changed due to refresh of roots
            if (node.InfoRef.Tree.TranspositionRoots.TryGetValue(node.ZobristHash, out transpositionNodeIndex))
            {
              node.InfoRef.TranspositionRootNodeIndex = new MCTSNodeStructIndex(transpositionNodeIndex);
              //sMCTSEventSource.TestCounter1++;
            }
          }

#if DEBUG
          //ref MCTSNodeInfo nodeInfoRef = ref Unsafe.AsRef(node.InfoRef);
          bool found = context.Tree.TranspositionRoots.TryGetValue(node.ZobristHash, out int siblingTanspositionNodeIndex);
          if (!found || siblingTanspositionNodeIndex != node.InfoRef.TranspositionRootNodeIndex.Index)
          {
            throw new Exception("Internal error: transposition root inconsistency " + siblingTanspositionNodeIndex + " " + node.InfoRef.TranspositionRootNodeIndex.Index);
          }
#endif

          // Only apply if subtree is deeper than this subtree.
          if (transpositionRootNodeRef.N > (node.N + (numToApply / 2) + 1))
          {
            const float wtSubtree = 1.0f;
            const float wtThisNode = 1.0f - wtSubtree;

            if (wtThisNode == 0)
            {
              node.W    += numToApply * transpositionRootNodeRef.Q;
              node.mSum += (FP16)(numToApply * transpositionRootNodeRef.MAvg);
              node.dSum += numToApply * transpositionRootNodeRef.DAvg;
            }
            else
            {
              node.W    +=        numToApply * (wtThisNode * vToApply + wtSubtree * transpositionRootNodeRef.Q);
              node.mSum += (FP16)(numToApply * (wtThisNode * mToApply + wtSubtree * transpositionRootNodeRef.MAvg));
              node.dSum +=        numToApply * (wtThisNode * dToApply + wtSubtree * transpositionRootNodeRef.DAvg);
            }

            //MCTSEventSource.TestCounter1++;
            haveApplied = true;
          }
        }

        if (!haveApplied)
        {
          if (numToApply == 1)
          {
            node.W += vToApply;
            node.mSum += (FP16)mToApply;
            node.dSum += dToApply;
          }
          else
          {
            node.W += vToApply * numToApply;
            node.mSum += (FP16)(mToApply * numToApply);
            node.dSum += dToApply * numToApply;
          }
        }

#if FEATURE_UNCERTAINTY
        // Update uncertainty (exponentially weighted moving average)
        float absDiff = MathF.Abs(vToApply - (float)node.Q);
        node.Uncertainty = (FP16)(LAMBDA * node.Uncertainty + (1.0f - LAMBDA) * absDiff);
#endif

        if (MCTSParamsFixed.TRACK_NODE_TREND)
        {
          if (node.QUpdatesWtdVariance == 0)
          {
            // First initialization. Start with a typical variance estimate.
            node.QUpdatesWtdVariance = 0.5f * 0.5f;
          }
          else
          {
            float diff = vToApply - (float)node.Q;;
            node.QUpdatesWtdAvg = LAMBDA * node.QUpdatesWtdAvg + (1.0f - LAMBDA) * diff;
            node.QUpdatesWtdVariance = LAMBDA * node.QUpdatesWtdVariance + (1.0f - LAMBDA) * diff * diff;
          }

        }

        if (node.IsRoot)
        {
          return;
        }
        else
        {
          if (parentRef.IsRoot)
          {
            indexOfChildDescendentFromRoot = node.Index;
            MCTSManager.ThreadSearchContext.RootMoveTracker.UpdateQValue(IndexInParent, vToApply, numToApply);
          }
          else
          {
            // Initiate a prefetch of parent's parent.
            // This happens far early enough that the memory access 
            // should be complete by the time we need the data.
            PrefetchNode(ref nodes[parentRef.ParentIndex.Index]);
          }

#if NOT
//          bool thisNodeExploratory = valuePriorAvgIsBetter < -0.03f; // -0.03f
//          bool thisNodeMuchWorse = valueThisSampleBetter < -0.20f; // -0.20f

// somewhat promising in suites but loses moderately badly in games
          if (node.N > 1 && node.ParentRef.N > 0 && !double.IsNaN(node.ParentRef.Q) && thisNodeMuchWorse && thisNodeExploratory && SearchManager.ThreadSearchContext.ParamsSearch.TEST_FLAG)
            vToApply = (float)node.Q;
#endif

          if (first)
          {
            vToApply = vToApplyNonFirst;
            dToApply = dToApplyNonFirst;
            first = false;
          }

          // Backup in tree 
          vToApply *= -1; // flip sign to change perspective
          mToApply++;     // moves left will look one greater from parent's perspective

          node = ref parentRef;
        }
      }
    }


    /// <summary>
    /// Applies specified update to W field (only) of node and all ancestors.
    /// </summary>
    /// <param name="wDelta"></param>
    public void BackupApplyWDeltaOnly(float wDelta)
    {
      Span<MCTSNodeStruct> nodes = MCTSNodeStoreContext.Nodes.Span;

      ref MCTSNodeStruct node = ref this;
      while (true)
      {
        // TODO: updated needed to other fields, sucha s VSumSquares
        node.W += wDelta;

        if (node.IsRoot)
        {
          return;
        }
        else
        {
          // Backup in tree (flip sign for change in perspective)
          wDelta *= -1;
          node = ref nodes[node.ParentIndex.Index];
        }

      }
    }

    /// <summary>
    /// Forcibly resets state to be that of a drawn position.
    /// </summary>
    internal void ResetToDraw()
    {
      WinP = 0;
      LossP = 0;
      Terminal = GameResult.Draw;
      MPosition = 1;

      W = 0;
      dSum = DepthInTree;
      mSum = N;

      if (!IsRoot)
      {
        ParentRef.DrawKnownToExistAmongChildren = true;
      }
    }


    /// <summary>
    /// Processes a node which has been determined to be a proven loss.
    /// Propagates this upward to the parent since parent's best move 
    /// is now obviously this one.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="lossP"></param>
    /// <param name="m"></param>
    internal void SetProvenLossAndPropagateToParent(FP16 lossP, FP16 m)
    {
      WinP = 0;
      LossP = lossP;
      MPosition = (byte)MathF.Round(m, 0);

      if (!IsRoot)
      {
        // This checkmate will obviously be chosen by the opponent
        // Therefore propagate the result up to the opponent as a victory,
        // overriding such that the Q for that node reflects the certain loss
        ref MCTSNodeStruct parentRef = ref this.ParentRef;
        parentRef.WinP = lossP;
        parentRef.LossP = 0;
        parentRef.W = parentRef.V * parentRef.N; // Make Q come out to be same as V (which has already been set to the sure win)
        parentRef.MPosition = (byte)MathF.Round(m + 1, 0);

        parentRef.CheckmateKnownToExistAmongChildren = true;
      }
    }


#endregion

    /// <summary>
    /// Returns the index of this node in the array of parent's children.
    /// </summary>
    public int IndexInParent => miscFields.IndexInParent;

    public void SetIndexInParent(int indexInParent) => miscFields.IndexInParent = (byte)indexInParent;

    /// <summary>
    /// Computes the power mean over all children Q values using specified coefficient.
    /// The calculation via shift/scaling to make all Q values in range [0..1] to avoid negative numbers which are not valid for poer mean.
    /// </summary>
    /// <param name="p"></param>
    /// <param name="inverted"></param>
    /// <returns></returns>
    public float QPowerMean(double p, bool inverted)
    {
      if (Terminal != Chess.GameResult.Unknown)
      {
        return (float)Q;
      }

#if NOT
      // NOTE: this code was never tried, and left abandoned.
      //       It turns out any children with low visit count will have little impact
      //       because of use of weighted average.

      // We do not include any children with very low number of visits
      // (these could be noisy estimates with spurious maxima/minima)
      const int MIN_N_FOR_AVERAGE = 500;
      // Count number of eligible children visits
      int eligibleN = 0;
      for (int i = 0; i < NumChildrenExpanded; i++)
      {
        ref MCTSNodeStruct countChildNode = ref ChildAtIndex(i).ChildRef;
        if (countChildNode.N > MIN_N_FOR_AVERAGE)
          eligibleN += countChildNode.N;
      }
#endif
      PowerMeanCalculator pwc = new PowerMeanCalculator(N - 1, p);

      for (int i = 0; i < NumChildrenExpanded; i++)
      {
        MCTSNodeStructChild child = ChildAtIndex(i);
        if (child.IsExpanded)
        {
          ref MCTSNodeStruct childNode = ref child.ChildRef;

          if (childNode.N > 0)
          {
            double adjustedQ = 0.55 + 0.5 * (inverted ? -childNode.Q : childNode.Q);
            pwc.AddValue(adjustedQ, childNode.N);
          }
        }
      }

      float pm = (float)pwc.PowerMean;
      float adjustedPM = (pm - 0.55f) * 2.0f;
      float ret = inverted ? adjustedPM : -adjustedPM;
      //      Console.WriteLine($" {N}  PowerMean({p}) " + Q + " ==> " + ret);
      return ret;
    }

    /// <summary>
    /// Returns the new value for a variance accumulator with exponential weighting
    /// that reflects an update with a specified new update (repeated thisN times).
    /// </summary>
    static float NewEMWVarianceAcc(float priorAcc, float priorN, float squaredDeviation, int thisN, float lambda)
    {
      float newAcc = priorAcc;
      for (int i = 0; i < thisN; i++)
      {
        float priorVariance = newAcc / (priorN + i);

        float newVariance = priorVariance * (1.0f - lambda)
                          + squaredDeviation * lambda;
        newAcc = newVariance * (priorN + i + 1);
      }

      // Return the variance accumulator value which would now return 
      // our new variance target after the current sample is recorded
      return newAcc;
    }


  }

}

#if EXPERIMENTAL
    // This is an alternate version which uses Avx2.GatherVector256
    // It seems to be correct, but in extensive tests August 2020 did not seem any faster

    public unsafe void PrefetchChildren(MCTSNodeStore store, MCTSNodeStructIndex nodeIndex, int minIndex, int maxIndex)
    {
      Debug.Assert(maxIndex - minIndex < 8);

      // The AVX instructions load from array of floats, but we are actually a
      int FLOATS_PER_NODE = Marshal.SizeOf <MCTSNodeStruct>() / sizeof(float);

      // Prepare array of indices at which these nodes exist
      Span<int> indices = stackalloc int[8];

      Span<MCTSNodeStructChild> childSpan = store.Children.SpanForNode(in store.Nodes.Span[nodeIndex.Index]); // TODO: maybe just pass in the struct here instead of node Index?
      for (int i = minIndex; i < maxIndex; i++)
      {
        MCTSNodeStructChild child = childSpan[i];
        if (child.IsExpanded)
        {
          indices[i - minIndex] = child.ChildIndex.Index * FLOATS_PER_NODE;
        }
      }

      // Prefetch node data
      void* nodePtr = Unsafe.AsPointer(ref store.Nodes.Span[nodeIndex.Index]);
      PrefetchDataAt(nodePtr);


      fixed (int* indicesPtr = &indices[0])
      {
        Vector256<int> indicesVector = Avx2.LoadVector256(indicesPtr);
        Vector256<float> result = Avx2.GatherVector256((float*)store.Nodes.RawMemory, indicesVector, 4);
      }

    }
    // --------------------------------------------------------------------------------------------
    public void PossiblyPrefetchNodeAndChildrenInRange(MCTSNodeStore store, MCTSNodeStructIndex nodeIndex,
                                                       int firstChildIndex, int numChildren)
    {
      int numDone = 0;
      while ((numChildren - numDone) > 3)
      {
        int numThisLoop = Math.Min(8, numChildren - numDone);
        PrefetchChildren(store, nodeIndex, firstChildIndex + numDone, firstChildIndex + numDone + numThisLoop - 1);
        numDone += numThisLoop;
      }
    }

#endif

#if EXPERIMENTAL
    public (int indexBest, int indexSecondBest) IndicesMaxExpandedChildren(MCTSNodeStructMetricFunc rankFunc)
    {
      float bestV = float.MinValue;
      int bestI = -1;
      float nextBestV = float.MinValue;
      int nextBestI = -1;
      for (int i=0; i<NumChildrenExpanded; i++)
      {
        float value = rankFunc(in ChildAtIndexRef(i));
        if (value > bestV)
        {
          nextBestV = bestV;
          nextBestI = bestI;
          bestV = value;
          bestI = i;
        }
        else if (value > nextBestV)
        {
          nextBestV = value;
          nextBestI = i;
        }
      }

      return (bestI, nextBestI);
    }
#endif


#if NOT

    /// <summary>
    /// Returns a span which re-interprets the child array as an array of FP16
    /// to be used as a V buffer.
    /// </summary>
    /// <param name="store"></param>
    /// <returns></returns>
    internal unsafe Span<FP16> ChildrenArrayAsVBuffer(MCTSNodeStore store)
    {
      // Determine how many V scores we have room for
      int maxVScores = NumPolicyMoves * (Marshal.SizeOf<MCTSNodeStructChild>() / Marshal.SizeOf<FP16>());
      return new Span<FP16>(Unsafe.AsPointer(ref store.Children.childIndices[ChildStartIndex]), maxVScores);
    }


    public unsafe void FillChidrenWithVScores(MCTSNodeStore store, ref MCTSNodeStruct source)
    {
      if (NumPolicyMoves > 0)
      {
        // We expect child array to e already allocated
        Debug.Assert(ChildStartBlockIndex > 0);

        // Interpet the 
        Span<FP16> vScores = ChildrenArrayAsVBuffer(store);

        int count = 0;
        MCTSNodeSequentialVisitor visitor = new MCTSNodeSequentialVisitor(store, source.Index);
        foreach (MCTSNodeStructIndex childNodeIndex in visitor.Iterate)
        {
          if (count >= vScores.Length)
            break;
          else
            vScores[count++] = store.Nodes.nodes[childNodeIndex.Index].V;
        }

        // If we did not fill V array completely, set the 
        // next element as an "end" market (NaN)
        if (count < vScores.Length) vScores[count] = FP16.NaN;
      }
    }

#endif

