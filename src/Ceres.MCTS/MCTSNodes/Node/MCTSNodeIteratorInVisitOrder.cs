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

using Ceres.Chess;
using Ceres.Base.DataTypes;
using Ceres.Chess.MoveGen;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.MTCSNodes
{
  /// <summary>
  /// Class that facilitates visitation of all nodes in a subtree
  /// in the same order in which they were visited.
  /// 
  /// Also the state of the iteration is captured in a data structure
  /// so it can be stored and subsequent retrieved and resumed.
  /// 
  /// TODO: Improve class to possibly handle the case of a tree 
  ///       still being actively modified by search in progress
  ///       (e.g. stop once an in-flight node is reached).
  /// </summary>
  public class MCTSNodeIteratorInVisitOrder : ICloneable
  {
    /// <summary>
    /// Underlying node store.
    /// </summary>
    MCTSNodeStore Store;

    /// <summary>
    /// Root index.
    /// </summary>
    public MCTSNodeStructIndex Root;

    /// <summary>
    /// Sorted set of all pending branches which have been traversed
    /// need to be revisisted.
    /// </summary>
    SortedSet<MCTSNodeStructIndex> pendingBranches;

    /// <summary>
    /// Statistic holding the maximum number of branches seen so far.
    /// </summary>
    public int MaxBranches = 0;

    /// <summary>
    /// Statistic holding the number of nodes returned so far in this iteration.
    /// </summary>
    public int NumReturned = 0;

    /// <summary>
    /// Index of node where the iteration is currently visiting.
    /// </summary>
    MCTSNodeStructIndex currentNode;


    /// <summary>
    /// Constructor to begin iteration within specified at a specified node.
    /// </summary>
    /// <param name="store"></param>
    /// <param name="root"></param>
    public MCTSNodeIteratorInVisitOrder(MCTSNodeStore store, MCTSNodeStructIndex root)
    {
      Store = store;
      Root = root;

      currentNode = root;
      pendingBranches = new SortedSet<MCTSNodeStructIndex>();
    }

    /// <summary>
    /// Returns the next branch to be iterated.
    /// </summary>
    /// <param name="thisNodeIndex"></param>
    /// <param name="boundSeqNum"></param>
    /// <returns></returns>
    MCTSNodeStructIndex GetNextBranch(MCTSNodeStructIndex thisNodeIndex, int boundSeqNum)
    {
      int bestIndex1 = -1;
      int bestSeq = int.MaxValue;

      ref MCTSNodeStruct thisNode = ref Store.Nodes.nodes[thisNodeIndex.Index];
      
      int numChildren = thisNode.NumChildrenExpanded;
      if (numChildren == 0)
      {
        return new MCTSNodeStructIndex(0);
      }

      Span<MCTSNodeStructChild> children = thisNode.Children;
      for (int i = 0; i < numChildren; i++)
      {
        ref MCTSNodeStruct childNode = ref children[i].ChildRef;
        if (childNode.Index.Index > boundSeqNum)
        {
          if (childNode.Index.Index < bestSeq)
          {
            bestIndex1 = i;
            bestSeq = childNode.Index.Index;
          }
        }
      }

      if (bestIndex1 == -1)
        return new MCTSNodeStructIndex(0);
      else
        return children[bestIndex1].ChildIndex;
    }


    /// <summary>
    /// Switches from current branch to next pending branch.
    /// </summary>
    void SwitchToNextBranch()
    {
      // Peel off the node to switch to
      MCTSNodeStructIndex nextBranch = pendingBranches.Min;
      ref MCTSNodeStruct nextBranchRef = ref Store.Nodes.nodes[nextBranch.Index];

      // Remove this from the set of pending branches
      pendingBranches.Remove(nextBranch);

      // But first record next most compelling pending branch from parent
      MCTSNodeStructIndex nextBranch2 = GetNextBranch(nextBranchRef.ParentIndex, nextBranch.Index);

      if (!nextBranch2.IsNull)
      {
        pendingBranches.Add(nextBranch2);
        if (pendingBranches.Count > MaxBranches)
        {
          MaxBranches = pendingBranches.Count;
        }
      }

      // Switch to this branch
      currentNode = nextBranch;
    }


    /// <summary>
    /// Iterator method to return next available branch.
    /// </summary>
    /// <returns></returns>
    public MCTSNodeStructIndex GetNext()
    {
      while (true)
      {
        if (currentNode.IsNull)
        {
          if (pendingBranches.Count == 0)
            return MCTSNodeStructIndex.Null; // completely exhausted
          else
          {
            SwitchToNextBranch();
          }
        }

        // Check the current node to see if it is no longer on the minimum sequence path
        if (pendingBranches.Count > 0 && currentNode.Index >= pendingBranches.Min.Index)
        {
          SwitchToNextBranch();
        }
        else
        {
          // We'll return this node next
          MCTSNodeStructIndex nodeToReturn = currentNode;

          // But first record next most compelling pending branch
          MCTSNodeStructIndex nextBranch1 = GetNextBranch(currentNode, currentNode.Index);
          MCTSNodeStructIndex nextBranch2 = GetNextBranch(currentNode, currentNode.Index + 1);

          if (!nextBranch2.IsNull)
          {
            pendingBranches.Add(nextBranch2);
            if (pendingBranches.Count > MaxBranches)
            {
              MaxBranches = pendingBranches.Count;
            }
          }

          currentNode = nextBranch1;

          NumReturned++;
          return nodeToReturn;
        }
      }
    }

    /// <summary>
    /// Main public method which returns an IEnumerable
    /// that can be used to traverse the subtree.
    /// </summary>
    public IEnumerable<MCTSNodeStructIndex> Iterate
    {
      get
      {
        while (true)
        {
          MCTSNodeStructIndex nextValue = GetNext();
          if (nextValue.IsNull)
          {
            yield break;
          }
          else
          {
            yield return nextValue;
          }
        }
      }
    }

    /// <summary>
    /// Helper method to clone the current state of the iteatore into another iterator.
    /// </summary>
    /// <returns></returns>
    public object Clone()
    {
      MCTSNodeIteratorInVisitOrder ret = new MCTSNodeIteratorInVisitOrder(Store, Root);

      // Clone the sorted set
      ret.pendingBranches = new SortedSet<MCTSNodeStructIndex>();
      foreach (MCTSNodeStructIndex entry in pendingBranches)
      {
        ret.pendingBranches.Add(entry);
      }

      ret.MaxBranches = MaxBranches;
      ret.NumReturned = NumReturned;
      ret.currentNode = currentNode;

      return ret;
    }

#if NOT
    // --------------------------------------------------------------------------------------------
    public static void TestMCTSNodeSequentialVisitor()
    {
      MCTSNodeStore store = new MCTSNodeStore(50_000, MGMoveSequence.StartPos);
      MCTSNodeStorageSerialize.Restore(store, @"c:\temp", "TESTSORE");

      SearchContext context = new SearchContext(store, null, new ParamsNN(), new ParamsSearch(new ParamsNN()), new ParamsSelect(true), null);
      Console.WriteLine(store.Nodes.NumTotalNodes);
      int countSeen = 0;
      using (new SearchContextExecutionBlock(context))
      {
        MCTSNodeSequentialVisitor visitor = new MCTSNodeSequentialVisitor(store, store.RootIndex);
        foreach (var index in visitor.Iterate)
        {
          //      ref MCTSNodeStruct node = ref store.Nodes.nodes[index.Index];
          countSeen++;
          if (index.Index != countSeen)
            Console.WriteLine("ERROR " + index.Index);
        }

        Console.WriteLine(countSeen);
        Console.WriteLine(visitor.MaxBranches);
      }
    }

#endif    
  }
}
