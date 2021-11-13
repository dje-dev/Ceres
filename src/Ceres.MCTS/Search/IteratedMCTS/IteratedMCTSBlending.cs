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

using Ceres.Chess.EncodedPositions;
using Ceres.Chess.PositionEvalCaching;
using Ceres.MCTS.MTCSNodes;
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.Search.IteratedMCTS
{
  public static class IteratedMCTSBlending
  {
    /// <summary>
    /// Extracts subset of nodes (filtered to those with N sufficiently large)
    /// into a cache to be reused in a subsequent interation of an iterated search.
    /// </summary>
    /// <param name="root"></param>
    /// <param name="minN"></param>
    /// <param name="maxWeightEmpiricalPolicy"></param>
    /// <param name="treeModificationType"></param>
    /// <returns></returns>
    public static PositionEvalCache ModifyNodeP(MCTSNode root, int minN, float maxWeightEmpiricalPolicy, 
                                                IteratedMCTSDef.TreeModificationType treeModificationType)
    {
      bool cache = treeModificationType == IteratedMCTSDef.TreeModificationType.DeleteNodesMoveToCache;

      PositionEvalCache posCache = cache ? new PositionEvalCache() : null;

      root.StructRef.TraverseSequential(root.Store, (ref MCTSNodeStruct nodeRef, MCTSNodeStructIndex index) =>
      {
        bool shouldBlend = nodeRef.N >= minN;
        if (shouldBlend || treeModificationType == IteratedMCTSDef.TreeModificationType.DeleteNodesMoveToCache)
        {
          MCTSNode node = root.Tree.GetNode(index);

          bool rewriteInTree = treeModificationType == IteratedMCTSDef.TreeModificationType.ClearNodeVisits 
                            && nodeRef.N >= minN;

          float weightEmpirical = maxWeightEmpiricalPolicy * ((float)nodeRef.N / (float)root.N);

          ProcessNode(posCache, node, weightEmpirical,  cache, rewriteInTree);
        }

        return true;
      });
      
      return posCache;
    }


    // --------------------------------------------------------------------------------------------
    static void ProcessNode(PositionEvalCache cache, MCTSNode node, float weightEmpirical, 
                            bool saveToCache, bool rewriteNodeInTree)
    {
      Span<MCTSNodeStructChild> children = node.StructRef.Children;

      // TODO: optimize this away if saveToCache is false
      ushort[] probabilities = new ushort[node.NumPolicyMoves];
      ushort[] indices = new ushort[node.NumPolicyMoves];

      // Compute empirical visit distribution
      float[] nodeFractions = new float[node.NumPolicyMoves];
      for (int i = 0; i < node.NumChildrenExpanded; i++)
        nodeFractions[i] = (float)node.ChildAtIndex(i).N / (float)node.N;

      // Determine P of first unexpanded node
      // We can't allow any child to have a new P less than this
      // since we need to keep them in order by P and the resorting logic below
      // can only operate over expanded nodes
      float minP = 0;
      if (node.NumChildrenExpanded < node.NumPolicyMoves)
        minP = node.ChildAtIndexInfo(node.NumChildrenExpanded).p;

      // Add each move to the policy vector with blend of prior and empirical values
      for (int i = 0; i < node.NumChildrenExpanded; i++)
      {
        (MCTSNode node, EncodedMove move, FP16 p) info = node.ChildAtIndexInfo(i);
        indices[i] = (ushort)info.move.IndexNeuralNet;

        float newValue = (1.0f - weightEmpirical) * info.p
                       + weightEmpirical * nodeFractions[i];
        if (newValue < minP) newValue = minP;
        probabilities[i] = CompressedPolicyVector.EncodedProbability(newValue);

        if (rewriteNodeInTree && weightEmpirical != 0)
        {
          MCTSNodeStructChild thisChild = children[i];
          if (thisChild.IsExpanded)
          {
            ref MCTSNodeStruct childNodeRef = ref thisChild.ChildRef;
            thisChild.ChildRef.P = (FP16)newValue;
          }
          else
          {
            node.StructRef.ChildAtIndex(i).SetUnexpandedPolicyValues(thisChild.Move, (FP16)newValue);
          }
        }

      }

      // TODO: is it true the new P may no longer sum to 1.0f, if so do we need to normalize?

      if (rewriteNodeInTree)
      {
        // Sort expanded nodes based on their new P (bubblesort)
        bool foundUnsorted;
        do
        {
          foundUnsorted = false;
          for (int i = 1; i < node.NumChildrenExpanded; i++)
          {
            if (node.ChildAtIndex(i).P > node.ChildAtIndex(i - 1).P)
            {
              foundUnsorted = true;
              MCTSNodeStructChild temp = children[i - 1];
              children[i - 1] = children[i];
              children[i] = temp;
            }
          }
        } while (foundUnsorted);
      }

      if (saveToCache)
      {
        // Initialize and sort policy entries
        CompressedPolicyVector newPolicy = default;
        CompressedPolicyVector.Initialize(ref newPolicy, indices, probabilities);

        // Save back to cache
        // TODO: possibly blend in the search Q to the WinP/LossP (possibly M too?)
        cache.Store(node.StructRef.ZobristHash, node.Terminal, node.WinP, node.LossP, node.MPosition, in newPolicy);
      }
    }

  }
}
