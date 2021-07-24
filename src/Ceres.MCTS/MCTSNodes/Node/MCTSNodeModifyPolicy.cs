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
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;

#endregion


namespace Ceres.MCTS.MTCSNodes
{
  public unsafe sealed partial class MCTSNode
  {
    /// <summary>
    /// Modifies a specified node by blending in 
    /// fractionally another specified policy .
    /// 
    /// TODO: Slight imperfections:
    ///         - do we need to renormalize probabilities?
    ///         - what if the the new policy as a move not already present (currently ignored)
    ///         - nodes which are transposition linked are ignored
    /// </summary>
    public void BlendPolicy(in CompressedPolicyVector otherPolicy, float fracOther)
    {
      if (IsTranspositionLinked)
      {
        // TODO: should we apply this update to the transposition root?
        return;
      }

      if (fracOther == 0)
      {
        return;
      }

      float softmaxValue = Context.ParamsSelect.PolicySoftmax;

      const bool DEBUG = false;
      CompressedPolicyVector thisPolicy = default;
      if (DEBUG)
      {
        MCTSNodeStructUtils.ExtractPolicyVector(softmaxValue, in Ref, ref thisPolicy);
        Console.WriteLine(thisPolicy.ToString());
        Console.WriteLine(otherPolicy.ToString());
      }

      // Process expanded nodes
      for (int i = 0; i < NumPolicyMoves; i++)
      {
        var childInfo = ChildAtIndexInfo(i);
        int indexOther = otherPolicy.IndexOfMove(childInfo.move);

        // Only try to modify policy if the move was already in the policy
        if (indexOther != -1)
        {
          float otherProbabilityRaw = otherPolicy.PolicyInfoAtIndex(indexOther).Probability;
          FP16 newPolicy = BlendedP(softmaxValue, fracOther, childInfo.p, otherProbabilityRaw, ParamsSelect.MinPolicyProbability);

          if (i < NumChildrenExpanded)
          {
            // Directly set the new policy on the node structure
            ChildAtIndex(i).Ref.P = newPolicy;
          }
          else
          {
            // Cause the new p to be set in the children list
            ChildAtIndexRef(i).SetUnexpandedPolicyValues(childInfo.move, newPolicy);
          }
        }

      }

      // Make sure the children are ordered by policy after the operation
      // (this is expected by the leaf selection logic).
      EnsureChildrenOrderedByPolicy();

      if (DEBUG)
      {
        CompressedPolicyVector modifiedPolicy = default;

        MCTSNodeStructUtils.ExtractPolicyVector(softmaxValue, in Ref, ref modifiedPolicy);

        Console.WriteLine(modifiedPolicy.ToString());

        var p1 = modifiedPolicy.DecodedAndNormalized;
        var p2 = thisPolicy.DecodedAndNormalized;
        for (int i = 0; i < 1858; i++)
          if (MathF.Abs(p1[i] - p2[i]) > 0.075)
          {
            Console.WriteLine("*** BIG DISCREPANCY");
          }

        Console.WriteLine();
      }
    }


    /// <summary>
    /// Resorts the array of children to be in descending by policy (if necessary).
    /// </summary>
    /// <param name="node"></param>
    public void EnsureChildrenOrderedByPolicy()
    {
      bool didSwap = false;
      int numSwapped;
      Span<MCTSNodeStructChild> childrenRaw = Ref.Children;
      do
      {
        numSwapped = 0;
        for (int i = 0; i < NumPolicyMoves - 1; i++)
        {
          float p1 = ChildAtIndexInfo(i).p;
          float p2 = ChildAtIndexInfo(i + 1).p;
          if (p2 > p1)
          {
            MCTSNodeStructChild temp = childrenRaw[i];
            childrenRaw[i] = childrenRaw[i + 1];
            childrenRaw[i + 1] = temp;
            numSwapped++;
            didSwap = true;
          }

        }
      } while (numSwapped > 0);

      // The leaf selection code expects the set of children which are expanded
      // to be contiguous starting at the first child.
      // Restore this property if necessary.
      if (didSwap)
      {
        bool sawExpanded = false;
        for (int i = NumPolicyMoves - 1; i >= 0; i--)
        {
          sawExpanded |= ChildAtIndexInfo(i).node != null;
          if (sawExpanded && ChildAtIndexInfo(i).node == null)
          {
            //              node.Ref.NumChildrenVisited = (byte)Math.Max(node.NumChildrenVisited, i+1);
            //              node.Ref.NumChildrenExpanded = (byte)Math.Max(node.NumChildrenVisited, i + 1);
            CreateChild(i, true);
          }
        }
      }
    }


    /// <summary>
    /// Computes a blended policy probability value (after softmax)
    /// given two specified probabilties.
    /// </summary>
    static FP16 BlendedP(float softmaxValue, float fracP2, 
                         float p1Softmaxed, float p2Raw, 
                         float minProbability)
    {
      float p1Raw = MathF.Pow(p1Softmaxed, softmaxValue);
//      p2Raw = MathF.Pow(p2Raw, 1.0f / softmaxValue);
      float avgRaw = (1.0f - fracP2) * p1Raw
                            + fracP2 * p2Raw;
      float softmaxed = MathF.Pow(avgRaw, 1.0f / softmaxValue);
      return (FP16)MathF.Max(softmaxed, minProbability);
    }

  }

}
