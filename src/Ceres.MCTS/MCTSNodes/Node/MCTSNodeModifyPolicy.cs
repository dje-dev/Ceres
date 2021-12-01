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
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;

#endregion


namespace Ceres.MCTS.MTCSNodes
{
  public unsafe partial struct MCTSNode
  {
    /// <summary>
    /// Modifies a specified node by blending in 
    /// fractionally another specified policy .
    /// 
    /// TODO: Imperfections/possible improvements:
    ///         - do we need to renormalize probabilities?
    ///         - what if the the new policy as a move not already present (currently ignored)
    ///         - could we detect that another node in the transposition equivalence class was blended
    ///           and copy this blending?
    /// </summary>
    public void BlendPolicy(in CompressedPolicyVector otherPolicy, float fracOther)
    {
      if (IsTranspositionLinked)
      {
        return;
      }

      if (fracOther == 0)
      {
        return;
      }

      float softmaxValue = Context.ParamsSelect.PolicySoftmax;

      const bool VERBOSE = false;
      CompressedPolicyVector thisPolicyBeforeBlend = default;
      if (VERBOSE)
      {
        MCTSNodeStructUtils.ExtractPolicyVector(softmaxValue, in StructRef, ref thisPolicyBeforeBlend);
        Console.WriteLine("CURRENT: " + thisPolicyBeforeBlend.DumpStrShort(0, 14));
        Console.WriteLine("OTHER  : " + otherPolicy.DumpStrShort(0, 14));
      }

      // Process all children.
      for (int i = 0; i < NumPolicyMoves; i++)
      {
        (MCTSNode node, EncodedMove move, FP16 p) childInfo = ChildAtIndexInfo(i);
        int indexOther = otherPolicy.IndexOfMove(childInfo.move);

        // Only try to modify policy if the move was already in the policy
        // TODO: should we possibly replace some unexpanded move in this node
        //       with this move if this other policy is relatively high? (but maybe this is rare)
        if (indexOther != -1)
        {
          float otherProbabilityRaw = otherPolicy.PolicyInfoAtIndex(indexOther).Probability;
          FP16 newPolicy = BlendedP(softmaxValue, fracOther, childInfo.p, otherProbabilityRaw, ParamsSelect.MinPolicyProbability);
          Debug.Assert(!FP16.IsNaN(newPolicy));

          if (i < NumChildrenExpanded)
          {
            // Directly set the new policy in the node structure.
            ChildAtIndex(i).StructRef.P = newPolicy;
          }
          else
          {
            // Replace the P in the (not-yet-expanded) children list.
            ChildAtIndexRef(i).SetUnexpandedPolicyValues(childInfo.move, newPolicy);
          }
        }
      }

      // Generally children are laid out in ascending policy order.
      // However the order of children may have now changed.
      // It is not possible to fully restore this order because Ceres
      // leaf selection logic assumes that expanded children are all contiguous at beginning of array.

      // However we are free to reorder the unexpanded children to be back in policy order.
      // This is actually what is important from a leaf selection perspective, since
      // all expanded childre are already always included for consideration in expansion
      // and the fact that they may be out of order does not change behavior.
      // However this method gets the most promising unexpanded children shifted to lower indices
      // and this is important so they are the first candidate unexpanded children to be considered in future visits.
      ReorderUnexpandedChildren();

      if (VERBOSE)
      {
        float sumPolicy = 0;

        // Extract new policy of this node as a CompressedPolicyVector.
        CompressedPolicyVector modifiedPolicy = default;
        MCTSNodeStructUtils.ExtractPolicyVector(softmaxValue, in StructRef, ref modifiedPolicy);

        Console.WriteLine("NEW    : " + modifiedPolicy.DumpStrShort(0, 14));

        float[] p1 = modifiedPolicy.DecodedAndNormalized;
        float[] p1Raw = modifiedPolicy.DecodedNoValidate;
        float[] p2 = thisPolicyBeforeBlend.DecodedAndNormalized;
        //float[] p2Raw = thisPolicy.DecodedNoValidate;
        for (int i = 0; i < 1858; i++)
        {
          sumPolicy += p1Raw[i];
          if (MathF.Abs(p1[i] - p2[i]) > 0.075)
          {
            Console.WriteLine($"*** BIG DIFFERENCE {EncodedMove.FromNeuralNetIndex(i) }: {p1[i],6:F3} {p2[i],6:F3}");
          }
        }
        Console.WriteLine($"Sum modified policy={sumPolicy,6:F3}  N= { N}  NumExpanded= {NumChildrenExpanded}");
        Console.WriteLine();
      }
    }

    public void ReorderUnexpandedChildren()
    {
      bool foundOutOfOrder;
      do
      {
        // Bubble sort.
        foundOutOfOrder = false;
        for (int i=NumChildrenExpanded;i<NumPolicyMoves  - 1; i++)
        {
          ref MCTSNodeStructChild infoLeft = ref ChildAtIndexRef(i);
          ref MCTSNodeStructChild infoRight = ref ChildAtIndexRef(i + 1);

          if (infoLeft.P < infoRight.p)
          {
            MCTSNodeStructChild temp = infoLeft;
            infoLeft = infoRight;
            infoRight = temp;

            foundOutOfOrder = true;
          }
        }
      } while (foundOutOfOrder);  
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
