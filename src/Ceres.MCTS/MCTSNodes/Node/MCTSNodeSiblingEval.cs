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
using System.Runtime.CompilerServices;
using Ceres.Base.Math;
using Ceres.Base.Threading;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.TBBackends.Fathom;
using Ceres.MCTS.Environment;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;

#endregion


namespace Ceres.MCTS.MTCSNodes
{
  /// <summary>
  /// Information relating to the evaluation a node which 
  /// is a (lower probability) sibling of a node currently in flight.
  /// </summary>
  public readonly struct MCTSNodeSiblingEval
  {
    /// <summary>
    /// Total number of sibling evaluations blended in
    /// to backed up values at nonzero weight.
    /// </summary>
    public static AccumulatorMultithreaded CountSiblingEvalsUsed;

    /// <summary>
    /// Maximum number of children with lower probability to
    /// potentially to be scanned (at a cost of processor resources).
    /// </summary>
    const int MAX_SCAN = 10;

    /// <summary>
    /// Maximum inferiority of child which will be scanned.
    /// </summary>
    const float MAX_P_INFERIORITY = 0.05f;

 
    /// <summary>
    /// Number of visits over which this node's Q was determined
    /// (will be 1 if not transposition linked, otherwise N of transposition root).
    /// </summary>
    public readonly int SelfN;

    /// <summary>
    /// Magnitude by which the selected sibling had lower prior policy probability.
    /// </summary>
    public readonly float SiblingPInferiority;

    /// <summary>
    /// The Q of the sibling node (typically obtained from transposition or tablebase).
    /// </summary>
    public readonly float SiblingQ;

    /// <summary>
    /// The D of the sibling node (typically obtained from transposition or tablebase).
    /// </summary>
    public readonly float SiblingD;

    /// <summary>
    /// The N of the sibling node in the transposition root node.
    /// </summary>
    public readonly short SiblingN;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="nSelf"></param>
    /// <param name="siblingQ"></param>
    /// <param name="siblingD"></param>
    /// <param name="siblingN"></param>
    public MCTSNodeSiblingEval(int nSelf, float siblingPInferiority, float siblingQ, float siblingD, short siblingN)
    {
      SelfN = nSelf;
      SiblingPInferiority = siblingPInferiority;
      SiblingQ = siblingQ;
      SiblingD = siblingD;
      SiblingN = siblingN;
    }


    /// <summary>
    /// Fractional weight to be used of a sibling node (rather than the node actually in flight)
    /// to be used in backup.
    /// </summary>
    public float WeightSiblingForBackup(bool hadBetterQ)
    {
      // The sibling had a better value (from the perspective of the node)
      // so blend this in at a possibly considerable weight
      // (based on a minimax perspective).
      if (SiblingN == short.MaxValue)
      {
        // Was terminal (e.g. from tablebases).
        return 1.0f;
      }
      else
      {
        // Potentially we could give sibling eval lower weight if was not better 
        // (minimax method) but in practice this seemed distinctly worse.
        //float qBetterScaling = hadBetterQ ? 1.0f : 0.75f;
        const float qBetterScaling = 1.0f;

        // Give more weight to sibling the closer it is to this node's policy
        const float SIBLING_SCALING_FACTOR = 15f;
        float distanceScaling = SIBLING_SCALING_FACTOR * (MAX_P_INFERIORITY - SiblingPInferiority);

        // The more visits were used as the basis for estimating the sibling N
        // compared to the number of visit used to evaluate this node,
        // the more weight we give to the sibling.
        float CalcConfidenceAdjustment(float a, float b)
        {
          Debug.Assert(a >= b);

          // Naively assume uncertainty reduced with some fractional power of number of visits.
          float relativeConfidence = MathF.Pow(a / b, 0.75f);

          // We want adjustments [0..0.2] as the relative confidences runs from [1...4_or_above];
          const float MAX_ADJ = 0.2f;
          relativeConfidence = StatUtils.Bounded(relativeConfidence, 1, 4);
          return (relativeConfidence - 1) * (MAX_ADJ / 3);
        }

        // Weight centered at naive default of 0.5.
        float confidenceScaling = 0.5f + ((SiblingN > SelfN) ? CalcConfidenceAdjustment(SiblingN, SelfN)
                                                             : -CalcConfidenceAdjustment(SelfN, SiblingN));

        return confidenceScaling * distanceScaling * qBetterScaling;
      }

    }



    /// <summary>
    /// Returns the value to be backed up in the tree to the
    /// parent of the node argument and above, based on the
    /// specified normal MCTS value but also possibly blending
    /// in the influence of the sibling node.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="vToApplyFromChildOfNode"></param>
    /// <returns></returns>
    public (float, float) BackupValueForNode(MCTSNode node, float vToApplyFromChildOfNode, float dToApplyFromChildOfNode)
    {
      bool siblingWasBetter = SiblingQ < vToApplyFromChildOfNode;
      float siblingWt = WeightSiblingForBackup(siblingWasBetter);
      
      if (siblingWt > 0)
      {
        CountSiblingEvalsUsed.Add(1, node.Index);
        return (
                 (SiblingQ * siblingWt) + (vToApplyFromChildOfNode * (1.0f - siblingWt)),
                 (SiblingD * siblingWt) + (dToApplyFromChildOfNode * (1.0f - siblingWt))
               );

      }
      else
      {
        return (vToApplyFromChildOfNode, dToApplyFromChildOfNode);
      }
    }


    /// <summary>
    /// Possibly sets the PendingSiblingEval by looking at siblings not yet evaluated
    /// for ones which have either:
    ///   - a tablebase hit, or
    ///   - a transposition hit
    /// </summary>
    /// <param name="node"></param>
    /// <param name="numCacheHashPositions"></param>
    /// <exception cref="Exception"></exception>
    internal static unsafe void TrySetPendingSiblingValue(MCTSNode node, int numCacheHashPositions)
    {
      if (numCacheHashPositions != 1)
      {
        throw new Exception("Implementation restriction, the sibling transposition lookup feature not yet implemented for history length > 1");
      }

      ref readonly MCTSNodeStruct parentRef = ref node.Parent.StructRef;
      int parentChildIndex = node.IndexInParentsChildren;
      Span<MCTSNodeStruct> nodes = node.Store.Nodes.Span;


      int maxChildIndex = Math.Min(node.Parent.NumPolicyMoves - 1, parentChildIndex + MAX_SCAN);
      float minChildP = node.P - MAX_P_INFERIORITY;

      float? bestQ = null;
      float bestD = float.NaN;
      short bestN = 0;
      float pInferiority = 0;
      for (int childIndex = parentChildIndex + 1; childIndex <= maxChildIndex; childIndex++)
      {
        MCTSNodeStructChild childInfo = parentRef.ChildAtIndex(childIndex);

        // Stop scanning if we see a child that has P much lower than child being evaluated.
        if (childInfo.P < minChildP)
        {
          break;
        }

        pInferiority = node.P - childInfo.P;

        // Determine the position arrived at after following this sibling.
        MGPosition posMG = node.Parent.Annotation.PosMG;
        posMG.MakeMove(ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(childInfo.Move, in posMG));
        Position pos = posMG.ToPosition;

        // Possibly probe tablebases for proven loss.
        if (pos.PieceCount <= FathomTB.MaxPieces)
        {
          const float NO_TABLEBASE_PROBE_ABOVE_Q = 0.70f;
          bool alreadyAlmostWon = parentRef.Q > NO_TABLEBASE_PROBE_ABOVE_Q;
          if (!alreadyAlmostWon && FathomTB.ProbeWDL(in pos) == FathomWDLResult.Loss)
          {
            bestQ = -ParamsSelect.LossPForProvenLoss(1, false);
            bestD = 0;
            bestN = short.MaxValue;
            break;
          }
        }

        // Compute hash of position and lookup in transposition table.
        ulong siblingHash = pos.CalcZobristHash(node.Context.EvaluatorDef.HashMode);
        if (node.Tree.TranspositionRoots.TryGetValue(siblingHash, out int siblingTanspositionNodeIndex))
        {
          ref readonly MCTSNodeStruct siblingTranspositionNodeRef = ref nodes[siblingTanspositionNodeIndex];

          // Update bestQ so far if this is an evaluated node which would look better from perspective of parent (i.e. more negative).
          if (siblingTranspositionNodeRef.IsEvaluated 
            && !siblingTranspositionNodeRef.HasRepetitions // ignore if repetitions since might have different repetition history
            )
          {
            if (bestQ is null || siblingTranspositionNodeRef.Q < bestQ.Value)
            {
              bestQ = (float)siblingTranspositionNodeRef.Q;
              bestD = siblingTranspositionNodeRef.DAvg;
              bestN = (short)Math.Min(short.MaxValue - 1, siblingTranspositionNodeRef.N);
            }
          }
        }
      }

      if (bestQ.HasValue)
      {
        int nSelf = 1;
        if (node.IsTranspositionLinked)
        {
          ref readonly MCTSNodeStruct transpositionRef = ref nodes[node.TranspositionRootIndex];
          nSelf = transpositionRef.N;
        }
        node.SiblingEval = new MCTSNodeSiblingEval(nSelf, pInferiority, bestQ.Value, bestD, bestN);
      }
    }


    [ModuleInitializer]
    internal static void ModuleInit()
    {
      CountSiblingEvalsUsed.Initialize();
    }
  }
}
