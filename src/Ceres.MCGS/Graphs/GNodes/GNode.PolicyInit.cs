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
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Ceres.Base.DataType;
using Ceres.Base.DataTypes;

using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.MCGS.Graphs.GEdgeHeaders;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

public readonly partial struct GNode : IComparable<GNode>, IEquatable<GNode>
{
  // Although we retrieve up to 64 moves from neural network, 
  // the encoding used in the raw search node is compressed such that 
  // it can represent values only in the range [0..63].
  internal const int MAX_MOVES_RETAIN = 63;

  /// <summary>
  /// Represents one  as FP16. This field is constant.
  /// </summary>
  static readonly FP16 FP16_ONE = FP16.ToHalf(1);

  /// <summary>
  /// Populates the children moves and probabilities from a CompressedPolicyVector.
  /// </summary>
  /// <param name="posMG"></param>
  /// <param name="policyVector"></param>
  /// <param name="movesMG">Will return sorted in the same way as the legal policy move</param>
  /// <param name="childrenProbabilites"></param>
  /// <returns >Number of valid moves returned in childrenProbabilites</returns>
  static internal int InitializeFromPolicyInfo(float policySoftmax, float minPolicyProbability,
                                               in MGPosition posMG, in CompressedPolicyVector policyVector,
                                               MGMoveList movesMG, Span<FP16> childrenProbabilites,
                                               Span<EncodedMove> validLZMovesUsed,
                                               bool returnedMovesAreInSameOrderAsMGMoveList = false)
  {
    // Check for either of two special cases
    if (movesMG.NumMovesUsed == 0)
    {
      // Special case 1: Nothing to do if there were no moves
      return 0;
    }
    else if (movesMG.NumMovesUsed == 1)
    {
      // Special case 2: Move must have probability 1 if only legal move
      childrenProbabilites[0] = FP16_ONE;
      if (!validLZMovesUsed.IsEmpty)
      {
        MGMove onlyMove = movesMG.MovesArray[0];
        validLZMovesUsed[0] = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(onlyMove);
      }
      return 1;
    }
    else
    {
      return DoInitializeFromPolicyInfo(policySoftmax, minPolicyProbability, in posMG, in policyVector,
                                        movesMG, childrenProbabilites, validLZMovesUsed,
                                        returnedMovesAreInSameOrderAsMGMoveList);
    }
  }


  [SkipLocalsInit]
  static private int DoInitializeFromPolicyInfo(float policySoftmax, float minPolicyProbability,
                                                in MGPosition posMG, in CompressedPolicyVector policyVector,
                                                MGMoveList movesMG, Span<FP16> childrenProbabilites,
                                                Span<EncodedMove> validLZMovesUsed,
                                                bool returnedMovesAreInSameOrderAsMGMoveList = false)
  {
    // Ignore moves with extremely small probabilities
    // They will be too small to be ever selected, and often may actually be illegal moves
    const float THRESHOLD_IGNORE_MOVE = 0.00005f;

    // Special processing if random values requested
    (EncodedMove firstMove, _) = policyVector.PolicyInfoAtIndex(0);
    if (firstMove.RawValue == CompressedPolicyVector.SPECIAL_VALUE_RANDOM_NARROW)
    {
      return DoInitializeRandom(false, movesMG, childrenProbabilites, validLZMovesUsed);
    }
    else if (firstMove.RawValue == CompressedPolicyVector.SPECIAL_VALUE_RANDOM_WIDE)
    {
      return DoInitializeRandom(true, movesMG, childrenProbabilites, validLZMovesUsed);
    }

    //      int policyMoveCount = policyVector.ProbabilitySummaryList(movesFromPolicyTempBufferLocal, THRESHOLD_IGNORE_MOVE, topN: MAX_MOVES_RETAIN);

    Span<float> probabilitiesTemp = stackalloc float[Math.Min(MAX_MOVES_RETAIN, movesMG.NumMovesUsed)];

    float probabilitySumBeforeAdjust = 0;
    int countPolicyMovesProcessed = 0;

    // For efficiency, get a move array as direct variable
    MGMove[] legalMoveArray = movesMG.MovesArray;

    bool blackToMove = posMG.SideToMove == SideType.Black;
    float lastP = float.NaN;

    foreach ((int policyMoveIndex, EncodedMove thisPolicyMove, float thisPolicyProb) in policyVector.EnumerateProbabilityTuples(THRESHOLD_IGNORE_MOVE, MAX_MOVES_RETAIN))
    {
      // Policy values are expected to be sorted descending
      Debug.Assert(float.IsNaN(lastP) || thisPolicyProb <= lastP);

      // Find index of this move in the MG moves
      int indexLegalMove;
      if (returnedMovesAreInSameOrderAsMGMoveList)
      {
        indexLegalMove = policyMoveIndex;
      }
      else
      {
        indexLegalMove = MoveInMGMovesArrayLocator.FindMoveInMGMoves(in posMG, legalMoveArray, thisPolicyMove, countPolicyMovesProcessed, movesMG.NumMovesUsed, blackToMove);
      }

      // The NN might output illegal moves (not found in list of legal moves), ignore these
      if (indexLegalMove != -1)
      {
        probabilitySumBeforeAdjust += thisPolicyProb;

        probabilitiesTemp[countPolicyMovesProcessed] = thisPolicyProb;

        // Make the legal move array order agree with that of policy vector
        if (countPolicyMovesProcessed != indexLegalMove)
        {
          Debug.Assert(indexLegalMove > countPolicyMovesProcessed);
          (legalMoveArray[countPolicyMovesProcessed], legalMoveArray[indexLegalMove]) 
            = (legalMoveArray[indexLegalMove], legalMoveArray[countPolicyMovesProcessed]);
        }

        if (!validLZMovesUsed.IsEmpty)
        {
          validLZMovesUsed[countPolicyMovesProcessed] = thisPolicyMove;
        }

        countPolicyMovesProcessed++;
      }
    }

    if (countPolicyMovesProcessed == 0)
    {
      // Unfortunately the NN did not find any of the legal moves (or we are asked to generate random moves)
      // Work around this by assuming equal probability for all legal moves
      return PoliciesInitializedUniform(movesMG, childrenProbabilites, validLZMovesUsed, MAX_MOVES_RETAIN);
    }

    if (probabilitySumBeforeAdjust == 0 || float.IsNaN(probabilitySumBeforeAdjust))
    {
      throw new Exception("NaN policy probability returned from NN. Hash collision? " + posMG.ToPosition.FEN);
    }

    int numProbsToUse = Math.Min(countPolicyMovesProcessed, childrenProbabilites.Length);

    // Fill in all children probabilities (for valid moves found by NN)
    if (policySoftmax == 1.0f)
    {
      // Save probabilities (scaled to sum to 1.0)
      float scaleFactor = 1.0f / probabilitySumBeforeAdjust;
      Span<float> probs = probabilitiesTemp.Slice(0, numProbsToUse);
      if (numProbsToUse >= 16)
      {
        TensorPrimitives.Multiply(probs, scaleFactor, probs);
        TensorPrimitives.ConvertToHalf(probs, MemoryMarshal.Cast<FP16, Half>(childrenProbabilites));
      }
      else
      {
        for (int i = 0; i < numProbsToUse; i++)
        {
          childrenProbabilites[i] = new FP16(scaleFactor * probs[i]);
        }
      }
    }
    else
    {
      // Apply softmax operator with optional minimum probability enforcement
      ApplySoftmax(policySoftmax, minPolicyProbability, probabilitiesTemp, numProbsToUse, childrenProbabilites);
    }

    return numProbsToUse;
  }


  /// <summary>
  /// Applies softmax operator with optional minimum probability enforcement.
  /// Uses TensorPrimitives for vectorized operations where beneficial.
  /// </summary>
  [SkipLocalsInit]
  private static void ApplySoftmax(float policySoftmax, float minPolicyProbability,
                                   Span<float> probabilitiesTemp, int numProbsToUse,
                                   Span<FP16> childrenProbabilites)
  {
    Span<float> probs = probabilitiesTemp.Slice(0, numProbsToUse);
    float invPolicySoftmax = 1.0f / policySoftmax;

    // Apply softmax: Log, Multiply, Exp
    TensorPrimitives.Log(probs, probs);
    TensorPrimitives.Multiply(probs, invPolicySoftmax, probs);
    TensorPrimitives.Exp(probs, probs);

    // Compute sum after softmax
    float probabilitySumAfterAdjust = TensorPrimitives.Sum(probs);

    // Inline minimum probability enforcement and final scaling+conversion
    if (minPolicyProbability > 0)
    {
      // Enforce minimum probability - requires sequential processing due to ordering constraint
      float minProbability = minPolicyProbability * probabilitySumAfterAdjust;
      const float FRAC_RETAIN_PROBABILITY = 0.1f;

      // Adjust probabilities and recompute sum
      for (int i = 0; i < numProbsToUse; i++)
      {
        float thisProbability = probs[i];
        if (thisProbability < minProbability)
        {
          float replacementProbability = minPolicyProbability + FRAC_RETAIN_PROBABILITY * thisProbability;

          // Preserve order with prior entries
          if (i > 0 && replacementProbability > probs[i - 1])
          {
            replacementProbability = probs[i - 1];
          }

          probabilitySumAfterAdjust += (replacementProbability - thisProbability);
          probs[i] = replacementProbability;
        }
      }

      // Scale and convert - use TensorPrimitives for larger arrays
      float scaleFactor = 1.0f / probabilitySumAfterAdjust;
      if (numProbsToUse >= 16)
      {
        TensorPrimitives.Multiply(probs, scaleFactor, probs);
        TensorPrimitives.ConvertToHalf(probs, MemoryMarshal.Cast<FP16, Half>(childrenProbabilites));
      }
      else
      {
        for (int i = 0; i < numProbsToUse; i++)
        {
          childrenProbabilites[i] = new FP16(scaleFactor * probs[i]);
        }
      }
    }
    else
    {
      // No minimum probability enforcement needed
      float scaleFactor = 1.0f / probabilitySumAfterAdjust;

      if (numProbsToUse >= 16)
      {
        TensorPrimitives.Multiply(probs, scaleFactor, probs);
        TensorPrimitives.ConvertToHalf(probs, MemoryMarshal.Cast<FP16, Half>(childrenProbabilites));
      }
      else
      {
        for (int i = 0; i < numProbsToUse; i++)
        {
          childrenProbabilites[i] = new FP16(scaleFactor * probs[i]);
        }
      }
    }
  }


  /// <summary>
  /// Allocates children nodes and initializes them with policy probabilities
  /// taken from a specified CompressedPolicyVector, also applying softmax operation.
  /// </summary>
  /// <param name="policySoftmax"></param>
  /// <param name="minPolicyProbability"></param>
  /// <param name="mgPos"></param>
  /// <param name="moves"></param>
  /// <param name="policyVector"></param>
  [SkipLocalsInit]
  public void SetPolicy(float policySoftmax, float minPolicyProbability,
                        in MGPosition mgPos, MGMoveList moves,
                        in CompressedPolicyVector policyVector,
#if ACTION_ENABLED
                          bool hasAction,
                          in CompressedActionVector actionVector,
#endif

                        bool returnedMovesAreInSameOrderAsMGMoveList)
  {
    // Create new move array which also includes unpacked associated policy vector probabilities
    Span<FP16> childrenProbabilites = stackalloc FP16[moves.NumMovesUsed];

    // Actually unpack
    Span<EncodedMove> validLZMovesUsed = stackalloc EncodedMove[moves.NumMovesUsed];
    int numUsedPolicyMoves = InitializeFromPolicyInfo(policySoftmax, minPolicyProbability,
                                                      in mgPos, in policyVector, moves, childrenProbabilites,
                                                      validLZMovesUsed, returnedMovesAreInSameOrderAsMGMoveList);

    // Allocate children

    //      NodeRef. SetNumPolicyMovesAndAllocateChildInfo(Tree, numUsedPolicyMoves);

    // Finally, set these in the child policy vector
    if (numUsedPolicyMoves > 0)
    {
      ref readonly GNodeStruct thisRef = ref NodeRef;
      Span<GEdgeHeaderStruct> children = AllocatedEdgeHeaders(numUsedPolicyMoves).HeaderStructsSpan;
      //        Span<MoveInfoStruct> children = GraphStore.MoveInfos.SpanAtIndex(thisRef.ChildInfo.ChildInfoStartIndex(GraphStore), thisRef.NumPolicyMoves);
      for (int i = 0; i < numUsedPolicyMoves; i++)
      {
#if ACTION_ENABLED
          FP16 actionV = FP16.NaN;
          FP16 actionU = FP16.NaN;
          if (hasAction)
          {
            actionV = (FP16)((float)actionVector[i].W - (float)actionVector[i].L);
//            actionU = actionVector[i].U;
          }
#endif
        children[i].SetUnexpandedValues(ArrayUtils.GetItem(validLZMovesUsed, i),
                                              ArrayUtils.GetItem(childrenProbabilites, i),
#if ACTION_ENABLED
                                                actionV, actionU);
#else
                                              FP16.NaN, FP16.NaN);
#endif
      }
    }
  }

  #region Nonstandard initialization

  /// <summary>
  /// Initializes policy values randomly.
  /// </summary>
  /// <param name="wide"></param>
  /// <param name="movesMG"></param>
  /// <param name="childrenProbabilites"></param>
  /// <param name="validLZMovesUsed"></param>
  /// <returns></returns>
  static int DoInitializeRandom(bool wide, MGMoveList movesMG, Span<FP16> childrenProbabilites, Span<EncodedMove> validLZMovesUsed)
  {
    Span<MGMove> shuffledMoves = stackalloc MGMove[movesMG.NumMovesUsed];
    new Span<MGMove>(movesMG.MovesArray)[..movesMG.NumMovesUsed].CopyTo(shuffledMoves);

    // Move probabilities must be descending, decay each by some multiplier
    float DECAY_MULTIPLIER = wide ? 0.85f : 0.60f;

    int numMoves = Math.Min(32, movesMG.NumMovesUsed);
    float prob = 1.0f;
    float sumProb = 0.0f;
    for (int i = 0; i < numMoves; i++)
    {
      // Convert move
      validLZMovesUsed[i] = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(shuffledMoves[i]);

      // Set probabilities in some reasonable range, here unnormalized values are uniformly integers over [0, 20]
      childrenProbabilites[i] = prob < 0.001 ? 0 : (FP16)prob;
      sumProb += prob;

      prob *= DECAY_MULTIPLIER;
    }

    // Normalize to sum to 1.0
    for (int i = 0; i < numMoves; i++)
    {
      childrenProbabilites[i] = (FP16)(childrenProbabilites[i] / sumProb);
    }

    return numMoves;
  }


  /// <summary>
  /// Initializes policy to uniform distribution (for the rare case of no valid policy information).
  /// </summary>
  /// <param name="movesMG"></param>
  /// <param name="childrenProbabilites"></param>
  /// <param name="validLZMovesUsed"></param>
  /// <param name="maxMovesToReturn"></param>
  /// <returns></returns>
  private static int PoliciesInitializedUniform(MGMoveList movesMG, Span<FP16> childrenProbabilites,
                                                Span<EncodedMove> validLZMovesUsed, int maxMovesToReturn)
  {
    // TODO: below we arbitrarily omit any moves beyond 63rd; is there a better way to sort them based on probability?
    int numMovesToSave = Math.Min(Math.Min(childrenProbabilites.Length, movesMG.NumMovesUsed), maxMovesToReturn);
    FP16 uniformProbability = (FP16)(1.0f / numMovesToSave);
    for (int i = 0; i < numMovesToSave; i++)
    {
      childrenProbabilites[i] = uniformProbability;

      if (!validLZMovesUsed.IsEmpty)
      {
        validLZMovesUsed[i] = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(movesMG.MovesArray[i]);
      }
    }
    return numMovesToSave;
  }

  #endregion
}
