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

using Ceres.Chess;
using Ceres.Base.DataTypes;

using Ceres.Chess.MoveGen;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.MCTS.MTCSNodes.Struct;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.MCTS.MTCSNodes
{
  /// <summary>
  /// Methods at MCTSNode involved with initializing policy vectors.
  /// </summary>
  public unsafe partial struct MCTSNodeInfo
  {
    [ThreadStatic]
    static (EncodedMove move, float probability)[] movesFromPolicyTempBuffer;

    /// <summary>
    ///
    /// </summary>
    /// <param name="posMG"></param>
    /// <param name="policyVector"></param>
    /// <param name="movesMG">Will return sorted in the same way as the legal policy move</param>
    /// <param name="childrenProbabilites"></param>
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
        childrenProbabilites[0] = (FP16)1.0f;
        if (validLZMovesUsed != null)
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
      // Get local copy of the ThreadStatic buffer (for efficiency)
      if (movesFromPolicyTempBuffer == null) movesFromPolicyTempBuffer = new (EncodedMove, float)[64];
      (EncodedMove move, float probability)[] movesFromPolicyTempBufferLocal = movesFromPolicyTempBuffer;

      // Ignore moves with extremely small probabilities
      // They will be too small to be ever selected, and often may actually be illegal moves
      const float THRESHOLD_IGNORE_MOVE = 0.00005f;

      // Although we retrieve up to 64 moves from neural netowrk, 
      // the encoding used in the raw search node is compressed such that 
      // it can represent values only in the range [0..63].
      const int MAX_MOVES_RETAIN = 63;
      int policyMoveCount = policyVector.ProbabilitySummaryList(movesFromPolicyTempBufferLocal, THRESHOLD_IGNORE_MOVE, topN:MAX_MOVES_RETAIN);

      // Special processing if random values requested
      if (movesFromPolicyTempBuffer[0].move.RawValue == CompressedPolicyVector.SPECIAL_VALUE_RANDOM_NARROW)
      {
        return DoInitializeRandom(false, movesMG, childrenProbabilites, validLZMovesUsed);
      }
      else if (movesFromPolicyTempBuffer[0].move.RawValue == CompressedPolicyVector.SPECIAL_VALUE_RANDOM_WIDE)
      {
        return DoInitializeRandom(true, movesMG, childrenProbabilites, validLZMovesUsed);
      }

      Span<float> probabilitiesTemp = stackalloc float[policyMoveCount];

      float probabilitySumBeforeAdjust = 0;
      int countPolicyMovesProcessed = 0;

      // For efficiency, get a move array as direct variable
      MGMove[] legalMoveArray = movesMG.MovesArray;

      bool blackToMove = posMG.SideToMove == SideType.Black;
      float lastP = float.NaN;
      for (int policyMoveIndex = 0; policyMoveIndex < policyMoveCount; policyMoveIndex++)
      {
        (EncodedMove thisPolicyMove, float thisPolicyProb) = movesFromPolicyTempBufferLocal[policyMoveIndex];

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
            MGMove temp = legalMoveArray[indexLegalMove];
            legalMoveArray[indexLegalMove] = legalMoveArray[countPolicyMovesProcessed];
            legalMoveArray[countPolicyMovesProcessed] = temp;
          }

          if (validLZMovesUsed != null)
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
        float scaleProbabilityFactor = 1.0f / probabilitySumBeforeAdjust;
        for (int policyMoveIndex = 0; policyMoveIndex < numProbsToUse; policyMoveIndex++)
        {
          childrenProbabilites[policyMoveIndex] = new FP16(probabilitiesTemp[policyMoveIndex] * scaleProbabilityFactor);
        }
      }
      else
      {
        // Apply softmax operator
        float probabilitySumAfterAdjust = 0;
        float policySoftmaxReciprocal = 1.0f / policySoftmax;
        for (int policyMoveIndex = 0; policyMoveIndex < numProbsToUse; policyMoveIndex++)
        {
          float thisProb = probabilitiesTemp[policyMoveIndex];

          float thisProbAdj = MathF.Pow(thisProb, policySoftmaxReciprocal);
          probabilitySumAfterAdjust += thisProbAdj;
          probabilitiesTemp[policyMoveIndex] = thisProbAdj;
        }

        // Enforce minimum probability (if any)
        probabilitySumAfterAdjust = PossiblyEnforceMinProbability(minPolicyProbability, probabilitiesTemp, numProbsToUse, probabilitySumAfterAdjust);

        // Scale to 100%
        // Note that we do the softmax above first
        // In the case dropped (illegal moves) this produces less sharp distribution,
        // which is desirable given the NN output was flawed and therefore there is more uncertainty around the policy estimates
        float scaleProbabilityFactor = 1.0f / probabilitySumAfterAdjust;
        for (int policyMoveIndex = 0; policyMoveIndex < numProbsToUse; policyMoveIndex++)
        {
          childrenProbabilites[policyMoveIndex] = new FP16(scaleProbabilityFactor * probabilitiesTemp[policyMoveIndex]);
        }
      }

      return numProbsToUse;
    }


    /// <summary>
    /// Iterates over all moves and possibly increases their probability
    /// if below some specified minimum value.
    /// </summary>
    /// <param name="minPolicyProbability"></param>
    /// <param name="probabilitiesTemp"></param>
    /// <param name="numProbsToUse"></param>
    /// <param name="probabilitySumAfterAdjust"></param>
    /// <returns></returns>
    private static float PossiblyEnforceMinProbability(float minPolicyProbability, Span<float> probabilitiesTemp, int numProbsToUse, float probabilitySumAfterAdjust)
    {
      if (minPolicyProbability > 0)
      {
        float minProbability = minPolicyProbability * probabilitySumAfterAdjust;
        for (int policyMoveIndex = 0; policyMoveIndex < numProbsToUse; policyMoveIndex++)
        {
          float thisProbability = probabilitiesTemp[policyMoveIndex];
          if (thisProbability < minProbability)
          {
            // Set a value close to the minimum probability, 
            // but retain the ordering by adding back a small fraction of original probability
            const float FRAC_RETAIN_PROBABILITY = 0.1f;
            float replacementProbability = minPolicyProbability + FRAC_RETAIN_PROBABILITY * thisProbability;

            // Make sure to preserve order with prior entries
            // (that the addition of the fraction did not change the order)
            if (policyMoveIndex > 0)
            {
              float priorProbability = probabilitiesTemp[policyMoveIndex - 1];
              if (replacementProbability > priorProbability)
                replacementProbability = priorProbability;
            }

            float probabilityToAdd = replacementProbability - thisProbability;

            probabilitySumAfterAdjust += probabilityToAdd;
            probabilitiesTemp[policyMoveIndex] = replacementProbability;
          }
        }
      }

      return probabilitySumAfterAdjust;
    }


    /// <summary>
    /// Allocates children nodes and initialzes them with policy probabilities
    /// taken from a speciifed CompressedPolicyVector, also applying softmax operation.
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
      Ref.SetNumPolicyMovesAndAllocateChildInfo(Tree, numUsedPolicyMoves);

      if (numUsedPolicyMoves > 0)
      {
        Span<MCTSNodeStructChild> children = Tree.Store.Children.SpanForNode(in this.Ref);

        // Finally, set these in the child policy vector
        for (int i = 0; i < numUsedPolicyMoves; i++)
        {
          children[i].SetUnexpandedPolicyValues(validLZMovesUsed[i], childrenProbabilites[i]);
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
      new Span<MGMove>(movesMG.MovesArray).Slice(0, movesMG.NumMovesUsed).CopyTo(shuffledMoves);

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
        if (prob < 0.001)
          childrenProbabilites[i] = 0;
        else
          childrenProbabilites[i] = new FP16(prob);
        sumProb += prob;

        prob *= DECAY_MULTIPLIER;
      }

      // Normalize to sum to 1.0
      for (int i = 0; i < numMoves; i++)
        childrenProbabilites[i] = new FP16(childrenProbabilites[i] / sumProb);

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

        if (validLZMovesUsed != null)
          validLZMovesUsed[i] = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(movesMG.MovesArray[i]);
      }
      return numMovesToSave;
    }

    #endregion
  }
}
